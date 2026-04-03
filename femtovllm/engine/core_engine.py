from pathlib import Path
from typing import Optional

import torch
from transformers import Qwen3Config

import femtovllm.ops
from femtovllm.engine.model_runner import ModelRunner
from femtovllm.engine.scheduler import Scheduler, SchedulerV3
from femtovllm.engine.sequence import Sequence
from femtovllm.protocol import ReqId, SamplingParams, StepDelta, StopReason


class CoreEngine:
    """ """

    @classmethod
    def calc_max_kv_len_non_split(cls) -> int:
        """ """
        major, minor = torch.cuda.get_device_capability()

        # sizeof(float) * 1024 * 6 = 24KB
        if major <= 7:
            return 1024 * 6

        # sizeof(float) * 1024 * 8 = 32KB
        return femtovllm.ops.MAX_KV_LEN_NON_SPLIT

    def __init__(
        self,
        max_seqs: int,
        max_tokens: int,
        max_tokens_per_seq: int,
        num_blocks: int,
        block_size: int,
        hf_config: Qwen3Config,
        weights_dir: Path,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ):
        """ """
        ##########
        ##### [PARSE: user config]
        ##### [TODO: class EngineConfig]
        ##########
        max_seqs = int(max_seqs)
        max_tokens = int(max_tokens)
        max_tokens_per_seq = int(max_tokens_per_seq)

        ##### [STEP: block]
        num_blocks = int(num_blocks)

        # static block_size
        block_size = int(block_size)
        if block_size != femtovllm.ops.TILE_SIZE:
            raise NotImplementedError(
                f"Dynamic block sizes are not yet supported (got {block_size}). "
                f"Please set block_size={femtovllm.ops.TILE_SIZE} "
                "to match the hardware-aligned tile size of the custom GEMM/GEMV kernels."
            )
        ##### [STEP: block]

        if not isinstance(hf_config, Qwen3Config):
            raise TypeError(f"{type(hf_config)=}")

        weights_dir = Path(weights_dir).resolve()

        ##### [STEP: dtype]
        dtype = torch.bfloat16 if (dtype is None) else dtype

        # cast to torch.dtype
        if isinstance(dtype, str):
            dtype = {
                "half": torch.half,
                #####
                "fp16": torch.float16,
                "float16": torch.float16,
                #####
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
            }[dtype.strip().casefold()]
        elif isinstance(dtype, torch.dtype):
            pass
        else:
            raise TypeError(f"{type(dtype)=}")

        # invalid value
        if dtype == torch.float16:
            raise RuntimeError(f"{dtype=} possibly leads to weights overflow")

        self.dtype = dtype
        ##### [STEP: dtype]

        ##### [STEP: device]
        device = "cuda" if (device is None) else device

        # invalid type
        if not isinstance(device, (str, torch.device)):
            raise TypeError(f"{type(device)=}")

        self.device = device
        ##### [STEP: device]

        ##########
        ##### [PARSE: user config]
        ##########

        # [STEP: scheduler]
        max_kv_len_non_split = self.calc_max_kv_len_non_split()

        self.scheduler = {
            1: Scheduler,
            3: SchedulerV3,
        }[femtovllm._DEV.scheduler_version](
            max_seqs=max_seqs,
            max_tokens=max_tokens,
            max_tokens_per_seq=max_tokens_per_seq,
            num_blocks=num_blocks,
            block_size=block_size,
            max_kv_len_non_split=max_kv_len_non_split,
        )

        # [STEP: init model and move weights]
        self.model_runner = ModelRunner(
            hf_config=hf_config,
            weights_dir=weights_dir,
            dtype=dtype,
            device=device,
        )

        # [STEP: clean cache for further kv_cache]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ##### [STEP: gen huge kv_cache tensors as pools]
        n_kv_heads = hf_config.num_key_value_heads
        d_head = hf_config.head_dim
        n_layers = hf_config.num_hidden_layers

        kv_cache_pool_shape = (
            num_blocks,
            n_kv_heads,
            block_size,
            d_head,
        )

        self.k_cache_pools: list[torch.Tensor] = [
            torch.empty(
                kv_cache_pool_shape,
                dtype=dtype,
                device=device,
            )
            for _ in range(n_layers)
        ]
        self.v_cache_pools: list[torch.Tensor] = [
            torch.empty(
                kv_cache_pool_shape,
                dtype=dtype,
                device=device,
            )
            for _ in range(n_layers)
        ]
        ##### [STEP: gen huge kv_cache tensors as pools]

    def step(self):
        """ """
        scheduled, aborted = self.scheduler.step()
        token_ids_next = self.model_runner.step(
            scheduled_const=scheduled,
            k_cache_pools=self.k_cache_pools,
            v_cache_pools=self.v_cache_pools,
            raw_block_tables=[
                #####
                self.scheduler.get_block_table(x)
                for x, _ in scheduled
            ],
        )
        step_deltas: list[StepDelta] = []

        # [STEP: scheduled]
        if len(scheduled) != len(token_ids_next):
            raise RuntimeError(f"{scheduled=} {token_ids_next=}")

        for i, (seq, num_step_tokens) in enumerate(scheduled):
            ##############################
            ##### [STEP: Post-Forward State Update & Commit]
            ##### The transformer has successfully computed KV caches for `num_step_tokens`.
            ##### 1. Advance the logical computed length.
            ##### 2. Commit these newly computed blocks to the global Prefix Tree.
            ##### This critical timepoint ensures the blocks are immediately
            ##### available for cross-sequence sharing and scheduler prioritization.
            ##############################
            seq.num_computed_tokens += num_step_tokens
            self.scheduler.commit_blocks(seq)

            ##############################
            ##### [STEP: Chunked Prefill Check]
            ##### If the sequence is still in the prefill phase (e.g., prompt is
            ##### split into multiple chunks), no new token is generated yet.
            ##############################
            if seq.is_prefilling:
                continue

            ##############################
            ##### [STEP: Append Predicted Token]
            ##### Append the newly decoded token.
            ##### State Transition: The sequence now has 1 uncomputed token again,
            ##### making it ready for the next decode step.
            ##############################
            token_id = token_ids_next[i]
            seq.append(token_id)

            ##############################
            ##### [STEP: Check Termination Conditions]
            ##### Free resources immediately if the sequence hits EOS or length limits.
            ##############################
            if token_id in seq.stop_token_ids_set:
                self.scheduler.free_and_finish(seq, StopReason.EOS)
            elif seq.num_new_tokens >= seq.sampling_params.max_new_tokens:
                self.scheduler.free_and_finish(seq, StopReason.LENGTH)

            ##############################
            ##### [STEP: Construct Stream Output]
            ##### Record the state delta for the client response stream.
            ##############################
            step_deltas.append(
                StepDelta(
                    req_id=seq.req_id,
                    seq_id=seq.seq_id,
                    new_token_id=token_id,
                    stop_reason=seq.stop_reason,
                )
            )

        # [STEP: aborted]
        step_deltas.extend(
            StepDelta(
                req_id=x.req_id,
                seq_id=x.seq_id,
                new_token_id=None,
                stop_reason=x.stop_reason,
            )
            for x in aborted
        )

        return step_deltas

    def has_unfinished_requests(self):
        return self.scheduler.has_unfinished_sequences()

    def add_request(
        self,
        req_id: ReqId,
        token_ids: list[int],
        sampling_params: SamplingParams,
    ):
        """ """
        # TODO: SequenceGroup
        self.scheduler.add_sequence(
            Sequence(
                req_id=req_id,
                seq_id=f"{req_id}_0",
                token_ids=token_ids,
                sampling_params=sampling_params,
            )
        )
