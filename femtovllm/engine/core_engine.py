from pathlib import Path
from typing import Optional

import torch
from transformers import Qwen3Config

from femtovllm.engine.kv_cache_manager import KVCacheManager
from femtovllm.engine.model_runner import ModelRunner
from femtovllm.engine.request_queue import RequestQueue
from femtovllm.engine.scheduler import Scheduler
from femtovllm.engine.sequence import Sequence
from femtovllm.engine.step_budget import StepBudget
from femtovllm.protocol import SamplingParams, StepDelta


class CoreEngine:
    """ """

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
        device: Optional[str] = None,
    ):
        """ """
        #####
        # [PARSE: user config]
        # [TODO: class EngineConfig]
        #####
        max_seqs = int(max_seqs)
        max_tokens = int(max_tokens)
        max_tokens_per_seq = int(max_tokens_per_seq)

        num_blocks = int(num_blocks)
        block_size = int(block_size)

        if not isinstance(hf_config, Qwen3Config):
            raise TypeError(f"{type(hf_config)=}")

        weights_dir = Path(weights_dir).resolve()

        if dtype is not None:
            if not isinstance(dtype, (str, torch.dtype)):
                raise TypeError(f"{type(dtype)=}")
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

        if device is not None:
            if not isinstance(device, str):
                raise TypeError(f"{type(device)=}")
        #####
        # [PARSE: user config]
        #####

        kv_cache_manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )

        self.scheduler = Scheduler(
            step_budget=StepBudget(
                max_seqs=max_seqs,
                max_tokens=max_tokens,
                max_tokens_per_seq=max_tokens_per_seq,
            ),
            request_queue=RequestQueue(),
            kv_cache_manager=kv_cache_manager,
        )

        # [STEP: init model and move weights]
        self.model_runner = ModelRunner(
            hf_config=hf_config,
            weights_dir=weights_dir,
            kv_cache_manager=kv_cache_manager,
            dtype=dtype,
            device=device,
        )

        # [STEP: clean cache for further kv_cache]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # [STEP: gen huge kv_cache tensor]
        # TODO

    def step(self):
        """ """
        scheduled, aborted = self.scheduler.step()
        token_ids_next = self.model_runner.step(
            scheduled_const=scheduled,
        )
        step_deltas: list[StepDelta] = []

        # [STEP: scheduled]
        if len(scheduled) != len(token_ids_next):
            raise RuntimeError(f"{scheduled=} {token_ids_next=}")

        for i, (seq, num_step_tokens) in enumerate(scheduled):
            seq.num_computed_tokens += num_step_tokens
            if seq.is_prefilling:
                continue

            token_id = token_ids_next[i]
            seq.append(token_id)

            if token_id in seq.stop_token_ids_set:
                self.scheduler.free_and_finish(seq, "EOS")
            elif seq.num_new_tokens >= seq.sampling_params.max_new_tokens:
                self.scheduler.free_and_finish(seq, "LENGTH")

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
        req_id: str,
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
