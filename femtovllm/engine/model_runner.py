from pathlib import Path

import torch
from transformers import Qwen3Config

from femtovllm.engine.sampler import Sampler
from femtovllm.engine.sequence import Sequence
from femtovllm.models import QwenForCausalLM
from femtovllm.protocol import VarlenAttnMetadata


class ModelRunner:
    """
    Currently support:
    - Qwen3
    """

    def __init__(
        self,
        hf_config: Qwen3Config,
        weights_dir: Path,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        """ """
        self.hf_config = hf_config
        self.weights_dir = weights_dir

        self.d_model = hf_config.hidden_size

        self.dtype = dtype
        self.device = device

        self.model = QwenForCausalLM(hf_config)
        self.model.load_weights(self.weights_dir)

        self.model.to(
            dtype=self.dtype,
            device=self.device,
        )
        self.model.eval()

        self.sampler = Sampler()

    def pad_block_tables(
        self,
        raw_block_tables: list[list[int]],
    ):
        """ """
        if not raw_block_tables:
            return None

        max_blocks = max(
            #####
            len(x)
            for x in raw_block_tables
        )

        # pad -1 rather than 0
        block_tables = []
        for raw_table in raw_block_tables:
            block_tables.append(
                #####
                raw_table + [-1] * (max_blocks - len(raw_table))
            )

        return torch.tensor(
            block_tables,
            dtype=torch.int32,
            device=self.device,
        )

    @torch.inference_mode()
    def step(
        self,
        scheduled_const: list[tuple[Sequence, int]],
        k_cache_pools: list[torch.Tensor],
        v_cache_pools: list[torch.Tensor],
        raw_block_tables: list[list[int]],
    ):
        """ """
        if not scheduled_const:
            return []

        ##########
        ##### route seqs
        ##########
        prefill_scheduled = []
        prefill_indices = []
        prefill_tables = []

        decode_scheduled = []
        decode_indices = []
        decode_tables = []

        for i, (seq_bundle, block_table) in enumerate(
            zip(scheduled_const, raw_block_tables),
        ):
            seq, _ = seq_bundle

            if seq.is_prefilling:
                prefill_scheduled.append(seq_bundle)
                prefill_indices.append(i)
                prefill_tables.append(block_table)
            else:
                decode_scheduled.append(seq_bundle)
                decode_indices.append(i)
                decode_tables.append(block_table)

        ##########
        ##### calc routed hidden
        ##########
        B = len(scheduled_const)
        C = self.d_model
        hidden = torch.empty(
            (B, C),
            dtype=self.dtype,
            device=self.device,
        )

        if prefill_scheduled:
            hidden[prefill_indices] = self._route_prefill(
                prefill_scheduled,
                k_cache_pools,
                v_cache_pools,
                prefill_tables,
            )

        if decode_scheduled:
            hidden[decode_indices] = self._route_decode(
                decode_scheduled,
                k_cache_pools,
                v_cache_pools,
                decode_tables,
            )

        ##########
        ##### hidden
        ##### => logits
        ##### => token_ids
        ##########
        # (B, C)
        # (B, vocab_size)
        logits_next = self.model.lm_head(hidden)

        # (B,)
        token_ids_next: list[int] = self.sampler(
            logits_next,
            scheduled_const,
        )
        return token_ids_next

    def _route_prefill(
        self,
        scheduled_const: list[tuple[Sequence, int]],
        k_cache_pools: list[torch.Tensor],
        v_cache_pools: list[torch.Tensor],
        raw_block_tables: list[list[int]],
    ):
        """ """
        if not scheduled_const:
            raise RuntimeError("Empty input")

        flatten = []
        raw_positions = []
        raw_cu_seqlens = [0]
        raw_kv_lens = []

        max_q_len = -1
        max_kv_len = -1
        for seq_const, num_tokens in scheduled_const:
            pos = seq_const.num_computed_tokens
            kv_len = pos + num_tokens

            flatten.extend(seq_const.token_ids[pos:kv_len])
            raw_positions.extend(range(pos, kv_len))
            raw_cu_seqlens.append(raw_cu_seqlens[-1] + num_tokens)
            raw_kv_lens.append(kv_len)

            max_q_len = max(max_q_len, num_tokens)
            max_kv_len = max(max_kv_len, kv_len)

        flatten = torch.tensor(
            flatten,
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.tensor(
            raw_positions,
            dtype=torch.int32,
            device=self.device,
        )
        cu_seqlens = torch.tensor(
            raw_cu_seqlens,
            dtype=torch.int32,
            device=self.device,
        )
        kv_lens = torch.tensor(
            raw_kv_lens,
            dtype=torch.int32,
            device=self.device,
        )

        # (B, C)
        hidden = self.model.forward_varlen(
            idx_flatten=flatten,
            varlen_attn_metadata=VarlenAttnMetadata(
                positions=positions,
                raw_positions=raw_positions,
                cu_seqlens=cu_seqlens,
                raw_cu_seqlens=raw_cu_seqlens,
                max_q_len=max_q_len,
                k_cache_pools=k_cache_pools,
                v_cache_pools=v_cache_pools,
                kv_lens=kv_lens,
                max_kv_len=max_kv_len,
                block_tables=self.pad_block_tables(raw_block_tables),
                raw_block_tables=raw_block_tables,
                is_decoding=False,
            ),
            skip_lm_head=True,
        )

        return hidden

    def _route_decode(
        self,
        scheduled_const: list[tuple[Sequence, int]],
        k_cache_pools: list[torch.Tensor],
        v_cache_pools: list[torch.Tensor],
        raw_block_tables: list[list[int]],
    ):
        """ """
        if not scheduled_const:
            raise RuntimeError("Empty input")

        flatten = []
        raw_positions = []
        raw_kv_lens = []

        max_kv_len = -1
        for seq_const, _ in scheduled_const:
            pos = seq_const.num_computed_tokens
            kv_len = pos + 1

            flatten.append(seq_const.token_ids[pos])
            raw_positions.append(pos)
            raw_kv_lens.append(kv_len)

            max_kv_len = max(max_kv_len, kv_len)

        flatten = torch.tensor(
            flatten,
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.tensor(
            raw_positions,
            dtype=torch.int32,
            device=self.device,
        )
        kv_lens = torch.tensor(
            raw_kv_lens,
            dtype=torch.int32,
            device=self.device,
        )

        # (B, C)
        hidden = self.model.forward_varlen(
            idx_flatten=flatten,
            varlen_attn_metadata=VarlenAttnMetadata(
                positions=positions,
                raw_positions=raw_positions,
                cu_seqlens=None,
                raw_cu_seqlens=None,
                max_q_len=1,
                k_cache_pools=k_cache_pools,
                v_cache_pools=v_cache_pools,
                kv_lens=kv_lens,
                max_kv_len=max_kv_len,
                block_tables=self.pad_block_tables(raw_block_tables),
                raw_block_tables=raw_block_tables,
                is_decoding=True,
            ),
            skip_lm_head=True,
        )

        return hidden
