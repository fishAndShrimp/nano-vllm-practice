from pathlib import Path

import torch
from transformers import Qwen3Config

from femtovllm.engine.kv_cache_manager import KVCacheManager
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
        kv_cache_manager: KVCacheManager,
        dtype: torch.dtype,
        device: str,
    ):
        """ """
        self.hf_config = hf_config
        self.weights_dir = weights_dir
        self.kv_cache_manager = kv_cache_manager
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
        if len(scheduled_const) <= 0:
            return []

        flatten = []
        positions = []
        raw_cu_seqlens = [0]

        q_len_max = -1
        for seq_const, num_tokens in scheduled_const:
            i_pos = seq_const.num_computed_tokens

            flatten.extend(seq_const.token_ids[i_pos : i_pos + num_tokens])
            positions.extend(range(i_pos, i_pos + num_tokens))
            raw_cu_seqlens.append(raw_cu_seqlens[-1] + num_tokens)

            q_len_max = max(q_len_max, num_tokens)

        flatten = torch.tensor(
            flatten,
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.tensor(
            positions,
            dtype=torch.int32,
            device=self.device,
        )
        cu_seqlens = torch.tensor(
            raw_cu_seqlens,
            dtype=torch.int32,
            device=self.device,
        )

        # (B, vocab_size)
        logits_next = self.model.forward_varlen(
            idx_flatten=flatten,
            varlen_attn_metadata=VarlenAttnMetadata(
                positions=positions,
                cu_seqlens=cu_seqlens,
                raw_cu_seqlens=raw_cu_seqlens,
                q_len_max=q_len_max,
                k_cache_pools=k_cache_pools,
                v_cache_pools=v_cache_pools,
                block_tables=self.pad_block_tables(raw_block_tables),
                raw_block_tables=raw_block_tables,
            ),
        )

        # (B,)
        token_ids_next: list[int] = self.sampler(
            logits_next,
            scheduled_const,
        )

        return token_ids_next
