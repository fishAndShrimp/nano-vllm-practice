from pathlib import Path
from typing import Optional

import torch
from transformers import Qwen3Config

from femtovllm.engine.kv_cache_manager import KVCacheManager
from femtovllm.engine.sampler import Sampler
from femtovllm.engine.sequence import Sequence
from femtovllm.models import QwenForCausalLM


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
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ):
        """ """
        self.hf_config = hf_config
        self.weights_dir = weights_dir
        self.kv_cache_manager = kv_cache_manager
        self.dtype = torch.float16 if (dtype is None) else dtype
        self.device = "cuda" if (device is None) else device

        self.model = QwenForCausalLM(hf_config)
        self.model.load_weights(self.weights_dir)

        self.model.to(
            dtype=self.dtype,
            device=self.device,
        )
        self.model.eval()

        self.sampler = Sampler()

    @torch.inference_mode()
    def step(
        self,
        scheduled_const: list[tuple[Sequence, int]],
    ):
        """ """
        flatten = []
        positions = []
        cu_seqlens = [0]

        for seq_const, num_tokens in scheduled_const:
            i_pos = seq_const.num_computed_tokens

            flatten.extend(seq_const.token_ids[i_pos : i_pos + num_tokens])
            positions.extend(range(i_pos, i_pos + num_tokens))
            cu_seqlens.append(cu_seqlens[-1] + num_tokens)

        flatten = torch.tensor(
            flatten,
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.tensor(
            positions,
            dtype=torch.long,
            device=self.device,
        )
        cu_seqlens = torch.tensor(
            cu_seqlens,
            dtype=torch.int32,
            device=self.device,
        )

        # (B, vocab_size)
        logits_next = self.model.forward_varlen(flatten, positions, cu_seqlens)

        # (B,)
        token_ids_next: list[int] = self.sampler(
            logits_next,
            scheduled_const,
        )

        return token_ids_next
