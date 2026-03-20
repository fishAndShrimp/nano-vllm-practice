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

    def add_request(
        self,
        req_id: str,
        token_ids: list[int],
    ):
        """ """
        seq_id = f"{req_id}_0"
        self.scheduler.add_sequence(
            Sequence(seq_id, token_ids),
        )
