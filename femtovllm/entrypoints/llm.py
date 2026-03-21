import itertools
from pathlib import Path
from typing import Optional

import torch
from transformers import Qwen3Config

from femtovllm.engine.core_engine import CoreEngine
from femtovllm.inputs.input_builder import InputBuilder
from femtovllm.protocol import SamplingParams


class LLM:
    """ """

    def __init__(
        self,
        max_seqs: int,
        max_tokens: int,
        max_tokens_per_seq: int,
        num_blocks: int,
        block_size: int,
        hf_config: Qwen3Config,
        weights_dir: str | Path,
        dtype: Optional[str | torch.dtype] = None,
        device: Optional[str] = None,
    ):
        """ """
        self.input_builder = InputBuilder(weights_dir)

        self.core_engine = CoreEngine(
            max_seqs=max_seqs,
            max_tokens=max_tokens,
            max_tokens_per_seq=max_tokens_per_seq,
            num_blocks=num_blocks,
            block_size=block_size,
            hf_config=hf_config,
            weights_dir=weights_dir,
            dtype=dtype,
            device=device,
        )

        self._request_counter = itertools.count()

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: Optional[SamplingParams] = None,
    ):
        """ """
        if isinstance(prompts, str):
            prompts = [
                prompts,
            ]

        for prompt in prompts:
            token_ids, new_sampling_params = self.input_builder.build(
                prompt, sampling_params
            )

            self.core_engine.add_request(
                req_id="req_{}".format(
                    next(self._request_counter),
                ),
                token_ids=token_ids,
                sampling_params=new_sampling_params,
            )
