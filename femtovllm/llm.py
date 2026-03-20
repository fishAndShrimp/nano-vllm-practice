import itertools
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config

from femtovllm.engine.core_engine import CoreEngine


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
        #####
        # [PARSE: user input]
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
        # [PARSE: user input]
        #####

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            weights_dir
        )

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
    ):
        """ """
        if isinstance(prompts, str):
            prompts = [
                prompts,
            ]

        for prompt in prompts:
            req_id = "req_{}".format(
                next(self._request_counter),
            )
            token_ids = self.tokenizer.encode(prompt)

            self.core_engine.add_request(req_id, token_ids)

        # TODO: offline decoding
