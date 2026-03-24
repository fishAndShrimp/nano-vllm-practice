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

        self._texts = {}

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: Optional[SamplingParams] = None,
        keep_prompts: bool = True,
        stream: bool = True,
    ):
        """ """
        self.clear()
        req_ids = self.enqueue(
            prompts=prompts,
            sampling_params=sampling_params,
            keep_prompts=keep_prompts,
        )

        generator = self.stream_outputs()
        if stream:
            return generator, req_ids

        for _ in generator:
            pass
        return self._texts

    def clear(self):
        """ """
        self._texts.clear()

    def enqueue(
        self,
        prompts: str | list[str],
        sampling_params: Optional[SamplingParams] = None,
        keep_prompts: bool = True,
    ):
        """ """
        if isinstance(prompts, str):
            prompts = [
                prompts,
            ]

        req_ids = []
        for prompt in prompts:
            req_id = "req_{}".format(
                next(self._request_counter),
            )
            req_ids.append(req_id)
            self._texts[req_id] = prompt if keep_prompts else ""

            token_ids, new_sampling_params = self.input_builder.build(
                prompt, sampling_params
            )

            self.core_engine.add_request(
                req_id=req_id,
                token_ids=token_ids,
                sampling_params=new_sampling_params,
            )

        return req_ids

    def stream_outputs(self):
        """ """
        while self.core_engine.has_unfinished_requests():
            step_deltas = self.core_engine.step()

            text_deltas = []
            for step_delta in step_deltas:
                if step_delta.new_token_id is None:
                    continue

                token_str = self.input_builder.tokenizer.decode(step_delta.new_token_id)

                text_deltas.append(
                    (step_delta.req_id, token_str),
                )
                self._texts[step_delta.req_id] += token_str

            if len(text_deltas) > 0:
                yield text_deltas
