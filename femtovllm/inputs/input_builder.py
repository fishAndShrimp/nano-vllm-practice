from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from femtovllm.protocol import SamplingParams


class InputBuilder:
    """ """

    def __init__(self, weights_dir: str | Path):
        """ """
        weights_dir = Path(weights_dir).resolve()
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            weights_dir
        )

        eos_token_ids = self._resolve_eos_from_config(
            self.tokenizer.eos_token_id,
        )

        for token_str in [
            "<|endoftext|>",
            "<|im_end|>",
        ]:
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            if token_id is not None:
                if token_id != self.tokenizer.unk_token_id:
                    eos_token_ids.append(token_id)

        self.eos_token_ids = list(set(eos_token_ids))

    def _resolve_eos_from_config(
        self,
        eos_from_config: Optional[int | list[int]],
    ) -> list[int]:
        if eos_from_config is None:
            return []
        if isinstance(eos_from_config, int):
            return [eos_from_config]
        if isinstance(eos_from_config, list):
            return [int(x) for x in eos_from_config]

        raise TypeError(f"{eos_from_config=}")

    def build(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """ """
        if not isinstance(prompt, str):
            raise TypeError(f"{type(prompt)=}")

        if sampling_params is None:
            sampling_params = SamplingParams()
        else:
            sampling_params = sampling_params.clone()

        stop_token_ids = sampling_params.stop_token_ids + self.eos_token_ids
        stop_token_ids = list(set(stop_token_ids))
        sampling_params.stop_token_ids = stop_token_ids

        return (
            self.tokenizer.encode(prompt),
            sampling_params,
        )
