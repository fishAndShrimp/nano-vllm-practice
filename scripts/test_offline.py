from pathlib import Path

import torch
from transformers import Qwen3Config

from femtovllm import LLM, SamplingParams

weights_dir = (
    #####
    Path(__file__).resolve().parent.parent
    / "femtovllm"
    / "weights"
    / "qwen3_0.6b_weights"
)


llm = LLM(
    max_seqs=10,
    max_tokens=1000,
    max_tokens_per_seq=100,
    num_blocks=200,
    block_size=100,
    hf_config=Qwen3Config.from_pretrained(weights_dir),
    weights_dir=weights_dir,
    dtype=torch.bfloat16,
)


print(
    llm.generate(
        [
            "The capital of France is",
            "The capital city of England is",
            "The capital city of the United Kingdom is",
        ],
        sampling_params=SamplingParams(
            temperature=0,
            max_new_tokens=1,
        ),
        stream=False,
    )
)
