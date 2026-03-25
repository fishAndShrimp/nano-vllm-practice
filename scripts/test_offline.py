from pathlib import Path

import torch
from transformers import Qwen3Config

import femtovllm
from femtovllm import LLM, SamplingParams

# -----
TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""
# -----


weights_dir = (
    #####
    Path(__file__).resolve().parent.parent
    / "femtovllm"
    / "weights"
    / "qwen3_0.6b_weights"
)


femtovllm._DEV.varlen_attn_impl = "pytorch"
llm = LLM(
    max_seqs=10,
    max_tokens=1000,
    max_tokens_per_seq=100,
    num_blocks=200,
    block_size=64,
    hf_config=Qwen3Config.from_pretrained(weights_dir),
    weights_dir=weights_dir,
    dtype=torch.bfloat16,
)


print(
    llm.generate(
        [
            # TEMPLATE.format(x)
            x
            for x in (
                "The capital of France is",
                "The capital city of England is",
                "The capital city of the United Kingdom is",
            )
        ],
        sampling_params=SamplingParams(
            temperature=0,
            presence_penalty=1,
            max_new_tokens=5,
        ),
        stream=False,
    )
)
