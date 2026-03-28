from pathlib import Path

import torch
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from transformers import Qwen3Config

import femtovllm
from femtovllm import LLM, SamplingParams

torch.cuda.set_sync_debug_mode("warn")


# ==========================================
# Inputs & Templates
# Define the chat template and prepare the batch of prompts
# ==========================================
TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""


prompts = [
    TEMPLATE.format(x)
    for x in (
        "The capital of France is",
        "The capital city of England is",
        "The capital city of the United Kingdom is",
    )
]


# ==========================================
# Engine Initialization & Generation Setup
# Load model weights, configure memory blocks, and start the generation task
# ==========================================
weights_dir = (
    #####
    Path(__file__).resolve().parent.parent
    / "femtovllm"
    / "weights"
    / "qwen3_0.6b_weights"
)


femtovllm._DEV.varlen_attn_impl = "custom_gemm_gemv"
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


# Start generation and retrieve both the generator and the assigned request IDs
stream_generator, req_ids = llm.generate(
    prompts,
    sampling_params=SamplingParams(
        temperature=0.5,
        presence_penalty=1,
        max_new_tokens=1000,
    ),
)


# ==========================================
# Rich UI Rendering & Streaming Output
# Dynamically render concurrent outputs in the terminal using Rich
# ==========================================
accumulated_texts = {req_id: "" for req_id in req_ids}


def format_content(raw_text: str) -> str:
    text = raw_text.replace("<|im_end|>", "[bold red] ⏹ [/bold red]")

    if "<think>" in text and "</think>" not in text:
        text = text.replace("<think>", "[dim]<think>") + "[/dim]"
    elif "<think>" in text and "</think>" in text:
        text = text.replace("<think>", "[dim]<think>")
        text = text.replace("</think>", "</think>[/dim]")

    return text


def generate_layout():
    panels = []
    for req_id in req_ids:
        formatted_text = format_content(accumulated_texts[req_id])
        panel = Panel(
            formatted_text,
            title=f"[bold cyan]{req_id}[/bold cyan]",
            border_style="green",
        )
        panels.append(panel)
    return Columns(panels)


with Live(generate_layout(), refresh_per_second=15) as live:
    for text_deltas in stream_generator:
        for req_id, token_str in text_deltas:
            accumulated_texts[req_id] += token_str
        live.update(generate_layout())
