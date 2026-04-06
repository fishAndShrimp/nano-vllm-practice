import math
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Import the installed package
import femtovllm


def manual_attn(q, k, v):
    """PyTorch manual implementation (Baseline)"""
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    return att @ v


def femtovllm_attn(q, k, v):
    """Our pure SIMT fused attention operator"""
    return femtovllm._C.FlashAttentionWarpCuda(q, k, v)


def benchmark_fn(fn, *args, num_repeats=100, num_warmups=10, sdp_backend=None):
    """Professional CUDA benchmarking function with warmup"""
    # Use PyTorch's latest sdpa_kernel context manager
    if sdp_backend is not None:
        # Convert to list if a single backend is provided (required by new API)
        backends = [sdp_backend] if not isinstance(sdp_backend, list) else sdp_backend
        context = sdpa_kernel(backends)
    else:
        context = nullcontext()

    with context:
        # Warmup: ensure GPU reaches max clock state and kernel initialization is done
        for _ in range(num_warmups):
            fn(*args)
        torch.cuda.synchronize()

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_repeats):
            fn(*args)
        end_event.record()
        torch.cuda.synchronize()

    # Return average latency per execution in milliseconds (ms)
    return start_event.elapsed_time(end_event) / num_repeats


def main():
    B, H, C = 2, 4, 128
    seq_lens = [16, 32, 64, 128, 256]

    # Store latency results for different backends
    results = {
        "Manual": [],
        "femtovllm": [],
        "SDPA_Math": [],
        "SDPA_MemEff": [],
        "SDPA_Flash": [],
    }

    print(
        f"=== Starting Performance Benchmark (B={B}, H={H}, C={C}, dtype=float16) ===\n"
    )

    # Print Markdown table header
    print(
        "| Seq_Len | Manual (ms) | femtovllm (Ours) (ms) | SDPA Math (ms) | xFormers (ms) | FlashAttention (ms) |"
    )
    print(
        "|---------|-------------|-----------------------|----------------|---------------|---------------------|"
    )

    for T in seq_lens:
        q = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)
        k = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)
        v = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)

        # 1. Manual
        t_manual = benchmark_fn(manual_attn, q, k, v)

        # 2. femtovllm (Ours)
        t_femto = benchmark_fn(femtovllm_attn, q, k, v)

        # 3. SDPA - Math (PyTorch underlying C++ Math implementation)
        t_math = benchmark_fn(
            F.scaled_dot_product_attention,
            q,
            k,
            v,
            sdp_backend=SDPBackend.MATH,
        )

        # 4. SDPA - Memory Efficient (xFormers)
        t_memeff = benchmark_fn(
            F.scaled_dot_product_attention,
            q,
            k,
            v,
            sdp_backend=SDPBackend.EFFICIENT_ATTENTION,
        )

        # 5. SDPA - FlashAttention
        t_flash = benchmark_fn(
            F.scaled_dot_product_attention,
            q,
            k,
            v,
            sdp_backend=SDPBackend.FLASH_ATTENTION,
        )

        # Record data
        results["Manual"].append(t_manual)
        results["femtovllm"].append(t_femto)
        results["SDPA_Math"].append(t_math)
        results["SDPA_MemEff"].append(t_memeff)
        results["SDPA_Flash"].append(t_flash)

        # Print Markdown table row
        print(
            f"| {T:<7} | {t_manual:<11.4f} | {t_femto:<21.4f} | {t_math:<14.4f} | {t_memeff:<13.4f} | {t_flash:<19.4f} |"
        )

    print("\n")

    # --- Plot Performance Comparison ---
    plt.figure(figsize=(12, 7))

    # Baselines
    plt.plot(
        seq_lens,
        results["Manual"],
        marker="o",
        linestyle="--",
        color="gray",
        label="PyTorch Manual (Python)",
    )
    plt.plot(
        seq_lens,
        results["SDPA_Math"],
        marker="x",
        linestyle=":",
        color="black",
        label="SDPA Math (C++ Baseline)",
    )

    # Your Operator (Ours)
    plt.plot(
        seq_lens,
        results["femtovllm"],
        marker="s",
        linewidth=3.0,
        color="red",
        label="femtovllm (Ours - Pure SIMT Fused)",
    )

    # Advanced Hardware Optimizations (Tensor Cores)
    plt.plot(
        seq_lens,
        results["SDPA_MemEff"],
        marker="v",
        linestyle="-.",
        color="orange",
        label="SDPA xFormers (Mem Efficient)",
    )
    plt.plot(
        seq_lens,
        results["SDPA_Flash"],
        marker="^",
        linewidth=2.0,
        linestyle="-",
        color="blue",
        label="SDPA FlashAttention (Hardware Limit)",
    )

    plt.title(
        "Attention Backend Performance Comparison (FP16)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Sequence Length (T)", fontsize=13)
    plt.ylabel("Latency (ms)", fontsize=13)
    plt.xscale("log", base=2)
    plt.xticks(seq_lens, seq_lens)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()

    # Save the plot
    plt.savefig("attention_benchmark.png", dpi=300)
    print("=== Benchmark Complete! Chart saved as attention_benchmark.png ===")


if __name__ == "__main__":
    main()
