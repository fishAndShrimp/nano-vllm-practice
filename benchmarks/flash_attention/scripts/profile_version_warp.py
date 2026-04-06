import torch

# Import the installed package
import femtovllm


def main():
    # --- Golden Ratio Dimensions for Profiling ---
    # B=2, H=4, T=1024, C=128
    # Assuming Block Size X is 64, then Grid X = 1024/64 = 16
    # Total Blocks = 16 (T) * 4 (H) * 2 (B) = 128 Blocks
    # 128 Blocks provide enough workload to fill the SMs without
    # making the NCU profiling process take too long.
    B, H, T, C = 2, 4, 1024, 128

    print(f"Initializing tensors: B={B}, H={H}, T={T}, C={C}")

    # Initialize tensors with float16 to match the benchmark
    q = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)

    print("Warming up...")
    for _ in range(10):
        _ = femtovllm._C.FlashAttentionWarpCuda(q, k, v)
    torch.cuda.synchronize()

    print("Running profiled execution...")

    # Start the CUDA profiler and push an NVTX range for easy identification
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("Profile_FlashAttentionWarpCuda")

    # Execute the kernel to be profiled
    _ = femtovllm._C.FlashAttentionWarpCuda(q, k, v)

    # Synchronize and stop profiling
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    print("Profiling execution finished.")


if __name__ == "__main__":
    main()
