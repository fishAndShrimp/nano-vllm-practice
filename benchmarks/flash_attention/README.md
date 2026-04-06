# FlashAttention Benchmarks & Profiling / 基准测试与性能分析

This directory contains scripts and results for performance evaluation, correctness testing, and low-level hardware profiling of our custom FlashAttention CUDA kernels.
本目录用于自定义 FlashAttention CUDA 算子的性能评估、正确性验证以及底层硬件指标抓取。

---

## 📊 Performance Results / 当前测试结果

**Analysis / 性能分析:**
At short sequence lengths, our pure SIMT implementation achieves performance comparable to PyTorch SDPA. However, as `seqlen` increases, the performance gap widens due to the absence of Tensor Core utilization in our current version.
当序列长度（seqlen）较小时，本算子（纯 SIMT 实现）性能与 PyTorch SDPA 接近；随着 seqlen 增大，由于尚未引入 Tensor Cores 加速，与极致优化的基线相比性能差距逐渐显现。

**Note on Variance / 关于波动的说明:**
GPU execution times are subject to system scheduling and clock scaling. The data below shows three independent runs of `benchmark.py` (averaged over 100 iterations per run). **Note that RUN 1 and RUN 2 are cherry-picked optimal results.**
受系统调度与频率缩放影响，GPU 测速存在客观波动。以下为 `benchmark.py`（单次循环 100 遍取平均）的三次独立运行结果，**其中 RUN 1 与 RUN 2 为经过挑选的较优结果。**

### RUN 1

![attention_benchmark_run1](./attention_benchmark_run1.png)
| Seq_Len | Manual (ms) | femtovllm (Ours) (ms) | SDPA Math (ms) | xFormers (ms) | FlashAttention (ms) |
|---------|-------------|-----------------------|----------------|---------------|---------------------|
| 16      | 0.0865      | 0.0408                | 0.7229         | 0.0583        | 0.0629              |
| 32      | 0.2648      | 0.0326                | 0.6386         | 0.0564        | 0.0652              |
| 64      | 0.2607      | 0.0755                | 0.6709         | 0.0544        | 0.0654              |
| 128     | 0.2728      | 0.1140                | 0.2247         | 0.0166        | 0.0181              |
| 256     | 0.0575      | 0.3764                | 0.2560         | 0.0225        | 0.0304              |

### RUN 2

![attention_benchmark_run2](./attention_benchmark_run2.png)
| Seq_Len | Manual (ms) | femtovllm (Ours) (ms) | SDPA Math (ms) | xFormers (ms) | FlashAttention (ms) |
|---------|-------------|-----------------------|----------------|---------------|---------------------|
| 16      | 0.0912      | 0.0212                | 0.7344         | 0.0581        | 0.0621              |
| 32      | 0.2380      | 0.0426                | 0.6575         | 0.0611        | 0.0523              |
| 64      | 0.2591      | 0.0757                | 0.6488         | 0.0616        | 0.0638              |
| 128     | 0.2635      | 0.1287                | 0.6659         | 0.0518        | 0.0493              |
| 256     | 0.2349      | 0.3420                | 0.3657         | 0.0225        | 0.0400              |

### RUN 3

![attention_benchmark_run3](./attention_benchmark_run3.png)
| Seq_Len | Manual (ms) | femtovllm (Ours) (ms) | SDPA Math (ms) | xFormers (ms) | FlashAttention (ms) |
|---------|-------------|-----------------------|----------------|---------------|---------------------|
| 16      | 0.2820      | 0.0220                | 0.3323         | 0.0231        | 0.0193              |
| 32      | 0.0568      | 0.0306                | 0.3508         | 0.0358        | 0.0168              |
| 64      | 0.0983      | 0.0590                | 0.3399         | 0.0159        | 0.0215              |
| 128     | 0.0966      | 0.1398                | 0.3035         | 0.0153        | 0.0262              |
| 256     | 0.1015      | 0.3918                | 0.2835         | 0.0308        | 0.0438              |

---

## 🛠️ Environment Setup / 测试环境搭建

Profiling workflows vary significantly across platforms. The following setup is based on **Windows 11 + WSL2**.
性能分析工具的配置因系统而异。以下经验基于 **Windows 11 + WSL2** 环境：

1. **Tool Installation / 工具安装:**
   Download Nsight Systems (`nsys`) and Nsight Compute (`ncu`) from [NVIDIA Developer Tools](https://developer.nvidia.com/tools-overview). It is required to install them on **both** the Windows host and inside WSL2.
   请前往 NVIDIA 官网下载并安装 `nsys` 和 `ncu`。建议在 Windows 宿主机与 WSL2 内部**双端均进行安装**。

2. **WSL2 Permissions / WSL2 权限配置 (Critical):**
   To profile inside WSL2, you must grant GPU counter access on the Windows host. Open the **NVIDIA Control Panel** -> **Developer** -> **Manage GPU Performance Counters**, and select **"Allow access to the GPU performance counters to all users"**.
   在 WSL2 内部执行 Profiling 前，必须在 Windows 宿主机的 **NVIDIA 控制面板** -> **开发者** -> **管理 GPU 性能计数器** 中，勾选 **“允许所有用户访问 GPU 性能计数器”**，否则 `ncu` 会因权限不足报错。

---

## 🚀 Profiling Commands / 性能抓取命令速查

Execute the following commands in the **current directory**. Profiling reports will be saved in the `reports/` subdirectory (ensure this directory exists).
请在**当前目录**下执行以下命令，生成的分析报告将统一保存在 `reports/` 子文件夹中（需预先创建）。

### Nsight Systems (`nsys`) - Macro Timeline / 宏观时间线
Profiles CPU-GPU interactions and kernel launch overheads.
抓取 CPU-GPU 交互与算子启动开销。

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --export=none \
    --output=reports/nsys_version_warp \
    --force-overwrite=true \
    python scripts/profile_version_warp.py
```

### Nsight Compute (`ncu`) - Micro Kernel Analysis / 微观算子分析
Profiles detailed hardware metrics (e.g., memory bandwidth, SM occupancy) for a specific kernel.
深入抓取特定算子的显存带宽、SM 占用率等底层硬件指标。

*Note: Change the `-k` parameter to match the specific kernel name you want to profile.*
*注：可通过修改 `-k` 参数来指定抓取不同的 Kernel。*

```bash
/usr/local/NVIDIA-Nsight-Compute/ncu --set full \
    -k "FlashAttentionWarpKernel" \
    -s 10 \
    -c 1 \
    -o reports/ncu_version_warp \
    -f \
    python scripts/profile_version_warp.py
```
