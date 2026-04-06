
## Profiling Commands / 性能抓取命令速查

请在**当前目录**下执行以下命令，生成的分析报告将统一保存在 `reports/` 子文件夹中。该文件夹需要预先创建好

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
