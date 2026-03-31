# KVCacheManager

Manage Paged KV Cache block tables for sequences, utilizing underlying block allocators to maximize GPU memory efficiency through prefix sharing and reuse.
为序列管理 Paged KV Cache 物理块表，利用底层块分配器，通过前缀共享与复用机制最大化显存效率。

## Roadmap / 演进路线图

- [x] **v1: Simplest Block Manager**
  Basic allocation and deallocation without prefix sharing.
  基础的分配与回收，无前缀共享机制。

- [ ] ~~**v2: Hash-based Block Reuse** (Skipped)~~
  Abandoned due to strict block alignment requirements and potential hash collisions.
  因严格的块边界对齐限制和哈希碰撞风险而直接跳过。

- [x] **v3: Python Block-Aligned Prefix Tree**
  A simplified Radix Tree tailored for Python. It deliberately avoids heavy Copy-on-Write (CoW) and node splitting to minimize CPU/GIL overhead, focusing on pure block-level reuse.
  专为 Python 优化的简化版基数树（Radix Tree）。刻意摒弃了繁重的写时复制（CoW）与节点分裂逻辑，以最小化 CPU 与 GIL 开销，专注纯粹的块级复用。

  | Core Feature / 核心特性 | Technical Description / 技术描述 | Architecture Benefit / 架构收益 |
  | :--- | :--- | :--- |
  | **Fast/Slow Path Scheduling**<br>*(快慢路径调度)* | `can_allocate` uses $O(1)$ conservative estimation without tree traversal. `allocate` performs the actual tree matching. <br> `can_allocate` 采用 $O(1)$ 保守预估（不查树），仅在 `allocate` 时进行真实树匹配。 | Prevents Python CPU bottlenecks during high-concurrency scheduling.<br> 彻底避免高并发调度时的 Python CPU 阻塞。 |
  | **Zero-Copy Block Reuse**<br>*(零拷贝块级复用)* | Strict block-aligned matching. Reuses identical blocks directly; unmatched tails fallback to standard prefill.<br> 严格的块对齐匹配。直接复用完全相同的物理块，未命中的尾部回退至常规 Prefill。 | Eliminates memory copy overhead and complex node management.<br> 消除显存拷贝开销与复杂的树节点管理。 |
  | **Lazy Eviction Mechanism**<br>*(延迟驱逐机制)* | Finished sequences only decrement ref-counts. Nodes are physically evicted only when memory is strictly depleted.<br> 序列结束后仅扣减引用计数，仅在物理显存枯竭且必须驱逐时才释放树节点。 | Maximizes cache hit rates (e.g., System Prompts) safely.<br> 在保障显存安全的前提下，最大化系统提示词等缓存命中率。 |

- [ ] **v4: C++ Radix Tree Backend** (TODO)
  High-performance C++ extension to replace the Python tree. Eliminates GIL bottlenecks and pointer chasing overheads for extreme concurrency.
  高性能 C++ 后端重写。消除 Python GIL 瓶颈与指针追逐开销，支撑极致并发。
