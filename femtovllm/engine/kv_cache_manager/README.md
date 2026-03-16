# BlockAllocator

Allocate Paged KV Cache blocks to sequences, and reuse blocks to maximize GPU memory efficiency.
为序列分配 Paged KV Cache 物理块，并通过复用机制最大化显存效率。

## Roadmap / 演进路线图

- [x] **v1: Simplest Block Manager**
  Basic allocation and deallocation without prefix sharing.
  基础的分配与回收，无前缀共享机制。

- [ ] ~~**v2: Hash-based Block Reuse** (Skipped)~~
  Abandoned due to strict block alignment requirements and potential hash collisions.
  因严格的块边界对齐限制和哈希碰撞风险而直接跳过。

- [ ] **v3: Python Radix Tree Block Reuse** (TODO)
  Pure Python implementation of Radix Tree (from SGLang) for algorithmic verification. Zero-collision sharing for System Prompts and chat history.
  纯 Python 实现的基数树（来自 SGLang），用于验证核心算法。实现 System Prompt 和历史对话的零碰撞共享。

- [ ] **v4: C++ Radix Tree Backend** (TODO)
  High-performance C++ extension to replace the Python tree. Eliminates GIL bottlenecks and pointer chasing overheads for extreme concurrency.
  高性能 C++ 后端重写。消除 Python GIL 瓶颈与指针追逐开销，支撑极致并发。
