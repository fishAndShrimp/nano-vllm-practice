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

- [x] **v3: Prefix-Tree Native Scheduler & Block-Aligned Cache**
  A paradigm shift where the `KVCacheManager` and `Scheduler` become irreversibly coupled. The KV Cache actively dictates scheduling decisions through a block-aligned Prefix Tree. It deliberately avoids heavy Copy-on-Write (CoW) and node splitting to minimize Python CPU/GIL overhead, focusing on pure block-level state management.
  架构范式的转移：`KVCacheManager` 与 `Scheduler` 形成不可逆的深度耦合。KV Cache 通过块对齐的前缀树主动主导调度决策。刻意摒弃了繁重的写时复制（CoW）与节点分裂逻辑，以最小化 Python CPU/GIL 开销，专注纯粹的物理块级状态管理。

  | Core Mechanism / 核心机制 | Architectural Implementation / 架构实现 | Engineering Benefit / 工程收益 |
  | :--- | :--- | :--- |
  | **Native Prompt Caching**<br>*(原生提示词缓存)* | The Scheduler natively queries the Prefix Tree via `fast_forward_prefix` to advance `num_computed_tokens`, directly skipping redundant computation.<br> 调度器通过 `fast_forward_prefix` 原生查询前缀树，直接快进 `num_computed_tokens`，跳过冗余计算。 | Drastically reduces TTFT (Time To First Token) for shared prefixes.<br> 极大地降低了共享前缀的首字延迟 (TTFT)。 |
  | **The (L-1) Rule & State Healing**<br>*(L-1 规则与状态自愈)* | Forces at least one forward pass by retreating 1 token if fully matched. Protects trailing KV blocks from eviction during the temporary state misalignment, healing automatically after the forward pass.<br> 完全匹配时强制回退 1 个 Token 以确保至少执行一次前向传播。在短暂的状态错位期保护尾部物理块免遭驱逐，并在前向传播后自动“自愈”。 | Balances strict mathematical correctness with maximum VRAM protection.<br> 在严格的数学正确性与最大化显存保护之间取得完美平衡。 |
  | **Dynamic Deduplication**<br>*(动态物理块去重)* | `commit_blocks` bridges computation and storage. Newly computed blocks are dynamically merged into the tree, and redundant physical blocks are immediately freed.<br> `commit_blocks` 桥接了计算与存储。新计算的物理块被动态合并入树，冗余的物理块被立即释放。 | Achieves Zero-Copy reuse and prevents memory fragmentation.<br> 实现零拷贝复用，并彻底防止显存碎片化。 |
  | **Lazy FIFO Eviction**<br>*(延迟 FIFO 驱逐)* | Finished sequences unpin nodes. Nodes enter an `OrderedDict` and are physically evicted (FIFO) only when memory is strictly depleted during `allocate`.<br> 序列结束后解除节点锁定。节点进入 `OrderedDict`，仅在 `allocate` 触发显存枯竭时才执行 FIFO 物理驱逐。 | Maximizes cache hit rates safely without proactive GC overhead.<br> 安全最大化缓存命中率，且无主动垃圾回收（GC）开销。 |

- [ ] **v4: RadixTree-Native Scheduler** (End of Standalone KVCacheManager)
  **[Architectural Shift]** From v4 onwards, `KVCacheManager` will no longer exist as an independent module. The KV Cache is no longer just passive storage; it actively dictates scheduling. The Scheduler and Prefix Tree will be irreversibly coupled into a single RadixTree-native architecture to natively support Prompt Caching (fast-forwarding `num_computed_tokens`) and dynamic block deduplication.
  **[架构范式转移]** 从 v4 开始，`KVCacheManager` 将不再作为独立模块存在。KV Cache 不再是单纯的被动存储，而是主动决定调度策略。调度器与前缀树将不可逆地深度耦合，演进为单一的原生基数树调度器（RadixTree-native Scheduler），以原生支持 Prompt Caching（快进计算进度）与动态物理块去重。
