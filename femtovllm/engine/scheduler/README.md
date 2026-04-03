# Scheduler / 核心调度器

The brain of the inference engine. It orchestrates the lifecycle of sequences, dynamically allocating resources while strictly enforcing multi-dimensional constraints.
推理引擎的大脑。负责编排序列的生命周期，在严格执行多维约束的前提下，动态分配计算与存储资源。

## 📐 Multi-dimensional Constraints / 多维调度约束

During each `step()`, the scheduler truncates sequence execution based on the strictest bottleneck:
在每次 `step()` 调度中，调度器会基于最严格的短板来截断序列的执行长度：

- **Computation Limit (计算约束)**: Governed by `StepBudget`. Caps the maximum tokens processed per iteration to prevent GPU compute saturation.
  由 `StepBudget` 管控。限制单次迭代处理的最大 Token 数，防止 GPU 算力过载。
- **Storage Limit (存储约束)**: Governed by `KVCacheManager`. Ensures sufficient physical blocks exist for the requested sequence length.
  由 `KVCacheManager` 管控。确保有足够的物理块来支撑请求的序列长度。
- **Hardware Limit (硬件约束)**: Governed by `max_kv_len_non_split`. Prevents sequences from exceeding the model's maximum supported context window.
  由 `max_kv_len_non_split` 管控。防止单条序列突破模型支持的最大上下文窗口。
- **[TODO] Encoder/Multimodal Limit (多模态/编码器约束)**: Future support for Vision/Audio encoders. Will manage burst token generation from image patches and isolated Cross-Attention cache budgets.
  未来对视觉/音频编码器的支持。将管理图像 Patch 带来的突发 Token 洪峰，以及独立的交叉注意力（Cross-Attention）缓存预算。

---

## ⚙️ Core Mechanics / 核心调度机制

| Mechanism / 机制 | Description / 逻辑描述 |
| :--- | :--- |
| **Two-Phase Scheduling**<br>*(双阶段调度)* | 1. `_schedule_running()`: Prioritizes active sequences. <br>2. `_schedule_waiting()`: Fills remaining budget with new requests.<br> 优先保障运行中序列的资源，剩余预算再分配给等待队列。 |
| **Tail Preemption**<br>*(尾部抢占)* | When OOM occurs, evicts the tail (newest/lowest priority) of the running queue to guarantee the head (oldest) finishes.<br> 触发 OOM 时，驱逐运行队列尾部（最新请求），保底头部请求顺利完成。 |
| **State Healing**<br>*(状态自愈)* | Enforces the `(L-1)` rule during prompt caching. Retreats 1 token to force a forward pass, protecting trailing blocks from eviction.<br> 提示词缓存时强制回退 1 个 Token 执行前向传播，以保护尾部物理块免遭驱逐。 |

---

## 🚀 Roadmap / 演进路线图

- [x] **v1: Iteration-Level Scheduler**
  A reactive scheduler. Treats KV Cache as passive storage. Every token must be computed (No Prompt Caching).
  响应式调度器。将 KV Cache 视为被动存储，所有 Token 均需真实计算（无提示词缓存）。

- [x] **v3: Prefix-Tree Native Scheduler**
  Irreversibly coupled with `KVCacheManagerV3`. Natively queries the Prefix Tree (`fast_forward_prefix`) to skip redundant computation and bridges storage via `commit_blocks` for dynamic deduplication.
  与 `KVCacheManagerV3` 形成不可逆耦合。原生查询前缀树（`fast_forward_prefix`）跳过冗余计算，并通过 `commit_blocks` 桥接存储实现动态去重。

- [ ] **v4: RadixTree-Native Scheduler** (TODO)
  **[Architectural Shift]** The end of standalone KV Cache Managers. The Scheduler itself becomes a Radix Tree. Scheduling decisions, memory allocation, and multimodal chunking will be unified under a single RadixTree-native C++ backend for extreme concurrency.
  **[架构范式转移]** 终结独立的 KV Cache 管理器，调度器本身即是一棵基数树。调度决策、显存分配与多模态分块将被统一收敛至单一的 RadixTree-native C++ 后端，支撑极致并发。
