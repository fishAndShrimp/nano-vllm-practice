# Engine Architecture & System Design / 引擎架构与系统设计

This document defines the core architecture, component boundaries, and API contracts for the Scheduler.
本文档定义了引擎的核心架构、组件权责以及调度器的 API 契约。

---

## 1. Architecture Evolution / 架构演进

Inspired by `nanovllm`, this engine integrates modern LLM serving concepts from industry-leading frameworks:
本引擎基于 `nanovllm` 极简骨架，并深度融合了现代推理框架的核心理念：

- **Unified Token Scheduling (vLLM v1):** Eliminates the strict Prefill/Decode boundary. Everything is treated as "tokens catching up", enabling native Chunked Prefill.
  **统一 Token 调度:** 消除严格的 Prefill/Decode 边界，统一视为“Token 追赶”，原生支持 Chunked Prefill。
- **RadixTree Prefix Caching (SGLang):** Planned for zero-collision KV cache sharing across system prompts and chat histories.
  **基数树前缀缓存:** 计划引入 RadixTree 管理物理块，实现 System Prompt 和多轮对话的零碰撞显存复用。

---

## 2. Component Boundaries / 组件权责与状态归属

To prevent state corruption and memory leaks, `Sequence` mutations are strictly isolated across components.
为防止状态机混乱和内存泄漏，各组件对 `Sequence` 的操作权限被严格物理隔离。

| Target / 目标属性 | Component / 组件 | Access / 读写权限 | Description / 状态流转说明 |
| :--- | :--- | :--- | :--- |
| `Sequence.status` | **RequestQueue** | Read / Write | `WAITING` <=> `RUNNING` (Queue management / 队列管理) |
| `Sequence.status` | **Scheduler** | Read / Write | `RUNNING` => `FINISHED` (Lifecycle termination / 生命周期终止) |
| `Sequence` | **Scheduler** | Read / Write | Appends tokens, updates progress / 追加 Token，更新计算进度 |
| `Sequence` | **KVCacheManager** | **Read Only** | Strictly read-only, uses `seq_id` for mapping / 绝对只读，仅使用 `seq_id` 查表 |
| `Sequence.stop_reason` | **ModelRunner** / **Scheduler** | Read / Write | Sets tombstone for termination (e.g., EOS, OOM) / 设置终止墓碑标记（如自然生成 EOS 或被调度器 OOM 截断） |

---

## 3. Scheduling as a Multi-Dimensional Knapsack / 多维背包问题与两阶段调度

A scheduling step is a **Multi-Dimensional Knapsack Problem** (Compute Budget, KV Cache, and future Multimodal Encoders). The actual implementation handles many edge cases, but the core logic is a two-phase greedy allocation:
调度本质上是一个**多维背包问题**（计算预算、KV Cache、以及未来的多模态编码器）。尽管实际代码需要处理诸多边界条件，但其核心逻辑是一个两阶段的贪心分配：

### 📝 Core Scheduling Logic (Pseudo-code) / 核心调度逻辑伪代码

```python
# Phase 1: Schedule Running (Prioritize existing sequences / 优先保障运行中序列)
for seq in running_queue:
    # 1. Truncate by computation limit / 截断计算量
    num_tokens = min(seq.uncomputed, limit_computation)
    if not fit_budget(num_tokens): 
        break

    # 2. Resolve storage limit (KV Cache) / 解决存储限制
    while not fit_kv_cache(seq, num_tokens):
        if is_tail(seq):
            # Tail sequence: truncate tokens or OOM / 队尾序列：截断 Token 或触发 OOM
            num_tokens = min(num_tokens, limit_kv_cache)
            if num_tokens == 0 and is_head(seq): 
                force_finish(seq, "OOM")
            break
        else:
            # Non-tail sequence: preempt the tail to free space / 非队尾序列：抢占队尾腾出空间
            preempt_tail()
            
    allocate(seq, num_tokens)

# Phase 2: Schedule Waiting (Admit new sequences / 在资源充裕时接入新序列)
for seq in waiting_queue:
    # Truncate by BOTH computation and storage / 同时受限于计算量与物理块限制
    num_tokens = min(seq.uncomputed, limit_computation, limit_kv_cache)
    
    if num_tokens > 0 and fit_budget and fit_kv_cache:
        pop_from_waiting(seq)
        allocate(seq, num_tokens)
    else:
        # Stop immediately if any resource is exhausted / 任何资源耗尽则直接停止接入
        break 
```

### 🔍 Strategy Breakdown / 策略解析

- **Phase 1 (Running Queue):** We aggressively protect sequences that have already started. If KV cache is insufficient, we preempt the lowest priority sequence (the tail) to make room. If the sequence itself is the tail, we truncate its tokens to fit the remaining blocks.
  **第一阶段（运行队列）：** 我们极力保护已经开始生成的序列。如果显存不足，我们会抢占优先级最低的序列（队尾）来腾出空间。如果当前序列就是队尾，我们会截断它的 Token 以适应剩余的物理块。
- **Phase 2 (Waiting Queue):** We only admit new sequences if there is surplus compute budget and KV cache. If a waiting sequence cannot fit, we immediately stop checking the rest of the queue.
  **第二阶段（等待队列）：** 只有在计算预算和显存都有盈余时，才会接入新序列。如果队首的等待序列无法分配到资源，我们会立刻停止遍历剩余的等待队列。

---

## 4. Atomic Operations / 调度器原子操作

The core state transitions of the Scheduler are composed of a strict set of Atomic Operations. These operations guarantee strong consistency between Queue Status and Physical Resources.
Scheduler 的核心状态流转由一组严格的原子操作构成。这些操作保证了**队列状态**与**物理资源**的强一致性，绝不出现“状态改了但显存没释放”的幽灵状态。

- **`_preempt()`**
  [Atomic] `RUNNING` => `WAITING`. Resets computed tokens and fully frees KV cache.
  [原子操作] `RUNNING` => `WAITING`。将队尾序列踢回等待队列，清空其已计算的 Token 计数并全额释放物理块。

- **`_allocate(seq, num_tokens)`**
  [Atomic] Decreases resources. Consumes computation budget and allocates KV cache blocks.
  [原子操作] 减少资源。为序列分配当前 Step 的计算预算（Token 数量）并向 KVCacheManager 申请物理块。

- **`_finish(seq)`**
  [Atomic] `*` => `FINISHED`. Frees all KV cache blocks and marks sequence as finished.
  [原子操作] `*` => `FINISHED`。序列自然结束，释放其占用的所有物理块，并将状态标记为 FINISHED。

- **`_force_finish(seq, stop_reason)`**
  [Atomic] `*` => `FINISHED`. Sets `stop_reason` (e.g., "OOM") and forcefully calls `_finish()` to release resources, preserving the error context.
  [原子操作] `*` => `FINISHED`。异常终止，写入 `stop_reason`（如 "OOM"），并强制调用 `_finish()` 释放资源，保留异常现场。

- **`_sweep_stopped_sequences()`**
  [Atomic] Scans running queue, finishes sequences with `stop_reason`, and removes them from the queue.
  [原子操作] 垃圾回收。每 Step 开始时扫描运行队列，将带有 `stop_reason` 的序列执行 `_finish()` 并从队列中彻底剔除。

---

## 5. State Machine Constraints / 状态机流转约束

- **Tombstone Mechanism / 墓碑机制**
  Any sequence assigned a `stop_reason` (regardless of its current status) will be unconditionally reclaimed and destroyed by `_sweep_stopped_sequences` in the next step.
  任何序列只要被写入了 `stop_reason`，无论其当前处于什么状态，都会在下一个 Step 被 `_sweep_stopped_sequences` 无条件回收资源并销毁。

- **Lazy Preemption / 延迟抢占**
  When a non-head sequence hits a memory bottleneck, the Scheduler stalls it instead of preempting immediately. Destructive `_preempt()` is only triggered when higher-priority requests actually need the physical blocks.
  当非队首序列遇到显存瓶颈时，Scheduler 会优先将其挂起（Stall）而不是立即抢占，直到高优先级请求确实需要物理块时，才会触发破坏性的 `_preempt()`。
