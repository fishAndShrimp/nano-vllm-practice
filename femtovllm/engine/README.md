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

---

## 3. Scheduling as a Multi-Dimensional Knapsack / 多维背包问题与两阶段提交

A scheduling step is a **Multi-Dimensional Knapsack Problem** (Compute Budget, KV Cache, and future Multimodal Encoders). We strictly separate `can_consume` (Check) and `consume` (Commit) instead of using `try_consume`. This **Two-Phase Commit** prevents partial state pollution when one resource fails after another succeeds.
调度步本质是**多维背包问题**（算力预算、显存、及未来的多模态编码器）。我们严格分离 `can_consume`（检查）和 `consume`（提交），弃用 `try_consume`。这种**两阶段提交**可防止多资源校验时的部分状态污染。

代码范例 (Code Example):
    # Phase 1: Check (Pure functions, no side effects / 纯函数，无副作用)
    can_fit_budget = budget.can_consume(num_tokens)
    can_fit_kv = kv_manager.can_allocate(seq, num_tokens)
    # future: can_fit_encoder = encoder_manager.can_allocate(...)

    if can_fit_budget and can_fit_kv:
        # Phase 2: Commit (Atomic, guaranteed to succeed / 原子操作，必然成功)
        budget.consume(num_tokens)
        kv_manager.allocate(seq, num_tokens)

---

## 4. Scheduler Atomic Operations / 调度器原子操作

To prevent memory leaks or deadlocks, the Scheduler must execute these exact combinations when altering a sequence's lifecycle.
为防止显存泄漏或死锁，调度器在改变序列生命周期时，必须严格执行以下组合拳。

| Action / 触发动作 | Atomic Operations / 必须执行的组合拳 (严格按顺序) |
| :--- | :--- |
| **Preempt / 抢占**<br>*(Evict due to low resources / 因资源不足踢出)* | 1. `seq = req_queue.preempt_running_tail()`<br>2. `kv_manager.free(seq)`<br>3. `seq.num_computed_tokens = 0` |
| **Allocate / 分配**<br>*(Advance computation / 推进计算进度)* | 1. `kv_manager.allocate(seq, tokens)`<br>2. `budget.consume(tokens)` |
| **Finish / 完成**<br>*(Generation done / 生成结束)* | 1. `seq.finish()`<br>2. `kv_manager.free(seq)`<br>3. `req_queue.clean_finished_running()` |
