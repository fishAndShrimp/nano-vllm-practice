# FemtoVLLM 🚀

**FemtoVLLM** is a minimalist, high-performance LLM inference engine built from scratch. It serves as a testbed for deconstructing and verifying state-of-the-art inference optimizations—including **PagedAttention**, **Prefix Caching**, **Chunked Prefill**, and **Custom CUDA/Triton Kernels**—designed specifically for single-GPU environments.

![Offline Stream Demo](assets/test_offline_stream_v3.gif)

## ✨ Core Features & Architecture

- **Unified Token Scheduling (Native Chunked Prefill)**: Eliminates the strict Prefill/Decode boundary. Everything is treated as "tokens catching up", naturally supporting Chunked Prefill to eliminate TTFT (Time To First Token) spikes for ultra-long contexts.
- **Dual-Backend Operator Routing**: 
  - **Prefill (Compute-Bound)**: Powered by **Triton GEMM** for high-throughput 1D/2D block scheduling and Tensor Core utilization.
  - **Decode (Memory-Bound)**: Powered by **CUDA C++ GEMV** with warp-level reductions (`__shfl_down_sync`) to squeeze every drop of memory bandwidth.
- **Prefix-Tree Native Caching**: The Scheduler natively integrates a block-aligned Prefix Tree. It achieves Zero-Copy prompt caching and dynamic physical block deduplication, backed by a mathematically strict `(L-1)` state healing mechanism.
- **Lazy Tensorization**: Inputs remain as lightweight `list[int]` until the exact moment of execution, where Varlen (variable-length) tensors are dynamically flattened and built.

## 🗺️ Roadmap

- [x] **Phase 1**: Study `nanoGPT` for Transformer basics and auto-regressive generation.
- [x] **Phase 2**: Study `nano-vllm` for KV Cache and basic PagedAttention concepts.
- [x] **Phase 3**: Implement custom pure SIMT CUDA kernels (FlashAttention, PagedAttention GEMM/GEMV) and a Prefix-Tree Native Scheduler (v3) for Prompt Caching.
- [x] **Phase 4**: Port core attention kernels to **OpenAI Triton** and implement a unified dispatching mechanism to achieve higher throughput and lower TTFT compared to naive implementations.
- [ ] **Phase 5 (WIP)**: Upgrade to a RadixTree-Native Scheduler, unifying scheduling and memory allocation into a single C++ backend.

## 📚 References

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
