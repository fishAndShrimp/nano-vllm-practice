#pragma once

namespace femtovllm {

// ============================================================================
// Global Constants
// ============================================================================
inline constexpr int kDimHead = 128;
inline constexpr int kWarpSize = 32;

// ============================================================================
// Paged Attention GEMM (Prefill)
// ============================================================================
// In the GEMM kernel, computation is performed in 2D tiles
// to maximize ALU utilization and shared memory reuse.
// `kTileSize` dictates the dimension of these computation
// tiles. A larger tile size (e.g., 64 or 128) generally
// improves compute intensity but consumes more shared
// memory per thread block.
inline constexpr int kTileSize = 64;

// ============================================================================
// Paged Attention Memory Management (KV Cache)
// ============================================================================
// `kKVLenPerPage` defines the physical block size (number
// of tokens per page) managed by the KV Cache Manager. A
// smaller page size (e.g., 16) significantly reduces
// internal memory fragmentation, allowing for higher batch
// sizes and better VRAM utilization.
//
// Both GEMM (Prefill) and GEMV (Decode) kernels are fully
// decoupled from this physical layout and dynamically map
// logical token indices to physical pages using this
// constant.
inline constexpr int kKVLenPerPage = 16;

// To ensure efficient memory loading without complex
// boundary checks within a single computation tile, the
// tile size must perfectly span an integer number of
// physical pages.
static_assert(
    kTileSize % kKVLenPerPage == 0,
    "Architecture Error: kTileSize must be an exact "
    "multiple of kKVLenPerPage "
    "to ensure aligned memory access across physical pages."
);
inline constexpr int kNumPagesPerTile =
    kTileSize / kKVLenPerPage;

inline constexpr int kThreadsPerBlock = 256;
inline constexpr int kWarpsPerBlock =
    kThreadsPerBlock / kWarpSize;

// ============================================================================
// Paged Attention GEMV (Decode)
// ============================================================================
// Maximum sequence length supported by the non-split GEMV
// decode kernel. Sequences longer than this will require
// split-K reduction (to be implemented).
inline constexpr int kMaxKVLenNonSplit = 8192;
static_assert(
    kThreadsPerBlock > kDimHead,
    "When GEMV, "
    "kThreadsPerBlock must be greater than kDimHead to "
    "ensure enough threads "
    "are available to compute the accumulated V in "
    "parallel without looping."
);
static_assert(
    kThreadsPerBlock % kWarpSize == 0,
    "When GEMV, "
    "kThreadsPerBlock must be a multiple of kWarpSize (32) "
    "to avoid partial warps."
);
static_assert(
    kWarpsPerBlock <= kWarpSize,
    "When GEMV, "
    "kWarpsPerBlock cannot exceed kWarpSize (32) because "
    "the final block reduction is performed by a single "
    "warp."
);

}  // namespace femtovllm
