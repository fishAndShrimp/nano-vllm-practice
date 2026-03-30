#pragma once

namespace femtovllm {

// ============================================================================
// Paged Attention GEMM (Prefill)
// ============================================================================
// In the GEMM kernel, the computation tile is structurally
// bound to the physical memory page. The kernel directly
// uses the tile index to fetch the page index (i.e.,
// `page_idx = page_table[tile_idx]`). Because 1 Tile MUST
// exactly equal 1 Page, they cannot be configured
// independently. Hence, the GEMM kernel exclusively uses
// `kTileSize` to dictate both.
inline constexpr int kTileSize = 32;

// ============================================================================
// Paged Attention GEMV (Decode)
// ============================================================================
// In the GEMV kernel, computation is tiled by warps
// (`kWarpsPerBlock`), which is completely decoupled from
// the physical page layout. It locates data dynamically via
// math (i.e., `page_table[col_kt / kKVLenPerPage]`).
// Because it can tolerate ANY arbitrary page size, it only
// needs `kKVLenPerPage` to perform memory addressing.
// (Aliased to kTileSize to match the GEMM's memory pool
// layout).
inline constexpr int kKVLenPerPage = kTileSize;

// ============================================================================
// Global Constants
// ============================================================================
inline constexpr int kDimHead = 128;
inline constexpr int kWarpSize = 32;

}  // namespace femtovllm
