#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "./constants.cuh"

namespace femtovllm {

/*
 * [Architecture Decision]: Why no template<typename T> for
 * Warp Reduce?
 *
 * In LLM inference (e.g., Attention Softmax, RMSNorm),
 * while weights (Q, K, V) are often in low precision
 * (FP16/BF16) to save memory bandwidth, the accumulation
 * and reduction of intermediate scores MUST strictly remain
 * in FP32 (float).
 *
 * 1. Overflow/Underflow: Summing large arrays in FP16/BF16
 * causes severe rounding errors.
 * 2. Exponential Explosion: Reductions for Softmax (Max &
 * Sum) are highly sensitive; precision loss here will be
 * exponentially amplified by exp().
 *
 * Therefore, we explicitly restrict these primitives to
 * `float` to enforce numerical stability and prevent
 * accidental low-precision reductions.
 */

constexpr unsigned int kWarpMaskFull = 0xffffffff;

__device__ __forceinline__ float WarpReduceSum(float val) {
    // reduce sum to thread lane_0 only

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0;
         offset /= 2) {
        val += __shfl_down_sync(kWarpMaskFull, val, offset);
    }

    return val;
}

__device__ __forceinline__ float WarpAllReduceSum(
    float val
) {
    // reduce sum to all threads lane_0~lane_31 via
    // hypercube routing

#pragma unroll
    for (int cube_edge = 1; cube_edge < kWarpSize;
         cube_edge *= 2) {
        val +=
            __shfl_xor_sync(kWarpMaskFull, val, cube_edge);
    }

    return val;
}

__device__ __forceinline__ float WarpReduceMax(float val) {
    // reduce max to thread lane_0 only

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0;
         offset /= 2) {
        val = fmaxf(
            val,
            __shfl_down_sync(kWarpMaskFull, val, offset)
        );
    }

    return val;
}

__device__ __forceinline__ float WarpAllReduceMax(
    float val
) {
    // reduce max to all threads lane_0~lane_31 via
    // hypercube routing

#pragma unroll
    for (int cube_edge = 1; cube_edge < kWarpSize;
         cube_edge *= 2) {
        val = fmaxf(
            val,
            __shfl_xor_sync(kWarpMaskFull, val, cube_edge)
        );
    }

    return val;
}

}  // namespace femtovllm
