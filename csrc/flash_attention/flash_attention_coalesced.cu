#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/constants.cuh"
#include "../utils/cuda_check.cuh"

using femtovllm::kDimHead;
using femtovllm::kTileSize;

template <typename scalar_t>
__global__ void FlashAttentionCoalescedKernel(
    const scalar_t* __restrict__ q_batched,
    const scalar_t* __restrict__ k_t_batched,
    const scalar_t* __restrict__ v_batched,
    scalar_t* __restrict__ out_batched,
    int dim_t,
    int dim_d
) {
    int batch_idx = (blockIdx.z) * gridDim.y + (blockIdx.y);
    auto q = q_batched + dim_t * dim_d * batch_idx;
    auto k_t = k_t_batched + dim_t * dim_d * batch_idx;
    auto v = v_batched + dim_t * dim_d * batch_idx;
    auto out = out_batched + dim_t * dim_d * batch_idx;

    scalar_t q_tile[kTileSize];
    // use previous k_tile to help load q and v
    __shared__ union {
        scalar_t q[kTileSize][kTileSize + 1];
        scalar_t k[kTileSize][kTileSize + 1];
        scalar_t v[kTileSize][kTileSize + 1];
    } sdata;
    float hidden[kDimHead];

#pragma unroll
    for (int c = 0; c < kDimHead; c++) {
        hidden[c] = 0.0;
    }

    auto ly = threadIdx.x;
    auto gy_base = blockDim.x * blockIdx.x;
    auto gy = gy_base + ly;

    float m_softmax = -INFINITY;
    float sum_softmax = 0.0;

    for (int tile_idx = 0; kTileSize * tile_idx < dim_t;
         tile_idx++) {
        // sw: scores then weights
        float sw[kTileSize];

#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            sw[lx] = 0.0;
        }

        // [STEP: phases]
        for (int phase = 0; kTileSize * phase < dim_d;
             phase++) {
            // [STEP: load q]
            for (int row = 0; row < kTileSize; row++) {
                auto col = threadIdx.x;
                if ((gy_base + row) < dim_t &&
                    (kTileSize * phase + col) < dim_d) {
                    sdata.q[row][col] =
                        q[(gy_base + row) * dim_d +
                          (kTileSize * phase + col)];
                } else {
                    sdata.q[row][col] =
                        static_cast<scalar_t>(0);
                }
            }
            __syncthreads();
#pragma unroll
            for (int lx = 0; lx < kTileSize; lx++) {
                q_tile[lx] = sdata.q[ly][lx];
            }
            __syncthreads();

            // [STEP: load k_t]
            for (int row = 0; row < kTileSize; row++) {
                auto col = threadIdx.x;
                if ((kTileSize * phase + row) < dim_d &&
                    (kTileSize * tile_idx + col) < dim_t) {
                    sdata.k[row][col] =
                        k_t[(kTileSize * phase + row) *
                                dim_t +
                            (kTileSize * tile_idx + col)];
                } else {
                    sdata.k[row][col] =
                        static_cast<scalar_t>(0);
                }
            }
            __syncthreads();

            // [STEP: calc q@k_t]
#pragma unroll
            for (int k = 0; k < kTileSize; k++) {
                auto q_val = q_tile[k];

#pragma unroll
                for (int lx = 0; lx < kTileSize; lx++) {
                    sw[lx] += static_cast<float>(q_val) *
                              sdata.k[k][lx];
                }
            }
            __syncthreads();
        }

        // [STEP: FIND m_new]
        auto sqrt_d = static_cast<float>(sqrt(dim_d));
        auto m_new = m_softmax;
#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            sw[lx] /= sqrt_d;

            // !!! [CRITICAL: MASKING] !!!
            // -INF must be given before calc m_new
            auto gx = kTileSize * tile_idx + lx;
            if (!(gx < dim_t)) {
                sw[lx] = -INFINITY;
            }

            m_new = max(m_new, sw[lx]);
        }

        // [STEP: maintain with m_new]
        auto exp_delta = exp(m_softmax - m_new);
        sum_softmax *= exp_delta;
#pragma unroll
        for (int c = 0; c < kDimHead; c++) {
            hidden[c] *= exp_delta;
        }
        m_softmax = m_new;

        // [STEP: convert scores to weights]
#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            // scores => weights
            // !!! [CRITICAL: MASKING] !!!
            // A rare bounds check required for math
            // correctness, not memory safety. Even without
            // array overflow, out-of-bounds default scores
            // (0.0) would evaluate to exp(0.0 - m_new) > 0,
            // silently corrupting the sum_softmax
            // denominator.

            auto gx = kTileSize * tile_idx + lx;
            if (gx < dim_t) {
                sw[lx] = exp(sw[lx] - m_new);
            } else {
                sw[lx] = 0.0;
            }

            sum_softmax += sw[lx];
        }

        // [STEP: phases]
        for (int phase = 0; kTileSize * phase < dim_d;
             phase++) {
            // [STEP: load v]
            for (int row = 0; row < kTileSize; row++) {
                auto col = threadIdx.x;
                if ((kTileSize * tile_idx + row) < dim_t &&
                    (kTileSize * phase + col) < dim_d) {
                    sdata.v[row][col] =
                        v[(kTileSize * tile_idx + row) *
                              dim_d +
                          (kTileSize * phase + col)];
                } else {
                    sdata.v[row][col] =
                        static_cast<scalar_t>(0);
                }
            }
            __syncthreads();

            // [STEP: add weighted v]
#pragma unroll
            for (int lx = 0; lx < kTileSize; lx++) {
                auto weight = sw[lx];

                // !!! [CRITICAL: REGISTER ALLOCATION] !!!
                // We must iterate the full kDimHead to
                // maintain static indexing. Dynamic
                // indexing would force `hidden` to spill to
                // slow HBM.
#pragma unroll
                for (int c = 0; c < kDimHead; c++) {
                    auto lc = c - kTileSize * phase;
                    if (0 <= lc && lc < kTileSize) {
                        hidden[c] +=
                            weight * sdata.v[lx][lc];
                    }
                }
            }
            __syncthreads();
        }

        //         // [DEBUG: output softmax instead]
        // #pragma unroll
        //         for (int lx = 0; lx < kTileSize; lx++) {
        //             auto gx = kTileSize * tile_idx + lx;
        //             if (gx < dim_t) {
        //                 hidden[gx] = sw[lx];
        //             }
        //         }
    }

#pragma unroll
    for (int c = 0; c < kDimHead; c++) {
        if ((gy) < dim_t && (c) < dim_d) {
            out[gy * dim_d + c] = static_cast<scalar_t>(
                hidden[c] / sum_softmax
            );
        }
    }

    //     // [DEBUG: output without sum_softmax]
    // #pragma unroll
    //     for (int c = 0; c < kDimHead; c++) {
    //         if ((gy) < dim_t && (c) < dim_d) {
    //             out[gy * dim_d + c] = hidden[c];
    //         }
    //     }
}

torch::Tensor FlashAttentionCoalescedCuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    TORCH_CHECK_EQ(q.is_cuda(), true);
    TORCH_CHECK_EQ(k.is_cuda(), true);
    TORCH_CHECK_EQ(v.is_cuda(), true);
    TORCH_CHECK_EQ(q.is_contiguous(), true);
    TORCH_CHECK_EQ(k.is_contiguous(), true);
    TORCH_CHECK_EQ(v.is_contiguous(), true);

    TORCH_CHECK_EQ(q.dim(), 4);
    TORCH_CHECK_EQ(k.dim(), 4);
    TORCH_CHECK_EQ(v.dim(), 4);
    for (int i = 0; i < 4; i++) {
        TORCH_CHECK_EQ(q.size(i), k.size(i));
        TORCH_CHECK_EQ(q.size(i), v.size(i));
    }

    int dim_b = q.size(0);
    int dim_h = q.size(1);
    // dim_t == T == q_len
    int dim_t = q.size(2);
    int dim_d = q.size(3);
    TORCH_CHECK_LE(dim_d, kDimHead);

    auto out = torch::empty(
        {dim_b, dim_h, dim_t, dim_d},
        q.options()
    );
    auto k_t = k.transpose(-2, -1).contiguous();

    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        q.scalar_type(),
        "FlashAttentionCoalescedCuda",
        ([&] {
            FlashAttentionCoalescedKernel<scalar_t>
                <<<dim3(
                       (dim_t + kTileSize - 1) / kTileSize,
                       dim_h,
                       dim_b
                   ),
                   kTileSize>>>(
                    q.data_ptr<scalar_t>(),
                    k_t.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    dim_t,
                    dim_d
                );
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        })
    );

    return out;
}
