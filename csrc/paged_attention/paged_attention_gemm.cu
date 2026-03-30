#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/constants.cuh"
#include "../utils/cuda_check.cuh"

using femtovllm::kDimHead;
using femtovllm::kTileSize;

template <typename scalar_t>
__global__ void PagedAttentionGemmKernel(
    const scalar_t* __restrict__ q_batched,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    int pool_stride_0,
    int pool_stride_1,
    int pool_stride_2,
    scalar_t* __restrict__ out_batched,
    const int32_t* __restrict__ cu_seqlens,
    int q_len_flatten,
    const int32_t* __restrict__ kv_page_tables,
    int num_pages_per_seq,
    const int32_t* __restrict__ kv_lens,
    int dim_d,
    int n_rep,
    const int32_t* __restrict__ positions
) {
    auto seq_idx = blockIdx.y;
    auto q_begin = cu_seqlens[seq_idx];
    auto q_end = cu_seqlens[seq_idx + 1];

    auto ly = threadIdx.x;
    auto gy_base = q_begin + blockDim.x * blockIdx.x;
    auto gy = gy_base + ly;
    if (gy_base >= q_end) {
        return;
    }

    auto page_table =
        kv_page_tables + num_pages_per_seq * seq_idx;
    auto kv_len = kv_lens[seq_idx];

    auto position = kv_len;
    if (gy < q_end) {
        position = positions[gy];
    }

    auto head_idx = blockIdx.z;
    auto kv_head_idx = head_idx / n_rep;

    auto q = q_batched + q_len_flatten * dim_d * head_idx;
    auto out =
        out_batched + q_len_flatten * dim_d * head_idx;

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

    float m_softmax = -INFINITY;
    float sum_softmax = 0.0;

    for (int tile_idx = 0; kTileSize * tile_idx < kv_len;
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
                if ((gy_base + row) < q_end &&
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

            // [STEP: load k from pool]
            for (int col = 0; col < kTileSize; col++) {
                auto row = threadIdx.x;
                auto page_idx = page_table[tile_idx];

                if ((kTileSize * phase + row) < dim_d &&
                    (kTileSize * tile_idx + col) < kv_len) {
                    // [STEP: pick k from k_pool]
                    // k_pool now is (..., block_size,
                    // dim_d)
                    // We need q@k.T therefore k_pool is
                    // picked by a transposed order
                    sdata.k[row][col] = k_pool
                        [pool_stride_0 * page_idx +
                         pool_stride_1 * kv_head_idx +
                         pool_stride_2 * col +
                         (kTileSize * phase + row)];
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

        // [STEP: SCALE by sqrt(dim_d)]
        // [STEP: MASK by -INF]
        // [STEP: FIND m_new]
        auto sqrt_d = static_cast<float>(sqrt(dim_d));
        auto m_new = m_softmax;
#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            sw[lx] /= sqrt_d;

            // !!! [CRITICAL: MASKING] !!!
            // -INF must be given before calc m_new
            auto gx = kTileSize * tile_idx + lx;
            if (!(gx < kv_len && gx <= position)) {
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
            if (gx < kv_len && gx <= position) {
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
                auto page_idx = page_table[tile_idx];

                if ((kTileSize * tile_idx + row) < kv_len &&
                    (kTileSize * phase + col) < dim_d) {
                    // pick v from v_pool
                    sdata.v[row][col] = v_pool
                        [pool_stride_0 * page_idx +
                         pool_stride_1 * kv_head_idx +
                         pool_stride_2 * row +
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
        if ((gy) < q_end && (c) < dim_d) {
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

torch::Tensor PagedAttentionGemmCuda(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor cu_seqlens,
    int max_q_len,
    torch::Tensor kv_page_tables,
    torch::Tensor kv_lens,
    torch::Tensor positions
) {
    TORCH_CHECK_EQ(q.is_cuda(), true);
    TORCH_CHECK_EQ(k_pool.is_cuda(), true);
    TORCH_CHECK_EQ(v_pool.is_cuda(), true);
    TORCH_CHECK_EQ(q.is_contiguous(), true);
    TORCH_CHECK_EQ(k_pool.is_contiguous(), true);
    TORCH_CHECK_EQ(v_pool.is_contiguous(), true);

    TORCH_CHECK_EQ(cu_seqlens.scalar_type(), torch::kInt32);
    TORCH_CHECK_EQ(
        kv_page_tables.scalar_type(),
        torch::kInt32
    );
    TORCH_CHECK_EQ(kv_lens.scalar_type(), torch::kInt32);
    TORCH_CHECK_EQ(positions.scalar_type(), torch::kInt32);

    int num_seqs = cu_seqlens.numel() - 1;
    TORCH_CHECK_EQ(num_seqs, kv_page_tables.size(0));
    TORCH_CHECK_EQ(num_seqs, kv_lens.size(0));

    TORCH_CHECK_EQ(q.dim(), 3);
    TORCH_CHECK_EQ(k_pool.dim(), 4);
    TORCH_CHECK_EQ(v_pool.dim(), 4);
    TORCH_CHECK_EQ(q.size(-1), k_pool.size(-1));
    TORCH_CHECK_EQ(q.size(-1), v_pool.size(-1));
    // (num_blocks, n_kv_heads, block_size, d_head)
    for (int i = 0; i < 4; i++) {
        TORCH_CHECK_EQ(k_pool.size(i), v_pool.size(i));
    }

    // (dim_h, q_len_flatten, dim_d)
    // (n_heads, q_len_flatten, d_head)
    int dim_h = q.size(0);
    int q_len_flatten = q.size(1);
    int dim_d = q.size(2);
    TORCH_CHECK_LE(dim_d, kDimHead);
    TORCH_CHECK_EQ(q_len_flatten, positions.numel());

    int n_kv_heads = k_pool.size(1);
    TORCH_CHECK_EQ(dim_h % n_kv_heads, 0);
    int n_rep = dim_h / n_kv_heads;

    auto out = torch::empty(
        {dim_h, q_len_flatten, dim_d},
        q.options()
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "PagedAttentionGemmCuda",
        ([&] {
            PagedAttentionGemmKernel<scalar_t><<<
                dim3(
                    (max_q_len + kTileSize - 1) / kTileSize,
                    num_seqs,
                    dim_h
                ),
                kTileSize>>>(
                q.data_ptr<scalar_t>(),
                k_pool.data_ptr<scalar_t>(),
                v_pool.data_ptr<scalar_t>(),
                k_pool.stride(0),
                k_pool.stride(1),
                k_pool.stride(2),
                out.data_ptr<scalar_t>(),
                cu_seqlens.data_ptr<int32_t>(),
                q_len_flatten,
                kv_page_tables.data_ptr<int32_t>(),
                static_cast<int>(kv_page_tables.size(-1)),
                kv_lens.data_ptr<int32_t>(),
                dim_d,
                n_rep,
                positions.data_ptr<int32_t>()
            );

            // // Comment this after finishing DEBUG
            // CUDA_CHECK(cudaGetLastError());
            // cudaDeviceSynchronize();
        })
    );

    return out;
}
