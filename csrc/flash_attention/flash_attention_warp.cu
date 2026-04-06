#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/constants.cuh"
#include "../utils/cuda_check.cuh"
#include "../utils/warp_utils.cuh"

using femtovllm::kDimHead;

using femtovllm::kThreadsPerBlock;
using femtovllm::kWarpSize;
using femtovllm::kWarpsPerBlock;

constexpr int kDimPerThread = kDimHead / kWarpSize;

constexpr int kQTileSize = 32;
static_assert(kQTileSize % kWarpsPerBlock == 0);

constexpr int kKVTileSize = 32;
static_assert(kKVTileSize % kWarpsPerBlock == 0);

template <typename scalar_t>
__global__ void FlashAttentionWarpKernel(
    const scalar_t* __restrict__ q_batched,
    const scalar_t* __restrict__ k_batched,
    const scalar_t* __restrict__ v_batched,
    scalar_t* __restrict__ out_batched,
    int n_rep,
    int q_len,
    int kv_len
) {
    constexpr int kPackSize =
        sizeof(float4) / sizeof(scalar_t);
    static_assert(kDimHead % kPackSize == 0);
    constexpr int kPacksPerHead = kDimHead / kPackSize;
    const float scale_softmax = 1.0 / sqrt(kDimHead);

    auto batch_idx = blockIdx.y;
    auto head_idx = blockIdx.z;
    auto kv_head_idx = head_idx / n_rep;

    auto dim_h = gridDim.z;
    auto dim_h_kv = dim_h / n_rep;

    auto q = q_batched + (batch_idx * dim_h + head_idx) *
                             q_len * kDimHead;
    auto out =
        out_batched +
        (batch_idx * dim_h + head_idx) * q_len * kDimHead;

    auto k =
        k_batched + (batch_idx * dim_h_kv + kv_head_idx) *
                        kv_len * kDimHead;
    auto v =
        v_batched + (batch_idx * dim_h_kv + kv_head_idx) *
                        kv_len * kDimHead;

    __shared__ scalar_t
        q_s[kQTileSize][kDimHead + kPackSize];
    __shared__ scalar_t
        k_s[kKVTileSize][kDimHead + kPackSize];
    __shared__ scalar_t
        v_s[kKVTileSize][kDimHead + kPackSize];

    auto lx = threadIdx.x;
    auto ly = threadIdx.y;
    auto tid = threadIdx.y * kWarpSize + lx;
    auto q_row_base = blockIdx.x * kQTileSize;

    auto q_s_vec = reinterpret_cast<float4*>(q_s);
    auto q_vec = reinterpret_cast<const float4*>(q);
    for (int phase = 0; phase * kThreadsPerBlock <
                        kQTileSize * kPacksPerHead;
         phase++) {
        auto flat = phase * kThreadsPerBlock + tid;
        auto col = flat % kPacksPerHead;
        auto row = flat / kPacksPerHead;

        //////////////////
        //   (kDimHead + kPackSize)
        //             / kPackSize
        // = (kPacksPerHead + 1)
        //////////////////
        if ((q_row_base + row) < q_len) {
            q_s_vec[row * (kPacksPerHead + 1) + col] = q_vec
                [(q_row_base + row) * kPacksPerHead + col];
        } else {
            q_s_vec[row * (kPacksPerHead + 1) + col] =
                make_float4(0.0, 0.0, 0.0, 0.0);
        }
    }
    __syncthreads();

    for (int coarse = 0;
         coarse * kWarpsPerBlock < kQTileSize;
         coarse++) {
        //////////////////
        // coarse in q
        //////////////////

        //////////////////
        // init state per thread for this q tile
        //////////////////
        float hidden[kDimPerThread];
#pragma unroll
        for (int i = 0; i < kDimPerThread; i++) {
            hidden[i] = 0.0;
        }

        scalar_t q_w[kDimPerThread];
#pragma unroll
        for (int i = 0; i < kDimPerThread; i++) {
            q_w[i] = q_s[coarse * kWarpsPerBlock + ly]
                        [i * kWarpSize + lx];
        }

        float m_softmax = -INFINITY;
        float sum_softmax = 0.0;

        for (int tile_idx = 0;
             tile_idx * kKVTileSize < kv_len;
             tile_idx++) {
            auto kv_row_base = tile_idx * kKVTileSize;

            //////////////////
            // load k tile
            //////////////////
            auto k_s_vec = reinterpret_cast<float4*>(k_s);
            auto k_vec = reinterpret_cast<const float4*>(k);
            for (int phase = 0; phase * kThreadsPerBlock <
                                kKVTileSize * kPacksPerHead;
                 phase++) {
                auto flat = phase * kThreadsPerBlock + tid;
                auto col = flat % kPacksPerHead;
                auto row = flat / kPacksPerHead;

                if ((kv_row_base + row) < kv_len) {
                    k_s_vec
                        [row * (kPacksPerHead + 1) + col] =
                            k_vec
                                [(kv_row_base + row) *
                                     kPacksPerHead +
                                 col];
                } else {
                    k_s_vec
                        [row * (kPacksPerHead + 1) + col] =
                            make_float4(0.0, 0.0, 0.0, 0.0);
                }
            }
            // ensure k only
            // next step needs k only
            __syncthreads();

            //////////////////
            // load v tile
            //////////////////
            auto v_s_vec = reinterpret_cast<float4*>(v_s);
            auto v_vec = reinterpret_cast<const float4*>(v);
            for (int phase = 0; phase * kThreadsPerBlock <
                                kKVTileSize * kPacksPerHead;
                 phase++) {
                auto flat = phase * kThreadsPerBlock + tid;
                auto col = flat % kPacksPerHead;
                auto row = flat / kPacksPerHead;

                if ((kv_row_base + row) < kv_len) {
                    v_s_vec
                        [row * (kPacksPerHead + 1) + col] =
                            v_vec
                                [(kv_row_base + row) *
                                     kPacksPerHead +
                                 col];
                } else {
                    v_s_vec
                        [row * (kPacksPerHead + 1) + col] =
                            make_float4(0.0, 0.0, 0.0, 0.0);
                }
            }

            //////////////////
            // HERE sw[2] is hardcoded for
            // kKVTileSize <= 64
            //////////////////
            float sw[2] = {-INFINITY, -INFINITY};
            for (int kt_col = 0; kt_col < kKVTileSize;
                 kt_col++) {
                if (tile_idx * kKVTileSize + kt_col >=
                    kv_len) {
                    break;
                }

                float score = 0.0;

#pragma unroll
                for (int i = 0; i * kWarpSize < kDimHead;
                     i++) {
                    // no if, ensure kDimHead = 32 * k
                    score +=
                        q_w[i] *
                        k_s[kt_col][i * kWarpSize + lx];
                }

                score = femtovllm::WarpAllReduceSum(score);

                score *= scale_softmax;
                if (kt_col == lx) {
                    sw[0] = score;
                } else if (kt_col == lx + kWarpSize) {
                    sw[1] = score;
                }
            }

            auto m_sub = fmaxf(sw[0], sw[1]);
            m_sub = femtovllm::WarpAllReduceMax(m_sub);

            if (m_softmax < m_sub) {
                auto exp_delta = expf(m_softmax - m_sub);

#pragma unroll
                for (int i = 0; i < kDimPerThread; i++) {
                    hidden[i] *= exp_delta;
                }
                sum_softmax *= exp_delta;

                m_softmax = m_sub;
            }

            sw[0] = expf(sw[0] - m_softmax);
            sw[1] = expf(sw[1] - m_softmax);
            auto sum_sub = sw[0] + sw[1];

            sum_sub = femtovllm::WarpAllReduceSum(sum_sub);
            sum_softmax += sum_sub;
            //////////////////
            // HERE sw[2] is hardcoded for
            // kKVTileSize <= 64
            //////////////////

            // ensure v
            // next step needs v
            __syncthreads();

            for (int v_row = 0; v_row < kKVTileSize;
                 v_row++) {
                float weight = sw[0];
                if (kWarpSize <= v_row) {
                    weight = sw[1];
                }

                weight = __shfl_sync(
                    femtovllm::kWarpMaskFull,
                    weight,
                    v_row % kWarpSize
                );

#pragma unroll
                for (int i = 0; i < kDimPerThread; i++) {
                    hidden[i] +=
                        weight *
                        v_s[v_row][i * kWarpSize + lx];
                }
            }
        }

        {
            auto q_coarse = (coarse * kWarpsPerBlock + ly);

            if ((q_row_base + q_coarse) < q_len) {
#pragma unroll
                for (int i = 0; i < kDimPerThread; i++) {
                    out[(q_row_base + q_coarse) * kDimHead +
                        (i * kWarpSize + lx)] =
                        hidden[i] / sum_softmax;
                }
            }
        }

        //////////////////
        // coarse in q
        //////////////////
    }
}

torch::Tensor FlashAttentionWarpCuda(
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
    TORCH_CHECK_EQ(
        reinterpret_cast<uintptr_t>(q.data_ptr()) %
            sizeof(float4),
        0
    );
    TORCH_CHECK_EQ(
        reinterpret_cast<uintptr_t>(k.data_ptr()) %
            sizeof(float4),
        0
    );
    TORCH_CHECK_EQ(
        reinterpret_cast<uintptr_t>(v.data_ptr()) %
            sizeof(float4),
        0
    );

    TORCH_CHECK_EQ(q.dim(), 4);
    TORCH_CHECK_EQ(k.dim(), 4);
    TORCH_CHECK_EQ(v.dim(), 4);

    TORCH_CHECK_EQ(q.size(0), k.size(0));
    TORCH_CHECK_EQ(q.size(0), v.size(0));
    int dim_b = q.size(0);

    TORCH_CHECK_EQ(k.size(1), v.size(1));
    int dim_h = q.size(1);
    int n_kv_heads = k.size(1);
    TORCH_CHECK_EQ(dim_h % n_kv_heads, 0);
    int n_rep = dim_h / n_kv_heads;

    TORCH_CHECK_EQ(k.size(2), v.size(2));
    int q_len = q.size(2);
    int kv_len = k.size(2);

    TORCH_CHECK_EQ(q.size(3), k.size(3));
    TORCH_CHECK_EQ(q.size(3), v.size(3));
    int dim_d = q.size(3);
    TORCH_CHECK_EQ(dim_d, kDimHead);

    auto out = torch::empty_like(q);

    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        q.scalar_type(),
        "FlashAttentionWarpCuda",
        ([&] {
            FlashAttentionWarpKernel<<<
                dim3(
                    (q_len + kQTileSize - 1) / kQTileSize,
                    dim_b,
                    dim_h
                ),
                dim3(kWarpSize, kWarpsPerBlock)>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                n_rep,
                q_len,
                kv_len
            );
        })
    );

    return out;
}
