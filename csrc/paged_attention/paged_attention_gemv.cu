#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/constants.cuh"
#include "../utils/cuda_check.cuh"
#include "../utils/warp_utils.cuh"

using femtovllm::kDimHead;
using femtovllm::kKVLenPerPage;
using femtovllm::kWarpSize;

constexpr int kThreadsPerBlock = 256;
static_assert(
    kThreadsPerBlock > kDimHead,
    "kThreadsPerBlock must be greater than kDimHead to "
    "ensure enough threads "
    "are available to compute the accumulated V in "
    "parallel without looping."
);
static_assert(
    kThreadsPerBlock % kWarpSize == 0,
    "kThreadsPerBlock must be a multiple of kWarpSize (32) "
    "to avoid partial warps."
);
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
static_assert(
    kWarpsPerBlock <= kWarpSize,
    "kWarpsPerBlock cannot exceed kWarpSize (32) because "
    "the final block reduction is performed by a single "
    "warp."
);

template <typename scalar_t>
__global__ void PagedAttentionGemvKernel(
    const scalar_t* __restrict__ q_batched,
    int q_len_flatten,
    scalar_t* __restrict__ out_batched,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    int pool_stride_0,
    int pool_stride_1,
    int pool_stride_2,
    const int32_t* __restrict__ kv_page_tables,
    int num_pages_per_seq,
    const int32_t* __restrict__ kv_lens,
    int dim_d,
    int n_rep
) {
    auto seq_idx = blockIdx.x;
    auto head_idx = blockIdx.y;
    auto kv_head_idx = head_idx / n_rep;

    auto q = q_batched + head_idx * q_len_flatten * dim_d;
    auto page_table =
        kv_page_tables + seq_idx * num_pages_per_seq;
    auto kv_len = kv_lens[seq_idx];

    auto lx = threadIdx.x;
    auto warp_idx = threadIdx.y;

    // [STEP: load q]
    // one block for one seq
    __shared__ scalar_t q_seq[kDimHead];
    // (kThreadsPerBlock > kDimHead) hsa be asserted
    // no need to iterate
    {
        auto col = warp_idx * kWarpSize + lx;

        if (col < kDimHead) {
            if (col < dim_d) {
                q_seq[col] = q[seq_idx * dim_d + col];
            } else {
                q_seq[col] = static_cast<scalar_t>(0);
            }
        }
    }
    __syncthreads();

    // [STEP: q@k.T]
    // scores and weights
    // TODO dyna allocate
    __shared__ float sw[1024];

    // each warp maintains its own m and sum for softmax
    float m_softmax_warp = -INFINITY;
    float sum_softmax_warp = 0.0;

    for (int tile_idx = 0;
         tile_idx * kWarpsPerBlock < kv_len;
         tile_idx++) {
        // col for k.T
        auto col_kt = tile_idx * kWarpsPerBlock + warp_idx;

        if (col_kt >= kv_len) {
            break;
        }

        auto page_idx = page_table[col_kt / kKVLenPerPage];
        auto col_page = col_kt % kKVLenPerPage;

        float score = 0.0;
        for (int phase = 0; phase * kWarpSize < dim_d;
             phase++) {
            // row for k.T
            // hence
            // score[seq_idx][col_kt] = sum(
            //     q[seq_idx][row_kt] *
            //            k_t[row_kt][col_kt]
            // )
            auto row_kt = phase * kWarpSize + lx;

            // note that
            // (row_kt, col_kt) is for k.T
            // hence in untransposed condition,
            // (col_kt, row_kt) is for k
            if (row_kt < dim_d) {
                score +=
                    (q_seq[row_kt] *
                     k_pool
                         [pool_stride_0 * page_idx +
                          pool_stride_1 * kv_head_idx +
                          pool_stride_2 * col_page +
                          row_kt]);
            }
        }

        // reduce sum to lane_0
        score = femtovllm::WarpReduceSum(score);

        // only lane_0 to maintain the following
        if (lx == 0) {
            // Do NOT forget to scale
            score /= sqrt(dim_d);

            sw[col_kt] = score;

            if (m_softmax_warp < score) {
                sum_softmax_warp *=
                    expf(m_softmax_warp - score);
                m_softmax_warp = score;
            }

            sum_softmax_warp +=
                expf(score - m_softmax_warp);
        }
    }

    // reduce m and sum for softmax between warps
    __shared__ float m_softmax[kWarpsPerBlock];
    __shared__ float sum_softmax[kWarpsPerBlock];
    if (lx == 0) {
        // transfer m and sum from lane_0 to shared
        m_softmax[warp_idx] = m_softmax_warp;
        sum_softmax[warp_idx] = sum_softmax_warp;
    }
    __syncthreads();

    if (warp_idx == 0) {
        // collect all m and sum in warp_0
        // reuse registers:
        // - m_softmax_warp
        // - sum_softmax_warp

        if (lx < kWarpsPerBlock) {
            m_softmax_warp = m_softmax[lx];
            sum_softmax_warp = sum_softmax[lx];
        } else {
            m_softmax_warp = -INFINITY;
            sum_softmax_warp = 0.0;
        }

        // [NO __syncthreads here]
        // warp_0 will absolutely read shared
        // and then write shared

        // reduce m and sum in warp_0
        auto m_old = m_softmax_warp;
        m_softmax_warp =
            femtovllm::WarpAllReduceMax(m_softmax_warp);

        sum_softmax_warp *= expf(m_old - m_softmax_warp);
        sum_softmax_warp =
            femtovllm::WarpAllReduceSum(sum_softmax_warp);

        if (lx < kWarpsPerBlock) {
            m_softmax[lx] = m_softmax_warp;
            sum_softmax[lx] = sum_softmax_warp;
        }
    }
    __syncthreads();

    // all threads collect final m and max
    // this will NOT cause bank conflict
    // e.g.
    // shared[0] will be collected only once
    // and then broadcast to all lane_0~lane_31
    m_softmax_warp = m_softmax[warp_idx];
    sum_softmax_warp = sum_softmax[warp_idx];
    __syncthreads();

    // now all threads have the final correct m and sum for
    // safe-softmax
    for (int i = 0; i * kThreadsPerBlock < kv_len; i++) {
        auto col = i * kThreadsPerBlock +
                   warp_idx * kWarpSize + lx;

        if (col < kv_len) {
            sw[col] = expf(sw[col] - m_softmax_warp) /
                      sum_softmax_warp;
        }
    }

    // [STEP: weights@v]
    auto out = out_batched +
               head_idx * q_len_flatten * dim_d +
               seq_idx * dim_d;
    {
        auto col_v = warp_idx * kWarpSize + lx;

        if (col_v < dim_d) {
            float attn = 0.0;

            for (int row_v = 0; row_v < kv_len; row_v++) {
                auto page_idx =
                    page_table[row_v / kKVLenPerPage];
                auto row_page = row_v % kKVLenPerPage;

                attn +=
                    sw[row_v] *
                    v_pool
                        [pool_stride_0 * page_idx +
                         pool_stride_1 * kv_head_idx +
                         pool_stride_2 * row_page + col_v];
            }

            out[col_v] = static_cast<scalar_t>(attn);
        }
    }
}

torch::Tensor PagedAttentionGemvCuda(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor kv_page_tables,
    torch::Tensor kv_lens
) {
    TORCH_CHECK_EQ(q.is_cuda(), true);
    TORCH_CHECK_EQ(k_pool.is_cuda(), true);
    TORCH_CHECK_EQ(v_pool.is_cuda(), true);
    TORCH_CHECK_EQ(q.is_contiguous(), true);
    TORCH_CHECK_EQ(k_pool.is_contiguous(), true);
    TORCH_CHECK_EQ(v_pool.is_contiguous(), true);

    TORCH_CHECK_EQ(q.dim(), 3);
    TORCH_CHECK_EQ(k_pool.dim(), 4);
    TORCH_CHECK_EQ(v_pool.dim(), 4);

    TORCH_CHECK_EQ(q.size(-1), k_pool.size(-1));
    TORCH_CHECK_EQ(q.size(-1), v_pool.size(-1));
    for (int i = 0; i < 4; i++) {
        TORCH_CHECK_EQ(k_pool.size(i), v_pool.size(i));
    }

    TORCH_CHECK_EQ(
        kv_page_tables.scalar_type(),
        torch::kInt32
    );
    TORCH_CHECK_EQ(kv_lens.scalar_type(), torch::kInt32);

    int dim_h = q.size(0);
    int q_len_flatten = q.size(1);
    int dim_d = q.size(2);
    TORCH_CHECK_LE(dim_d, kDimHead);
    // Will be better, but not necessarily
    // TORCH_CHECK_EQ(dim_d % kWarpSize, 0);

    TORCH_CHECK_EQ(q_len_flatten, kv_page_tables.size(0));
    TORCH_CHECK_EQ(q_len_flatten, kv_lens.size(0));

    int n_kv_heads = k_pool.size(1);
    // page_size means block_size in paged attn
    int page_size = k_pool.size(2);

    TORCH_CHECK_EQ(dim_h % n_kv_heads, 0);
    int n_rep = dim_h / n_kv_heads;

    auto out = torch::empty_like(q);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "PagedAttentionGemvCuda",
        ([&] {
            PagedAttentionGemvKernel<scalar_t>
                <<<dim3(q_len_flatten, dim_h),
                   dim3(kWarpSize, kWarpsPerBlock)>>>(
                    q.data_ptr<scalar_t>(),
                    q_len_flatten,
                    out.data_ptr<scalar_t>(),
                    k_pool.data_ptr<scalar_t>(),
                    v_pool.data_ptr<scalar_t>(),
                    k_pool.stride(0),
                    k_pool.stride(1),
                    k_pool.stride(2),
                    kv_page_tables.data_ptr<int32_t>(),
                    kv_page_tables.size(-1),
                    kv_lens.data_ptr<int32_t>(),
                    dim_d,
                    n_rep
                );

            CUDA_CHECK(cudaGetLastError());
        })
    );

    return out;
}
