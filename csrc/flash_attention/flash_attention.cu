#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

constexpr int kTileSize = 32;
constexpr int kDimHead = 128;

template <typename scalar_t>
__global__ void FlashAttentionKernel(
    const scalar_t* __restrict__ q_batched,
    const scalar_t* __restrict__ k_t_batched,
    const scalar_t* __restrict__ v_batched,
    scalar_t* __restrict__ out_batched,
    int dim_t,
    int dim_c
) {
    int batch_idx = (blockIdx.z) * gridDim.y + (blockIdx.y);
    auto q = q_batched + dim_t * dim_c * batch_idx;
    auto k_t = k_t_batched + dim_t * dim_c * batch_idx;
    auto v = v_batched + dim_t * dim_c * batch_idx;
    auto out = out_batched + dim_t * dim_c * batch_idx;

    scalar_t q_tile[kTileSize];
    __shared__ scalar_t k_tile[kTileSize][kTileSize + 1];

    auto ly = threadIdx.x;
    auto gy = blockDim.x * blockIdx.x + ly;

    for (int tile_idx = 0; kTileSize * tile_idx < dim_t;
         tile_idx++) {
        float pvalues[kTileSize];

#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            pvalues[lx] = 0.0;
        }

        for (int phase = 0; kTileSize * phase < dim_c;
             phase++) {
#pragma unroll
            for (int lx = 0; lx < kTileSize; lx++) {
                if ((gy) < dim_t &&
                    (kTileSize * phase + lx) < dim_c) {
                    q_tile[lx] =
                        q[(gy)*dim_c +
                          (kTileSize * phase + lx)];
                } else {
                    q_tile[lx] = static_cast<scalar_t>(0);
                }
            }

            for (int row = 0; row < kTileSize; row++) {
                auto col = threadIdx.x;
                if ((kTileSize * phase + row) < dim_c &&
                    (kTileSize * tile_idx + col) < dim_t) {
                    k_tile[row][col] =
                        k_t[(kTileSize * phase + row) *
                                dim_t +
                            (kTileSize * tile_idx + col)];
                } else {
                    k_tile[row][col] =
                        static_cast<scalar_t>(0);
                }
            }
            __syncthreads();

#pragma unroll
            for (int k = 0; k < kTileSize; k++) {
                auto q_val = q_tile[k];

#pragma unroll
                for (int lx = 0; lx < kTileSize; lx++) {
                    pvalues[lx] +=
                        static_cast<float>(q_val) *
                        k_tile[k][lx];
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            auto gx = kTileSize * tile_idx + lx;
            if (gy < dim_t && gx < dim_t) {
                out[(gy)*dim_t + (gx)] =
                    static_cast<scalar_t>(pvalues[lx]);
            }
        }
    }
}

torch::Tensor FlashAttentionCuda(
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
    int dim_t = q.size(2);
    int dim_c = q.size(3);

    auto out = torch::empty(
        {dim_b, dim_h, dim_t, dim_c},
        q.options()
    );
    auto k_t = k.transpose(-2, -1).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "FlashAttentionCuda",
        ([&] {
            FlashAttentionKernel<scalar_t>
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
                    dim_c
                );
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        })
    );

    return out;
}
