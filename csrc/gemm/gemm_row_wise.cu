#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

constexpr int kTileSize = 32;

template <typename scalar_t>
__global__ void GemmRowWiseKernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int dim_m,
    int dim_k,
    int dim_n
) {
    scalar_t a_tile[kTileSize];
    __shared__ scalar_t b_tile[kTileSize][kTileSize + 1];

    auto ly = threadIdx.x;
    auto gy = blockDim.x * blockIdx.x + ly;

    for (int tile_idx = 0; kTileSize * tile_idx < dim_n;
         tile_idx++) {
        float pvalues[kTileSize];

#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            pvalues[lx] = 0.0;
        }

        for (int phase = 0; kTileSize * phase < dim_k;
             phase++) {
#pragma unroll
            for (int lx = 0; lx < kTileSize; lx++) {
                if ((gy) < dim_m &&
                    (kTileSize * phase + lx) < dim_k) {
                    a_tile[lx] =
                        a[(gy)*dim_k +
                          (kTileSize * phase + lx)];
                } else {
                    a_tile[lx] = static_cast<scalar_t>(0);
                }
            }

            for (int row = 0; row < kTileSize; row++) {
                auto col = threadIdx.x;
                if ((kTileSize * phase + row) < dim_k &&
                    (kTileSize * tile_idx + col) < dim_n) {
                    b_tile[row][col] =
                        b[(kTileSize * phase + row) *
                              dim_n +
                          (kTileSize * tile_idx + col)];
                } else {
                    b_tile[row][col] =
                        static_cast<scalar_t>(0);
                }
            }
            __syncthreads();

            // if (blockIdx.x == 0 && blockIdx.y == 0 &&
            //     threadIdx.x == 0 && threadIdx.y == 0) {
            //     printf(">>> [Debug] a_tile Row\n");
            //     for (int k = 0; k < kTileSize; ++k) {
            //         printf("%f ", a_tile[k]);
            //     }
            //     printf("\n");

            //     printf(">>> [Debug] b_tile Col0\n");
            //     for (int k = 0; k < kTileSize; ++k) {
            //         printf("%f ", b_tile[k][0]);
            //     }
            //     printf("\n");

            //     printf(">>> [Debug] b_tile Col1\n");
            //     for (int k = 0; k < kTileSize; ++k) {
            //         printf("%f ", b_tile[k][1]);
            //     }
            //     printf("\n");
            // }
            // __syncthreads();

#pragma unroll
            for (int k = 0; k < kTileSize; k++) {
                auto a_val = a_tile[k];

#pragma unroll
                for (int lx = 0; lx < kTileSize; lx++) {
                    pvalues[lx] +=
                        static_cast<float>(a_val) *
                        b_tile[k][lx];
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int lx = 0; lx < kTileSize; lx++) {
            auto gx = kTileSize * tile_idx + lx;
            if (gy < dim_m && gx < dim_n) {
                c[(gy)*dim_n + (gx)] =
                    static_cast<scalar_t>(pvalues[lx]);
            }
        }
    }
}

torch::Tensor
GemmRowWiseCuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK_EQ(a.is_cuda(), true);
    TORCH_CHECK_EQ(b.is_cuda(), true);
    TORCH_CHECK_EQ(a.is_contiguous(), true);
    TORCH_CHECK_EQ(b.is_contiguous(), true);

    TORCH_CHECK_EQ(a.dim() >= 2, true);
    TORCH_CHECK_EQ(b.dim() >= 2, true);
    TORCH_CHECK_EQ(a.size(-1), b.size(-2));

    int dim_m = a.size(-2);
    int dim_k = a.size(-1);
    int dim_n = b.size(-1);
    TORCH_CHECK_EQ(
        a.numel() / dim_m / dim_k,
        b.numel() / dim_k / dim_n
    );

    auto c = torch::empty(
        //
        {dim_m, dim_n},
        a.options()
    );

    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        a.scalar_type(),
        "GemmRowWiseCuda",
        ([&] {
            GemmRowWiseKernel<scalar_t>
                <<<(dim_m + kTileSize - 1) / kTileSize,
                   kTileSize>>>(
                    a.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    c.data_ptr<scalar_t>(),
                    dim_m,
                    dim_k,
                    dim_n
                );
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        })
    );

    return c;
}
