#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

constexpr int kTileSize = 32;

template <typename scalar_t>
__global__ void GemmKernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int dim_n,
    int dim_m,
    int dim_p
) {
    auto lx = threadIdx.x;
    auto gx = blockDim.x * blockIdx.x + lx;
    auto ly = threadIdx.y;
    auto gy = blockDim.y * blockIdx.y + ly;

    __shared__ scalar_t a_tile[kTileSize][kTileSize + 1];
    __shared__ scalar_t b_tile[kTileSize][kTileSize + 1];

    // scalar_t pvalue = static_cast<scalar_t>(0);
    float pvalue = 0;
    for (int phase = 0; phase * kTileSize < dim_m;
         phase++) {
        if ((gy) < dim_n &&
            (phase * kTileSize + lx) < dim_m) {
            a_tile[ly][lx] =
                a[(gy)*dim_m + (phase * kTileSize + lx)];
        } else {
            a_tile[ly][lx] = static_cast<scalar_t>(0);
        }

        if ((phase * kTileSize + ly) < dim_m &&
            (gx) < dim_p) {
            b_tile[ly][lx] =
                b[(phase * kTileSize + ly) * dim_p + (gx)];
        } else {
            b_tile[ly][lx] = static_cast<scalar_t>(0);
        }
        __syncthreads();

        for (int k = 0; k < kTileSize; k++) {
            pvalue += static_cast<float>(a_tile[ly][k]) *
                      b_tile[k][lx];
        }
        __syncthreads();
    }

    if (gy < dim_n && gx < dim_p) {
        c[gy * dim_p + gx] = static_cast<scalar_t>(pvalue);
    }
}

torch::Tensor GemmCuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK_EQ(a.is_cuda(), true);
    TORCH_CHECK_EQ(b.is_cuda(), true);
    TORCH_CHECK_EQ(a.is_contiguous(), true);
    TORCH_CHECK_EQ(b.is_contiguous(), true);

    TORCH_CHECK_EQ(a.dim() >= 2, true);
    TORCH_CHECK_EQ(b.dim() >= 2, true);
    TORCH_CHECK_EQ(a.size(-1), b.size(-2));

    int dim_n = a.size(-2);
    int dim_m = a.size(-1);
    int dim_p = b.size(-1);

    TORCH_CHECK_EQ(
        a.numel() / dim_n / dim_m,
        b.numel() / dim_m / dim_p
    );

    auto c = torch::empty(
        //
        {dim_n, dim_p},
        a.options()
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "GemmCuda",
        ([&] {
            GemmKernel<scalar_t>
                <<<dim3(
                       (dim_p + kTileSize - 1) / kTileSize,
                       (dim_n + kTileSize - 1) / kTileSize
                   ),
                   dim3(kTileSize, kTileSize)>>>(
                    a.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    c.data_ptr<scalar_t>(),
                    dim_n,
                    dim_m,
                    dim_p
                );
            CUDA_CHECK(cudaGetLastError());
        })
    );

    return c;
}
