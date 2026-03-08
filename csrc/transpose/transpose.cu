#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

constexpr int kTileDim = 32;

template <typename scalar_t>
__global__ void TransposeKernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    int dim_y,
    int dim_x
) {
    __shared__ scalar_t tile[kTileDim][kTileDim + 1];

    int lx = threadIdx.x;
    int gx = blockDim.x * blockIdx.x + lx;
    int ly = threadIdx.y;
    int gy = blockDim.y * blockIdx.y + ly;

    if (gy < dim_y && gx < dim_x) {
        tile[ly][lx] = a[gy * dim_x + gx];
    } else {
        tile[ly][lx] = static_cast<scalar_t>(0);
    }

    __syncthreads();

    int gy2 = blockDim.x * blockIdx.x + ly;
    int gx2 = blockDim.y * blockIdx.y + lx;

    if (gy2 < dim_x && gx2 < dim_y) {
        b[gy2 * dim_y + gx2] = tile[lx][ly];
    }
}

torch::Tensor TransposeCuda(torch::Tensor a) {
    TORCH_CHECK_EQ(a.is_cuda(), true);
    TORCH_CHECK_EQ(a.is_contiguous(), true);
    TORCH_CHECK_EQ(a.dim(), 2);

    int dim_y = a.size(0);
    int dim_x = a.size(1);

    auto b = torch::empty(
        //
        {dim_x, dim_y},
        a.options()
    );

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "TransposeCuda",
        ([&] {
            TransposeKernel<scalar_t>
                <<<dim3(
                       (dim_x + kTileDim - 1) / kTileDim,
                       (dim_y + kTileDim - 1) / kTileDim
                   ),
                   dim3(kTileDim, kTileDim)>>>(
                    a.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    dim_y,
                    dim_x
                );
            CUDA_CHECK(cudaGetLastError());
        })
    );

    return b;
}