#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

constexpr int kBlockSize = 1024;

template <typename scalar_t>
__global__ void ReduceKernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    int size
) {
    __shared__ scalar_t sdata[kBlockSize];

    int lx = threadIdx.x;
    int gx = blockDim.x * blockIdx.x + lx;

    if (gx < size) {
        sdata[lx] = a[gx];
    } else {
        sdata[lx] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    int offset = blockDim.x / 2;
    while (offset > 0) {
        if (lx < offset) {
            sdata[lx] += sdata[lx + offset];
        }
        __syncthreads();
        offset /= 2;
    }

    if (lx == 0) {
        b[0] = sdata[0];
    }
}

torch::Tensor ReduceCuda(torch::Tensor a) {
    TORCH_CHECK_EQ(a.is_cuda(), true);
    TORCH_CHECK_EQ(a.is_contiguous(), true);
    TORCH_CHECK_EQ(a.numel() <= kBlockSize, true);

    auto b = torch::empty(
        //
        {1},
        a.options()
    );

    int size = a.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ReduceCuda",
        ([&] {
            ReduceKernel<scalar_t><<<1, kBlockSize>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                size
            );
            CUDA_CHECK(cudaGetLastError());
        })
    );

    return b;
}