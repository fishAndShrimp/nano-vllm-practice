#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

template <typename scalar_t>
__global__ void ReluKernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    int size
) {
    int gx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gx < size) {
        scalar_t v = a[gx];
        b[gx] = (v > static_cast<scalar_t>(0))
                    ? v
                    : static_cast<scalar_t>(0);
    }
}

torch::Tensor ReluCuda(torch::Tensor a) {
    TORCH_CHECK_EQ(a.is_cuda(), true);
    TORCH_CHECK_EQ(a.is_contiguous(), true);

    auto b = torch::empty_like(a);

    int size = a.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        a.scalar_type(),
        "ReluCuda",
        ([&] {
            ReluKernel<scalar_t><<<blocks, threads>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                size
            );

            CUDA_CHECK(cudaGetLastError());
        })
    );

    return b;
}
