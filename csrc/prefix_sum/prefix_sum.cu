#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

constexpr int kBlockSize = 1024;

template <typename scalar_t>
__global__ void PrefixSumKernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    int size
) {
    __shared__ scalar_t sdata[kBlockSize + 1];

    auto lx = threadIdx.x;
    auto gx = blockDim.x * blockIdx.x + lx;

    if (gx < size) {
        sdata[lx] = a[gx];
    } else {
        sdata[lx] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int ax = (2 * offset) * (lx + 1) - offset - 1;
        int bx = (2 * offset) * (lx + 1) - 1;
        if (bx < blockDim.x) {
            sdata[bx] += sdata[ax];
        }
        __syncthreads();
    }

    if (lx == blockDim.x - 1) {
        sdata[lx + 1] = sdata[lx];
        sdata[lx] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0;
         offset /= 2) {
        int ax = (2 * offset) * (lx + 1) - offset - 1;
        int bx = (2 * offset) * (lx + 1) - 1;
        if (bx < blockDim.x) {
            auto tmp = sdata[ax];
            sdata[ax] = sdata[bx];
            sdata[bx] += tmp;
        }
        __syncthreads();
    }

    if (gx < size) {
        b[gx] = sdata[lx + 1];
    }
}

torch::Tensor PrefixSumCuda(torch::Tensor a) {
    TORCH_CHECK_EQ(a.is_cuda(), true);
    TORCH_CHECK_EQ(a.is_contiguous(), true);
    TORCH_CHECK_EQ(a.numel() <= kBlockSize, true);

    auto b = torch::empty_like(a);

    int threads = kBlockSize;
    TORCH_CHECK_EQ((threads & (threads - 1)) == 0, true);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "PrefixSumCuda",
        ([&] {
            PrefixSumKernel<scalar_t><<<1, threads>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                a.numel()
            );
            CUDA_CHECK(cudaGetLastError());
        })
    );

    return b;
}
