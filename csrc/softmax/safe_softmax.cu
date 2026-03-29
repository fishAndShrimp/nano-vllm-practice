#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

constexpr int kThreadsPerBlock = 256;

template <typename scalar_t>
__global__ void SafeSoftmaxKernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    int size
) {
    __shared__ scalar_t sdata[kThreadsPerBlock];

    int lx = threadIdx.x;

    scalar_t m = static_cast<scalar_t>(-INFINITY);
    scalar_t m_thread = static_cast<scalar_t>(-INFINITY);
    for (int phase = 0; blockDim.x * phase < size;
         phase++) {
        int gx = blockDim.x * phase + lx;
        if (gx < size) {
            m_thread = max(m_thread, a[gx]);
        }
    }
    sdata[lx] = m_thread;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0;
         offset /= 2) {
        if (lx < offset) {
            sdata[lx] = max(sdata[lx], sdata[lx + offset]);
        }
        __syncthreads();
    }
    m = max(m, sdata[0]);
    __syncthreads();

    scalar_t sum = static_cast<scalar_t>(0);
    scalar_t sum_thread = static_cast<scalar_t>(0);
    for (int phase = 0; blockDim.x * phase < size;
         phase++) {
        int gx = blockDim.x * phase + lx;
        if (gx < size) {
            sum_thread +=
                static_cast<scalar_t>(exp(a[gx] - m));
        }
    }
    sdata[lx] = sum_thread;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0;
         offset /= 2) {
        if (lx < offset) {
            sdata[lx] += sdata[lx + offset];
        }
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    for (int phase = 0; blockDim.x * phase < size;
         phase++) {
        int gx = blockDim.x * phase + lx;
        if (gx < size) {
            b[gx] =
                static_cast<scalar_t>(exp(a[gx] - m)) / sum;
        }
    }
}

torch::Tensor SafeSoftmaxCuda(torch::Tensor a) {
    TORCH_CHECK_EQ(a.is_cuda(), true);
    TORCH_CHECK_EQ(a.is_contiguous(), true);

    auto b = torch::empty_like(a);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "SafeSoftmaxCuda",
        ([&] {
            SafeSoftmaxKernel<scalar_t><<<
                //
                1,
                kThreadsPerBlock>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                a.numel()
            );
            CUDA_CHECK(cudaGetLastError());
        })
    );

    return b;
}
