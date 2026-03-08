#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/cuda_check.cuh"

template <typename scalar_t>
__global__ void VecAddKernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int size
) {
    int gx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gx < size) {
        c[gx] = a[gx] + b[gx];
        // printf("c:%f a:%f b:%f", c[gx], a[gx], b[gx]);
    }
}

torch::Tensor VecAddCuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK_EQ(a.is_cuda(), true);
    TORCH_CHECK_EQ(b.is_cuda(), true);
    TORCH_CHECK_EQ(a.is_contiguous(), true);
    TORCH_CHECK_EQ(b.is_contiguous(), true);
    TORCH_CHECK_EQ(a.numel(), b.numel());

    torch::Tensor c = torch::empty_like(a);

    int size = a.numel();

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "VecAddCuda",
        ([&] {
            VecAddKernel<<<blocks, threads>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                c.data_ptr<scalar_t>(),
                size
            );

            CUDA_CHECK(cudaGetLastError());
        })
    );

    return c;
}

torch::Tensor
VecAddRaw(torch::Tensor a_cpu, torch::Tensor b_cpu) {
    TORCH_CHECK_EQ(a_cpu.is_cpu(), true);
    TORCH_CHECK_EQ(b_cpu.is_cpu(), true);
    TORCH_CHECK_EQ(a_cpu.is_contiguous(), true);
    TORCH_CHECK_EQ(b_cpu.is_contiguous(), true);
    TORCH_CHECK_EQ(a_cpu.numel(), b_cpu.numel());
    TORCH_CHECK_EQ(a_cpu.scalar_type(), torch::kFloat32);
    TORCH_CHECK_EQ(b_cpu.scalar_type(), torch::kFloat32);

    int size = a_cpu.numel();
    int volume = size * sizeof(float);
    auto c_cpu = torch::empty_like(a_cpu);

    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&a_d, volume));
    CUDA_CHECK(cudaMalloc((void**)&b_d, volume));
    CUDA_CHECK(cudaMalloc((void**)&c_d, volume));

    CUDA_CHECK(cudaMemcpy(
        a_d,
        a_cpu.data_ptr<float>(),
        volume,
        cudaMemcpyDefault
    ));
    CUDA_CHECK(cudaMemcpy(
        b_d,
        b_cpu.data_ptr<float>(),
        volume,
        cudaMemcpyDefault
    ));

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    VecAddKernel<<<blocks, threads>>>(a_d, b_d, c_d, size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        c_cpu.data_ptr<float>(),
        c_d,
        volume,
        cudaMemcpyDefault
    ));

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(c_d));

    return c_cpu;
}
