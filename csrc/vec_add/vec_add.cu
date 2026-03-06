#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

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
    }
}

torch::Tensor VecAddCuda(torch::Tensor a, torch::Tensor b) {
    torch::Tensor c = torch::empty_like(a);

    int size = a.numel();

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        a.scalar_type(),
        "VecAddCuda",
        ([&] {
            VecAddKernel<scalar_t><<<blocks, threads>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                c.data_ptr<scalar_t>(),
                size
            );
        })
    );

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("VecAddCuda", &VecAddCuda);
}
