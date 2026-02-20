#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void VecAddKernel(

    const scalar_t* a, const scalar_t* b, scalar_t* c, int size

) {
    int gx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gx < size) {
        c[gx] = a[gx] + b[gx];
    }
}

torch::Tensor VecAddCuda(

    torch::Tensor a, torch::Tensor b

) {
    return a + b;
}
