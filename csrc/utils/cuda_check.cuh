#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void _CudaCheckImpl(
    cudaError_t code,
    const char* file,
    int line
) {
    if (code != cudaSuccess) {
        std::string msg = std::string("CUDA ERROR: ") +
                          cudaGetErrorString(code) +
                          " at " + file + ":" +
                          std::to_string(line);
        throw std::runtime_error(msg);
    }
}

#define CUDA_CHECK(val) \
    _CudaCheckImpl((val), __FILE__, __LINE__)
