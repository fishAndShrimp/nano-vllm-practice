import multiprocessing
import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the absolute path of the current directory to locate source files
ROOT_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------------
# Build Environment Configuration
# ------------------------------------------------------------------------

# 1. Parallel Compilation
# Dynamically set MAX_JOBS to utilize all available CPU cores (leaving 1 for the OS)
# This significantly speeds up the build process.
num_jobs = max(1, multiprocessing.cpu_count() - 1)

# Cap MAX_JOBS to 4 to prevent Out-Of-Memory (OOM) errors and severe OS swapping.
# Compiling PyTorch/CUDA C++ templates is highly memory-intensive,
# often consuming ~2.5GB of RAM per 'nvcc' process.
num_jobs = min(4, num_jobs)

os.environ["MAX_JOBS"] = str(num_jobs)

# 2. Target GPU Architecture
# If not explicitly set, target the architecture of the current GPU to save compile time.
# If no GPU is present, PyTorch will fallback to compiling for all supported architectures.
if os.environ.get("FAST_BUILD") == "1" and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

# 3. C++11 ABI Compatibility
# Detect the ABI version used by the current PyTorch installation.
# Keep this available in case manual ABI alignment is needed later.
abi_version = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

# ------------------------------------------------------------------------


setup(
    name="femtovllm",
    version="0.1.0",
    # Automatically find the "femtovllm" python package in this directory
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            # The binary will be compiled as a submodule: femtovllm._C
            name="femtovllm._C",
            sources=[
                str(ROOT_DIR / "csrc" / x)
                for x in [
                    "bindings.cpp",
                    "vec_add/vec_add.cu",
                    "relu/relu.cu",
                    "transpose/transpose.cu",
                    "reduce/reduce.cu",
                    "prefix_sum/prefix_sum.cu",
                    "softmax/safe_softmax.cu",
                    "softmax/online_softmax.cu",
                    "softmax/batched_online_softmax.cu",
                    "gemm/gemm.cu",
                    "gemm/gemm_row_wise.cu",
                    "flash_attention/flash_attention.cu",
                    "flash_attention/flash_attention_coalesced.cu",
                    "paged_attention/paged_attention_gemm.cu",
                    "paged_attention/paged_attention_gemv.cu",
                    # Future kernels (e.g., gemm.cu) will be added here
                ]
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    # f"-D_GLIBCXX_USE_CXX11_ABI={abi_version}",
                    # f"-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI={abi_version}",
                ],
                "nvcc": [
                    "-O3",
                    # f"-D_GLIBCXX_USE_CXX11_ABI={abi_version}",
                    # f"-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI={abi_version}",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
