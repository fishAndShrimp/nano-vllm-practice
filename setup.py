import multiprocessing
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the absolute path of the current directory to locate source files
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


num_jobs = max(1, multiprocessing.cpu_count() - 1)
os.environ["MAX_JOBS"] = str(num_jobs)


if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


abi_version = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


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
                os.path.join(ROOT_DIR, "csrc", "vec_add", "vec_add.cu"),
                # Future kernels (e.g., gemm.cu) will be added here
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
