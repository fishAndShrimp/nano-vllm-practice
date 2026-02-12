import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the absolute path of the current directory to locate source files
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
