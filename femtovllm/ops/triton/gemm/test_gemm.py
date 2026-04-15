import torch

import femtovllm.ops.triton

M = 23
K = 31
N = 41
DTYPE = torch.bfloat16


a = torch.rand((M, K), dtype=DTYPE, device="cuda")
b = torch.rand((K, N), dtype=DTYPE, device="cuda")


out_torch = a @ b
out_triton = femtovllm.ops.triton.gemm_triton(a, b)


print(a)
print(b)
print(out_torch)
print(out_triton)
print(
    f"{torch.allclose(out_torch, out_triton)=}",
)
print(
    f"{torch.allclose(out_torch, out_triton,rtol=1e-2,atol=1e-2)=}",
)
print(f"{(out_torch - out_triton).abs().max()=}")
