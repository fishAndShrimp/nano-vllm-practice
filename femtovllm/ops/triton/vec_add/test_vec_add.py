import torch

import femtovllm.ops.triton

SIZE = 2000


a = torch.rand(SIZE, dtype=torch.bfloat16, device="cuda")
b = torch.rand(SIZE, dtype=torch.bfloat16, device="cuda")


out_torch = a + b
out_triton = femtovllm.ops.triton.vec_add_triton(a, b)


print(a)
print(b)
print(out_torch)
print(out_triton)
print(
    f"{torch.allclose(out_torch, out_triton)=}",
)
