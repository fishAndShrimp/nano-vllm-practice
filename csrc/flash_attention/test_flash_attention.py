import torch
import torch.nn.functional as F

import femtovllm

B = 256
H = 8
T = 512
C = 128


q = torch.randn((B, H, T, C), device="cuda")
k = torch.randn((B, H, T, C), device="cuda")
v = torch.randn((B, H, T, C), device="cuda")


out1 = q @ k.transpose(-2, -1)
out1 /= C**0.5
out1 = F.softmax(out1, dim=-1)
out1 = out1 @ v


out2: torch.Tensor = femtovllm._C.FlashAttentionCuda(q, k, v)
out3: torch.Tensor = femtovllm._C.FlashAttentionCoalescedCuda(q, k, v)


print(out1[-1, -1])
print(out2[-1, -1])
print(out3[-1, -1])
# print(out1)
# print(out2)
# print(out3)
# print(out1 / out2)
# print(out2 / out1)
# print(out1 / out3)
# print(out3 / out1)
print(f"{torch.allclose(out1,out2)=}")
print(f"{torch.allclose(out1,out3)=}")
print(f"{torch.allclose(out2,out3)=}")
print(f"{torch.allclose(out1,out2,atol=1e-3,rtol=1e-3)=}")
print(f"{torch.allclose(out1,out3,atol=1e-3,rtol=1e-3)=}")
print(f"{torch.allclose(out2,out3,atol=1e-3,rtol=1e-3)=}")
print(f"{(out1 - out2).abs().max()=}")
print(f"{(out1 - out3).abs().max()=}")
print(f"{(out2 - out3).abs().max()=}")
