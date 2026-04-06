import torch
from torch.nn import functional as F

import femtovllm

print(F.__name__)


B = 4
H = 2
T = 64
C = 128


def main():
    q = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, T, C), device="cuda", dtype=torch.float16)

    # raw = [0] * 128
    # raw[0] = 1
    # raw[1] = 2
    # raw[2] = 3

    # q = torch.tensor(raw, device="cuda", dtype=torch.float16).reshape(1, 1, 1, 128)
    # k = torch.tensor(raw * 5, device="cuda", dtype=torch.float16).reshape(1, 1, 5, 128)
    # v = torch.tensor(raw * 5, device="cuda", dtype=torch.float16).reshape(1, 1, 5, 128)

    out1 = q @ k.transpose(-2, -1)
    out1 /= C**0.5
    out1 = F.softmax(out1, dim=-1)
    out1 = out1 @ v

    out2: torch.Tensor = femtovllm._C.FlashAttentionWarpCuda(q, k, v)
    # out3: torch.Tensor = femtovllm._C.FlashAttentionCoalescedCuda(q, k, v)
    torch.cuda.synchronize()

    print(out1[-1, -1])
    print(out2[-1, -1])
    # print(out3[-1, -1])
    # print(out1)
    # print(out2)
    # print(out3)
    # print(out1 / out2)
    # print(out2 / out1)
    # print(out1 / out3)
    # print(out3 / out1)
    print(f"{torch.allclose(out1,out2)=}")
    # print(f"{torch.allclose(out1,out3)=}")
    # print(f"{torch.allclose(out2,out3)=}")
    print(f"{torch.allclose(out1,out2,atol=1e-3,rtol=1e-3)=}")
    # print(f"{torch.allclose(out1,out3,atol=1e-3,rtol=1e-3)=}")
    # print(f"{torch.allclose(out2,out3,atol=1e-3,rtol=1e-3)=}")
    print(f"{(out1 - out2).abs().max()=}")
    # print(f"{(out1 - out3).abs().max()=}")
    # print(f"{(out2 - out3).abs().max()=}")


if __name__ == "__main__":
    main()
