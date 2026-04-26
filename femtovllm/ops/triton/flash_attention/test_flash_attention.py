import torch
import torch.nn.functional as F

import femtovllm.ops.triton

B = 16
H = 12
T = 256
D = 128
DTYPE = torch.bfloat16


def main():
    q = torch.rand(
        (B, H, T, D),
        dtype=DTYPE,
        device="cuda",
    )
    k = torch.rand(
        (B, H, T, D),
        dtype=DTYPE,
        device="cuda",
    )
    v = torch.rand(
        (B, H, T, D),
        dtype=DTYPE,
        device="cuda",
    )

    out_torch = F.scaled_dot_product_attention(q, k, v)
    out_triton = femtovllm.ops.triton.flash_attention_triton(q, k, v)

    print(out_torch.view(-1, D)[-1])
    print(out_triton.view(-1, D)[-1])

    print(
        f"{torch.allclose(out_torch, out_triton, rtol=1e-2, atol=1e-2)=}",
    )
    print(f"{(out_torch - out_triton).abs().max()=}")


if __name__ == "__main__":
    main()
