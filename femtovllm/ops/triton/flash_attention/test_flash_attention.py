import torch
import torch.nn.functional as F

import femtovllm.ops.triton

B = 1
H = 1
T = 2
D = 32


def main():
    q = torch.rand(
        (B, H, T, D),
        dtype=torch.float32,
        device="cuda",
    )
    k = torch.rand(
        (B, H, T, D),
        dtype=torch.float32,
        device="cuda",
    )
    v = torch.rand(
        (B, H, T, D),
        dtype=torch.float32,
        device="cuda",
    )

    out_torch = F.scaled_dot_product_attention(q, k, v)
    out_triton = femtovllm.ops.triton.flash_attention_triton(q, k, v)

    dim_d = q.shape[-1]
    print(out_torch.view(-1, dim_d)[-1])
    print(out_triton.view(-1, dim_d)[-1])
    print(
        f"{torch.allclose(out_torch, out_triton)=}",
    )
    print(f"{(out_torch - out_triton).abs().max()=}")


if __name__ == "__main__":
    main()
