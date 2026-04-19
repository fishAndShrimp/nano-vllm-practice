import torch
import torch.nn.functional as F

import femtovllm.ops.triton


def main():
    a = torch.rand(
        (19, 23),
        dtype=torch.bfloat16,
        device="cuda",
    )
    # a = torch.tensor(
    #     [[1, 1]],
    #     dtype=torch.bfloat16,
    #     device="cuda",
    # )

    out_torch = F.softmax(a, dim=-1)
    out_triton = femtovllm.ops.triton.online_softmax_triton(a)

    dim_d = a.shape[-1]

    print(a.view(-1, dim_d)[-1])
    print(out_torch.view(-1, dim_d)[-1])
    print(out_triton.view(-1, dim_d)[-1])
    print(
        f"{torch.allclose(out_torch, out_triton)=}",
    )
    print(f"{(out_torch - out_triton).abs().max()=}")


if __name__ == "__main__":
    main()
