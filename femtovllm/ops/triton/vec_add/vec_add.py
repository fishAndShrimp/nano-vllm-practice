import torch
import triton
import triton.language as tl


@triton.jit
def vec_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    """ """
    pid = tl.program_id(axis=0)

    offset_base = pid * BLOCK_SIZE

    offsets = offset_base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a + b

    tl.store(out_ptr + offsets, out, mask=mask)


def vec_add_triton(a: torch.Tensor, b: torch.Tensor):
    """ """
    assert a.is_cuda
    assert b.is_cuda

    assert a.is_contiguous()
    assert b.is_contiguous()

    assert a.numel() == b.numel()
    size = a.numel()

    def grid(meta):
        return (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    out = torch.empty_like(a)
    vec_add_kernel[grid](a, b, out, size, BLOCK_SIZE=1024)
    return out
