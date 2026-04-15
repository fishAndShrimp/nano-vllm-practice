import torch
import triton
import triton.language as tl

TILE_SIZE: tl.constexpr = 32


@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    dim_m,
    dim_k,
    dim_n,
    out_dtype: tl.constexpr,
):
    """ """
    offs_y_base = tl.program_id(0) * TILE_SIZE
    offs_x_base = tl.program_id(1) * TILE_SIZE

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(dim_m, dim_k),
        strides=(dim_k, 1),
        offsets=(offs_y_base, 0),
        block_shape=(TILE_SIZE, TILE_SIZE),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(dim_k, dim_n),
        strides=(dim_n, 1),
        offsets=(0, offs_x_base),
        block_shape=(TILE_SIZE, TILE_SIZE),
        order=(1, 0),
    )

    pvalues = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    for phase in tl.range(0, dim_k, TILE_SIZE, num_stages=2):
        a_block = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b_block = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")

        pvalues = tl.dot(a_block, b_block, pvalues)

        a_block_ptr = a_block_ptr.advance((0, TILE_SIZE))
        b_block_ptr = b_block_ptr.advance((TILE_SIZE, 0))

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(dim_m, dim_n),
        strides=(dim_n, 1),
        offsets=(offs_y_base, offs_x_base),
        block_shape=(TILE_SIZE, TILE_SIZE),
        order=(1, 0),
    )
    tl.store(
        out_block_ptr,
        pvalues.to(out_dtype),
        boundary_check=(0, 1),
    )


def gemm_triton(a: torch.Tensor, b: torch.Tensor):
    """ """
    assert a.is_cuda
    assert b.is_cuda

    assert a.is_contiguous()
    assert b.is_contiguous()

    assert a.shape[-1] == b.shape[-2]
    dim_m, dim_k = a.shape
    _, dim_n = b.shape

    out = torch.empty(
        (dim_m, dim_n),
        dtype=a.dtype,
        device=a.device,
    )
    gemm_kernel[
        (
            triton.cdiv(dim_m, TILE_SIZE),
            triton.cdiv(dim_n, TILE_SIZE),
        )
    ](
        a,
        b,
        out,
        dim_m,
        dim_k,
        dim_n,
        out_dtype={
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }[out.dtype],
    )

    return out
