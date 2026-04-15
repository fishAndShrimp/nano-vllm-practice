import torch
import triton
import triton.language as tl


@triton.jit
def safe_softmax_kernel(
    a_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """ """
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)

    for row in tl.range(pid, n_rows, n_programs, num_stages=4):
        offset_base = row * n_cols

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        offsets += offset_base

        a = tl.load(
            a_ptr + offsets,
            mask=mask,
            other=-torch.inf,
        )

        m_softmax = tl.max(a)
        a -= m_softmax
        a = tl.exp(a)

        sum_softmax = tl.sum(a)
        a /= sum_softmax

        tl.store(out_ptr + offsets, a, mask=mask)


def safe_softmax_triton(a: torch.Tensor):
    """ """
    assert a.is_cuda
    assert a.is_contiguous()

    n_cols = a.size(-1)
    n_rows = a.numel() // n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    NUM_BLOCKS = 128

    out = torch.empty_like(a)
    safe_softmax_kernel[(NUM_BLOCKS,)](
        a,
        out,
        n_rows,
        n_cols,
        BLOCK_SIZE,
    )

    return out
