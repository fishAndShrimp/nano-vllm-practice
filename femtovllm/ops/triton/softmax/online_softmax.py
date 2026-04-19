import torch
import triton
import triton.language as tl

Q_TILE_SIZE: tl.constexpr = 8
KV_TILE_SIZE: tl.constexpr = 32


@triton.jit
def online_softmax_kernel(
    a_ptr,
    out_ptr,
    q_len,
    kv_len,
    # debug_buffer_ptr,
    dtype: tl.constexpr,
):
    """ """
    for q_tile_idx in tl.range(tl.cdiv(q_len, Q_TILE_SIZE)):
        ####################
        ##### pass-1
        ####################
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(q_len, kv_len),
            strides=(kv_len, 1),
            offsets=(q_tile_idx * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, KV_TILE_SIZE),
            order=(1, 0),
        )

        m_softmax = tl.zeros((Q_TILE_SIZE, 1), tl.float32) - float("inf")
        sum_softmax = tl.zeros_like(m_softmax)

        for kv_tile_idx in tl.range(tl.cdiv(kv_len, KV_TILE_SIZE)):
            a_block = tl.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            offs_q = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            offs_kv = kv_tile_idx * KV_TILE_SIZE + tl.arange(0, KV_TILE_SIZE)
            a_block = tl.where(
                (offs_q < q_len)[:, None] & (offs_kv < kv_len)[None, :],
                a_block,
                float("-inf"),
            )

            # maintain m and sum
            m_new = tl.max(a_block, axis=1, keep_dims=True)

            m_old = m_softmax
            m_softmax = tl.maximum(m_softmax, m_new)

            sum_softmax = sum_softmax * tl.exp(m_old - m_softmax)

            # calc sum
            sum_softmax = sum_softmax + tl.sum(
                tl.exp(a_block - m_softmax),
                axis=1,
                keep_dims=True,
            )

            a_block_ptr = a_block_ptr.advance((0, KV_TILE_SIZE))

            # tl.store(
            #     tl.make_block_ptr(
            #         base=debug_buffer_ptr,
            #         shape=(Q_TILE_SIZE, KV_TILE_SIZE),
            #         strides=(KV_TILE_SIZE, 1),
            #         offsets=(0, 0),
            #         block_shape=(Q_TILE_SIZE, KV_TILE_SIZE),
            #         order=(1, 0),
            #     ),
            #     tl.cast(a_block, dtype),
            #     boundary_check=(0, 1),
            # )

        ####################
        ##### pass-2
        ####################
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(q_len, kv_len),
            strides=(kv_len, 1),
            offsets=(q_tile_idx * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, KV_TILE_SIZE),
            order=(1, 0),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(q_len, kv_len),
            strides=(kv_len, 1),
            offsets=(q_tile_idx * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, KV_TILE_SIZE),
            order=(1, 0),
        )
        for kv_tile_idx in tl.range(tl.cdiv(kv_len, KV_TILE_SIZE)):
            a_block = tl.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            offs_q = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            offs_kv = kv_tile_idx * KV_TILE_SIZE + tl.arange(0, KV_TILE_SIZE)
            a_block = tl.where(
                (offs_q < q_len)[:, None] & (offs_kv < kv_len)[None, :],
                a_block,
                float("-inf"),
            )

            a_block = tl.exp(a_block - m_softmax) / sum_softmax
            tl.store(
                out_block_ptr,
                tl.cast(a_block, dtype),
                boundary_check=(0, 1),
            )

            a_block_ptr = a_block_ptr.advance((0, KV_TILE_SIZE))
            out_block_ptr = out_block_ptr.advance((0, KV_TILE_SIZE))


def online_softmax_triton(
    a: torch.Tensor,
):
    """
    impl with block level to learn flash attention
    therefore grid is (1,1)
    """
    assert a.is_cuda
    assert a.is_contiguous()

    q_len, kv_len = a.shape
    out = torch.empty_like(a)

    # debug_buffer = torch.zeros(
    #     (Q_TILE_SIZE, KV_TILE_SIZE),
    #     dtype=a.dtype,
    #     device=a.device,
    # )
    online_softmax_kernel[(1, 1)](
        a,
        out,
        q_len,
        kv_len,
        # debug_buffer,
        dtype={
            torch.bfloat16: tl.bfloat16,
            torch.float16: tl.float16,
            torch.float32: tl.float32,
        }[a.dtype],
    )
    # print("\n\n##### debug_buffer")
    # print(debug_buffer)

    return out
