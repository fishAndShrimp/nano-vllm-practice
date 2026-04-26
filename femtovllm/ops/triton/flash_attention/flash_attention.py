import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr_batched,
    k_ptr_batched,
    v_ptr_batched,
    out_ptr_batched,
    n_heads,
    n_kv_heads,
    q_len,
    kv_len,
    n_rep,
    scale_softmax,
    dtype: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    KV_TILE_SIZE: tl.constexpr,
    DIM_HEAD: tl.constexpr,
):
    """ """
    batch = tl.program_id(2)

    head = tl.program_id(1)
    kv_head = head // n_rep

    q_ptr = q_ptr_batched + (batch * n_heads + head) * q_len * DIM_HEAD
    k_ptr = k_ptr_batched + (batch * n_kv_heads + kv_head) * kv_len * DIM_HEAD
    v_ptr = v_ptr_batched + (batch * n_kv_heads + kv_head) * kv_len * DIM_HEAD
    out_ptr = out_ptr_batched + (batch * n_heads + head) * q_len * DIM_HEAD

    q_tile_idx = tl.program_id(0)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(q_len, DIM_HEAD),
        strides=(DIM_HEAD, 1),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM_HEAD),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(q_len, DIM_HEAD),
        strides=(DIM_HEAD, 1),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM_HEAD),
        order=(1, 0),
    )

    for _ in tl.range(1):
        q_block = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_softmax = tl.full(
            (Q_TILE_SIZE, 1),
            float("-inf"),
            tl.float32,
        )
        sum_softmax = tl.zeros_like(m_softmax)

        k_block_ptr = tl.make_block_ptr(
            base=k_ptr,
            # shape=(kv_len, DIM_HEAD),
            # strides=(DIM_HEAD, 1),
            # offsets=(kv_tile_idx * KV_TILE_SIZE, 0),
            # block_shape=(KV_TILE_SIZE, DIM_HEAD),
            # order=(1, 0),
            shape=(DIM_HEAD, kv_len),
            strides=(1, DIM_HEAD),
            offsets=(0, 0),
            block_shape=(DIM_HEAD, KV_TILE_SIZE),
            order=(0, 1),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr,
            shape=(kv_len, DIM_HEAD),
            strides=(DIM_HEAD, 1),
            offsets=(0, 0),
            block_shape=(KV_TILE_SIZE, DIM_HEAD),
            order=(1, 0),
        )

        attn = tl.zeros((Q_TILE_SIZE, DIM_HEAD), tl.float32)
        for kv_tile_idx in tl.range(tl.cdiv(kv_len, KV_TILE_SIZE)):
            k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

            sw = tl.dot(q_block, k_block) * scale_softmax
            sw = tl.where(
                kv_tile_idx * KV_TILE_SIZE + tl.arange(0, KV_TILE_SIZE) < kv_len,
                sw,
                float("-inf"),
            )

            m_new = tl.max(sw, 1, keep_dims=True)
            m_old = m_softmax
            m_softmax = tl.maximum(m_old, m_new)

            exp_delta = tl.exp(m_old - m_softmax)
            sum_softmax *= exp_delta
            attn *= exp_delta

            sw = tl.exp(sw - m_softmax)
            sum_softmax += tl.sum(sw, 1, keep_dims=True)

            tl.static_print("  m_softmax.shape =", m_softmax.shape)
            tl.static_print("sum_softmax.shape =", sum_softmax.shape)

            v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            sw_cast = tl.cast(sw, dtype)
            attn += tl.dot(sw_cast, v_block)

            k_block_ptr = k_block_ptr.advance(
                (0, KV_TILE_SIZE),
            )
            v_block_ptr = v_block_ptr.advance(
                (KV_TILE_SIZE, 0),
            )

        attn /= sum_softmax
        tl.store(
            out_block_ptr,
            tl.cast(attn, dtype),
            boundary_check=(0, 1),
        )


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """
    B,H,T,D
    """
    Q_TILE_SIZE = 32

    for ele in [q, k, v]:
        assert ele.is_cuda
        assert ele.is_contiguous()

    assert k.shape[0] == v.shape[0] == q.shape[0]
    assert k.shape[1] == v.shape[1]
    assert k.shape[2] == v.shape[2]
    assert k.shape[3] == v.shape[3] == q.shape[3]

    dim_b, n_heads, q_len, DIM_HEAD = q.shape
    _, n_kv_heads, kv_len, _ = k.shape
    assert n_heads % n_kv_heads == 0
    n_rep = n_heads // n_kv_heads

    out = torch.empty_like(q)
    flash_attention_kernel[
        (
            triton.cdiv(q_len, Q_TILE_SIZE),
            n_heads,
            dim_b,
        )
    ](
        q,
        k,
        v,
        out,
        n_heads,
        n_kv_heads,
        q_len,
        kv_len,
        n_rep,
        DIM_HEAD ** (-0.5),
        dtype={
            torch.bfloat16: tl.bfloat16,
            torch.float16: tl.float16,
        }[q.dtype],
        Q_TILE_SIZE=Q_TILE_SIZE,
        KV_TILE_SIZE=64,
        DIM_HEAD=DIM_HEAD,
    )

    return out
