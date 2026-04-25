import torch
import triton
import triton.language as tl

Q_TILE_SIZE: tl.constexpr = 32
KV_TILE_SIZE: tl.constexpr = 64
DIM_HEAD: tl.constexpr = 128


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
    dim_d,
    n_rep,
    dtype: tl.constexpr,
):
    """ """
    batch = tl.program_id(1)

    head = tl.program_id(0)
    kv_head = head // n_rep

    q_ptr = q_ptr_batched + (batch * n_heads + head) * q_len * dim_d
    k_ptr = k_ptr_batched + (batch * n_kv_heads + kv_head) * kv_len * dim_d
    v_ptr = v_ptr_batched + (batch * n_kv_heads + kv_head) * kv_len * dim_d
    out_ptr = out_ptr_batched + (batch * n_heads + head) * q_len * dim_d

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(q_len, dim_d),
        strides=(dim_d, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM_HEAD),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(q_len, dim_d),
        strides=(dim_d, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM_HEAD),
        order=(1, 0),
    )

    m_softmax = tl.zeros((Q_TILE_SIZE, 1), tl.float32)
    sum_softmax = tl.zeros_like(m_softmax)
    attn = tl.zeros((Q_TILE_SIZE, DIM_HEAD), tl.float32)

    for q_tile_idx in tl.range(tl.cdiv(q_len, Q_TILE_SIZE)):
        q_block = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_softmax = m_softmax * 0.0 - float("inf")
        sum_softmax *= 0.0

        k_block_ptr = tl.make_block_ptr(
            base=k_ptr,
            # shape=(kv_len, dim_d),
            # strides=(dim_d, 1),
            # offsets=(kv_tile_idx * KV_TILE_SIZE, 0),
            # block_shape=(KV_TILE_SIZE, dim_d),
            # order=(1, 0),
            shape=(dim_d, kv_len),
            strides=(1, dim_d),
            offsets=(0, 0),
            block_shape=(DIM_HEAD, KV_TILE_SIZE),
            order=(0, 1),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr,
            shape=(kv_len, dim_d),
            strides=(dim_d, 1),
            offsets=(0, 0),
            block_shape=(KV_TILE_SIZE, DIM_HEAD),
            order=(1, 0),
        )
        for kv_tile_idx in tl.range(tl.cdiv(kv_len, KV_TILE_SIZE)):
            k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

            sw = tl.dot(q_block, k_block) / tl.sqrt(dim_d * 1.0)
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

            v_block = tl.load(
                v_block_ptr,
                boundary_check=(0, 1),
            )
            attn += tl.dot(sw, v_block)

            k_block_ptr = k_block_ptr.advance(
                (0, KV_TILE_SIZE),
            )
            v_block_ptr = v_block_ptr.advance(
                (KV_TILE_SIZE, 0),
            )

        attn /= sum_softmax
        tl.store(out_block_ptr, attn, boundary_check=(0, 1))

        q_block_ptr = q_block_ptr.advance((Q_TILE_SIZE, 0))
        out_block_ptr = out_block_ptr.advance((Q_TILE_SIZE, 0))


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """
    B,H,T,D
    """
    for ele in [q, k, v]:
        assert ele.is_cuda
        assert ele.is_contiguous()

    assert k.shape[0] == v.shape[0] == q.shape[0]
    assert k.shape[1] == v.shape[1]
    assert k.shape[2] == v.shape[2]
    assert k.shape[3] == v.shape[3] == q.shape[3]

    dim_b, n_heads, q_len, dim_d = q.shape
    _, n_kv_heads, kv_len, _ = k.shape
    assert n_heads % n_kv_heads == 0
    n_rep = n_heads // n_kv_heads

    print(f"{q_len=}")
    print(f"{dim_d=}")

    out = torch.empty_like(q)
    flash_attention_kernel[
        (
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
        dim_d,
        n_rep,
        tl.float32,
    )

    return out
