import torch
import triton
import triton.language as tl


@triton.jit
def paged_attention_gemm_kernel(
    q_ptr_batched,
    q_len_flatten,
    k_pool_ptr,
    v_pool_ptr,
    pool_stride_0,
    pool_stride_1,
    pool_stride_2,
    out_ptr_batched,
    cu_seqlens_ptr,
    cu_q_tiles_ptr,
    q_tile_to_seq_idx_ptr,
    kv_page_tables_ptr,
    num_pages_per_seq,
    kv_lens_ptr,
    positions_ptr,
    n_rep,
    scale_softmax,
    DTYPE: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    KV_LEN_PER_PAGE: tl.constexpr,
    DIM_HEAD: tl.constexpr,
):
    """ """
    pid0 = tl.program_id(0)

    seq_idx = tl.load(q_tile_to_seq_idx_ptr + pid0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // n_rep

    q_ptr = q_ptr_batched + head_idx * q_len_flatten * DIM_HEAD
    out_ptr = out_ptr_batched + head_idx * q_len_flatten * DIM_HEAD

    q_begin = tl.load(cu_seqlens_ptr + seq_idx)
    q_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    q_len = q_end - q_begin

    q_tile_idx = pid0 - tl.load(cu_q_tiles_ptr + seq_idx)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_begin * DIM_HEAD,
        shape=(q_len, DIM_HEAD),
        strides=(DIM_HEAD, 1),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM_HEAD),
        order=(1, 0),
    )
    q_block = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + q_begin * DIM_HEAD,
        shape=(q_len, DIM_HEAD),
        strides=(DIM_HEAD, 1),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM_HEAD),
        order=(1, 0),
    )

    #########################
    #####
    #########################
    m_softmax = tl.full(
        (Q_TILE_SIZE, 1),
        float("-inf"),
        tl.float32,
    )
    sum_softmax = tl.zeros_like(m_softmax)
    acc = tl.zeros(
        (Q_TILE_SIZE, DIM_HEAD),
        tl.float32,
    )

    kv_len = tl.load(kv_lens_ptr + seq_idx)
    page_table_ptr = kv_page_tables_ptr + seq_idx * num_pages_per_seq

    q_tile_begin = q_begin + q_tile_idx * Q_TILE_SIZE
    offs_q = q_tile_begin + tl.arange(0, Q_TILE_SIZE)
    q_positions = tl.load(
        positions_ptr + offs_q,
        mask=offs_q < q_end,
        other=kv_len,
    )

    for page_logical_idx in tl.range(tl.cdiv(kv_len, KV_LEN_PER_PAGE)):
        page_physical_idx = tl.load(page_table_ptr + page_logical_idx)

        kt_block_ptr = tl.make_block_ptr(
            base=k_pool_ptr
            + page_physical_idx * pool_stride_0
            + kv_head_idx * pool_stride_1,
            # shape=(KV_LEN_PER_PAGE, DIM_HEAD),
            # strides=(pool_stride_2, 1),
            # offsets=(0, 0),
            # block_shape=(KV_LEN_PER_PAGE, DIM_HEAD),
            # order=(1, 0),
            shape=(DIM_HEAD, KV_LEN_PER_PAGE),
            strides=(1, pool_stride_2),
            offsets=(0, 0),
            block_shape=(DIM_HEAD, KV_LEN_PER_PAGE),
            order=(0, 1),
        )
        kt_block = tl.load(kt_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(q_block, kt_block) * scale_softmax
        offs_kv = page_logical_idx * KV_LEN_PER_PAGE + tl.arange(0, KV_LEN_PER_PAGE)
        qk = tl.where(
            offs_kv[None, :] < kv_len,
            qk,
            float("-inf"),
        )
        qk = tl.where(
            offs_kv[None, :] < q_positions[:, None],
            qk,
            float("-inf"),
        )

        ##### online softmax maintaining
        # 1. m
        m_new = tl.max(qk, axis=1, keep_dims=True)
        m_old = m_softmax
        m_softmax = tl.maximum(m_softmax, m_new)
        exp_delta = tl.exp(m_old - m_softmax)

        # 2. sum acc
        sum_softmax *= exp_delta
        acc *= exp_delta

        # 3. qk
        #    scores => weights
        qk = tl.exp(qk - m_softmax)

        # 4. sum
        sum_softmax += tl.sum(qk, axis=1, keep_dims=True)

        ##### weighted v
        v_block_ptr = tl.make_block_ptr(
            base=v_pool_ptr
            + page_physical_idx * pool_stride_0
            + kv_head_idx * pool_stride_1,
            shape=(KV_LEN_PER_PAGE, DIM_HEAD),
            strides=(pool_stride_2, 1),
            offsets=(0, 0),
            block_shape=(KV_LEN_PER_PAGE, DIM_HEAD),
            order=(1, 0),
        )
        v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # downcast
        weights = tl.cast(qk, DTYPE)
        acc += tl.dot(weights, v_block)

    acc /= sum_softmax
    tl.store(
        out_block_ptr,
        tl.cast(acc, DTYPE),
        boundary_check=(0, 1),
    )


def paged_attention_gemm_triton(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_q_tiles: torch.Tensor,
    q_tile_to_seq_idx: torch.Tensor,
    Q_TILE_SIZE: int,
    kv_page_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    positions: torch.Tensor,
):
    """
    - q: flatten, (n_heads, q_len_flatten, d_head)
    """
    # TODO assert checks

    n_heads, q_len_flatten, DIM_HEAD = q.shape
    _, n_kv_heads, KV_LEN_PER_PAGE, _ = k_pool.shape
    assert n_heads % n_kv_heads == 0
    n_rep = n_heads // n_kv_heads

    pool_stride_0, pool_stride_1, pool_stride_2, _ = k_pool.stride()

    out = torch.empty_like(q)
    paged_attention_gemm_kernel[
        (
            q_tile_to_seq_idx.numel(),
            n_heads,
        )
    ](
        q_ptr_batched=q,
        q_len_flatten=q_len_flatten,
        k_pool_ptr=k_pool,
        v_pool_ptr=v_pool,
        pool_stride_0=pool_stride_0,
        pool_stride_1=pool_stride_1,
        pool_stride_2=pool_stride_2,
        out_ptr_batched=out,
        cu_seqlens_ptr=cu_seqlens,
        cu_q_tiles_ptr=cu_q_tiles,
        q_tile_to_seq_idx_ptr=q_tile_to_seq_idx,
        kv_page_tables_ptr=kv_page_tables,
        num_pages_per_seq=kv_page_tables.stride(0),
        kv_lens_ptr=kv_lens,
        positions_ptr=positions,
        n_rep=n_rep,
        scale_softmax=DIM_HEAD ** (-0.5),
        DTYPE={
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }[q.dtype],
        Q_TILE_SIZE=Q_TILE_SIZE,
        KV_LEN_PER_PAGE=KV_LEN_PER_PAGE,
        DIM_HEAD=DIM_HEAD,
    )

    return out
