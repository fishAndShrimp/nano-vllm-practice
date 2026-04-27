import torch

import femtovllm
from femtovllm.protocol import ImplCustomKernel

try:
    from femtovllm import _C as _cuda_backend
except ImportError as e:
    raise ImportError(
        "🚨 Failed to load femtovllm CUDA backend!\n"
        "This project strictly requires compiled CUDA extensions.\n"
        "Please ensure you have a GPU, NVCC installed, and run:\n"
        "    pip install .\n"
        "or\n"
        "    python setup.py install"
    ) from e


from femtovllm.ops import triton as _triton_backend

MAX_KV_LEN_NON_SPLIT = _cuda_backend.kMaxKVLenNonSplit

KV_LEN_PER_PAGE = _cuda_backend.kKVLenPerPage
# The CUDA backend strictly operates on fixed kDimHead lengths for register allocation.
# Therefore, this is an exact DIM_HEAD, not a MAX_DIM_HEAD limit.
DIM_HEAD = _cuda_backend.kDimHead


def _ensure_valid_tensor(
    t: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """
    Internal helper: Ensure tensor is on CUDA and contiguous.
    """
    if not t.is_cuda:
        raise ValueError(
            f"Argument '{name}' must be a CUDA tensor. Got device={t.device}"
        )

    if not t.is_contiguous():
        return t.contiguous()

    return t


def vec_add(
    a: torch.Tensor,
    b: torch.Tensor,
    impl: str = "cuda",
) -> torch.Tensor:
    """
    Perform element-wise vector addition: c = a + b.

    Args:
        a (torch.Tensor): Input tensor A. Must be on GPU (CUDA).
        b (torch.Tensor): Input tensor B. Must be on GPU (CUDA) and same shape as A.
        impl (str, optional): Implementation backend to use.
                              Options: "cuda" (default), "triton".

    Returns:
        torch.Tensor: The result of a + b.

    Raises:
        ValueError: If inputs are not on CUDA.
        NotImplementedError: If the specified backend is not supported.
    """
    a = _ensure_valid_tensor(a, "a")
    b = _ensure_valid_tensor(b, "b")

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a={a.shape}, b={b.shape}")

    if impl == "cuda":
        return _cuda_backend.VecAddCuda(a, b)
    elif impl == "triton":
        raise NotImplementedError("Triton backend is not implemented yet.")
    else:
        raise NotImplementedError(f"Unknown backend implementation: {impl}")


def paged_attention_gemv(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    kv_page_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    max_kv_len: int,
    #####
    impl: ImplCustomKernel = None,
):
    """ """
    q = _ensure_valid_tensor(q, "q")

    impl = femtovllm._DEV.impl_custom_gemv if (impl is None) else impl

    if impl == ImplCustomKernel.CUDA:
        return _cuda_backend.PagedAttentionGemvCuda(
            q,
            k_pool,
            v_pool,
            kv_page_tables,
            kv_lens,
            max_kv_len,
        )
    else:
        raise NotImplementedError()


def paged_attention_gemm(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_q_len: int,
    Q_TILE_SIZE: int,
    cu_q_tiles: torch.Tensor,
    q_tile_to_seq_idx: torch.Tensor,
    kv_page_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    positions: torch.Tensor,
    #####
    impl: ImplCustomKernel = None,
):
    """ """
    q = _ensure_valid_tensor(q, "q")

    impl = femtovllm._DEV.impl_custom_gemm if (impl is None) else impl

    if impl == ImplCustomKernel.CUDA:
        # raise NotImplementedError("WIP")
        return _cuda_backend.PagedAttentionGemmCuda(
            q,
            k_pool,
            v_pool,
            cu_seqlens,
            max_q_len,
            kv_page_tables,
            kv_lens,
            positions,
        )
    elif impl == ImplCustomKernel.TRITON:
        return _triton_backend.paged_attention_gemm_triton(
            q=q,
            k_pool=k_pool,
            v_pool=v_pool,
            cu_seqlens=cu_seqlens,
            cu_q_tiles=cu_q_tiles,
            q_tile_to_seq_idx=q_tile_to_seq_idx,
            Q_TILE_SIZE=Q_TILE_SIZE,
            kv_page_tables=kv_page_tables,
            kv_lens=kv_lens,
            positions=positions,
        )
    else:
        raise NotImplementedError()
