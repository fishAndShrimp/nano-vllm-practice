import torch

try:
    from . import _C as _cuda_backend
except ImportError as e:
    raise ImportError(
        "CUDA extension is not compiled. Please run `pip install .`"
    ) from e


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
        return _cuda_backend.vec_add(a, b)
    elif impl == "triton":
        raise NotImplementedError("Triton backend is not implemented yet.")
    else:
        raise NotImplementedError(f"Unknown backend implementation: {impl}")
