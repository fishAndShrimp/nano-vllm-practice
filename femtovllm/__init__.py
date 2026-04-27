import dataclasses

from femtovllm.entrypoints.llm import LLM
from femtovllm.protocol import AttentionBackend, ImplCustomKernel, SamplingParams


@dataclasses.dataclass
class DevFlags:
    varlen_attn_backend: AttentionBackend = AttentionBackend.CUSTOM_GEMM_GEMV
    impl_custom_gemm: ImplCustomKernel = ImplCustomKernel.TRITON
    impl_custom_gemv: ImplCustomKernel = ImplCustomKernel.CUDA

    fake_varlen_by_batch: bool = False

    scheduler_version: int = 3

    @property
    def route_prefill_decode(self) -> bool:
        """ """
        return self.varlen_attn_backend == AttentionBackend.CUSTOM_GEMM_GEMV


_DEV = DevFlags()
