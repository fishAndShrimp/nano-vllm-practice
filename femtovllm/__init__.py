import dataclasses

from femtovllm.entrypoints.llm import LLM
from femtovllm.protocol import AttentionBackend, SamplingParams


@dataclasses.dataclass
class DevFlags:
    varlen_attn_backend: AttentionBackend = AttentionBackend.CUSTOM_GEMM_GEMV
    fake_varlen_by_batch: bool = False

    scheduler_version: int = 3

    @property
    def route_prefill_decode(self) -> bool:
        """ """
        return self.varlen_attn_backend == AttentionBackend.CUSTOM_GEMM_GEMV


_DEV = DevFlags()
