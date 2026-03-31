import dataclasses

from femtovllm.entrypoints.llm import LLM
from femtovllm.protocol import AttentionBackend, SamplingParams


@dataclasses.dataclass
class DevFlags:
    varlen_attn_impl: AttentionBackend = AttentionBackend.CUSTOM_GEMM_GEMV
    fake_varlen_by_batch: bool = False

    @property
    def route_prefill_decode(self) -> bool:
        """ """
        return self.varlen_attn_impl == AttentionBackend.CUSTOM_GEMM_GEMV


_DEV = DevFlags()
