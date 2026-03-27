import dataclasses

from femtovllm.entrypoints.llm import LLM
from femtovllm.protocol import SamplingParams


@dataclasses.dataclass
class DevFlags:
    varlen_attn_impl: str = (
        "pytorch",
        "custom_gemm",
        "custom_gemm_gemv",
    )[0]
    fake_varlen_by_batch: bool = False


_DEV = DevFlags()
