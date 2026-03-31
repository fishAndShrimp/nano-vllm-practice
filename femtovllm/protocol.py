import dataclasses
import enum
from typing import Optional

import torch


class StopReason(str, enum.Enum):
    """ """

    # normal
    EOS = "EOS"
    LENGTH = "LENGTH"

    # aborted
    OOM = "OOM"
    HARDWARE_LIMIT = "HARDWARE_LIMIT"


class AttentionBackend(str, enum.Enum):
    """ """

    PYTORCH = "pytorch"
    CUSTOM_GEMM = "custom_gemm"
    CUSTOM_GEMM_GEMV = "custom_gemm_gemv"


@dataclasses.dataclass
class SamplingParams:
    """
    schema shared by inputs and engine
    """

    # probability manipulation
    temperature: float = 1.0
    presence_penalty: float = 1.0

    # stopping criteria
    stop_token_ids: list[int] = dataclasses.field(
        default_factory=list,
    )
    max_new_tokens: int = 5000

    def clone(self):
        return dataclasses.replace(
            self,
            stop_token_ids=self.stop_token_ids.copy(),
        )


@dataclasses.dataclass
class VarlenAttnMetadata:
    positions: torch.Tensor
    raw_positions: list[int]

    cu_seqlens: torch.Tensor
    raw_cu_seqlens: list[int]
    max_q_len: int

    k_cache_pools: list[torch.Tensor]
    v_cache_pools: list[torch.Tensor]
    kv_lens: torch.Tensor
    max_kv_len: int

    block_tables: torch.Tensor
    raw_block_tables: list[list[int]]

    is_decoding: bool


class StepDelta:
    """
    streaming out shared by entrypoints and engine
    - use __slots__ rather than __dict__ to speed up
    """

    __slots__ = (
        "req_id",
        "seq_id",
        "new_token_id",
        "stop_reason",
    )

    def __init__(
        self,
        req_id: str,
        seq_id: str,
        new_token_id: int,
        stop_reason: Optional[StopReason],
    ):
        self.req_id = req_id
        self.seq_id = seq_id
        self.new_token_id = new_token_id
        self.stop_reason = stop_reason
