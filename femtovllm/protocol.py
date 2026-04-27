import dataclasses
import enum
from typing import NamedTuple, Optional, TypeAlias

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


class ImplCustomKernel(str, enum.Enum):
    """ """

    CUDA = "cuda"
    TRITON = "triton"


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

    Q_TILE_SIZE: int
    raw_cu_q_tiles: list[int]
    cu_q_tiles: torch.Tensor
    raw_q_tile_to_seq_idx: list[int]
    q_tile_to_seq_idx: torch.Tensor

    k_cache_pools: list[torch.Tensor]
    v_cache_pools: list[torch.Tensor]
    kv_lens: torch.Tensor
    max_kv_len: int

    block_tables: torch.Tensor
    raw_block_tables: list[list[int]]

    is_decoding: bool


ReqId: TypeAlias = int | str
SeqId: TypeAlias = int | str


class StepDelta(NamedTuple):
    """
    Streaming out shared by entrypoints and engine.
    Implemented as a NamedTuple for:
    1. Fastest instantiation (bypasses Python __init__)
    2. Zero memory overhead (no __dict__)
    3. Strict immutability (safe for streaming queues)
    """

    req_id: ReqId
    seq_id: SeqId
    new_token_id: int
    stop_reason: Optional[StopReason]
