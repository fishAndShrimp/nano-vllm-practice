import dataclasses

import torch


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
            stop_token_ids=[x for x in self.stop_token_ids],
        )


@dataclasses.dataclass
class VarlenAttnMetadata:
    positions: torch.Tensor

    cu_seqlens: torch.Tensor
    raw_cu_seqlens: list[int]
    q_len_max: int

    k_cache_pools: list[torch.Tensor]
    v_cache_pools: list[torch.Tensor]

    block_tables: torch.Tensor
    raw_block_tables: list[list[int]]


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
        stop_reason: str,
    ):
        self.req_id = req_id
        self.seq_id = seq_id
        self.new_token_id = new_token_id
        self.stop_reason = stop_reason
