import enum


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


class Sequence:
    """
    wrapper to a `idx` row from nanogpt

    `idx` is (B, T) in nanogpt

    - seq_id: seq id
    - token_ids: `idx` row
    - block_table: block indices
    """

    def __init__(
        self,
        seq_id: int | str,
        token_ids: list[int],
    ):
        self.seq_id = seq_id
        self.token_ids = token_ids
        self.block_table: list[int] = []
        self.status = SequenceStatus.WAITING

    @property
    def seq_len(self) -> int:
        return len(self.token_ids)

    def append(self, token_id: int):
        self.token_ids.append(token_id)
