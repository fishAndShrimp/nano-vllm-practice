import enum
import time


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
        # [PART: const]
        self.arrival_time = time.time()
        self.seq_id = seq_id
        # [PART: const]

        # [PART: modified by RequestQueue]
        self.status = SequenceStatus.WAITING
        # [PART: modified by RequestQueue]

        # [PART: modified by Scheduler]
        ## must copy
        self.token_ids = [x for x in token_ids]
        self.num_computed_tokens = 0

        self.prefix_matched_length = 0
        self.prefix_node = None
        # [PART: modified by Scheduler]

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def num_uncomputed_tokens(self):
        return self.num_tokens - self.num_computed_tokens

    def append(self, token_id: int):
        self.token_ids.append(token_id)

    def finish(self):
        self.status = SequenceStatus.FINISHED
