import enum
import time

from femtovllm.protocol import SamplingParams


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
        req_id: int | str,
        seq_id: int | str,
        token_ids: list[int],
        sampling_params: SamplingParams,
    ):
        # [PART: const]
        self.arrival_time = time.time()
        self.req_id = req_id
        self.seq_id = seq_id

        if not isinstance(sampling_params, SamplingParams):
            raise TypeError(f"{type(sampling_params)=}")
        self.sampling_params = sampling_params

        self.stop_token_ids_set = set(sampling_params.stop_token_ids)
        self.prompt_length = len(token_ids)
        # [PART: const]

        # [PART: RUNNING <=> WAITING by RequestQueue]
        # [PART: RUNNING => FINISHED by CoreEngine/Scheduler]
        self.status = SequenceStatus.WAITING
        self.stop_reason = None

        ## must copy
        self.token_ids = [x for x in token_ids]
        self.num_computed_tokens = 0

        self.prefix_matched_length = 0
        self.prefix_node = None

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def num_new_tokens(self):
        return self.num_tokens - self.prompt_length

    @property
    def num_uncomputed_tokens(self):
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_decoding(self):
        """
        It is possible that: num_computed_tokens == num_tokens + 1

        since token_ids.append is called after maintaining num_computed_tokens
        """
        return self.num_computed_tokens >= self.num_tokens

    @property
    def is_prefilling(self):
        return not self.is_decoding

    def append(self, token_id: int):
        self.token_ids.append(token_id)

    def finish(
        self,
        stop_reason: str,
    ):
        self.status = SequenceStatus.FINISHED
        self.stop_reason = stop_reason

    def is_running(self):
        return self.status == SequenceStatus.RUNNING

    def __repr__(self) -> str:
        """
        Dynamic repr for debugging. Truncates massive lists and resolves Enums.
        """
        attrs = []

        for k, v in self.__dict__.items():
            if k == "token_ids":
                # Truncate to prevent console flooding during OOM/Scheduler crashes
                content = ", ".join(
                    map(str, v[:3]),
                )
                attrs.append(f"{k}=[{content}... (len={len(v)})]")
            elif isinstance(v, enum.Enum):
                attrs.append(f"{k}={v.name}")
            else:
                attrs.append(f"{k}={v!r}")

        # Inject crucial computed properties missing from __dict__
        attrs.append(f"num_uncomputed={self.num_uncomputed_tokens}")

        return f"{self.__class__.__name__}({', '.join(attrs)})"
