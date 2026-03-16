import enum
from collections import deque

from femtovllm.engine.sequence import Sequence, SequenceStatus


class SchedulingPolicy(enum.Enum):
    FCFS = enum.auto()  # First Come First Serve


class RequestQueue:
    def __init__(self):
        self._running: list[Sequence] = []
        self._waiting: deque[Sequence] = deque()

        # TODO
        self._swapped: list[Sequence] = []

    def clean_finished_running(self):
        self._running = [
            #####
            x
            for x in self._running
            if x.status != SequenceStatus.FINISHED
        ]

    def sort_and_copy_running(self):
        self._running.sort(key=lambda x: x.arrival_time)
        return [x for x in self._running]

    def preempt_running_tail(self) -> Sequence:
        if not self._running:
            raise RuntimeError("preempt in empty running")

        seq = self._running.pop()
        seq.status = SequenceStatus.WAITING
        self._waiting.appendleft(seq)

        return seq

    def pop_waiting(self):
        seq = self._waiting.popleft()
        seq.status = SequenceStatus.RUNNING
        self._running.append(seq)

        return seq

    def peek_waiting(self):
        return self._waiting[0]
