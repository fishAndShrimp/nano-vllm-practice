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

    def purge_zombie_finished(self):
        self._running = [
            #####
            x
            for x in self._running
            if (x.status != SequenceStatus.FINISHED)
        ]
        self._waiting = deque(
            #####
            x
            for x in self._waiting
            if (x.status != SequenceStatus.FINISHED)
        )

    def sort_and_copy_running(self):
        self._running.sort(key=lambda x: x.arrival_time)
        return [x for x in self._running]

    def running_head_is(self, seq: Sequence):
        if not self._running:
            raise RuntimeError("peek head in empty running")
        return self._running[0] == seq

    def running_tail_is(self, seq: Sequence):
        if not self._running:
            raise RuntimeError("peek tail in empty running")
        return self._running[-1] == seq

    def preempt_running_tail(self) -> Sequence:
        if not self._running:
            raise RuntimeError("preempt tail in empty running")

        seq = self._running.pop()
        seq.status = SequenceStatus.WAITING
        self._waiting.appendleft(seq)

        return seq

    def push_waiting(self, seq: Sequence):
        self._waiting.append(seq)

    def pop_waiting(self):
        seq = self._waiting.popleft()
        seq.status = SequenceStatus.RUNNING
        self._running.append(seq)

        return seq

    def peek_waiting(self):
        return self._waiting[0]

    @property
    def size_waiting(self):
        return len(self._waiting)

    def is_empty(self):
        return (
            #####
            self.size_waiting <= 0
            and len(self._running) <= 0
            and len(self._swapped) <= 0
        )
