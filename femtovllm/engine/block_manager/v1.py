from collections import deque

from femtovllm.engine.sequence import Sequence


class BlockManager:
    """ """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.available_blocks: deque[int] = deque(range(num_blocks))

    def calc_needed_blocks(self, seq: Sequence):
        num_all = (seq.seq_len + self.block_size - 1) // self.block_size
        num_curr = len(seq.block_table)
        return max(0, num_all - num_curr)

    def can_allocate(self, seq: Sequence) -> bool:
        needed = self.calc_needed_blocks(seq)
        return needed <= len(self.available_blocks)

    def may_allocate(self, seq: Sequence):
        assert self.can_allocate(seq)

        needed = self.calc_needed_blocks(seq)
        for _ in range(needed):
            seq.block_table.append(self.available_blocks.popleft())

    def free(self, seq: Sequence):
        self.available_blocks.extend(seq.block_table)
        seq.block_table.clear()
