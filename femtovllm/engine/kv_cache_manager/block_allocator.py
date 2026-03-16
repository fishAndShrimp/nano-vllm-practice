from collections import deque


class BlockAllocator:
    """ """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.available_blocks = deque(range(num_blocks))

    def count_available(self):
        return len(self.available_blocks)

    def can_allocate(self, needed: int):
        return needed <= self.count_available()

    def allocate(self, needed: int):
        if not self.can_allocate(needed):
            raise RuntimeError("[BlockAllocator][Overflow]")

        return [self.available_blocks.popleft() for _ in range(needed)]

    def free(self, blocks: list[int]):
        self.available_blocks.extend(blocks)
