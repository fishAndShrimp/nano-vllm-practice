from femtovllm.engine.kv_cache_manager.block_allocator import BlockAllocator
from femtovllm.engine.sequence import Sequence


class KVCacheManager:
    """ """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.block_allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
        )

        self.block_tables: dict[int | str, list[int]] = {}

    def _calc_needed_blocks(
        self,
        seq_const: Sequence,
        num_scheduled_tokens: int,
    ):
        block_size = self.block_size
        block_table = self.get_block_table(seq_const)

        total_tokens = seq_const.num_computed_tokens + num_scheduled_tokens
        total_blocks = (total_tokens + block_size - 1) // block_size
        needed = total_blocks - len(block_table)

        return needed

    def can_allocate(
        self,
        seq_const: Sequence,
        num_scheduled_tokens: int,
    ):
        needed = self._calc_needed_blocks(
            seq_const,
            num_scheduled_tokens,
        )
        return self.block_allocator.can_allocate(needed)

    def allocate(
        self,
        seq_const: Sequence,
        num_scheduled_tokens: int,
    ):
        if not self.can_allocate(seq_const, num_scheduled_tokens):
            raise RuntimeError("[KVCacheManager][Overflow]")

        needed = self._calc_needed_blocks(
            seq_const,
            num_scheduled_tokens,
        )
        r = self.block_allocator.allocate(needed)

        if seq_const.seq_id not in self.block_tables:
            self.block_tables[seq_const.seq_id] = []
        self.get_block_table(seq_const).extend(r)

    def get_block_table(self, seq_const: Sequence) -> list[int]:
        return self.block_tables.get(seq_const.seq_id, [])

    def free(self, seq_const: Sequence):
        if seq_const.seq_id not in self.block_tables:
            return

        block_table = self.get_block_table(seq_const)
        self.block_allocator.free(block_table)
        block_table.clear()

        del self.block_tables[seq_const.seq_id]

    def swap_out(self, seq_const: Sequence):
        raise NotImplementedError()

    def calc_max_tokens_allocable(self, seq_const: Sequence):
        """
        sum of:
        - remaining slots of last_block
        - remaining blocks
        """
        block_size = self.block_size
        block_table = self.get_block_table(seq_const)

        num_blocks = len(block_table) + self.block_allocator.count_available()
        return (
            #####
            num_blocks * block_size - seq_const.num_computed_tokens
        )
