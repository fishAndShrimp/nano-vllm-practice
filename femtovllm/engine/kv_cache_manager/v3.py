from collections import deque
from typing import Optional

from femtovllm.engine.kv_cache_manager.block_allocator import BlockAllocator
from femtovllm.engine.sequence import Sequence
from femtovllm.protocol import SeqId


class PrefixTreeNode:
    """ """

    def __init__(
        self,
        physical_block_idx: int,
        token_id_chunk: tuple[int, ...],
        parent: Optional["PrefixTreeNode"],
    ):
        """ """
        self.physical_block_idx = physical_block_idx
        self.token_id_chunk = token_id_chunk
        self.parent = parent

        self.ref_count = 1
        self.children: dict[tuple[int, ...], PrefixTreeNode] = {}


class PrefixTree:
    """
    ONLY save computed blocks
    """

    def __init__(
        self,
        block_size: int,
    ):
        """ """
        self.block_size = block_size

        self.root = PrefixTreeNode(
            physical_block_idx=-1,
            token_id_chunk=None,
            parent=None,
        )
        # paths from the root (include) to leaves
        self.chains: dict[SeqId, list[PrefixTreeNode]] = {}

    def ensure_chain(self, seq_const: Sequence):
        if seq_const.seq_id in self.chains:
            return
        self.chains[seq_const.seq_id] = [
            self.root,
        ]
        self.root.ref_count += 1

    def merge_block_table(
        self,
        seq_const: Sequence,
        block_table_ref: list[int],
    ) -> list[int]:
        """
        Receive block_table_ref and modify it

        Return the physical block indices of the redundant
        """
        if seq_const.seq_id not in self.chains:
            raise RuntimeError("")

        block_size = self.block_size
        chain = self.chains[seq_const.seq_id]

        ##########
        ##### [  num_tokens ] 0 1 2 3 4 5 6 7 8 9101112131415
        ##### [        // 4 ] 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
        ##### [ +1 +4) // 4 ] 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5
        ##### [    +4) // 4 ] 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
        ##### [ -1 +4) // 4 ] 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4
        ##### [  num_tokens ] 0 1 2 3 4 5 6 7 8 9101112131415
        ##########
        num_blocks_computed = seq_const.num_computed_tokens // block_size
        # exclude the root
        num_blocks_saved = len(chain) - 1

        if num_blocks_computed == num_blocks_saved:
            return []
        if num_blocks_computed < num_blocks_saved:
            raise RuntimeError(
                f"State misalignment! num_blocks_computed ({num_blocks_computed}) < "
                f"num_blocks_saved ({num_blocks_saved}). "
                "This usually happens if commit_blocks() is called BEFORE the forward pass "
                "completes the tokens claimed by the (L-1) rule in fast_forward_prefix()."
            )

        redundant_block_indices: list[int] = []
        for logical_block_idx in range(num_blocks_saved, num_blocks_computed):
            depth = logical_block_idx + 1
            depth_prev = depth - 1

            node_prev = chain[depth_prev]
            ##########
            ##### block_table[logical_block_idx] = physical_idx
            #####
            ##### Suppose   block_size          is  4
            #####           logical_block_idx   is  2
            #####           depth in tree       is  3
            #####           depth_prev          is  2
            #####
            #####                               prev    curr
            ##### chains:       0_root  1_node  2_node  3_node  4_node
            ##### block_table:          0_block 1_block 2_block 3_block
            ##### token_ids:            0:4     4:8     8:12    12:16
            ##########

            ptr_curr = tuple(
                seq_const.token_ids[
                    (logical_block_idx) * block_size : (logical_block_idx + 1)
                    * block_size
                ]
            )

            if ptr_curr in node_prev.children:
                # merge
                node_curr = node_prev.children[ptr_curr]

                redundant_block_indices.append(block_table_ref[logical_block_idx])
                block_table_ref[logical_block_idx] = node_curr.physical_block_idx

                node_curr.ref_count += 1
            else:
                # new a node and connect
                node_curr = PrefixTreeNode(
                    physical_block_idx=block_table_ref[logical_block_idx],
                    token_id_chunk=ptr_curr,
                    parent=node_prev,
                )
                node_prev.children[ptr_curr] = node_curr

            # append existing or new
            chain.append(node_curr)

        return redundant_block_indices


class KVCacheManagerV3:
    """
    **v3: Python Block-Aligned Prefix Tree**

    A simplified Radix Tree tailored for Python.
    It deliberately avoids heavy Copy-on-Write (CoW)
    and node splitting to minimize CPU/GIL overhead,
    focusing on pure block-level reuse.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ):
        """ """
        self.num_blocks = num_blocks
        self.block_size = block_size

        # tree perspective
        self.prefix_tree = PrefixTree(
            block_size=block_size,
        )
        self.nodes_lazy: deque[PrefixTreeNode] = deque()

        # block perspective
        self.block_allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
        )
        self.block_tables: dict[SeqId, list[int]] = {}

    def _calc_needed_blocks(
        self,
        seq_const: Sequence,
        num_scheduled_tokens: int,
        effective_prefix_len: int = 0,
    ):
        block_size = self.block_size

        effective_prefix_len = max(
            effective_prefix_len,
            seq_const.num_computed_tokens,
        )
        num_curr_blocks = (effective_prefix_len + block_size - 1) // block_size

        num_total_tokens = effective_prefix_len + num_scheduled_tokens
        num_total_blocks = (num_total_tokens + block_size - 1) // block_size
        needed = num_total_blocks - num_curr_blocks

        return needed

    def can_allocate(
        self,
        seq_const: Sequence,
        num_scheduled_tokens: int,
        effective_prefix_len: int = 0,
    ):
        needed = self._calc_needed_blocks(
            seq_const,
            num_scheduled_tokens,
            effective_prefix_len=effective_prefix_len,
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

        self.ensure_block_table(seq_const)
        self.get_block_table(seq_const).extend(r)

    def ensure_block_table(self, seq_const: Sequence):
        if seq_const.seq_id in self.block_tables:
            return
        self.block_tables[seq_const.seq_id] = []

    def get_block_table(self, seq_const: Sequence) -> list[int]:
        return self.block_tables[seq_const.seq_id]

    def get_block_table_len(self, seq_const: Sequence) -> int:
        """ """
        # use .get() to decrease hash check times from 2 to 1
        return len(
            self.block_tables.get(seq_const.seq_id, []),
        )

    def free(self, seq_const: Sequence):
        """
        free:
        - block_tables

        - tree.chains
            - node.ref_count
        """
        num_blocks_committed = 0

        if seq_const.seq_id in self.prefix_tree.chains:
            chain = self.prefix_tree.chains[seq_const.seq_id]
            num_blocks_committed = len(chain) - 1

            for i_node in reversed(chain):
                i_node.ref_count -= 1
                if i_node.ref_count <= 0:
                    self.nodes_lazy.append(i_node)

            del self.prefix_tree.chains[seq_const.seq_id]

        if seq_const.seq_id in self.block_tables:
            block_table = self.get_block_table(seq_const)

            self.block_allocator.free(
                block_table[num_blocks_committed:],
            )

            block_table.clear()
            del self.block_tables[seq_const.seq_id]

    def swap_out(self, seq_const: Sequence):
        raise NotImplementedError()

    def calc_max_tokens_allocable(
        self,
        seq_const: Sequence,
        effective_prefix_len: int = 0,
    ):
        """
        sum of:
        - remaining slots of last_block
        - remaining blocks
        """
        block_size = self.block_size

        effective_prefix_len = max(
            effective_prefix_len,
            seq_const.num_computed_tokens,
        )
        num_curr_blocks = (effective_prefix_len + block_size - 1) // block_size

        max_usable_blocks = num_curr_blocks + self.block_allocator.count_available()

        return max(
            0,
            max_usable_blocks * block_size - effective_prefix_len,
        )
