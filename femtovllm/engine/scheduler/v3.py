from femtovllm.engine.kv_cache_manager import KVCacheManagerV3
from femtovllm.engine.kv_cache_manager.v3 import PrefixTreeNode
from femtovllm.engine.request_queue import RequestQueue
from femtovllm.engine.sequence import Sequence
from femtovllm.engine.step_budget import StepBudget
from femtovllm.protocol import StopReason


class Scheduler:
    """ """

    def __init__(
        self,
        step_budget: StepBudget,
        request_queue: RequestQueue,
        kv_cache_manager: KVCacheManagerV3,
        max_kv_len_non_split: int,
    ):
        self.step_budget = step_budget
        self.request_queue = request_queue
        self.kv_cache_manager = kv_cache_manager

        self.max_kv_len_non_split = max_kv_len_non_split

    def _preempt(self):
        """
        [Atomic]
        - running => waiting/swapped
        - increase resource
        """
        seq = self.request_queue.preempt_running_tail()
        seq.num_computed_tokens = 0
        self.kv_cache_manager.free(seq)

    def _allocate(self, seq, num_tokens):
        """
        [Atomic]
        - decrease resource
        """
        self.step_budget.consume(num_tokens)
        self.kv_cache_manager.allocate(seq, num_tokens)

    def free_and_finish(
        self,
        seq: Sequence,
        stop_reason: StopReason,
    ):
        """
        [Atomic]
        - increase resource
        """
        self.kv_cache_manager.free(seq)
        seq.finish(stop_reason)

    def _calc_limit_computation(self):
        """ """
        return min(
            self.step_budget.max_tokens_per_seq,
            self.step_budget.remaining_tokens,
        )

    def _calc_limit_hardware(self, seq: Sequence):
        """ """
        return self.max_kv_len_non_split - seq.num_tokens

    def _schedule_running(self):
        """ """
        scheduled: list[tuple[Sequence, int]] = []
        aborted: list[Sequence] = []
        has_resource = True

        for seq in self.request_queue.sort_and_copy_running():
            # [STEP: seqs[curr:] are all preempted]
            if not seq.is_running():
                has_resource = False
                break

            ##############################
            ##### determine num_tokens
            ##############################
            # fast forward
            self.fast_forward_prefix(
                seq,
                dry_run=False,
            )

            # [LIMIT: computation]
            # truncate to fit computation limit
            limit_computation = self._calc_limit_computation()
            if limit_computation <= 0:
                has_resource = False
                break

            num_tokens = min(
                seq.num_uncomputed_tokens,
                limit_computation,
            )
            if num_tokens <= 0:
                raise RuntimeError(f"{seq=} strange {seq.num_uncomputed_tokens=}")

            # [LIMIT: hardware]
            # truncate to fit hardware limit
            limit_hardware = self._calc_limit_hardware(seq)
            if limit_hardware <= 0:
                self.free_and_finish(seq, StopReason.HARDWARE_LIMIT)
                continue

            num_tokens = min(num_tokens, limit_hardware)

            ##############################
            ##### fulfill num_tokens
            ##############################
            # [LIMIT: computation]
            # [budget]
            fit_budget = self.step_budget.can_consume(num_tokens)
            if not fit_budget:
                has_resource = False
                break

            # [LIMIT: storage]
            # [kv_cache]
            fit_kv_cache = self.kv_cache_manager.can_allocate(seq, num_tokens)
            while not fit_kv_cache:
                has_resource = False

                if self.request_queue.running_tail_is(seq):
                    limit_kv_cache = self.kv_cache_manager.calc_max_tokens_allocable(
                        seq
                    )
                    if limit_kv_cache <= 0:
                        if self.request_queue.running_head_is(seq):
                            # this seq requires more than entire kv_cache
                            self.free_and_finish(seq, StopReason.OOM)
                            aborted.append(seq)
                        break

                    # truncate to fit kv_cache limit
                    num_tokens = min(num_tokens, limit_kv_cache)
                else:
                    self._preempt()

                fit_kv_cache = self.kv_cache_manager.can_allocate(seq, num_tokens)

            ##############################
            ##### consume
            ##############################
            if (
                #####
                seq.is_running() and (num_tokens > 0) and fit_budget and fit_kv_cache
            ):
                self._allocate(seq, num_tokens)
                scheduled.append(
                    (seq, num_tokens),
                )

        return scheduled, aborted, has_resource

    def _schedule_waiting(self):
        """ """
        scheduled: list[tuple[Sequence, int]] = []

        while self.request_queue.size_waiting > 0:
            # [STEP: highest priority waiting]
            seq = self.request_queue.peek_waiting()
            # [STEP: highest has been aborted]
            if not seq.is_waiting():
                self.request_queue.pop_waiting()
                continue

            ##############################
            ##### determine num_tokens
            ##############################
            # cache-aware effective length
            effective_prefix_len = self.fast_forward_prefix(
                seq,
                dry_run=True,
            )
            effective_uncomputed_len = seq.num_tokens - effective_prefix_len

            # [LIMIT: computation]
            # [LIMIT: storage]
            # truncate to fit both computation and storage limit
            limit_both = min(
                self._calc_limit_computation(),
                self.kv_cache_manager.calc_max_tokens_allocable(
                    seq,
                    effective_prefix_len=effective_prefix_len,
                ),
            )
            if limit_both <= 0:
                break

            num_tokens = min(
                effective_uncomputed_len,
                limit_both,
            )

            if num_tokens <= 0:
                raise RuntimeError(f"{seq=} strange {seq.num_uncomputed_tokens=}")

            # [LIMIT: hardware]
            # truncate to fit hardware limit
            limit_hardware = self._calc_limit_hardware(seq)
            if limit_hardware <= 0:
                self.free_and_finish(seq, StopReason.HARDWARE_LIMIT)
                continue

            num_tokens = min(num_tokens, limit_hardware)

            ##############################
            ##### fulfill num_tokens
            ##############################
            # [LIMIT: computation]
            # [budget]
            fit_budget = self.step_budget.can_consume(num_tokens)

            # [LIMIT: storage]
            # [kv_cache]
            fit_kv_cache = self.kv_cache_manager.can_allocate(
                seq,
                num_tokens,
                effective_prefix_len=effective_prefix_len,
            )

            ##############################
            ##### consume
            ##############################
            if (
                #####
                seq.is_waiting() and (num_tokens > 0) and fit_budget and fit_kv_cache
            ):
                self.fast_forward_prefix(
                    seq,
                    dry_run=False,
                )
                self.ensure_sequence_state(seq)

                seq = self.request_queue.pop_waiting()
                self._allocate(seq, num_tokens)
                scheduled.append(
                    (seq, num_tokens),
                )
            else:
                break

        return scheduled

    def step(self):
        """ """
        self.request_queue.purge_zombie_finished()
        self.step_budget.reset()

        scheduled, aborted, has_resource = self._schedule_running()
        if has_resource:
            scheduled.extend(
                self._schedule_waiting(),
            )

        return scheduled, aborted

    def has_unfinished_sequences(self):
        return not self.request_queue.is_empty()

    def add_sequence(self, seq: Sequence):
        """ """
        self.request_queue.push_waiting(seq)

    ##############################
    ##### [ ARCHITECTURAL NOTE: Prefix-Aware Scheduling ]
    ##### From this point forward, KVCacheManager and Scheduler are irreversibly coupled.
    ##### As the engine evolves, the KV Cache is no longer just passive storage;
    ##### it actively dictates scheduling decisions.
    #####
    ##### - Current: Deduplicates newly computed blocks by merging them into the tree.
    ##### - Future: The Scheduler will natively query the Prefix Tree to fast-forward
    #####   `num_computed_tokens` (Prompt Caching) and reorder sequences based on
    #####   prefix match lengths. The Scheduler MUST be Prefix-Tree native.
    ##############################
    def ensure_sequence_state(self, seq: Sequence):
        """ """
        self.kv_cache_manager.prefix_tree.ensure_chain(seq)
        self.kv_cache_manager.ensure_block_table(seq)

    def commit_blocks(self, seq: Sequence):
        """
        Commit the sequence's newly computed blocks to the global Prefix Tree.

        This method bridges computation and storage: it registers the logical
        tokens into the shared tree and immediately frees any physical blocks
        that are identified as redundant (i.e., already cached by other sequences).
        """
        redundant_block_indices = self.kv_cache_manager.prefix_tree.merge_block_table(
            seq_const=seq,
            block_table_ref=self.kv_cache_manager.get_block_table(seq),
        )

        # Free physical blocks that were allocated for computation
        # but are now deduplicated by the Prefix Tree.
        self.kv_cache_manager.block_allocator.free(redundant_block_indices)

    def fast_forward_prefix(
        self,
        seq: Sequence,
        dry_run: bool,
    ) -> int:
        """
        [State Transition: Prompt Caching]
        Query the global prefix tree to skip computation for cached tokens.
        Enforces the (L-1) rule to ensure the model always performs at least
        one forward pass to generate the logits for the next token.
        """
        prefix_tree = self.kv_cache_manager.prefix_tree
        block_size = prefix_tree.block_size

        ##############################
        ##### no node split mechanism
        ##### fast forward unit is block_size
        ##############################
        if seq.num_uncomputed_tokens < block_size:
            return seq.num_computed_tokens

        chain_old: list[PrefixTreeNode] = prefix_tree.chains.get(
            seq.seq_id,
            [prefix_tree.root],
        )
        chain_extend: list[PrefixTreeNode] = []

        num_cached_tokens = (len(chain_old) - 1) * block_size
        node_curr = chain_old[-1]

        while num_cached_tokens + block_size <= seq.num_tokens:
            ptr_next = tuple(
                seq.token_ids[num_cached_tokens : num_cached_tokens + block_size]
            )
            if ptr_next not in node_curr.children:
                break

            num_cached_tokens += block_size
            node_next = node_curr.children[ptr_next]
            chain_extend.append(node_next)
            node_curr = node_next

        # strictly no benefit
        if num_cached_tokens <= seq.num_computed_tokens:
            return seq.num_computed_tokens

        ##### (L-1) rule
        # =====================================================================
        # [ARCHITECTURAL NOTE: The (L-1) Rule & Harmless State Misalignment]
        #
        # Example: block_size=4, seq.num_tokens=8 (fully cached in tree).
        # The tree matches 8 tokens (2 blocks). The (L-1) rule forces
        # num_cached_tokens back to 7 to ensure at least one forward pass.
        #
        # This creates a temporary state misalignment:
        # - chain & block_table hold 2 full blocks (capacity for 8 tokens).
        # - num_computed_tokens = num_cached_tokens is only 7.
        #
        # THE TRADE-OFF (Why we don't truncate the chain):
        # If we discard the 2nd block to strictly align with 7 tokens, we lose
        # the cached KV for tokens 4, 5, and 6. They become vulnerable to eviction,
        # forcing expensive recomputation. By holding the block, we protect them.
        #
        # SELF-HEALING:
        # The model runs the forward pass for the 8th token, safely overwriting
        # the exact same KV values in the 2nd block (idempotent write).
        # num_computed_tokens becomes 8, and the state perfectly realigns.
        #
        # STRICT INVARIANT (The Taboo):
        # As a direct limitation of this (L-1) misalignment, it is STRICTLY
        # FORBIDDEN to call `commit_blocks()` or `merge_block_table()` BEFORE
        # the forward pass completes. Doing so will crash the engine
        # (num_blocks_computed < num_blocks_saved). The forward pass MUST be
        # executed to heal the state.
        # =====================================================================
        if num_cached_tokens == seq.num_tokens:
            num_cached_tokens -= 1

        # due to (L-1), but still no benefit
        if num_cached_tokens <= seq.num_computed_tokens:
            return seq.num_computed_tokens

        ##############################
        ##### confirm fast forward to num_cached_tokens
        ##############################

        ##############################
        ##### dry run
        ##############################
        if dry_run:
            return num_cached_tokens

        ##############################
        ##### actual run
        ##############################
        ##### ensure things we need to modify
        self.ensure_sequence_state(seq)

        # strange
        if not chain_extend:
            raise RuntimeError(
                "num_cached_tokens > seq.num_computed_tokens. \n"
                "but chain_extend is empty. \n"
            )

        ##############################
        ##### clean the dangling ends and concat the matched:
        ##### - chain
        ##### - block_table
        ##############################
        ##### must maintain:
        ##### - chain
        ##### - ref_count
        ##### - block_table
        ##### - num_computed_tokens
        ##### TOO MANY! combine chain ref_count in future scheduler
        ##############################

        ##### chain
        ##### current v3 scheduler will NOT have dangling nodes on chain
        chain_old = prefix_tree.chains[seq.seq_id]
        # exclude root
        num_blocks_solid = len(chain_old) - 1
        chain_old.extend(chain_extend)
        for i_node in chain_extend:
            i_node.ref_count += 1

        ##### block_table
        ##### clean dangling blocks
        block_table = self.kv_cache_manager.get_block_table(seq)

        table_solid = block_table[:num_blocks_solid]
        table_dangling = block_table[num_blocks_solid:]

        self.kv_cache_manager.block_allocator.free(table_dangling)
        self.kv_cache_manager.block_tables[seq.seq_id] = (
            #####
            table_solid + [x.physical_block_idx for x in chain_extend]
        )

        ##### num_computed_tokens
        seq.num_computed_tokens = num_cached_tokens

        return num_cached_tokens
