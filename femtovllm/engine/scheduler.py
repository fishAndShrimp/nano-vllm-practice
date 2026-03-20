from femtovllm.engine.kv_cache_manager import KVCacheManager
from femtovllm.engine.request_queue import RequestQueue
from femtovllm.engine.sequence import Sequence
from femtovllm.engine.step_budget import StepBudget


class Scheduler:
    """ """

    def __init__(
        self,
        step_budget: StepBudget,
        request_queue: RequestQueue,
        kv_cache_manager: KVCacheManager,
    ):
        self.step_budget = step_budget
        self.request_queue = request_queue
        self.kv_cache_manager = kv_cache_manager

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

    def _finish(self, seq: Sequence):
        """
        [Atomic]
        - increase resource
        """
        self.kv_cache_manager.free(seq)
        seq.finish()

    def _force_finish(self, seq: Sequence, stop_reason: str):
        """ """
        seq.stop_reason = stop_reason
        self._finish(seq)

    def _sweep_stopped_sequences(self):
        """
        [Atomic]
        - running (with stop_reason) => finished
        - increase resource
        - remove from running queue
        """
        for seq in self.request_queue.sort_and_copy_running():
            if seq.stop_reason is not None:
                self._finish(seq)

        self.request_queue.clean_finished_running()

    def _calc_limit_computation(self):
        """ """
        return min(
            self.step_budget.max_tokens_per_seq,
            self.step_budget.remaining_tokens,
        )

    def _schedule_running(self):
        """ """
        scheduled: list[tuple[Sequence, int]] = []
        has_resource = True

        for seq in self.request_queue.sort_and_copy_running():
            # [STEP: seqs[curr:] are all preempted]
            if not seq.is_running():
                has_resource = False
                break

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
                            self._force_finish(seq, "OOM")
                        break

                    # truncate to fit kv_cache limit
                    num_tokens = min(num_tokens, limit_kv_cache)
                else:
                    self._preempt()

                fit_kv_cache = self.kv_cache_manager.can_allocate(seq, num_tokens)

            # [STEP: consume]
            if (
                #####
                seq.is_running() and (num_tokens > 0) and fit_budget and fit_kv_cache
            ):
                self._allocate(seq, num_tokens)
                scheduled.append(
                    (seq, num_tokens),
                )

        return scheduled, has_resource

    def _schedule_waiting(
        self,
        scheduled: list[tuple[Sequence, int]],
    ):
        """ """
        while self.request_queue.size_waiting > 0:
            # [STEP: highest priority waiting]
            seq = self.request_queue.peek_waiting()

            # [LIMIT: computation]
            # [LIMIT: storage]
            # truncate to fit both computation and storage limit
            limit_both = min(
                self._calc_limit_computation(),
                self.kv_cache_manager.calc_max_tokens_allocable(seq),
            )
            if limit_both <= 0:
                break

            num_tokens = min(
                seq.num_uncomputed_tokens,
                limit_both,
            )
            if num_tokens <= 0:
                raise RuntimeError(f"{seq=} strange {seq.num_uncomputed_tokens=}")

            fit_budget = self.step_budget.can_consume(num_tokens)
            fit_kv_cache = self.kv_cache_manager.can_allocate(seq, num_tokens)
            # [STEP: consume]
            if (
                #####
                (num_tokens > 0) and fit_budget and fit_kv_cache
            ):
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
        self.step_budget.reset()
        self._sweep_stopped_sequences()

        scheduled, has_resource = self._schedule_running()
        if has_resource:
            scheduled = self._schedule_waiting(scheduled)

        return scheduled

    def add_sequence(self, seq: Sequence):
        """ """
        self.request_queue.push_waiting(seq)
