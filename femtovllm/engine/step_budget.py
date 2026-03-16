class StepBudget:
    def __init__(
        self,
        max_seqs: int,
        max_tokens: int,
    ):
        self.max_seqs = max_seqs
        self.max_tokens = max_tokens

        self.curr_seqs = 0
        self.curr_tokens = 0

        self.reset()

    def reset(self):
        self.curr_seqs = 0
        self.curr_tokens = 0

    def can_consume(self, num_tokens: int) -> bool:
        if self.curr_seqs + 1 > self.max_seqs:
            return False

        if self.curr_tokens + num_tokens > self.max_tokens:
            return False

        return True

    def consume(self, num_tokens: int):
        if not self.can_consume(num_tokens):
            raise RuntimeError("[StepBudget][Overflow]")

        self.curr_seqs += 1
        self.curr_tokens += num_tokens
