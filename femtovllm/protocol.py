import dataclasses


@dataclasses.dataclass
class SamplingParams:
    """
    schema shared by inputs and engine
    """

    # probability manipulation
    temperature: float = 1.0
    presence_penalty: float = 1.0

    # stopping criteria
    stop_token_ids: list[int] = dataclasses.field(
        default_factory=list,
    )

    def clone(self):
        return dataclasses.replace(
            self,
            stop_token_ids=[x for x in self.stop_token_ids],
        )
