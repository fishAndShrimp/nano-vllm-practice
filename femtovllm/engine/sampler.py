import torch
import torch.nn as nn
import torch.nn.functional as F

from femtovllm.engine.sequence import Sequence


class Sampler(nn.Module):
    """ """

    def __init__(self):
        """ """
        super().__init__()

    def forward(
        self,
        logits_next: torch.Tensor,
        scheduled_const: list[tuple[Sequence, int]],
    ):
        """ """
        B, vocab_size = logits_next.shape

        # (B,)
        presence_penalties = torch.tensor(
            [x.sampling_params.presence_penalty for x, _ in scheduled_const],
            device=logits_next.device,
        )
        if presence_penalties.abs().max() > 1e-5:
            # (B, vocab_size)
            # TODO: optimize GPU copy
            counts = torch.stack(
                [
                    torch.bincount(
                        torch.tensor(
                            x.token_ids,
                            device=logits_next.device,
                        ),
                        minlength=vocab_size,
                    )
                    for x, _ in scheduled_const
                ]
            )
            logits_next = (
                #####
                logits_next - presence_penalties[:, None] * (counts > 0)
            )

        # (B,)
        temperatures = torch.tensor(
            [x.sampling_params.temperature for x, _ in scheduled_const],
            device=logits_next.device,
        )

        token_ids_next = torch.empty(
            B,
            dtype=torch.long,
            device=logits_next.device,
        )

        mask_greedy = temperatures < 1e-5
        if mask_greedy.any():
            token_ids_next[mask_greedy] = torch.argmax(logits_next[mask_greedy], dim=-1)

        mask_sample = ~mask_greedy
        if mask_sample.any():
            logits_sample = logits_next[mask_sample]
            logits_sample /= (temperatures[mask_sample])[:, None]
            token_ids_next[mask_sample] = torch.multinomial(
                F.softmax(
                    logits_sample,
                    dim=-1,
                ),
                num_samples=1,
            ).squeeze(-1)

        return token_ids_next.tolist()
