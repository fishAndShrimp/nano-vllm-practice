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
        device = logits_next.device

        ##########
        ##### CPU control flow
        ##########

        ##### presence_penalty
        presence_penalties = [
            x.sampling_params.presence_penalty for x, _ in scheduled_const
        ]
        apply_presence_penalty = any(
            #####
            (abs(x) > 1e-5)
            for x in presence_penalties
        )

        ##### temperature
        temperatures = [x.sampling_params.temperature for x, _ in scheduled_const]
        greedy_indices = []
        sample_indices = []
        for i, t in enumerate(temperatures):
            if t < 1e-5:
                greedy_indices.append(i)
            else:
                sample_indices.append(i)

        ##########
        ##### CPU control flow
        ##########

        ##########
        ##### GPU presence_penalty
        ##########
        # TODO: optimize GPU copy
        if apply_presence_penalty:
            pad_id = vocab_size

            raw_token_id_tables = [x.token_ids for x, _ in scheduled_const]
            max_table_len = max(len(x) for x in raw_token_id_tables)
            token_id_tables = [
                #####
                x + [pad_id] * (max_table_len - len(x))
                for x in raw_token_id_tables
            ]
            token_id_tables = torch.tensor(
                token_id_tables, dtype=torch.long, device=device
            )

            counts = torch.zeros((B, vocab_size + 1), dtype=torch.long, device=device)
            counts.scatter_add_(
                dim=1,
                index=token_id_tables,
                src=torch.ones_like(token_id_tables),
            )
            counts = counts[:, :vocab_size]

            presence_penalties = torch.tensor(presence_penalties, device=device)
            logits_next = logits_next - presence_penalties[:, None] * (counts > 0)
        ##########
        ##### GPU presence_penalty
        ##########

        ##########
        ##### GPU temperature
        ##########
        token_ids_next = torch.empty(B, dtype=torch.long, device=device)

        if greedy_indices:
            greedy_indices = torch.tensor(
                greedy_indices, dtype=torch.long, device=device
            )
            token_ids_next[greedy_indices] = torch.argmax(
                logits_next[greedy_indices], dim=-1
            )

        if sample_indices:
            temperatures = torch.tensor(temperatures, device=device)
            sample_indices = torch.tensor(
                sample_indices, dtype=torch.long, device=device
            )
            logits_sample = logits_next[sample_indices]
            logits_sample /= (temperatures[sample_indices])[:, None]

            token_ids_next[sample_indices] = torch.multinomial(
                F.softmax(logits_sample, dim=-1),
                num_samples=1,
            ).squeeze(-1)

        ##########
        ##### GPU temperature
        ##########

        return token_ids_next.tolist()
