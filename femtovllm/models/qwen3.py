import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers import Qwen3Config


class SiluAndMul(nn.Module):
    """
    SwiGLU
    """

    def forward(self, x: torch.Tensor):
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y


class QwenRotaryEmbedding(nn.Module):
    """
    Qwen Rope
    """

    def __init__(
        self,
        max_seq_len: int,
        d_head: int,
        base: float = 1000000.0,
        device: str = None,
    ):
        super().__init__()

        if d_head % 2 != 0:
            raise ValueError("d_head")

        # (C//2)
        inv_freq = torch.arange(d_head // 2, device=device).float()
        inv_freq = 2 * inv_freq / d_head
        inv_freq = (1.0 / base) ** (inv_freq)

        # (T, C//2)
        theta = torch.arange(max_seq_len, device=device).float()
        theta = theta[:, None] * inv_freq[None, :]

        # (T, C)
        theta = torch.cat((theta, theta), dim=-1)

        # (T, C)
        self.register_buffer("sin", torch.sin(theta), persistent=False)
        self.register_buffer("cos", torch.cos(theta), persistent=False)

    def forward(
        self,
        q_or_k: torch.Tensor,
        position: int = None,
    ):
        T = q_or_k.shape[-2]

        x_y = q_or_k
        x, y = x_y.chunk(2, dim=-1)
        ny_x = torch.cat((-y, x), dim=-1)

        # (1, 1, T, C)
        if position is not None:
            w_sin = self.sin[position][None, None, None, :]
            w_cos = self.cos[position][None, None, None, :]
        else:
            w_sin = self.sin[:T][None, None, ...]
            w_cos = self.cos[:T][None, None, ...]

        return w_cos * x_y + w_sin * ny_x
