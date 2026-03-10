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
            raise ValueError(f"d_head must be even, now {d_head=}")

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


class QwenSelfAttention(nn.Module):
    """ """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"{(d_model % n_heads)=}")
        d_head = d_model // n_heads

        if n_heads % n_kv_heads != 0:
            raise ValueError(f"{(n_heads % n_kv_heads)=}")
        n_rep = n_heads // n_kv_heads

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_rep = n_rep

        self.dropout = dropout

        self.w_q = nn.Linear(d_model, n_heads * d_head, bias=True)
        self.w_k = nn.Linear(d_model, n_kv_heads * d_head, bias=True)
        self.w_v = nn.Linear(d_model, n_kv_heads * d_head, bias=True)

        self.rotary_embd = QwenRotaryEmbedding(max_seq_len, d_head)

        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_past=None,
        position: int = None,
    ):
        """ """
        B, T, C = x.shape
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        d_head = self.d_head
        n_rep = self.n_rep

        # (B, T, H, D)
        # (B, H, T, D)
        q = self.w_q(x).view(B, T, n_heads, d_head).transpose(1, 2)
        k = self.w_k(x).view(B, T, n_kv_heads, d_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, n_kv_heads, d_head).transpose(1, 2)

        q = self.rotary_embd(q, position)
        k = self.rotary_embd(k, position)

        if kv_past is not None:
            k_past, v_past = kv_past
            k = torch.cat((k_past, k), dim=-2)
            v = torch.cat((v_past, v), dim=-2)

        # [CRITICAL -1]
        # seqlen is no longer T
        k_rep = (
            k.unsqueeze(2)
            .expand(B, n_kv_heads, n_rep, -1, d_head)
            .reshape(B, n_kv_heads * n_rep, -1, d_head)
        )
        v_rep = (
            v.unsqueeze(2)
            .expand(B, n_kv_heads, n_rep, -1, d_head)
            .reshape(B, n_kv_heads * n_rep, -1, d_head)
        )

        # (B, H, T, D)
        out = F.scaled_dot_product_attention(
            q,
            k_rep,
            v_rep,
            is_causal=(T > 1),
            dropout_p=(self.dropout if self.training else 0.0),
        )

        # (B, T, H, D)
        # (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out), (k, v)
