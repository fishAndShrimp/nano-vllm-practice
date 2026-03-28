import collections
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm
from transformers import Qwen3Config

import femtovllm
from femtovllm.protocol import VarlenAttnMetadata


class QwenRotaryEmbedding(nn.Module):
    """
    Qwen RoPE (Rotary Position Embedding)
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

        # (D//2)
        inv_freq = torch.arange(d_head // 2, device=device).float()
        inv_freq = 2 * inv_freq / d_head
        inv_freq = (1.0 / base) ** (inv_freq)

        # (T, D//2)
        theta = torch.arange(max_seq_len, device=device).float()
        theta = theta[:, None] * inv_freq[None, :]

        # (T, D)
        theta = torch.cat((theta, theta), dim=-1)

        # (T, D)
        self.register_buffer("sin", torch.sin(theta), persistent=False)
        self.register_buffer("cos", torch.cos(theta), persistent=False)

    def forward(
        self,
        q_or_k: torch.Tensor,
        positions: int | torch.Tensor = None,
    ):
        """
        Applies Rotary Position Embedding (RoPE) to the Query or Key tensor.

        This method automatically routes the computation based on the dimensions of the
        input tensor `q_or_k`, enforcing strict type checks for the `positions` argument.

        1. Standard Padded Mode (4D)
            - q_or_k shape: `(B, H, T, D)`
            - positions: Must be `int` or `None`.
                - If `None`: Rotates using positions `[0 : T]`.
                - If `int` (e.g., p): Rotates using positions `[p : p + T]` (useful for decoding).

        2. Varlen / FlashAttention Mode (3D)
            - q_or_k shape: `(seqlen_total, H, D)`
            - positions: Must be a 1D `torch.Tensor`.
                - Shape `(seqlen_total,)` containing the exact absolute position IDs
                  for each token in the flattened batch.

        Args:
            q_or_k (torch.Tensor): The input Query or Key tensor to rotate.
            positions (int | torch.Tensor | None): The position index/indices.

        Returns:
            torch.Tensor: The rotated tensor, maintaining the exact same shape as `q_or_k`.

        Raises:
            RuntimeError: If the type of `positions` does not match the required type
                          for the detected operating mode (3D vs 4D).
        """

        # (B, H, T, D)
        # or (seqlen_total, H, D) varlen
        is_varlen = q_or_k.dim() == 3

        # [STEP: route checks]
        if is_varlen:
            required_positions_types = (torch.Tensor,)
        else:
            required_positions_types = (int, type(None))
        if not isinstance(positions, required_positions_types):
            raise RuntimeError(
                f"{positions=} {required_positions_types=} {q_or_k.dim()=}"
            )

        # [STEP: (x, y) and (negative_y, x)]
        x_y = q_or_k
        x, y = x_y.chunk(2, dim=-1)
        ny_x = torch.cat((-y, x), dim=-1)

        # [STEP: pick sin and cos]
        if is_varlen:
            # (seqlen_total, D) varlen
            w_sin = self.sin[positions]
            w_cos = self.cos[positions]
            # (seqlen_total, 1, D) varlen
            w_sin = w_sin[:, None, :]
            w_cos = w_cos[:, None, :]
        else:
            _, _, T, _ = q_or_k.shape

            # (T, D)
            if isinstance(positions, int):
                w_sin = self.sin[positions : positions + T]
                w_cos = self.cos[positions : positions + T]
            else:
                w_sin = self.sin[:T]
                w_cos = self.cos[:T]

            # (1, 1, T, D)
            w_sin = w_sin[None, None, ...]
            w_cos = w_cos[None, None, ...]

        # [STEP: broadcast]
        # (B, H, T, D)
        # *
        # (1, 1, T, D)
        # or
        # (seqlen_total, H, D) varlen
        # *
        # (seqlen_total, 1, D) varlen
        return w_cos * x_y + w_sin * ny_x


class QwenSelfAttention(nn.Module):
    """
    diff from nanogpt:
    - GQA
    - Rope
    - KVCache
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout_p: float = 0.1,
        config: Qwen3Config = None,
    ):
        super().__init__()
        self.varlen_attn_impl = femtovllm._DEV.varlen_attn_impl

        if d_model % n_heads != 0:
            raise ValueError(f"{(d_model % n_heads)=}")
        d_head = d_model // n_heads if (config is None) else config.head_dim

        if n_heads % n_kv_heads != 0:
            raise ValueError(f"{(n_heads % n_kv_heads)=}")
        n_rep = n_heads // n_kv_heads

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_rep = n_rep

        self.dropout_p = dropout_p

        self.w_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * d_head, bias=False)

        rms_norm_eps = 1e-6 if (config is None) else config.rms_norm_eps
        self.q_norm = nn.RMSNorm(d_head, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(d_head, eps=rms_norm_eps)
        self.rotary_embd = QwenRotaryEmbedding(max_seq_len, d_head)

        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def gen_right_bottom_attn_mask(
        self,
        q_len: int,
        kv_len: int,
        device: str | torch.device,
    ):
        """ """
        q_pos = torch.arange(q_len, device=device) - q_len + kv_len
        kv_pos = torch.arange(kv_len, device=device)
        mask = q_pos[:, None] >= kv_pos[None, :]
        return mask

    def forward_varlen_custom_gemm(
        self,
        x: torch.Tensor,
        k_cache_pool: torch.Tensor,
        v_cache_pool: torch.Tensor,
        varlen_attn_metadata: VarlenAttnMetadata,
    ):
        """ """
        B = len(varlen_attn_metadata.block_tables)
        seqlen_total, C = x.shape

        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        d_head = self.d_head
        n_rep = self.n_rep

        # (seqlen_total, H, D)
        # no longer transpose
        # ========================================================================
        # [CRITICAL: Optimal Memory Dataflow for Varlen Attention]
        # To eliminate memory movement overhead, we strictly defer any transpose
        # operations until the final caching stage. The elegant dataflow is:
        #
        # 1. Projection: Linear layer outputs contiguous (seqlen_flatten, H * D).
        # 2. Reshape: Zero-copy `.view()` into (seqlen_flatten, H, D).
        # 3. RoPE: Applied in-place on (seqlen_flatten, H, D) without moving memory.
        # 4. Caching (Final Destination): We only transpose at the exact moment of
        #    writing into the Paged KV Cache `[num_blocks, n_kv_heads, block_size, d_head]`.
        #    This allows PyTorch's underlying `copy_()` to fuse the memory rearrangement
        #    into a single step, avoiding expensive `.contiguous()` allocations.
        # ========================================================================
        q = self.w_q(x).view(seqlen_total, n_heads, d_head)
        k = self.w_k(x).view(seqlen_total, n_kv_heads, d_head)
        v = self.w_v(x).view(seqlen_total, n_kv_heads, d_head)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rotary_embd(q, varlen_attn_metadata.positions)
        k = self.rotary_embd(k, varlen_attn_metadata.positions)

        attn = []
        for batch in range(B):
            begin = int(varlen_attn_metadata.cu_seqlens[batch])
            end = int(varlen_attn_metadata.cu_seqlens[batch + 1])

            block_table = varlen_attn_metadata.block_tables[batch]
            _, _, block_size, _ = k_cache_pool.shape

            # (T, H, D)
            i_q = q[begin:end]
            T, _, _ = i_q.shape
            # (H, T, D)
            i_q = i_q.transpose(0, 1)

            # (H, kv_len, D)
            # reuse and append cache
            # cat in kv_len (dim=1)
            i_k = []
            i_v = []

            remaining_cache = int(varlen_attn_metadata.positions[begin])
            p_begin = begin
            for block_index in block_table.tolist():  # EXPENSIVE
                if block_index < 0:
                    break
                if p_begin >= end:
                    break

                if remaining_cache < block_size:
                    # current block has empty slots
                    # start to cache more
                    num_new_tokens = min(
                        # empty slots in current block
                        block_size - remaining_cache,
                        # remaining new kv
                        end - p_begin,
                    )

                    # (T, n_kv_heads, D)
                    # transpose
                    # (n_kv_heads, T, D)
                    k_cache_pool[
                        block_index,
                        :,
                        remaining_cache : remaining_cache + num_new_tokens,
                    ] = k[p_begin : p_begin + num_new_tokens].transpose(0, 1)
                    v_cache_pool[
                        block_index,
                        :,
                        remaining_cache : remaining_cache + num_new_tokens,
                    ] = v[p_begin : p_begin + num_new_tokens].transpose(0, 1)

                    remaining_cache += num_new_tokens
                    p_begin += num_new_tokens

                    i_k.append(k_cache_pool[block_index, :, :remaining_cache])
                    i_v.append(v_cache_pool[block_index, :, :remaining_cache])
                else:
                    i_k.append(k_cache_pool[block_index])
                    i_v.append(v_cache_pool[block_index])
                remaining_cache -= block_size

            # (n_kv_heads, T, D)
            # cat
            # (n_kv_heads, kv_len, D)
            i_k = torch.cat(i_k, dim=1)
            i_v = torch.cat(i_v, dim=1)

            # (n_kv_heads, kv_len, D)
            # replicate kv
            # (n_kv_heads, n_rep, kv_len, D)
            # reshape
            # (H, kv_len, D)
            #
            # [CRITICAL: The Replica Manner for GQA]
            # We must replicate the SAME kv_head `n_rep` times consecutively so that
            # the first `n_rep` query heads share the exact same kv_head.
            #
            # Example: 4 Q_heads, 2 KV_heads (n_rep = 2)
            # CORRECT memory layout after reshape: [KV0, KV0, KV1, KV1]
            #   -> Q0 matches KV0, Q1 matches KV0 | Q2 matches KV1, Q3 matches KV1
            #
            # WRONG layout (if unsqueeze(0) was used): [KV0, KV1, KV0, KV1]
            #   -> Q1 would wrongly match KV1, causing severe feature mismatch.
            i_k_rep = (
                i_k.unsqueeze(1)
                .expand(n_kv_heads, n_rep, -1, d_head)
                .reshape(n_kv_heads * n_rep, -1, d_head)
            )
            i_v_rep = (
                i_v.unsqueeze(1)
                .expand(n_kv_heads, n_rep, -1, d_head)
                .reshape(n_kv_heads * n_rep, -1, d_head)
            )
            _, kv_len, _ = i_k_rep.shape

            # (H, T, D)
            i_attn = F.scaled_dot_product_attention(
                # (H, T, D)
                i_q,
                # (H, kv_len, D)
                i_k_rep,
                i_v_rep,
                attn_mask=self.gen_right_bottom_attn_mask(T, kv_len, x.device),
                dropout_p=(self.dropout_p if self.training else 0.0),
            )
            # (T, C) = (T, H*D)
            i_attn = i_attn.transpose(0, 1).contiguous().view(T, n_heads * d_head)

            attn.append(i_attn)

        # (seqlen_total, C)
        attn = torch.cat(attn, dim=0)

        return self.o_proj(attn), None

    def forward_varlen_pytorch(
        self,
        x: torch.Tensor,
        k_cache_pool: torch.Tensor,
        v_cache_pool: torch.Tensor,
        varlen_attn_metadata: VarlenAttnMetadata,
    ):
        """ """
        B = len(varlen_attn_metadata.block_tables)
        seqlen_total, C = x.shape

        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        d_head = self.d_head
        n_rep = self.n_rep

        # (seqlen_total, H, D)
        # no longer transpose
        # ========================================================================
        # [CRITICAL: Optimal Memory Dataflow for Varlen Attention]
        # To eliminate memory movement overhead, we strictly defer any transpose
        # operations until the final caching stage. The elegant dataflow is:
        #
        # 1. Projection: Linear layer outputs contiguous (seqlen_flatten, H * D).
        # 2. Reshape: Zero-copy `.view()` into (seqlen_flatten, H, D).
        # 3. RoPE: Applied in-place on (seqlen_flatten, H, D) without moving memory.
        # 4. Caching (Final Destination): We only transpose at the exact moment of
        #    writing into the Paged KV Cache `[num_blocks, n_kv_heads, block_size, d_head]`.
        #    This allows PyTorch's underlying `copy_()` to fuse the memory rearrangement
        #    into a single step, avoiding expensive `.contiguous()` allocations.
        # ========================================================================
        q = self.w_q(x).view(seqlen_total, n_heads, d_head)
        k = self.w_k(x).view(seqlen_total, n_kv_heads, d_head)
        v = self.w_v(x).view(seqlen_total, n_kv_heads, d_head)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rotary_embd(q, varlen_attn_metadata.positions)
        k = self.rotary_embd(k, varlen_attn_metadata.positions)

        attn = []
        for batch in range(B):
            begin = varlen_attn_metadata.raw_cu_seqlens[batch]
            end = varlen_attn_metadata.raw_cu_seqlens[batch + 1]

            raw_block_table = varlen_attn_metadata.raw_block_tables[batch]
            _, _, block_size, _ = k_cache_pool.shape

            # (T, H, D)
            i_q = q[begin:end]
            T, _, _ = i_q.shape
            # (H, T, D)
            i_q = i_q.transpose(0, 1)

            # (H, kv_len, D)
            # reuse and append cache
            # cat in kv_len (dim=1)
            i_k = []
            i_v = []

            remaining_cache = varlen_attn_metadata.raw_positions[begin]
            p_begin = begin
            for block_index in raw_block_table:
                if block_index < 0:
                    break
                if p_begin >= end:
                    break

                if remaining_cache < block_size:
                    # current block has empty slots
                    # start to cache more
                    num_new_tokens = min(
                        # empty slots in current block
                        block_size - remaining_cache,
                        # remaining new kv
                        end - p_begin,
                    )

                    # (T, n_kv_heads, D)
                    # transpose
                    # (n_kv_heads, T, D)
                    k_cache_pool[
                        block_index,
                        :,
                        remaining_cache : remaining_cache + num_new_tokens,
                    ] = k[p_begin : p_begin + num_new_tokens].transpose(0, 1)
                    v_cache_pool[
                        block_index,
                        :,
                        remaining_cache : remaining_cache + num_new_tokens,
                    ] = v[p_begin : p_begin + num_new_tokens].transpose(0, 1)

                    remaining_cache += num_new_tokens
                    p_begin += num_new_tokens

                    i_k.append(k_cache_pool[block_index, :, :remaining_cache])
                    i_v.append(v_cache_pool[block_index, :, :remaining_cache])
                else:
                    i_k.append(k_cache_pool[block_index])
                    i_v.append(v_cache_pool[block_index])
                remaining_cache -= block_size

            # (n_kv_heads, T, D)
            # cat
            # (n_kv_heads, kv_len, D)
            i_k = torch.cat(i_k, dim=1)
            i_v = torch.cat(i_v, dim=1)

            # (n_kv_heads, kv_len, D)
            # replicate kv
            # (n_kv_heads, n_rep, kv_len, D)
            # reshape
            # (H, kv_len, D)
            #
            # [CRITICAL: The Replica Manner for GQA]
            # We must replicate the SAME kv_head `n_rep` times consecutively so that
            # the first `n_rep` query heads share the exact same kv_head.
            #
            # Example: 4 Q_heads, 2 KV_heads (n_rep = 2)
            # CORRECT memory layout after reshape: [KV0, KV0, KV1, KV1]
            #   -> Q0 matches KV0, Q1 matches KV0 | Q2 matches KV1, Q3 matches KV1
            #
            # WRONG layout (if unsqueeze(0) was used): [KV0, KV1, KV0, KV1]
            #   -> Q1 would wrongly match KV1, causing severe feature mismatch.
            i_k_rep = (
                i_k.unsqueeze(1)
                .expand(n_kv_heads, n_rep, -1, d_head)
                .reshape(n_kv_heads * n_rep, -1, d_head)
            )
            i_v_rep = (
                i_v.unsqueeze(1)
                .expand(n_kv_heads, n_rep, -1, d_head)
                .reshape(n_kv_heads * n_rep, -1, d_head)
            )
            _, kv_len, _ = i_k_rep.shape

            # (H, T, D)
            i_attn = F.scaled_dot_product_attention(
                # (H, T, D)
                i_q,
                # (H, kv_len, D)
                i_k_rep,
                i_v_rep,
                attn_mask=self.gen_right_bottom_attn_mask(T, kv_len, x.device),
                dropout_p=(self.dropout_p if self.training else 0.0),
            )
            # (T, C) = (T, H*D)
            i_attn = i_attn.transpose(0, 1).contiguous().view(T, n_heads * d_head)

            attn.append(i_attn)

        # (seqlen_total, C)
        attn = torch.cat(attn, dim=0)

        return self.o_proj(attn), None

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
        k_cache_pool: torch.Tensor = None,
        v_cache_pool: torch.Tensor = None,
        varlen_attn_metadata: VarlenAttnMetadata = None,
    ) -> tuple[
        torch.Tensor,
        Optional[list[tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """ """
        if varlen_attn_metadata is not None:
            if self.varlen_attn_impl == "custom_gemm":
                return self.forward_varlen_custom_gemm(
                    x=x,
                    k_cache_pool=k_cache_pool,
                    v_cache_pool=v_cache_pool,
                    varlen_attn_metadata=varlen_attn_metadata,
                )

            if self.varlen_attn_impl == "pytorch":
                return self.forward_varlen_pytorch(
                    x=x,
                    k_cache_pool=k_cache_pool,
                    v_cache_pool=v_cache_pool,
                    varlen_attn_metadata=varlen_attn_metadata,
                )

            raise NotImplementedError(f"{self.varlen_attn_impl=}")

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

        q = self.q_norm(q)
        k = self.k_norm(k)

        position = None
        if kv_cache is not None:
            k_past, v_past = kv_cache
            assert k_past.shape[-2] == v_past.shape[-2]
            position = k_past.shape[-2]

        q = self.rotary_embd(q, position)
        k = self.rotary_embd(k, position)

        # (B, H, kv_len, D)
        if kv_cache is not None:
            k = torch.cat((k_past, k), dim=-2)
            v = torch.cat((v_past, v), dim=-2)

        # [CRITICAL auto -1 length]
        # seqlen is no longer T
        # e.g. When decoding,
        # q.T is still 1,
        # but k.kv_len and v.kv_len is (kv_past_len + 1)
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
        _, _, kv_len, _ = k_rep.shape

        # (B, H, T, D)
        out = F.scaled_dot_product_attention(
            q,
            k_rep,
            v_rep,
            attn_mask=self.gen_right_bottom_attn_mask(T, kv_len, x.device),
            dropout_p=(self.dropout_p if self.training else 0.0),
        )

        # (B, T, H, D)
        # (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, n_heads * d_head)

        return self.o_proj(out), (k, v)


class SiluAndMul(nn.Module):
    """
    SwiGLU
    """

    def forward(self, x: torch.Tensor):
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y


class QwenFeedForward(nn.Module):
    """ """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_up_proj(x)),
        )


class QwenDecoderLayer(nn.Module):
    """ """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        intermediate_size: int,
        dropout_p: float = 0.1,
        config: Qwen3Config = None,
    ):
        super().__init__()

        rms_norm_eps = 1e-6 if (config is None) else config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(d_model, eps=rms_norm_eps)
        self.sa = QwenSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            dropout_p=dropout_p,
            config=config,
        )
        self.dropout_sa = nn.Dropout(dropout_p)

        self.post_attention_layernorm = nn.RMSNorm(d_model, eps=rms_norm_eps)
        self.ffn = QwenFeedForward(
            d_model=d_model,
            intermediate_size=intermediate_size,
        )
        self.dropout_ffn = nn.Dropout(dropout_p)

    def forward(
        self,
        x,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
        k_cache_pool: torch.Tensor = None,
        v_cache_pool: torch.Tensor = None,
        varlen_attn_metadata: VarlenAttnMetadata = None,
    ):
        """ """
        # (B, T, C)
        # or (seqlen_total, C) varlen
        y, kv_cache = self.sa(
            self.input_layernorm(x),
            kv_cache=kv_cache,
            k_cache_pool=k_cache_pool,
            v_cache_pool=v_cache_pool,
            varlen_attn_metadata=varlen_attn_metadata,
        )
        x = x + self.dropout_sa(y)

        x = x + self.dropout_ffn(
            self.ffn(
                self.post_attention_layernorm(x),
            )
        )

        return x, kv_cache


class QwenModel(nn.Module):
    """ """

    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(
                    d_model=config.hidden_size,
                    n_heads=config.num_attention_heads,
                    n_kv_heads=config.num_key_value_heads,
                    max_seq_len=config.max_position_embeddings,
                    intermediate_size=config.intermediate_size,
                    config=config,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        idx,
        all_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None,
        varlen_attn_metadata: VarlenAttnMetadata = None,
    ):
        """
        - all_kv_cache: legacy list[(k,v)] mechanism
        """
        # (B, T, C)
        # or (seqlen_total, C) varlen
        x = self.embed_tokens(idx)

        if all_kv_cache is None:
            all_kv_cache = [None for _ in self.layers]

        for i, layer in enumerate(self.layers):
            k_cache_pool = (
                None
                if (varlen_attn_metadata is None)
                else varlen_attn_metadata.k_cache_pools[i]
            )
            v_cache_pool = (
                None
                if (varlen_attn_metadata is None)
                else varlen_attn_metadata.v_cache_pools[i]
            )

            x, all_kv_cache[i] = layer(
                x,
                kv_cache=all_kv_cache[i],
                k_cache_pool=k_cache_pool,
                v_cache_pool=v_cache_pool,
                varlen_attn_metadata=varlen_attn_metadata,
            )

        return self.norm(x), all_kv_cache


class QwenForCausalLM(nn.Module):
    """
    Standard Qwen Causal LM wrapper,
    containing the base model and the LM head.

    NOTE FOR ENGINE DEVELOPERS:
    For high-performance varlen (flattened) inference,
    do NOT use this class's `forward` method directly.

    Instead, call `self.model` to get hidden states,
    extract only the tokens you need (e.g., the last token of each sequence),
    and apply `self.lm_head` manually to save computation.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        idx,
        all_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Standard forward pass for padded `(B, T)` inputs.

        WARNING: This method does NOT support varlen (flattened) inputs.

        Applying the `lm_head` to an entire `(seqlen_total, C)` flattened tensor,
        introduces massive computational waste,
        as we typically only need logits for the final token of each sequence.

        For varlen inference,
        use the base `self.model` instead.
        """

        # (B, T, C)
        x, all_kv_cache = self.model(
            idx,
            all_kv_cache=all_kv_cache,
        )

        # (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits, all_kv_cache

    def forward_varlen(
        self,
        # (T)
        idx_flatten: torch.Tensor,
        varlen_attn_metadata: VarlenAttnMetadata,
    ):
        """
        - input.shape: (seqlen_total,)
        - output.shape: (B, vocab_size)
        - B: len(cu_seqlens) - 1
        """
        if femtovllm._DEV.fake_varlen_by_batch:
            return self._forward_varlen_fake(
                idx_flatten=idx_flatten,
                varlen_attn_metadata=varlen_attn_metadata,
            )

        hidden, _ = self.model(
            idx=idx_flatten,
            varlen_attn_metadata=varlen_attn_metadata,
        )

        hidden_next = hidden[
            #####
            varlen_attn_metadata.cu_seqlens[1:] - 1
        ]
        logits_next: torch.Tensor = self.lm_head(hidden_next)

        return logits_next

    def _forward_varlen_fake(
        self,
        # (T)
        idx_flatten: torch.Tensor,
        varlen_attn_metadata: VarlenAttnMetadata,
    ):
        """
        WARNING: Temporary workaround to fast verify engine logic.

        This function reconstructs the flattened 1D input back into a padded `(B, T)` shape.

        CRITICAL LIMITATION:
        Because the `(B, T)` format fundamentally cannot handle variable-length historical KV caches,
        this implementation ONLY works for the VERY FIRST chunked prefill (when KV cache is empty).
        Any subsequent chunked prefills or decode steps will fail, as appending to unaligned,
        variable-length historical KV caches breaks the dense tensor shape.
        """

        _PAD_TOKEN = "<|endoftext|>"
        _PAD_TOKEN_ID = 151643

        raw_cu_seqlens = varlen_attn_metadata.raw_cu_seqlens

        B = len(varlen_attn_metadata.raw_cu_seqlens) - 1
        T = varlen_attn_metadata.q_len_max

        idx_batch = []
        idx_flatten_cpu = idx_flatten.tolist()  # EXPENSIVE
        for i in range(B):
            q_begin = raw_cu_seqlens[i]
            q_end = raw_cu_seqlens[i + 1]
            q_len = q_end - q_begin
            idx_batch.append(
                idx_flatten_cpu[q_begin:q_end] + [_PAD_TOKEN_ID] * (T - q_len)
            )

        idx_batch = torch.tensor(
            idx_batch, dtype=idx_flatten.dtype, device=idx_flatten.device
        )

        # (B, T, C)
        x, _ = self.model(idx_batch, None)

        x_flatten = []
        for i in range(B):
            q_begin = raw_cu_seqlens[i]
            q_end = raw_cu_seqlens[i + 1]
            q_len = q_end - q_begin
            x_flatten.append(x[i, :q_len])  # EXPENSIVE
        x_flatten = torch.cat(x_flatten, dim=0)

        # (B, C)
        x_next = x_flatten[
            #####
            varlen_attn_metadata.cu_seqlens[1:] - 1
        ]
        # (B, vocab_size)
        logits_next: torch.Tensor = self.lm_head(x_next)
        return logits_next

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.1,
        enable_kv_cache: bool = True,
        eos_token_ids: int | list[int] = None,
        pad_token_id: int = None,
        presence_penalty: float = 0.5,
    ):
        """ """

        def build_eos_pad_ids(
            eos_token_ids: int | list[int],
            pad_token_id: int,
        ):
            if isinstance(eos_token_ids, int):
                eos_token_ids = [eos_token_ids]
            assert len(eos_token_ids) > 0

            if pad_token_id is None:
                pad_token_id = eos_token_ids[0]

            eos_token_ids = torch.tensor(
                eos_token_ids,
                device=idx.device,
            )

            return eos_token_ids, pad_token_id

        original_mode = self.training
        self.eval()
        try:
            B, _ = idx.shape
            if eos_token_ids is not None:
                mask_finished: torch.Tensor = torch.zeros(
                    (B, 1),
                    dtype=torch.bool,
                    device=idx.device,
                )
                eos_token_ids, pad_token_id = build_eos_pad_ids(
                    eos_token_ids, pad_token_id
                )

            idx_next = idx
            all_kv_cache = None
            for _ in tqdm(
                range(max_new_tokens),
                desc="Generating Tokens",
            ):
                if enable_kv_cache:
                    logits, all_kv_cache = self(
                        idx_next,
                        all_kv_cache=all_kv_cache,
                    )
                else:
                    logits, all_kv_cache = self(idx)

                logits = logits[:, -1, :]

                if presence_penalty > 0:
                    for b in range(B):
                        ids_presence = torch.unique(idx[b])
                        logits[b, ids_presence] -= presence_penalty

                if temperature < 1e-5:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(
                        logits / temperature,
                        dim=-1,
                    )
                    idx_next = torch.multinomial(probs, num_samples=1)

                if eos_token_ids is not None:
                    idx_next.masked_fill_(mask_finished, pad_token_id)
                    mask_finished = mask_finished | (
                        #####
                        torch.isin(idx_next, eos_token_ids)
                    )

                idx = torch.cat([idx, idx_next], dim=-1)

                if (eos_token_ids is not None) and mask_finished.all():
                    break

        finally:
            self.train(original_mode)

        return idx

    def load_weights(
        self,
        local_weights_dir: Path | str,
    ):
        static_load_weights(
            self,
            local_weights_dir,
        )


def map_weight_key(hf_key: str):
    """ """
    return (
        hf_key.replace(".self_attn.", ".sa.")
        .replace(".q_proj.", ".w_q.")
        .replace(".k_proj.", ".w_k.")
        .replace(".v_proj.", ".w_v.")
        .replace(".mlp.", ".ffn.")
        .replace(".gate_proj.", ".gate_up_proj.")
        .replace(".up_proj.", ".gate_up_proj.")
    )


def static_load_weights(
    model: nn.Module,
    local_weights_dir: Path | str,
    ignore_hf_keys: set = None,
):
    """ """
    if ignore_hf_keys is None:
        ignore_hf_keys = {}

    state_dict = model.state_dict()
    my_keys_used = set()
    fusion = collections.defaultdict(lambda: [None, None])

    def merge_split_weights(my_key: str, hf_key: str, hf_tensor: torch.Tensor):
        if ".gate_up_proj." in my_key:
            idx = 0 if (".gate_proj." in hf_key) else 1
            fusion[my_key][idx] = hf_tensor

            if None not in fusion[my_key]:
                state_dict[my_key].copy_(torch.cat(fusion[my_key], dim=0))
                my_keys_used.add(my_key)
                del fusion[my_key]

            return True

        return False

    for path in Path(local_weights_dir).iterdir():
        if path.suffix != ".safetensors":
            continue

        with safe_open(path, framework="pt", device="cpu") as f:
            for hf_key in f.keys():
                if hf_key in ignore_hf_keys:
                    continue

                my_key = map_weight_key(hf_key)

                if my_key in state_dict:
                    hf_tensor = f.get_tensor(hf_key)
                    if not merge_split_weights(my_key, hf_key, hf_tensor):
                        try:
                            state_dict[my_key].copy_(hf_tensor)
                            my_keys_used.add(my_key)
                        except Exception as e:
                            raise RuntimeError(
                                f"ERROR when copy {my_key=} <= {hf_key=} {hf_tensor.shape=}"
                            ) from e
                else:
                    raise ValueError(f"UNEXPECTED {my_key=} {hf_key=}")

    if state_dict.keys() != my_keys_used:
        raise ValueError(f"MISSING {(state_dict.keys() - my_keys_used)=}")
