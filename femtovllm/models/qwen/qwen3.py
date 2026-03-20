import collections
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm
from transformers import Qwen3Config


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
            w_sin = self.sin[position : position + T][None, None, ...]
            w_cos = self.cos[position : position + T][None, None, ...]
        else:
            w_sin = self.sin[:T][None, None, ...]
            w_cos = self.cos[:T][None, None, ...]

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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
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

        q = self.q_norm(q)
        k = self.k_norm(k)

        position = None
        if kv_cache is not None:
            k_past, v_past = kv_cache
            assert k_past.shape[-2] == v_past.shape[-2]
            position = k_past.shape[-2]

        q = self.rotary_embd(q, position)
        k = self.rotary_embd(k, position)

        if kv_cache is not None:
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
        kv_cache=None,
    ):
        """ """
        # (B, T, C)
        #    (T, C) varlen
        y, kv_cache = self.sa(
            self.input_layernorm(x),
            kv_cache=kv_cache,
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
    ):
        """
        - all_kv_cache: legacy list[(k,v)] mechanism
        """
        # (B, T, C)
        #    (T, C) varlen
        x = self.embed_tokens(idx)

        if all_kv_cache is None:
            all_kv_cache = [None for _ in self.layers]

        for i, layer in enumerate(self.layers):
            x, all_kv_cache[i] = layer(
                x,
                kv_cache=all_kv_cache[i],
            )

        return self.norm(x), all_kv_cache


class QwenForCausalLM(nn.Module):
    """ """

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
        all_kv_cache=None,
    ):
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
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        k_cache_pool: torch.Tensor,
        v_cache_pool: torch.Tensor,
        block_tables: torch.Tensor,
    ):
        return self._forward_varlen_fake(
            idx_flatten,
            positions,
            cu_seqlens,
            k_cache_pool,
            v_cache_pool,
            block_tables,
        )

    def _forward_varlen_fake(
        self,
        # (T)
        idx_flatten: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        k_cache_pool: torch.Tensor,
        v_cache_pool: torch.Tensor,
        block_tables: torch.Tensor,
    ):
        _PAD_TOKEN = "<|endoftext|>"
        _PAD_TOKEN_ID = 151643

        B = len(cu_seqlens) - 1
        T = max(
            [
                #####
                (cu_seqlens[i + 1] - cu_seqlens[i])
                for i in range(B)
            ]
        )

        idx = torch.full(
            (B, T),
            fill_value=_PAD_TOKEN_ID,
            dtype=idx_flatten.dtype,
            device=idx_flatten.device,
        )

        for i in range(B):
            i_len = cu_seqlens[i + 1] - cu_seqlens[i]
            idx[i, :i_len] = idx_flatten[
                #####
                cu_seqlens[i] : cu_seqlens[i + 1]
            ]

        x, _ = self.model(idx, None)

        x_flatten = []
        for i in range(B):
            i_len = cu_seqlens[i + 1] - cu_seqlens[i]
            x_flatten.append(x[i, :i_len])
        x_flatten = torch.cat(x_flatten, dim=0)

        logits_flatten: torch.Tensor = self.lm_head(x_flatten)
        return logits_flatten

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
