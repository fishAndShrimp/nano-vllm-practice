# Frequent Errors

## cat kv_past

```python
if kv_past is not None:
    k_past, v_past = kv_past
    k = torch.cat((k_past, k), dim=-2)
    v = torch.cat((v_past, v), dim=-2)
```

---

## transpose -> contiguous -> view

```python
out = out.transpose(1, 2).contiguous().view(B, T, C)
```

---

## dropout in F.scaled_dot_product_attention

```python
# (B, H, T, D)
out = F.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=(T > 1),
    dropout_p=(self.dropout if self.training else 0.0),
)
```

---

## k_rep and v_rep is right before attention calc

use expand and -1

```python
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
```

---

## bias=False

```python
self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
self.down_proj = nn.Linear(d_model, intermediate_size, bias=False)
```

---

## final norm for pre-norm model

```python
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
                QwenBlock(
                    d_model=config.hidden_size,
                    n_heads=config.num_attention_heads,
                    n_kv_heads=config.num_key_value_heads,
                    max_seq_len=config.max_position_embeddings,
                    intermediate_size=config.intermediate_size,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
```

---

## remember the returned (x, kv_cache)

```python
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

    def forward(self, idx):
        # (B, T, C)
        x, all_kv_cache = self.model(idx)

        # (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits, all_kv_cache
```

---

## nn.Linear shape is [out_features, in_features]

when cat [gate_proj, up_proj], dim is 0

```python
def merge_split_weights(my_key: str, hf_key: str, hf_tensor: torch.Tensor):
    if ".gate_up_proj." in my_key:
        idx = 0 if (".gate_proj." in hf_key) else 1
        fuse[my_key][idx] = hf_tensor

        if None not in fuse[my_key]:
            state_dict[my_key].copy_(torch.cat(fuse[my_key], dim=0))
            my_keys_used.add(my_key)
            del fuse[my_key]

        return True

    return False
```

```python
>>> nn.Linear(3,4).weight.shape
torch.Size([4, 3])
```

---

## head_dim != hidden_size // num_attention_heads

```json
Qwen3Config {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "head_dim": 128,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
}
```

---

