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

