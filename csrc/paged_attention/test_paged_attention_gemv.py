import femtovllm._C
import torch
from torch.nn import functional as F

import femtovllm

##########
##### constants
##########
BLOCK_SIZE = 32


##########
##### hyperparams
##########
num_blocks = 32
n_heads = 8
n_kv_heads = 4
d_head = 16


##########
##### random
##########
q_len_flatten = 2
kv_lens = [
    BLOCK_SIZE * 4 - 5,
    BLOCK_SIZE * 3 - 4,
]
q = torch.randn((n_heads, q_len_flatten, d_head), device="cuda")
k_pool = torch.randn((num_blocks, n_kv_heads, BLOCK_SIZE, d_head), device="cuda")
v_pool = torch.randn((num_blocks, n_kv_heads, BLOCK_SIZE, d_head), device="cuda")


raw_block_tables = [
    [0, 4, 5, 6],
    [1, 8, 9],
]
max_blocks = max(len(x) for x in raw_block_tables)
block_tables = []
for raw_table in raw_block_tables:
    block_tables.append(
        raw_table + [-1] * (max_blocks - len(raw_table)),
    )
_block_tables = block_tables
block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")


max_kv_len = max(kv_lens)
_kv_lens = kv_lens
kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")


##########
##### test
##########
paged_attn = femtovllm._C.PagedAttentionGemvCuda(
    q, k_pool, v_pool, block_tables, kv_lens, max_kv_len
)
print(paged_attn)
print(paged_attn.shape)
print()


ref_attn = []
for i in range(q_len_flatten):
    q_len = 1
    i_q = q[:, i : i + 1, :]

    kv_len = _kv_lens[i]
    # i_block_table = _block_tables[i]
    # print(f"{i_block_table=}")
    i_block_table = [x for x in _block_tables[i] if (x >= 0)]
    # print(f"{i_block_table=}")

    i_k = torch.cat([k_pool[x] for x in i_block_table], dim=-2)[:, :kv_len, :]
    i_v = torch.cat([v_pool[x] for x in i_block_table], dim=-2)[:, :kv_len, :]

    i_sub_ref_attn = F.scaled_dot_product_attention(
        i_q,
        i_k,
        i_v,
        # q_len is 1
        # no difference between using causal or not
        is_causal=False,
        enable_gqa=True,
    )
    ref_attn.append(i_sub_ref_attn)
ref_attn = torch.cat(ref_attn, dim=-2)
print(ref_attn)
print(ref_attn.shape)
print()


print(f"{torch.allclose(paged_attn, ref_attn)=}")
diff_attn = paged_attn - ref_attn
# print(diff_attn)
print(f"{diff_attn.abs().max()=}")
