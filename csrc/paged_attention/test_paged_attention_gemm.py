import femtovllm._C
import torch
from torch.nn import functional as F

import femtovllm

##########
##### hyperparams
##########
num_blocks = 32
block_size = 16
n_heads = 8
n_kv_heads = 4
d_head = 128


##########
##### random
##########
cu_seqlens = [
    0,
    8,
    24,
]
q_len_max = max(
    #####
    (y - x)
    for x, y in zip(cu_seqlens[:-1], cu_seqlens[1:])
)
B = len(cu_seqlens) - 1
kv_lens = [
    block_size * 4 - 5,
    block_size * 3 - 4,
]
q = torch.randn((n_heads, cu_seqlens[-1], d_head), device="cuda")
k_pool = torch.randn((num_blocks, n_kv_heads, block_size, d_head), device="cuda")
v_pool = torch.randn((num_blocks, n_kv_heads, block_size, d_head), device="cuda")


raw_block_tables = [
    [0, 4, 5, 6],
    [1, 8, 9],
]
max_blocks = max(len(x) for x in raw_block_tables)
block_tables = []
for i_raw_tables in raw_block_tables:
    block_tables.append(
        i_raw_tables + [-1] * (max_blocks - len(i_raw_tables)),
    )
_block_tables = block_tables
block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")


positions = []
for i in range(B):
    q_len = cu_seqlens[i + 1] - cu_seqlens[i]
    kv_len = kv_lens[i]
    positions.append(
        [
            #####
            (kv_len - q_len + x)
            for x in range(q_len)
        ]
    )
positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
print(f"{positions=}\n")


_cu_seqlens = cu_seqlens
cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
_kv_lens = kv_lens
kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")


##########
##### test
##########
def gen_right_bottom_mask(q_len, kv_len):
    """ """
    q_pos = torch.arange(q_len, device="cuda") - q_len + kv_len
    kv_pos = torch.arange(kv_len, device="cuda")
    mask = q_pos[:, None] >= kv_pos[None, :]
    print("[ MASK ]")
    print(mask)
    return mask


paged_attn = femtovllm._C.PagedAttentionGemmCuda(
    q, k_pool, v_pool, cu_seqlens, q_len_max, block_tables, kv_lens, positions
)
print(paged_attn)
print(paged_attn.shape)
print()


ref_attn = []
for i in range(B):
    q_len = _cu_seqlens[i + 1] - _cu_seqlens[i]
    i_q = q[:, _cu_seqlens[i] : _cu_seqlens[i + 1], :]

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
        # # [BUG]
        # # when (q_len != kv_len) and (q_len > 1)
        # # always use attn_mask
        # is_causal=True,
        enable_gqa=True,
        attn_mask=gen_right_bottom_mask(q_len, kv_len),
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
