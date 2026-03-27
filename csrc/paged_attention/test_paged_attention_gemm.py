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
cu_seqlens = torch.tensor(
    [
        0,
        8,
        24,
    ],
    dtype=torch.int32,
    device="cuda",
)
q_len_max = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
B = len(cu_seqlens) - 1
kv_lens = torch.tensor(
    [
        block_size * 4 - 5,
        block_size * 3 - 4,
    ],
    dtype=torch.int32,
    device="cuda",
)
q = torch.randn((n_heads, cu_seqlens[-1], d_head), device="cuda")
k_pool = torch.randn((num_blocks, n_kv_heads, block_size, d_head), device="cuda")
v_pool = torch.randn((num_blocks, n_kv_heads, block_size, d_head), device="cuda")


block_tables = torch.ones((B, 8), dtype=torch.int32)
for i, i_rnd_table in enumerate(
    [
        [0, 4, 5, 6],
        [1, 8, 9],
    ]
):
    block_tables[i, : len(i_rnd_table)] = torch.tensor(i_rnd_table, dtype=torch.int32)
block_tables = block_tables.to(device="cuda")


positions = []
for i in range(B):
    q_len = int(cu_seqlens[i + 1] - cu_seqlens[i])
    i_sub_positions = int(kv_lens[i]) - q_len + torch.arange(q_len, dtype=torch.int32)
    positions.append(i_sub_positions)
positions = torch.cat(positions).to(device="cuda")
print(f"{positions=}\n")


##########
##### test
##########
def gen_right_bottom_mask(q_len, kv_len):
    """ """
    q_pos = torch.arange(q_len) - q_len + kv_len
    kv_pos = torch.arange(kv_len)
    mask = q_pos[:, None] >= kv_pos[None, :]
    print("[ MASK ]")
    print(mask)
    return mask.to(device="cuda")


paged_attn = femtovllm._C.PagedAttentionGemmCuda(
    q, k_pool, v_pool, cu_seqlens, q_len_max, block_tables, kv_lens, positions
)
print(paged_attn)
print(paged_attn.shape)
print()


ref_attn = []
for i in range(B):
    i_q = q[:, cu_seqlens[i] : cu_seqlens[i + 1], :]
    q_len = int(cu_seqlens[i + 1] - cu_seqlens[i])

    kv_len = int(kv_lens[i])
    # i_block_table = block_tables[i]
    # print(f"{i_block_table=}")
    i_block_table = [x for x in block_tables[i] if (x >= 0)]
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
