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
cu_seqlens = torch.tensor([0, 8, 32], dtype=torch.int32, device="cuda")
q_len_max = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
B = len(cu_seqlens) - 1
kv_lens = torch.tensor(
    [
        block_size * 4 - 5,
        block_size * 3 - 3,
    ],
    dtype=torch.int32,
    device="cuda",
)
q = torch.randn((n_heads, cu_seqlens[-1], d_head), device="cuda")
k_pool = torch.randn((num_blocks, n_kv_heads, block_size, d_head), device="cuda")
v_pool = torch.randn((num_blocks, n_kv_heads, block_size, d_head), device="cuda")


block_tables = torch.ones((B, 8), dtype=torch.int32, device="cuda")
for i, i_rnd_table in enumerate(
    [
        [0, 4, 5, 6],
        [1, 2, 3],
    ]
):
    block_tables[i, : len(i_rnd_table)] = torch.tensor(
        i_rnd_table, dtype=torch.int32, device="cuda"
    )


##########
##### test
##########
paged_attn = femtovllm._C.PagedAttentionGemmCuda(
    q, k_pool, v_pool, cu_seqlens, q_len_max, block_tables, kv_lens
)
print(paged_attn[-1])
print(paged_attn.shape)


list_ref_attn = []
for i in range(B):
    i_q = q[:, cu_seqlens[i] : cu_seqlens[i + 1], :]

    kv_len = int(kv_lens[i])
    # i_block_table = block_tables[i]
    # print(f"{i_block_table=}")
    i_block_table = [x for x in block_tables[i] if (x >= 0)]
    # print(f"{i_block_table=}")

    i_k = torch.cat([k_pool[x] for x in i_block_table], dim=-2)[:, :kv_len, :]
    i_v = torch.cat([v_pool[x] for x in i_block_table], dim=-2)[:, :kv_len, :]

    i_ref_attn = F.scaled_dot_product_attention(
        i_q,
        i_k,
        i_v,
        is_causal=False,
        enable_gqa=True,
    )
    list_ref_attn.append(i_ref_attn)
ref_attn = torch.cat(list_ref_attn, dim=-2)
print(ref_attn[-1])
print(ref_attn.shape)


print(f"{torch.allclose(paged_attn, ref_attn)=}")
diff_attn = paged_attn - ref_attn
# print(diff_attn)
print(f"{diff_attn.abs().max()=}")
