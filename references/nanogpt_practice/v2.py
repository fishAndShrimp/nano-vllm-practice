import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# v2 -----
n_embd = 32
dropout = 0.1
# -----

torch.manual_seed(1337)


# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# unique characters
chars = sorted(set(text))
vocab_size = len(chars)


# mapping integers <=> characters
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(x):
    return [stoi[ch] for ch in x]


def decode(x):
    return "".join([itos[i] for i in x])


# train test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y


# estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


class SingleHead(nn.Module):
    """one head self-attention"""

    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head

        self.w_q = nn.Linear(n_embd, d_head, bias=False)
        self.w_k = nn.Linear(n_embd, d_head, bias=False)
        self.w_v = nn.Linear(n_embd, d_head, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size,
                )
            ),
        )

    def forward(self, x: torch.Tensor):
        """ """
        B, T, C = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )
        wei = F.softmax(scores, dim=-1)

        out = wei @ v

        return out


class MultiHead(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_heads, d_head):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                #####
                SingleHead(d_head)
                for _ in range(n_heads)
            ]
        )

        n_embd = n_heads * d_head
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1,
        )
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """ """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x):
        """ """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer decoder layer

    - communication => attention
    - computation => FFN
    """

    def __init__(self, n_embd, n_heads):
        super().__init__()

        assert n_embd % n_heads == 0
        d_head = n_embd // n_heads

        self.sa = MultiHead(n_heads, d_head)
        self.ffn = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd,
        )
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            Block(n_embd, n_heads=4),
            Block(n_embd, n_heads=4),
            Block(n_embd, n_heads=4),
            nn.LayerNorm(n_embd),
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are (B,T)
        B, T = idx.shape

        # tok_emb: (B,T,C)
        tok_emb = self.token_embedding_table(idx)

        # pos_emb: (T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device),
        )

        # x: (B,T,C)
        x = tok_emb + pos_emb

        # (B,T,C) => (B,T,C)
        x = self.blocks(x)

        # logits: (B,T,vocab_size)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx -> context in batch
        # (B,T)

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # logits -> scores (B,T,C)
            logits, _ = self(idx_cond)

            # pick last, become (B,C)
            logits = logits[:, -1, :]

            # softmax -> probabilities
            # (B,C)
            probs = F.softmax(logits, dim=-1)

            # sample
            # idx_next: (B,1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # cat (B,T) (B,1)
            # get (B,T+1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel().to(device)
# m = model.to(device)


# optim
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)


for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device,
)
print(
    decode(
        model.generate(
            context,
            max_new_tokens=500,
        )[0].tolist()
    )
)
