from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config

from femtovllm.models import QwenForCausalLM

local_weights_dir = (
    Path(__file__).resolve().parent.parent.parent / "weights" / "qwen3_0.6b_weights"
)


## [STEP: init and load weights]
config = Qwen3Config.from_pretrained(local_weights_dir)
print(config)
model = QwenForCausalLM(config)
model.load_weights(local_weights_dir)
model.to("cuda")


## [STEP: disable dropout]
model.eval()


## [STEP: tokenize]
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(local_weights_dir)

text = "Yugioh"
idx = tokenizer(
    text,
    return_tensors="pt",
).input_ids.to("cuda")


## [STEP: gen]
temperature = 0.5


print(f"{text} <= input")
with torch.no_grad():
    for _ in range(30):
        out, all_kv_cache = model(idx)
        next_token_logits = out[:, -1, :]
        next_token_probs = F.softmax(
            next_token_logits / temperature,
            dim=-1,
        )

        # next_token_id = torch.argmax(
        #     next_token_logits,
        #     dim=-1,
        #     keepdim=True,
        # )
        next_token_id = torch.multinomial(
            next_token_probs,
            num_samples=1,
        )

        idx = torch.cat((idx, next_token_id), dim=-1)
        print(tokenizer.decode(idx[0]))
