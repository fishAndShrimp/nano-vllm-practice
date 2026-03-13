from pathlib import Path
from pprint import pp

from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config

from femtovllm.models.qwen3 import QwenForCausalLM, load_weights

# -----
USER_QUESTION = """
What is yu-gi-oh card game?
"""
# -----
TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""
TEXT = TEMPLATE.format(USER_QUESTION)
MAX_NEW_TOKENS = 1000
# -----


local_weights_dir = (
    #####
    Path(__file__).parent.parent / "weights" / "qwen3_0.6b_weights"
)


config = Qwen3Config.from_pretrained(local_weights_dir)
pp(config)
model = QwenForCausalLM(config)
load_weights(model, local_weights_dir)
model.to("cuda")
model.eval()


tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(local_weights_dir)
idx = tokenizer(
    TEXT,
    return_tensors="pt",
).input_ids.to("cuda")

eos_token_ids = [
    tokenizer(x).input_ids[0]
    for x in {
        tokenizer.eos_token,
        "<|im_end|>",
        "<|endoftext|>",
    }
]

idx_out = model.generate(
    idx,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.1,
    eos_token_ids=eos_token_ids,
    pad_token_id=tokenizer.pad_token_id,
    presence_penalty=1.0,
)
print(tokenizer.decode(idx_out[0]))
