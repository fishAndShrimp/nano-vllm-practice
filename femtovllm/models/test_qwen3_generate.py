from pathlib import Path
from pprint import pp

from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config

from femtovllm.models.qwen3 import QwenForCausalLM, load_weights

# -----
TEXT = "The capital city of France is"
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
idx_out = model.generate(
    idx,
    max_new_tokens=100,
    temperature=0.0000001,
)
pp(tokenizer.decode(idx_out))
