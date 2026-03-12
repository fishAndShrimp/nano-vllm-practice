from pathlib import Path
from pprint import pp

from safetensors import safe_open
from transformers import Qwen3Config

from femtovllm.models.qwen3 import QwenForCausalLM, map_weight_key

local_weights_dir = (
    Path(__file__).resolve().parent.parent / "weights" / "qwen3_0.6b_weights"
)


config = Qwen3Config.from_pretrained(local_weights_dir)
# print(config)


my_model = QwenForCausalLM(config)
my_keys = set(my_model.state_dict().keys())


hf_keys = set()
for path in Path(local_weights_dir).iterdir():
    if path.suffix != ".safetensors":
        continue

    with safe_open(path, framework="pt", device="cpu") as f:
        hf_keys = hf_keys.union(f.keys())


missing_keys = my_keys - hf_keys
unexpected_keys = hf_keys - my_keys


print("\n\n :: MISSING KEYS :: ")
pp(missing_keys)
print("\n\n :: UNEXPECTED KEYS :: ")
pp(unexpected_keys)


mapped_keys = set([map_weight_key(x) for x in unexpected_keys])
print("\n\n :: TEST MAP WEIGHT KEY :: MISSING :: ")
pp(missing_keys - mapped_keys)
print("\n\n :: TEST MAP WEIGHT KEY :: UNEXPECTED :: ")
pp(mapped_keys - missing_keys)
