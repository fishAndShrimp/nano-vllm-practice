from pathlib import Path

from huggingface_hub import snapshot_download

local_weights_dir = (
    Path(__file__).resolve().parent.parent.parent / "weights" / "qwen3_0.6b_weights"
)


r = snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=local_weights_dir,
)
print(r)
