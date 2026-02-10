import shutil
import subprocess
import sys

# ================= CONFIGURATION =================
# Base name for the output file (without extension)
BASE_NAME = "codebase_context"

# Active tool selection
CURRENT_TOOL = "repomix"  # Options: "repomix", "gitingest"

# Strategy definitions: Decouples tool logic from execution
STRATEGIES = {
    "repomix": {
        "cmd": ["repomix", "--output", "{out}", "--style", "xml"],
        "ext": ".xml",
    },
    "gitingest": {"cmd": ["gitingest", ".", "-o", "{out}"], "ext": ".txt"},
}
# =================================================


def main():
    # 1. Validate tool availability
    if not shutil.which(CURRENT_TOOL):
        sys.exit(f"❌ Error: '{CURRENT_TOOL}' is not installed or not in PATH.")

    # 2. Resolve strategy and filename
    config = STRATEGIES.get(CURRENT_TOOL)
    output_file = f"{BASE_NAME}{config['ext']}"

    # 3. Build command dynamically
    # Formats the command list by injecting the output filename
    cmd = [arg.format(out=output_file) for arg in config["cmd"]]

    print(f"🚀 Packing codebase into '{output_file}' using {CURRENT_TOOL}...")

    try:
        subprocess.run(cmd, check=True)
        print("✅ Done.")
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
