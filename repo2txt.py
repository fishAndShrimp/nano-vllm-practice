import shutil
import subprocess
import sys

OUTPUT_FILE = "codebase_context.xml"


def main():
    if not shutil.which("repomix"):
        sys.exit("❌ Error: 'repomix' is not installed.")

    # Repomix respects .gitignore by default.
    # We explicitly use --include to capture local environment details (e.g., nvcc, nvidia-smi)
    # that are excluded from version control but essential for LLM context.
    cmd = [
        "repomix",
        "--output",
        OUTPUT_FILE,
        "--style",
        "xml",
        "--include",
        ".for_ai/**/*",
    ]

    print(f"🚀 Packing codebase into '{OUTPUT_FILE}'...")

    try:
        subprocess.run(cmd, check=True)
        print("✅ Done.")
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
