import shutil
import subprocess
import sys

OUTPUT_FILE = "codebase_context.xml"


def main():
    if not shutil.which("repomix"):
        sys.exit("❌ Error: 'repomix' is not installed.")

    # Repomix respects .gitignore by default.
    # However, we use a '.repomixignore' file to explicitly "un-ignore" the
    # .for_ai directory. This ensures local context is captured in the XML
    # without needing complex CLI flags or polluting the Git history.
    cmd = [
        "repomix",
        "--output",
        OUTPUT_FILE,
        "--style",
        "xml",
    ]

    print(f"🚀 Packing codebase into '{OUTPUT_FILE}'...")

    try:
        subprocess.run(cmd, check=True)
        print("✅ Done.")
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
