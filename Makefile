.PHONY: install fastinstall clean

# ------------------------------------------------------------------------
# Build Commands
# ------------------------------------------------------------------------

# Default installation: Compiles for all supported GPU architectures.
# Safe for distribution and environments without a local GPU (e.g., CI/CD).
install:
	pip install -v -e . --no-build-isolation

# Fast installation: Compiles ONLY for the local GPU architecture.
# Drastically reduces compile time for local development.
# Note: Requires the FAST_BUILD logic to be implemented in setup.py.
fastinstall:
	FAST_BUILD=1 pip install -v -e . --no-build-isolation

# ------------------------------------------------------------------------
# Cleanup Commands
# ------------------------------------------------------------------------

# Remove all compiled binaries, build artifacts, and clear pip cache.
# Run this if you encounter weird linking errors or want a fresh build.
clean:
	rm -rf build/ *.egg-info/ femtovllm/*.so
	pip cache purge
