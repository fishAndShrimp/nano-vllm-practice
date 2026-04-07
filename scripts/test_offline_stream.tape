# =====================================================================
# ⚠️ EXECUTION CONTEXT: PROJECT ROOT
# This tape MUST be executed from the root directory of the repository.
# Usage: vhs < scripts/test_offline_stream.tape
#
# ℹ️ DEPENDENCY NOTE
# `vhs` is strictly for generating README demos and is NOT required to 
# run the project. Therefore, it is excluded from `environment.yml`.
# To install, visit: https://github.com/charmbracelet/vhs
# =====================================================================

# Output destination (relative to project root)
Output assets/test_offline_stream_v3.gif

# Terminal aesthetics: Optimized for Rich Columns layout
Set Margin 20
Set MarginFill "#282A36"
Set BorderRadius 10
Set WindowBar Colorful
Set Theme "Dracula"
Set FontSize 16
Set Width 1600
Set Height 800
Set Framerate 20 # Reduced framerate to optimize file size

# Execution flow: Initial pause for visual context
Sleep 1s

# Launch the generation script (path relative to root)
Type "python scripts/test_offline_stream.py"
Sleep 500ms
Enter

# Buffer time for model weight loading & initialization
# 🎬 Magic Cut: Completely hide the 8s loading phase to prevent UI flickering
Hide
Sleep 8s
Show

# Observation window for the Rich Live streaming output
Sleep 10s

# Graceful pause before the GIF loops
Sleep 2s
