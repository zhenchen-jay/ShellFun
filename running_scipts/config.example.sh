#!/usr/bin/env bash
# Configuration file for sequence_rendering.sh
# 
# Copy this file to config.sh and customize for your machine:
#   cp config.example.sh config.sh
#
# Then run:
#   source config.sh && ./sequence_rendering.sh

# ============================================
# Machine-specific configuration
# ============================================

# Conda base directory (optional - will auto-detect if not set)
# export CONDA_BASE="/path/to/conda"

# Input/output root directories
export INPUT_ROOT="C:/Users/YOUR_USERNAME/Dropbox/shell_test/Comparisons/CrashTest"
export OUTPUT_ROOT="C:/Users/YOUR_USERNAME/Dropbox/shell_test/Comparisons/CrashTest_Rendered_TopView"

# Render script (optional - defaults to crash_ball_render_top_view.py)
# Can be either:
#   - Just the filename (looks in render_scripts/): "my_render.py"
#   - Relative path from render_scripts/: "crash_ball_render.py"
#   - Absolute path: "/full/path/to/script.py"
export RENDER_SCRIPT="crash_ball_render_top_view.py"

# ============================================
# Examples for different machines/OS
# ============================================

# Windows (Git Bash / WSL with Windows paths):
# export INPUT_ROOT="C:/Users/username/Dropbox/shell_test/Comparisons/CrashTest"
# export OUTPUT_ROOT="C:/Users/username/Dropbox/shell_test/Comparisons/CrashTest_Rendered_TopView"

# macOS:
# export INPUT_ROOT="$HOME/Dropbox/shell_test/Comparisons/CrashTest"
# export OUTPUT_ROOT="$HOME/Dropbox/shell_test/Comparisons/CrashTest_Rendered_TopView"

# Linux:
# export INPUT_ROOT="/home/username/Dropbox/shell_test/Comparisons/CrashTest"
# export OUTPUT_ROOT="/home/username/Dropbox/shell_test/Comparisons/CrashTest_Rendered_TopView"
