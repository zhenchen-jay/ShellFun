#!/usr/bin/env bash
set -euo pipefail

# Usage: ./sequence_rendering.sh [input_root] [output_root] [render_script]
# Or set environment variables: INPUT_ROOT, OUTPUT_ROOT, RENDER_SCRIPT

# ============================================
# Configuration
# ============================================

# Auto-detect conda base if not set
if [[ -z "${CONDA_BASE:-}" ]]; then
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
fi

# Activate conda env (portable)
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate blender
else
  echo "Warning: Could not find conda. Assuming 'blender' environment is already active or not needed."
fi

# Get input/output roots and render script from:
# 1. Command line arguments (highest priority)
# 2. Environment variables
# 3. Default values (lowest priority)

if [[ $# -ge 2 ]]; then
  input_root="$1"
  output_root="$2"
  render_script="${3:-}"
elif [[ -n "${INPUT_ROOT:-}" && -n "${OUTPUT_ROOT:-}" ]]; then
  input_root="$INPUT_ROOT"
  output_root="$OUTPUT_ROOT"
  render_script="${RENDER_SCRIPT:-}"
else
  # Default paths - update these for your machine
  input_root="${INPUT_ROOT:-C:/Users/csyzz/Dropbox/shell_test/Comparisons/CrashTest}"
  output_root="${OUTPUT_ROOT:-C:/Users/csyzz/Dropbox/shell_test/Comparisons/CrashTest_Rendered_TopView}"
  render_script="${RENDER_SCRIPT:-}"
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set default render script if not specified
if [[ -z "$render_script" ]]; then
  render_script="$SCRIPT_DIR/../render_scripts/crash_ball_render_top_view.py"
else
  # If render_script is not an absolute path, make it relative to SCRIPT_DIR/../render_scripts/
  if [[ "$render_script" != /* ]]; then
    render_script="$SCRIPT_DIR/../render_scripts/$render_script"
  fi
fi

# Validate input directory exists
if [[ ! -d "$input_root" ]]; then
  echo "ERROR: Input directory not found: $input_root"
  echo ""
  echo "Usage: $0 <input_root> <output_root> [render_script]"
  echo "   Or: INPUT_ROOT=/path/to/input OUTPUT_ROOT=/path/to/output RENDER_SCRIPT=script.py $0"
  exit 1
fi

# Validate render script exists
if [[ ! -f "$render_script" ]]; then
  echo "ERROR: Render script not found: $render_script"
  echo ""
  echo "Usage: $0 <input_root> <output_root> [render_script]"
  echo "   Or: INPUT_ROOT=/path/to/input OUTPUT_ROOT=/path/to/output RENDER_SCRIPT=script.py $0"
  exit 1
fi

echo "================================================"
echo "Sequence Rendering"
echo "================================================"
echo "Input root:    $input_root"
echo "Output root:   $output_root"
echo "Render script: $(basename "$render_script")"
echo "================================================"
echo ""

folders=(
  # "Directional_StVK_Bending_Tan_RestFlat"
  "Directional_StVK_Bending_Tan"
  "Vertex_Based_Quadratic_Bending_Tan"
  "Discrete_Hinge_Bending_Tan"
  #"Directional_StVK_Bending_Angle"
  #"Directional_StVK_Bending_Angle_RestFlat"
  #"Directional_StVK_Bending_Sin"
  #"Directional_StVK_Bending_Sin_RestFlat"
)

# ============================================
# Pass 1: Normal renders
# ============================================
echo "Pass 1: Normal renders"
echo "================================================"
echo ""

for name in "${folders[@]}"; do
  input_path="${input_root}/${name}/area_2e-5"
  output_path="${output_root}/${name}/area_2e-5"

  echo "Processing: $name"
  echo "  Input:  $input_path"
  echo "  Output: $output_path"
  
  if [[ ! -d "$input_path" ]]; then
    echo "  Warning: Input directory not found, skipping..."
    echo ""
    continue
  fi
  
  # Normal render
  python "$render_script" -- -i "$input_path" -o "$output_path" --samples 300
  echo "  Done!"
  echo ""
done

# ============================================
# Pass 2: Zoomed-out renders
# ============================================
echo "================================================"
echo "Pass 2: Zoomed-out renders (focal_length=25)"
echo "================================================"
echo ""

for name in "${folders[@]}"; do
  input_path="${input_root}/${name}/area_2e-5"
  output_path_zoomed="${output_root}/${name}/area_2e-5_zoomed_out"

  echo "Processing: $name (zoomed-out)"
  echo "  Input:  $input_path"
  echo "  Output: $output_path_zoomed"
  
  if [[ ! -d "$input_path" ]]; then
    echo "  Warning: Input directory not found, skipping..."
    echo ""
    continue
  fi
  
  # Zoomed-out render with focal_length = 25
  python "$render_script" -- -i "$input_path" -o "$output_path_zoomed" --samples 300 --focal-length 25
  echo "  Done!"
  echo ""
done

echo "================================================"
echo "All sequences complete!"
echo "================================================"