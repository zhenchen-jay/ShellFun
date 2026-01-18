#!/usr/bin/env bash
set -euo pipefail

# Activate conda env (portable)
CONDA_BASE="C:/ProgramData/miniconda3"
if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: Could not find conda.sh. Update CONDA_BASE in this script."
  echo "Try: conda info --base"
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate blender

input_root="C:/Users/csyzz/Dropbox/shell_test/Comparisons/CrashTest"
output_root="C:/Users/csyzz/Dropbox/shell_test/Comparisons/CrashTest_Rendered_TopView"

folders=(
  "Directional_StVK_Bending_Tan"
  "Discrete_Hinge_Bending_Tan"
  "Vertex_Based_Quadratic_Bending_Tan"
  "Directional_StVK_Bending_Angle"
  "Directional_StVK_Bending_Sin"
)

for name in "${folders[@]}"; do
  input_path="${input_root}/${name}"
  output_path="${output_root}/${name}"

  python ../render_scripts/crash_ball_render_top_view.py -- -i "$input_path" -o "$output_path" --samples 300
done