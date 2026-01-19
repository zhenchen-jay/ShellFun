# Sequence Rendering Scripts

Scripts for batch rendering sequences across multiple machines.

## Setup for a New Machine

### Option 1: Using Environment Variables (Recommended)

1. Copy the example config:
   ```bash
   cd running_scipts
   cp config.example.sh config.sh
   ```

2. Edit `config.sh` with your machine-specific paths:
   ```bash
   # Edit INPUT_ROOT and OUTPUT_ROOT for your machine
   nano config.sh
   ```

3. Run the script:
   ```bash
   source config.sh && ./sequence_rendering.sh
   ```

### Option 2: Using Command Line Arguments

Run the script with input and output paths directly:
```bash
./sequence_rendering.sh /path/to/input /path/to/output
```

With custom render script:
```bash
./sequence_rendering.sh /path/to/input /path/to/output crash_ball_render.py
```

### Option 3: Set Environment Variables Inline

```bash
INPUT_ROOT=/path/to/input OUTPUT_ROOT=/path/to/output ./sequence_rendering.sh
```

With custom render script:
```bash
INPUT_ROOT=/path/to/input OUTPUT_ROOT=/path/to/output RENDER_SCRIPT=my_render.py ./sequence_rendering.sh
```

## Features

- **Auto-detects conda**: Automatically finds your conda installation using `conda info --base`
- **Portable paths**: Works across Windows, macOS, and Linux
- **Flexible configuration**: Multiple ways to specify paths and render scripts
- **Configurable render script**: Easy to switch between different rendering scripts
- **Error checking**: Validates directories and scripts exist before processing
- **Progress output**: Shows which folders are being processed

## Configuration Priority

The script looks for configuration in this order (highest to lowest priority):

1. Command line arguments: `./sequence_rendering.sh <input> <output> [render_script]`
2. Environment variables: `INPUT_ROOT`, `OUTPUT_ROOT`, `RENDER_SCRIPT`
3. Default values in the script

### Render Script Paths

The `RENDER_SCRIPT` can be specified as:
- **Just the filename**: `"crash_ball_render.py"` (looks in `render_scripts/` folder)
- **Absolute path**: `"/full/path/to/my_render.py"`

Default: `crash_ball_render_top_view.py`

## Git Ignore

Add `config.sh` to `.gitignore` so each person can have their own machine-specific config:

```bash
# In your .gitignore
running_scipts/config.sh
```

## Examples

### Windows (Git Bash)
```bash
./sequence_rendering.sh "C:/Users/username/Dropbox/data/input" "C:/Users/username/Dropbox/data/output"
```

### macOS
```bash
./sequence_rendering.sh "$HOME/Dropbox/data/input" "$HOME/Dropbox/data/output"
```

### Linux
```bash
./sequence_rendering.sh "/data/input" "/data/output"
```

## Switching Between Render Scripts

### Quick Switch via Environment Variable

```bash
# Use crash_ball_render.py instead
RENDER_SCRIPT=crash_ball_render.py ./sequence_rendering.sh

# Use cloth_on_sphere_render.py
RENDER_SCRIPT=cloth_on_sphere_render.py ./sequence_rendering.sh

# Or set it in your config.sh
echo 'export RENDER_SCRIPT="cloth_on_sphere_render.py"' >> config.sh
```

### Available Render Scripts

Common render scripts in the `render_scripts/` folder:
- `crash_ball_render.py`
- `crash_ball_render_top_view.py` (default)
- `cloth_on_sphere_render.py`
- `cloth_on_sphere_render_single_frame.py`
- `twisted_cylinder_render.py`
- `twisted_cylinder_render_single_frame.py`
- `pin_flip_cloth_render.py`
- `crush_can_render.py`
- `crush_cone_render.py`

## Customizing for Different Tasks

To create a new rendering task:

1. Copy `sequence_rendering.sh` to a new file (e.g., `my_rendering.sh`)
2. Update the `folders` array with your specific folder names
3. Adjust the python arguments as needed

Example customization:
```bash
# In my_rendering.sh, change the folders array:
folders=(
  "MyFolder1"
  "MyFolder2"
  "MyFolder3"
)

# Change render arguments:
python "$render_script" -- -i "$input_path" -o "$output_path" --samples 500 --resolution-x 4320
```
