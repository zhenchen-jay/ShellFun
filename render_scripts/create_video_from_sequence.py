#!/usr/bin/env python3
"""
Create video from image sequence.

Usage:
    python create_video_from_sequence.py -i "frame_*_cropped.png" --start 0 --end 100 --fps 30
    python create_video_from_sequence.py -i "*_suffix.png" --fps 30 -o output.mp4
    python create_video_from_sequence.py -i "*.png" --fps 24
    python create_video_from_sequence.py -i "*.jpg" --exclude "*_cropped*" --fps 30
"""

import argparse
import sys
import subprocess
from pathlib import Path
import glob
import re
import fnmatch


def natural_sort_key(s):
    """Natural sort key for human-like sorting (1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def create_video_ffmpeg(image_paths, output_path, fps=30, quality="high"):
    """
    Create video using ffmpeg.
    
    Args:
        image_paths: List of image file paths
        output_path: Output video path
        fps: Frames per second
        quality: 'high', 'medium', or 'low'
    
    Returns:
        True if successful, False otherwise
    """
    if not image_paths:
        print("ERROR: No images provided")
        return False
    
    # Create a temporary file list for ffmpeg
    temp_list = output_path.parent / "ffmpeg_input_list.txt"
    
    try:
        # Write file list
        with open(temp_list, 'w') as f:
            for img_path in image_paths:
                # Use absolute path and escape single quotes
                abs_path = img_path.resolve()
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {1.0/fps}\n")
            # Add last frame again to ensure proper duration
            f.write(f"file '{image_paths[-1].resolve()}'\n")
        
        # Set quality parameters
        quality_settings = {
            'high': ['-crf', '18', '-preset', 'slow'],
            'medium': ['-crf', '23', '-preset', 'medium'],
            'low': ['-crf', '28', '-preset', 'fast']
        }
        
        crf_preset = quality_settings.get(quality, quality_settings['high'])
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(temp_list),
            '-vsync', 'vfr',
            '-pix_fmt', 'yuv420p',
            *crf_preset,
            '-y',  # Overwrite output file
            str(output_path)
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error creating video: {e}")
        return False
    finally:
        # Clean up temp file
        if temp_list.exists():
            temp_list.unlink()


def extract_suffix_from_pattern(pattern):
    """
    Extract suffix from pattern like 'frame_*_cropped.png' -> 'cropped'
    
    Args:
        pattern: Glob pattern string
        
    Returns:
        Suffix string or None
    """
    # Remove extension
    pattern_no_ext = Path(pattern).stem
    
    # Find the suffix after the last '*'
    if '*' in pattern_no_ext:
        parts = pattern_no_ext.split('*')
        suffix = parts[-1]
        # Remove leading underscore if present
        if suffix.startswith('_'):
            suffix = suffix[1:]
        return suffix if suffix else None
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Create video from image sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i "frame_*_cropped.png" --fps 30
  %(prog)s -i "frame_*.png" --start 10 --end 100 --fps 24 -o output.mp4
  %(prog)s -i "*.jpg" --fps 30 --quality high
  %(prog)s -i "*.jpg" --exclude "*_cropped*" --fps 30  # Exclude cropped versions
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input image pattern (e.g., "frame_*.png", "*_cropped.png")')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output video path (default: auto-generated from suffix)')
    parser.add_argument('--start', type=int, default=None,
                        help='Start frame index (optional)')
    parser.add_argument('--end', type=int, default=None,
                        help='End frame index (optional)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--quality', choices=['high', 'medium', 'low'], default='high',
                        help='Video quality (default: high)')
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing images (default: current directory)')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Exclude files matching this pattern (e.g., "*_cropped*" to exclude cropped files)')
    
    args = parser.parse_args()
    
    # Resolve directory
    input_dir = Path(args.dir).resolve()
    if not input_dir.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        sys.exit(1)
    
    # Find matching images
    pattern = args.input
    full_pattern = str(input_dir / pattern)
    
    print(f"\n{'='*60}")
    print("Video Creator from Image Sequence")
    print(f"{'='*60}")
    print(f"Pattern: {pattern}")
    print(f"Directory: {input_dir}")
    print(f"Searching: {full_pattern}")
    
    # Expand glob pattern
    matching_files = glob.glob(full_pattern)
    
    if not matching_files:
        print(f"\nERROR: No files matching pattern: {pattern}")
        sys.exit(1)
    
    # Convert to Path objects and sort naturally
    image_paths = sorted([Path(f) for f in matching_files], key=natural_sort_key)
    
    print(f"\nFound {len(image_paths)} images")
    
    # Apply exclude pattern if specified
    if args.exclude:
        excluded_count = 0
        filtered_paths = []
        for img_path in image_paths:
            if fnmatch.fnmatch(img_path.name, args.exclude):
                excluded_count += 1
            else:
                filtered_paths.append(img_path)
        
        image_paths = filtered_paths
        if excluded_count > 0:
            print(f"Excluded {excluded_count} files matching pattern: {args.exclude}")
            print(f"Remaining: {len(image_paths)} images")
    
    # Apply start/end frame filtering if specified
    if args.start is not None or args.end is not None:
        start_idx = args.start if args.start is not None else 0
        end_idx = args.end if args.end is not None else len(image_paths)
        
        # Clamp to valid range
        start_idx = max(0, min(start_idx, len(image_paths)))
        end_idx = max(start_idx, min(end_idx, len(image_paths)))
        
        image_paths = image_paths[start_idx:end_idx]
        print(f"Using frames {start_idx} to {end_idx-1} ({len(image_paths)} frames)")
    
    if not image_paths:
        print("ERROR: No images to process after filtering")
        sys.exit(1)
    
    # Show first and last few files
    print("\nFirst images:")
    for i, p in enumerate(image_paths[:3]):
        print(f"  {i}: {p.name}")
    if len(image_paths) > 6:
        print("  ...")
    if len(image_paths) > 3:
        print("\nLast images:")
        for i, p in enumerate(image_paths[-3:], start=len(image_paths)-3):
            print(f"  {i}: {p.name}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = input_dir / output_path
    else:
        # Auto-generate from suffix
        suffix = extract_suffix_from_pattern(pattern)
        if suffix:
            output_name = f"{suffix}.mp4"
        else:
            output_name = "output.mp4"
        output_path = input_dir / output_name
    
    print(f"\n{'='*60}")
    print("Creating video")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"FPS: {args.fps}")
    print(f"Quality: {args.quality}")
    print(f"Duration: {len(image_paths) / args.fps:.2f} seconds")
    
    # Create video
    success = create_video_ffmpeg(image_paths, output_path, fps=args.fps, quality=args.quality)
    
    if success:
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"Video created: {output_path}")
        
        # Show file size
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")
    else:
        print(f"\n{'='*60}")
        print("FAILED!")
        print(f"{'='*60}")
        print("Could not create video. Make sure ffmpeg is installed.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
