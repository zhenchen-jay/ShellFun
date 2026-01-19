#!/usr/bin/env python3
"""
Crop multiple images to remove common white spaces.

Usage:
    python crop_images_tool.py -i image1.png image2.png image3.png
    python crop_images_tool.py -i *.png --padding 10 --suffix _crop
    python crop_images_tool.py -i *.png --overwrite
"""

import argparse
import sys
import glob
from pathlib import Path
import numpy as np
from PIL import Image


def find_crop_box(image_paths, padding=20):
    """
    Find common bounding box for all images by detecting content.
    
    Args:
        image_paths: List of image file paths
        padding: Pixels of padding around content (default: 20)
    
    Returns:
        Tuple (min_x, min_y, max_x, max_y) or None if no content found
    """
    if not image_paths:
        return None
    
    global_min_x, global_min_y = float('inf'), float('inf')
    global_max_x, global_max_y = 0, 0
    
    for img_path in image_paths:
        if not img_path.exists():
            print(f"  Warning: {img_path} not found, skipping...")
            continue
            
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Detect content based on image type
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA: use alpha channel
            content_mask = img_array[:, :, 3] > 200
        elif len(img_array.shape) == 3:
            # RGB: content is where sum is less than near-white
            content_mask = np.sum(img_array, axis=2) < 750
        else:
            # Grayscale: content is where value is less than near-white
            content_mask = img_array < 250
        
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            row_idx = np.where(rows)[0]
            col_idx = np.where(cols)[0]
            
            global_min_x = min(global_min_x, col_idx[0])
            global_min_y = min(global_min_y, row_idx[0])
            global_max_x = max(global_max_x, col_idx[-1] + 1)
            global_max_y = max(global_max_y, row_idx[-1] + 1)
    
    if global_min_x == float('inf'):
        return None
    
    return (
        max(0, int(global_min_x - padding)),
        max(0, int(global_min_y - padding)),
        int(global_max_x + padding),
        int(global_max_y + padding)
    )


def crop_images(image_paths, crop_box, suffix="_cropped", overwrite=False):
    """
    Crop all images to the same bounding box.
    
    Args:
        image_paths: List of image file paths
        crop_box: Tuple (min_x, min_y, max_x, max_y)
        suffix: Suffix to add to cropped filenames (default: "_cropped")
        overwrite: If True, overwrite original files (default: False)
    
    Returns:
        List of paths to cropped images
    """
    cropped_paths = []
    
    for img_path in image_paths:
        if not img_path.exists():
            continue
        
        img = Image.open(img_path)
        w, h = img.size
        
        # Ensure crop box is within image bounds
        box = (
            max(0, min(crop_box[0], w)),
            max(0, min(crop_box[1], h)),
            max(0, min(crop_box[2], w)),
            max(0, min(crop_box[3], h))
        )
        
        cropped = img.crop(box)
        
        if overwrite:
            cropped_path = img_path
        else:
            cropped_path = img_path.parent / f"{img_path.stem}{suffix}{img_path.suffix}"
        
        cropped.save(cropped_path)
        cropped_paths.append(cropped_path)
        
        print(f"  Cropped {img_path.name}: {w}x{h} -> {box[2]-box[0]}x{box[3]-box[1]}")
        if not overwrite:
            print(f"    Saved to: {cropped_path.name}")
    
    return cropped_paths


def main():
    """Main function for image cropping tool."""
    parser = argparse.ArgumentParser(
        description='Crop multiple images to remove common white spaces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i image1.png image2.png image3.png
  %(prog)s -i frame_*.png --padding 10
  %(prog)s -i *.png --overwrite
        """
    )
    
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Input image files')
    parser.add_argument('--padding', type=int, default=20,
                        help='Padding around content in pixels (default: 20)')
    parser.add_argument('--suffix', type=str, default='_cropped',
                        help='Suffix for output files (default: _cropped)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite original files instead of creating new ones')
    
    args = parser.parse_args()
    
    # Expand glob patterns and convert to Path objects
    image_paths = []
    for pattern in args.input:
        # Try to expand as glob pattern
        expanded = glob.glob(pattern)
        if expanded:
            # Glob found matches
            image_paths.extend([Path(p) for p in expanded])
        else:
            # No matches, treat as literal path
            image_paths.append(Path(pattern))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for p in image_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    image_paths = unique_paths
    
    # Filter out non-existent files
    existing_paths = [p for p in image_paths if p.exists()]
    
    if not existing_paths:
        print("ERROR: No valid image files found")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("Image Cropping Tool")
    print(f"{'='*50}")
    print(f"Found {len(existing_paths)} images:")
    for p in existing_paths:
        print(f"  - {p.name}")
    
    if len(existing_paths) != len(image_paths):
        print(f"\nWarning: {len(image_paths) - len(existing_paths)} files not found")
    
    # Find common crop box
    print(f"\n{'='*50}")
    print(f"Finding common bounding box (padding: {args.padding}px)")
    print(f"{'='*50}")
    
    crop_box = find_crop_box(existing_paths, padding=args.padding)
    
    if crop_box is None:
        print("ERROR: Could not find content in images")
        sys.exit(1)
    
    print(f"Crop box: ({crop_box[0]}, {crop_box[1]}) to ({crop_box[2]}, {crop_box[3]})")
    print(f"Size: {crop_box[2] - crop_box[0]} x {crop_box[3] - crop_box[1]}px")
    
    # Crop images
    print(f"\n{'='*50}")
    print("Cropping images")
    print(f"{'='*50}")
    
    cropped_paths = crop_images(existing_paths, crop_box, suffix=args.suffix, overwrite=args.overwrite)
    
    # Summary
    print(f"\n{'='*50}")
    print("COMPLETE!")
    print(f"{'='*50}")
    print(f"Cropped {len(cropped_paths)} images")
    if not args.overwrite:
        print(f"Output files saved with suffix: {args.suffix}")
    else:
        print(f"Original files overwritten")


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
