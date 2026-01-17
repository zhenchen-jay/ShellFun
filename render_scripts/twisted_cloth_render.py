"""
Custom render script for pin flip cloth simulation using BlenderToolbox.

Uses utility functions from render_meshes_bt.py

Usage:
    python twisted_cloth_render.py -- -i /path/to/meshes -o /path/to/output
    python twisted_cloth_render.py -- -i /path/to/meshes -o /path/to/output --flat-shading
    python twisted_cloth_render.py -- -i /path/to/meshes -o /path/to/output --video --fps 30
"""

import bpy
import sys
import os
import math
import argparse
import shutil
import time
import stat
from pathlib import Path

import blendertoolbox as bt

# Import utility functions from render_meshes_bt
from render_meshes_bt import (
    load_config,
    get_mesh_bounds,
    find_crop_box,
    crop_images,
    natural_sort_key,
    convert_png_to_jpg,
    convert_pngs_to_jpgs,
    create_gif,
    create_video,
    get_color_for_method,
    load_mesh_with_fallback,
    split_components_by_face_count,
    save_largest_and_minz_of_smallest,
    setup_gpu_rendering,
)
from setMat_doubleColor_with_wireframe_modifier import setMat_doubleColor_with_wireframe_modifier
from setLight_sun_with_strength import setLight_sun_with_strength
from setup_world import setup_world, get_blender_hdri
from set_invisible_ground import set_invisible_ground
from setMat_metal_wrapper import setMat_metal_wrapper

def remove_readonly_handler(func, path, exc_info):
    """
    Windows-specific handler to remove read-only files/directories.
    Called by shutil.rmtree when it encounters permission errors.
    """
    try:
        # Change permissions to allow deletion
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
        # Retry the original operation that failed
        func(path)
    except Exception:
        # If we still can't delete it after fixing permissions, 
        # let it propagate to the outer exception handler
        raise

def safe_rmtree(path, max_retries=3, delay=0.1):
    """
    Safely remove a directory tree on Windows, handling locked files and permissions.
    
    Args:
        path: Path to directory to remove
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Delay between retries in seconds (default: 0.1)
    
    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    if not path.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            # Remove read-only attributes from all files and directories
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    try:
                        os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    except (OSError, PermissionError):
                        pass
                for f in files:
                    file_path = os.path.join(root, f)
                    try:
                        os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    except (OSError, PermissionError):
                        pass
            
            # Try to remove the directory
            shutil.rmtree(path, onerror=remove_readonly_handler)
            return True
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"  Warning: Could not remove temp folder after {max_retries} attempts: {path}")
                print(f"    Error: {e}")
                print(f"    You may need to manually delete: {path}")
                return False
    return False

def parse_pin_flip_cloth_arguments():
    """Parse command line arguments with video export option."""
    parser = argparse.ArgumentParser(description='Render pin flip cloth meshes using BlenderToolbox')
    parser.add_argument('-i', '--input-folder', type=str, required=True,
                        help='Folder containing PLY/OBJ files')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Folder to save rendered images')
    parser.add_argument('--samples', type=int, default=128,
                        help='Render samples (default: 128)')
    parser.add_argument('--resolution-x', type=int, default=2160,
                        help='Render width (default: 2160)')
    parser.add_argument('--resolution-y', type=int, default=2160,
                        help='Render height (default: 2160)')
    parser.add_argument('--exposure', type=float, default=1.5,
                        help='Exposure (default: 1.5)')
    parser.add_argument('--focal-length', type=float, default=45.0,
                        help='Camera focal length in mm (default: 45.0)')
    parser.add_argument('--flat-shading', action='store_true', default=False,
                        help='Use flat shading instead of smooth')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop images to content')
    
    # Video export options
    parser.add_argument('--video', action='store_true', default=True,
                        help='Export video from rendered images (default: enabled)')
    parser.add_argument('--no-video', action='store_true', default=False,
                        help='Disable video export')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frame rate (default: 30)')
    parser.add_argument('--video-name', type=str, default='animation.mp4',
                        help='Output video filename (default: animation.mp4)')
    
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args([])
    
    return args

########################################################
# camera settings
# locattion: (0, 0, 0.18)
# rotation: (0, 0, 0)

# Sun light: 
# location: (0, 0, 0.2)
# rotation: (0, 0, 0)

########################################################

def main():
    """Custom main function for pin flip cloth rendering."""
    
    # ========================================
    # Parse arguments
    # ========================================
    args = parse_pin_flip_cloth_arguments()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    if not input_folder.exists():
        print(f"ERROR: Input folder not found: {input_folder}")
        sys.exit(1)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Use args for most settings
    resolution_x = args.resolution_x
    resolution_y = args.resolution_y
    samples = args.samples
    exposure = args.exposure
    focal_length = args.focal_length
    flat_shading = args.flat_shading
    do_crop = args.crop
    
    # Video options (enabled by default, use --no-video to disable)
    export_video = args.video and not args.no_video
    video_fps = args.fps
    video_name = args.video_name
    
    # ========================================
    # Object color
    # ========================================
    # rgba
    # get the method name from the input folder name
    method_name = input_folder.stem
    obj_color = get_color_for_method(method_name)
    
    print(f"Method name: {method_name}, Object color: {obj_color}")

    
    # Mesh transform
    mesh_location = (0, 0, 0)
    mesh_rotation = (90, 0, 0)  # Y-up to Z-up
    mesh_scale = (1, 1, 1)
    
    # ========================================
    # Print settings
    # ========================================
    print("\n" + "=" * 50)
    print("Pin Flip Cloth Renderer")
    print("=" * 50)
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Resolution: {resolution_x}x{resolution_y}")
    print(f"Samples: {samples}, Exposure: {exposure}")
    print(f"Mesh color: {obj_color} (custom)")
    print(f"Shading: {'flat' if flat_shading else 'smooth'}")
    if export_video:
        print(f"Video export: {video_name} @ {video_fps} fps")
    
    # ========================================
    # Find mesh files
    # ========================================
    mesh_files = []
    for ext in ['.ply', '.obj'] or ext in ['.PLY', '.OBJ']:
        mesh_files.extend(input_folder.glob(f'*{ext}'))
    # Natural sort: frame_1, frame_2, ..., frame_10 (not frame_1, frame_10, frame_2)
    mesh_files = sorted(mesh_files, key=natural_sort_key)
    
    if not mesh_files:
        print(f"\nERROR: No mesh files found in {input_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(mesh_files)} meshes:")
    for f in mesh_files:
        print(f"  - {f.name}")

    
    # ========================================
    # Camera settings (fixed)
    # ========================================
    print(f"\n{'='*50}")
    print("Camera settings (fixed)")
    print(f"{'='*50}")
    
    # Fixed camera position and rotation
    camera_location = (0.76, -3.3, 0.65)
    camera_rotation = (90, 0, 0)  # Euler rotation in degrees
    
    print(f"  Camera location: {camera_location}")
    print(f"  Camera rotation: {camera_rotation}")
    print(f"  Focal length: {focal_length}mm")
    
    # ========================================
    # Light settings (fixed)
    # ========================================
    # Sun light with fixed rotation
    light_rotation = (98.6883, -16.9346, -1.85937)  # Euler rotation in degrees
    light_location = (0.812986, -3.43492, 3.65679)
    light_strength = 4.0
    shadow_softness = 0.3
    
    print(f"\n  Sun light rotation: {light_rotation}")
    print(f"  Sun light strength: {light_strength}")
    
    # Material color (custom, not from args)
    print(f"  Material color: {obj_color}")
    meshColor_top = bt.colorObj(obj_color, 0.5, 1.0, 1.0, 0.0, 0.0)
    meshColor_bottom = bt.colorObj((0.3, 0.3, 0.3, 1.0), 0.5, 1.0, 1.0, 0.0, 0.0)
    ao_strength = 0.5

    # ========================================
    # World settings (auto-detect Blender HDRI)
    # ========================================
    # world_path = get_blender_hdri("forest")  # Options: forest, city, courtyard, interior, night, studio, sunrise, sunset
    
    
    # ========================================
    # PASS 1: Render each mesh
    # ========================================
    print(f"\n{'='*50}")
    print("Pass 2: Rendering each mesh")
    print(f"{'='*50}")
    tmp_dir = output_folder / '_tmp_converted'
    rendered_paths = []
    
    for i, mesh_file in enumerate(mesh_files):
        print(f"\n  [{i+1}/{len(mesh_files)}] Rendering: {mesh_file.name}")
        
        # Fresh scene
        bt.blenderInit(resolution_x, resolution_y, samples, exposure)
        setup_gpu_rendering()
        
        # Load mesh (with fallback to PyMeshLab conversion for problematic PLY files)
        mesh = load_mesh_with_fallback(bt, mesh_file, mesh_location, mesh_rotation, mesh_scale, tmp_dir)
        
        # Shading
        if flat_shading:
            bpy.ops.object.shade_flat()
        else:
            bpy.ops.object.shade_smooth()
        
        # subdivide the mesh
        # bt.subdivision(mesh, level = 2)
        
        # Material
        setMat_doubleColor_with_wireframe_modifier(mesh, meshColor_top, meshColor_bottom, AOStrength=ao_strength, edgeThickness=0.0002)
        
        # Camera (fixed position and rotation using direct Blender API)
        bpy.ops.object.camera_add(location=camera_location)
        cam = bpy.context.object
        cam.rotation_euler = tuple(math.radians(a) for a in camera_rotation)
        cam.data.lens = focal_length
        bpy.context.scene.camera = cam
        
        # Lighting (fixed rotation) - using direct Blender API for Blender 4.x compatibility
        # set_invisible_ground(location=(0, -ground_z, 0), rotation_euler=(90, 0, 0))

        # Sun light
        sun_light = setLight_sun_with_strength(location=light_location, rotation_euler=light_rotation, strength=light_strength, shadow_soft_size=shadow_softness)

        # set world
        # setup_world(world_path=world_path, world_name="World", strength=1.0, make_film_transparent=True, use_existing_world=True, set_as_scene_world=True)
    

        bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))
        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')
        
        # Save blend for the selected mesh
        if i == len(mesh_files) - 1:
            blend_path = output_folder / "scene.blend"
            bpy.ops.wm.save_mainfile(filepath=str(blend_path))
            print(f"    Saved blend: {blend_path}")
            
        
        # Render
        suffix = "_flat" if flat_shading else ""
        output_path = output_folder / f"{mesh_file.stem}{suffix}.png"
        print(f"    Rendering to: {output_path}")
        bt.renderImage(str(output_path), cam)
        rendered_paths.append(output_path)
        
        print(f"    ✓ Complete")
    
    # ========================================
    # Crop images
    # ========================================
    cropped_paths = rendered_paths  # Default to original if not cropping
    
    if do_crop and rendered_paths:
        print(f"\n{'='*50}")
        print("Cropping images")
        print(f"{'='*50}")
        
        crop_box = find_crop_box(rendered_paths)
        if crop_box:
            cropped_paths = crop_images(rendered_paths, crop_box)
    
    # ========================================
    # Export videos (both uncropped and cropped)
    # ========================================
    if export_video and rendered_paths:
        print(f"\n{'='*50}")
        print(f"Exporting videos @ {video_fps} fps")
        print(f"{'='*50}")
        
        # Parse video name
        video_name_path = Path(video_name)
        video_stem = video_name_path.stem
        video_suffix = video_name_path.suffix or '.mp4'
        
        # --- Uncropped video ---
        print("\n  [Uncropped video]")
        uncropped_video_name = f"{video_stem}_uncropped{video_suffix}"
        uncropped_video_path = output_folder / uncropped_video_name
        
        # Convert original PNGs to JPGs
        print("  Converting PNGs to JPGs...")
        uncropped_jpg_paths = convert_pngs_to_jpgs(rendered_paths, output_folder, suffix="_uncropped")
        
        print(f"  Creating video...")
        uncropped_result = create_video(uncropped_jpg_paths, uncropped_video_path, video_fps)
        if uncropped_result:
            print(f"  ✓ Video saved: {uncropped_video_path}")
        else:
            print(f"  ✗ Failed to create uncropped video")
        
        # --- Cropped video ---
        if do_crop and cropped_paths and cropped_paths != rendered_paths:
            print("\n  [Cropped video]")
            cropped_video_path = output_folder / video_name
            
            # Convert cropped PNGs to JPGs
            print("  Converting cropped PNGs to JPGs...")
            cropped_jpg_paths = convert_pngs_to_jpgs(cropped_paths, output_folder)
            
            print(f"  Creating video...")
            cropped_result = create_video(cropped_jpg_paths, cropped_video_path, video_fps)
            if cropped_result:
                print(f"  ✓ Video saved: {cropped_video_path}")
            else:
                print(f"  ✗ Failed to create cropped video")
    
    # ========================================
    # Cleanup temporary files
    # ========================================
    if tmp_dir.exists():
        if safe_rmtree(tmp_dir):
            print(f"\n  Cleaned up temp folder: {tmp_dir}")
        # else: warning already printed by safe_rmtree
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 50)
    print("COMPLETE!")
    print("=" * 50)
    print(f"Rendered {len(rendered_paths)} meshes")
    print(f"Output: {output_folder}")
    
    print("\nRendered images:")
    for p in rendered_paths:
        print(f"  - {p.name}")
    
    if export_video:
        video_name_path = Path(video_name)
        if video_name_path.is_absolute():
            video_path = video_name_path
        else:
            video_path = output_folder / video_name
        if video_path.exists():
            print(f"\nVideo: {video_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
