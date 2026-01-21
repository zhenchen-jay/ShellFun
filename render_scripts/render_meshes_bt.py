"""
Blender Script to Render All PLY/OBJ Files using BlenderToolbox

This script uses BlenderToolbox (https://github.com/HTDerekLiu/BlenderToolbox) for rendering.

Workflow:
1. Load all meshes first to compute combined bounding box
2. Compute camera/lighting settings based on combined bounds
3. For each mesh: init fresh scene with same settings, render

Features:
- Mesh rotation (90, 0, 0) to convert Y-up to Z-up coordinate system
- Two-sided material (different colors for front/back faces, great for cloth)
- Consistent camera positioning based on combined mesh bounds
- Shadow-casting sun light
- Invisible ground plane shadow catcher
- Automatic cropping to remove white space

Installation:
    conda create -n blender python=3.11
    source activate blender
    pip install bpy==4.3.0 --extra-index-url https://download.blender.org/pypi/
    pip install blendertoolbox

Usage:
    # Basic usage
    python render_meshes_bt.py -- -i /path/to/meshes -o /path/to/output
    
    # With flat shading
    python render_meshes_bt.py -- -i /path/to/meshes -o /path/to/output --flat-shading
    
    # With JSON config
    python render_meshes_bt.py -- -i /path/to/meshes -o /path/to/output --config config.json

Reference:
    Based on BlenderToolbox demo: https://github.com/HTDerekLiu/BlenderToolbox/blob/master/demos/demo_balloon.py
"""

import bpy
import sys
import re
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import pymeshlab

import blendertoolbox as bt
from setMat_doubleColorWire import setMat_doubleColorWire


# ============================================================
# Utility Functions (can be imported by other scripts)
# ============================================================

def natural_sort_key(path):
    """
    Natural sort key for file paths.
    Sorts frame_1, frame_2, ..., frame_10 correctly (not frame_1, frame_10, frame_2).
    
    Args:
        path: Path object or string
    
    Returns:
        List of parts for sorting
    """
    if isinstance(path, Path):
        name = path.stem
    else:
        name = str(path)
    # Extract numbers from the filename and convert to integers for proper sorting
    parts = re.split(r'(\d+)', name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]

# ============================================================
# Mesh conversion utilities (for problematic PLY files)
# ============================================================

def convert_ply_to_obj_pymeshlab(ply_path, obj_path=None):
    """
    Convert a PLY file to OBJ using PyMeshLab.
    
    This is useful for PLY files that Blender can't import directly
    (e.g., "Invalid face size" errors).
    
    Args:
        ply_path: Path to the input PLY file
        obj_path: Path for the output OBJ file. If None, uses same name with .obj extension
    
    Returns:
        Path to the OBJ file, or None if conversion failed
    """
    try:
        import pymeshlab
    except ImportError:
        print("  ERROR: pymeshlab not installed. Install with: pip install pymeshlab")
        return None
    
    ply_path = Path(ply_path)
    if obj_path is None:
        obj_path = ply_path.with_suffix('.obj')
    else:
        obj_path = Path(obj_path)
    
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(ply_path))
        ms.save_current_mesh(str(obj_path))
        return obj_path
    except Exception as e:
        print(f"  ERROR converting {ply_path.name} with pymeshlab: {e}")
        return None

def split_components_by_face_count(meshset: pymeshlab.MeshSet):
    """
    Returns a list of (face_count, mesh_id, mesh_name) for each component mesh in the MeshSet.
    Assumes the MeshSet currently contains exactly one mesh to be split.
    """
    if meshset.mesh_number() == 0:
        return []
    
    # Split into connected components; pymeshlab will create new meshes in the MeshSet
    # Use delete_source_mesh=False to avoid empty meshset issues
    meshset.generate_splitting_by_connected_components(delete_source_mesh=False)

    # After splitting, the original mesh is at index 0, new components follow
    # If only 1 component, there will be 2 meshes (original + 1 component)
    # We want to skip the original (index 0) and use the components
    num_meshes = meshset.mesh_number()
    
    if num_meshes <= 1:
        # No split happened or something went wrong
        print("  Warning: No components created after split.")
        return []

    info = []
    # Skip mesh 0 (original), iterate through component meshes (1 to num_meshes-1)
    for i in range(1, num_meshes):
        try:
            meshset.set_current_mesh(i)
            m = meshset.current_mesh()
            face_count = m.face_number()
            name = m.mesh_name() if hasattr(m, "mesh_name") else f"component_{i}"
            info.append((face_count, i, name))
        except Exception as e:
            print(f"  Warning: Could not access mesh {i}: {e}")
            continue

    # Sort by face count
    info.sort(key=lambda x: x[0])
    return info


def split_bellows_cone_sphere(input_path: str,
                              output_bellows_path: str,
                              output_cone_path: str,
                              output_sphere_path: str):
    """
    Split mesh into 3 connected components: bellows, cone, and sphere.
    
    Identification strategy:
    - Bellows: Largest vertex count
    - Cone vs Sphere: Distinguished by bounding box size (cone has larger bbox)
    
    Args:
        input_path: Path to input mesh file
        output_bellows_path: Path to save bellows mesh
        output_cone_path: Path to save cone mesh
        output_sphere_path: Path to save sphere mesh
    
    Returns:
        Dict with keys 'bellows', 'cone', 'sphere' containing info about each component,
        or None if splitting failed
    """
    import numpy as np
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    
    # Split into connected components
    ms.generate_splitting_by_connected_components(delete_source_mesh=False)
    
    num_meshes = ms.mesh_number()
    
    if num_meshes < 4:  # Original + 3 components
        print(f"  Warning: Expected 3 components, but got {num_meshes - 1}")
        return None
    
    # Collect info about each component (skip original at index 0)
    components = []
    for i in range(1, num_meshes):
        try:
            ms.set_current_mesh(i)
            m = ms.current_mesh()
            
            # Get vertex count
            vertex_count = m.vertex_number()
            
            # Get bounding box size
            bbox_min = m.bounding_box().min()
            bbox_max = m.bounding_box().max()
            bbox_size = np.linalg.norm(bbox_max - bbox_min)  # Diagonal length
            
            components.append({
                'index': i,
                'vertex_count': vertex_count,
                'bbox_size': bbox_size,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max
            })
        except Exception as e:
            print(f"  Warning: Could not access mesh {i}: {e}")
            continue
    
    if len(components) < 3:
        print(f"  Error: Could not identify 3 components")
        return None
    
    # Sort by vertex count to identify bellows (largest)
    components_by_vertices = sorted(components, key=lambda x: x['vertex_count'], reverse=True)
    bellows = components_by_vertices[0]
    
    # Remaining two components: cone and sphere
    remaining = [c for c in components if c['index'] != bellows['index']]
    
    # Sort by bbox size to distinguish cone (larger) from sphere (smaller)
    remaining_by_bbox = sorted(remaining, key=lambda x: x['bbox_size'], reverse=True)
    cone = remaining_by_bbox[0]
    sphere = remaining_by_bbox[1]
    
    # Save each component
    ms.set_current_mesh(bellows['index'])
    ms.save_current_mesh(output_bellows_path)
    
    ms.set_current_mesh(cone['index'])
    ms.save_current_mesh(output_cone_path)
    
    ms.set_current_mesh(sphere['index'])
    ms.save_current_mesh(output_sphere_path)
    
    result = {
        'bellows': {
            'path': output_bellows_path,
            'vertices': bellows['vertex_count'],
            'bbox_size': bellows['bbox_size']
        },
        'cone': {
            'path': output_cone_path,
            'vertices': cone['vertex_count'],
            'bbox_size': cone['bbox_size']
        },
        'sphere': {
            'path': output_sphere_path,
            'vertices': sphere['vertex_count'],
            'bbox_size': sphere['bbox_size']
        }
    }
    
    return result


def save_largest_and_minz_of_smallest(input_path: str,
                                     output_largest_path: str,
                                     output_smallest_path: str | None = None):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)

    # Split into connected components (creates multiple meshes in MeshSet)
    comps = split_components_by_face_count(ms)

    if len(comps) == 0:
        # Single component or empty mesh - reload and save as-is
        print(f"  Single component mesh, saving directly: {input_path}")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path)
        ms.set_current_mesh(0)
        m = ms.current_mesh()
        v = m.vertex_matrix()
        min_z = float(v[:, 2].min()) if v.shape[0] > 0 else float("nan")
        face_count = m.face_number()
        ms.save_current_mesh(output_largest_path)
        # For single component, smallest = largest
        if output_smallest_path is not None:
            ms.save_current_mesh(output_smallest_path)
        return {
            "num_components": 1,
            "largest_faces": face_count,
            "smallest_faces": face_count,
            "min_z_smallest": min_z,
        }

    smallest_faces, smallest_id, _ = comps[0]
    largest_faces, largest_id, _ = comps[-1]

    # Compute min Z of smallest component (over vertices)
    ms.set_current_mesh(smallest_id)
    v = ms.current_mesh().vertex_matrix()  # Nx3 numpy array
    min_z_smallest = float(v[:, 2].min()) if v.shape[0] > 0 else float("nan")

    # Save largest component
    ms.set_current_mesh(largest_id)
    ms.save_current_mesh(output_largest_path)

    # Optionally save smallest component too
    if output_smallest_path is not None:
        ms.set_current_mesh(smallest_id)
        ms.save_current_mesh(output_smallest_path)

    return {
        "num_components": len(comps),
        "largest_faces": int(largest_faces),
        "smallest_faces": int(smallest_faces),
        "min_z_smallest": min_z_smallest,
    }


def setup_gpu_rendering():
    """
    Configure Blender to use GPU rendering if available.
    
    Automatically detects available GPU compute types (CUDA, OPTIX, METAL, HIP, OPENCL)
    and enables all available devices.
    
    Should be called after bt.blenderInit() to ensure Cycles is active.
    """
    import bpy
    
    scene = bpy.context.scene
    
    try:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        
        # Try to detect available compute device types
        # Common types: OPTIX (NVIDIA RTX), CUDA (NVIDIA), METAL (macOS), HIP (AMD), OPENCL
        available_types = []
        for compute_type in ['OPTIX', 'CUDA', 'METAL', 'HIP', 'OPENCL']:
            try:
                prefs.compute_device_type = compute_type
                prefs.get_devices()
                if prefs.devices:
                    available_types.append(compute_type)
                    print(f"  GPU acceleration: {compute_type}")
                    break
            except (TypeError, AttributeError):
                continue
        
        if not available_types:
            print("  No GPU acceleration available, using CPU")
            scene.cycles.device = 'CPU'
        else:
            # Enable all available devices
            for device in prefs.devices:
                device.use = True
                print(f"    Enabled device: {device.name}")
    except Exception as e:
        print(f"  Warning: Could not configure GPU acceleration: {e}")
        print("  Falling back to CPU rendering")
        scene.cycles.device = 'CPU'


def load_mesh_with_fallback(bt, mesh_file, location, rotation, scale, tmp_dir=None):
    """
    Load a mesh file with Blender, falling back to PyMeshLab conversion if needed.
    
    Args:
        bt: BlenderToolbox module
        mesh_file: Path to the mesh file (PLY or OBJ)
        location: Mesh location tuple
        rotation: Mesh rotation tuple (degrees)
        scale: Mesh scale tuple
        tmp_dir: Directory for temporary converted files. If None, uses mesh_file's directory
    
    Returns:
        Loaded mesh object, or None if loading failed
    """
    import bpy
    
    mesh_file = Path(mesh_file)
    
    # First, try loading directly
    try:
        mesh = bt.readMesh(str(mesh_file), location, rotation, scale)
        return mesh
    except Exception as e:
        pass
    
    # Check if Blender reported an error (bt.readMesh might not raise exception)
    # We'll check if the mesh was actually loaded
    if bpy.context.object is None or bpy.context.object.type != 'MESH':
        # Loading failed, try PyMeshLab conversion for PLY files
        if mesh_file.suffix.lower() == '.ply':
            print(f"    Blender failed to load {mesh_file.name}, trying PyMeshLab conversion...")
            
            # Determine temp directory
            if tmp_dir is None:
                tmp_dir = mesh_file.parent / '_tmp_converted'
            tmp_dir = Path(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to OBJ
            obj_path = tmp_dir / f"{mesh_file.stem}.obj"
            converted = convert_ply_to_obj_pymeshlab(mesh_file, obj_path)
            
            if converted:
                try:
                    mesh = bt.readMesh(str(converted), location, rotation, scale)
                    print(f"    ✓ Loaded via PyMeshLab conversion")
                    return mesh
                except Exception as e:
                    print(f"    ERROR loading converted OBJ: {e}")
                    return None
            else:
                return None
    
    return bpy.context.object if bpy.context.object and bpy.context.object.type == 'MESH' else None

# ============================================================
# color maps from method prefix to rgba
# Keys are matched as prefixes, so "Directional_StVK_Bending_Tan" matches 
# "Directional_StVK_Bending_Tan_RestFlat", "Directional_StVK_Bending_Tan_v2", etc.
color_maps = {
    "Directional_StVK_Bending_Angle": (0.226, 0.358, 0.67, 1.0),
    "Directional_StVK_Bending_Sin": (0.6, 0.6, 0.4, 1.0),
    "Directional_StVK_Bending_Tan": (0.2, 0.4, 0.4, 1.0),
    "Discrete_Hinge_Bending_Tan": (0.3, 0.178, 0.5, 1.0),
    "StVK_Bending_Tan": (0.3, 0.3, 0.3, 1.0),
    "StVK_Bending_Sin": (0.2, 0.2, 0.2, 1.0),
    "Vertex_Based_Quadratic_Bending_Tan": (0.7, 0.1, 0, 1.0),
}

def get_color_for_method(method_name, default_color=(0.2, 0.4, 0.4, 1.0)):
    """
    Get color for a method name, supporting prefix matching.
    
    First tries exact match, then tries prefix matching (longest prefix wins).
    
    Args:
        method_name: The folder/method name to look up
        default_color: Color to use if no match found
    
    Returns:
        RGBA tuple for the color
    """
    # Exact match first
    if method_name in color_maps:
        return color_maps[method_name]
    
    # Prefix match - find longest matching prefix
    best_match = None
    best_len = 0
    for prefix in color_maps:
        if method_name.startswith(prefix) and len(prefix) > best_len:
            best_match = prefix
            best_len = len(prefix)
    
    if best_match:
        return color_maps[best_match]
    
    return default_color

def convert_png_to_jpg(png_file, jpg_file):
    """
    Converts a PNG image to JPG with a white background.
    
    Args:
        png_file: Path to the input PNG file
        jpg_file: Path to save the output JPG file
    
    Returns:
        Path to the saved JPG file
    """
    with Image.open(png_file) as img:
        # Create a white background image
        bg = Image.new("RGB", img.size, (255, 255, 255))
        # Paste the image onto the white background (use alpha as mask if available)
        if img.mode == 'RGBA':
            bg.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        else:
            bg.paste(img)
        # Save as JPG
        bg.save(jpg_file, "JPEG", quality=95)
    return Path(jpg_file)


def convert_pngs_to_jpgs(png_paths, output_folder=None, suffix=""):
    """
    Convert a list of PNG files to JPG with white background.
    
    Args:
        png_paths: List of paths to PNG files
        output_folder: Folder to save JPGs (default: same as PNG)
        suffix: Optional suffix to add to output filenames (e.g., "_uncropped")
    
    Returns:
        List of paths to JPG files
    """
    jpg_paths = []
    for png_path in png_paths:
        png_path = Path(png_path)
        jpg_name = f"{png_path.stem}{suffix}.jpg"
        if output_folder:
            jpg_path = Path(output_folder) / jpg_name
        else:
            jpg_path = png_path.parent / jpg_name
        convert_png_to_jpg(png_path, jpg_path)
        jpg_paths.append(jpg_path)
        print(f"  Converted: {png_path.name} -> {jpg_path.name}")
    return jpg_paths


def create_gif(image_files, output_file, duration=100, loop=0):
    """
    Creates a GIF from a sequence of image files.
    
    Args:
        image_files: List of file paths to the images
        output_file: Path to save the output GIF
        duration: Duration for each frame in milliseconds (default: 100ms = 10fps)
        loop: Number of loops (0 = infinite)
    
    Returns:
        Path to the saved GIF
    """
    if not image_files:
        print("  ERROR: No images to create GIF from")
        return None
    
    images = [Image.open(img_file) for img_file in image_files]
    
    # Convert to RGB if necessary (GIF doesn't support RGBA well)
    rgb_images = []
    for img in images:
        if img.mode == 'RGBA':
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            rgb_images.append(bg)
        else:
            rgb_images.append(img.convert('RGB'))
    
    rgb_images[0].save(
        output_file,
        save_all=True,
        append_images=rgb_images[1:],
        duration=duration,
        loop=loop
    )
    
    print(f"  ✓ GIF saved: {output_file}")
    return Path(output_file)


def create_video(image_files, output_video, fps=30):
    """
    Creates a video from a sequence of images using moviepy.
    
    Args:
        image_files: List of file paths to the images
        output_video: Path to save the output video
        fps: Frames per second for the video (default: 30)
    
    Returns:
        Path to the saved video, or None if failed
    
    Requirements:
        pip install moviepy
    """
    if not image_files:
        print("  ERROR: No images to create video from")
        return None
    
    try:
        # moviepy 2.x
        from moviepy import ImageSequenceClip
    except ImportError:
        try:
            # moviepy 1.x
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            print("  ERROR: moviepy not installed. Install with: pip install moviepy")
            return None
    
    # Convert paths to strings
    image_files_str = [str(f) for f in image_files]
    
    try:
        clip = ImageSequenceClip(image_files_str, fps=fps)
        # Use yuv420p pixel format for Windows compatibility (default yuv444p is not widely supported)
        # Also pad to even dimensions (required for yuv420p)
        clip.write_videofile(
            str(output_video), 
            codec='libx264',
            ffmpeg_params=[
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt", "yuv420p"
            ],
            logger=None
        )
        print(f"  ✓ Video saved: {output_video}")
        return Path(output_video)
    except Exception as e:
        print(f"  ERROR creating video: {e}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Render meshes using BlenderToolbox')
    parser.add_argument('-i', '--input-folder', type=str, required=True,
                        help='Folder containing PLY/OBJ files')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Folder to save rendered images')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file')
    parser.add_argument('--samples', type=int, default=128,
                        help='Render samples (default: 128)')
    parser.add_argument('--resolution-x', type=int, default=1080,
                        help='Render width (default: 1080)')
    parser.add_argument('--resolution-y', type=int, default=1080,
                        help='Render height (default: 1080)')
    parser.add_argument('--exposure', type=float, default=1.5,
                        help='Exposure (default: 1.5)')
    parser.add_argument('--focal-length', type=float, default=45.0,
                        help='Camera focal length in mm (default: 45.0)')
    parser.add_argument('--front-color', type=str, default='0.67,0.4,0.95',
                        help='Front face color R,G,B (default: 0.67,0.4,0.95)')
    parser.add_argument('--back-color', type=str, default='0.6,0.6,0.7',
                        help='Back face color R,G,B (default: 0.6,0.6,0.7)')
    parser.add_argument('--flat-shading', action='store_true', default=False,
                        help='Use flat shading instead of smooth')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop images to content')
    
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args([])
    
    return args


def load_config(config_path):
    """Load settings from JSON config file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR loading config: {e}")
        return None


def get_mesh_bounds(mesh_obj):
    """Calculate mesh bounding box."""
    import mathutils
    corners = [mesh_obj.matrix_world @ mathutils.Vector(c) for c in mesh_obj.bound_box]
    
    min_c = [min(c[i] for c in corners) for i in range(3)]
    max_c = [max(c[i] for c in corners) for i in range(3)]
    
    center = tuple((min_c[i] + max_c[i]) / 2 for i in range(3))
    size = tuple(max_c[i] - min_c[i] for i in range(3))
    
    return center, size


def find_crop_box(image_paths, padding=20):
    """Find common bounding box for all images."""
    if not image_paths:
        return None
    
    global_min_x, global_min_y = float('inf'), float('inf')
    global_max_x, global_max_y = 0, 0
    
    for img_path in image_paths:
        if not img_path.exists():
            continue
            
        img = Image.open(img_path)
        img_array = np.array(img)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            content_mask = img_array[:, :, 3] > 200
        elif len(img_array.shape) == 3:
            content_mask = np.sum(img_array, axis=2) < 750
        else:
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


def crop_images(image_paths, crop_box):
    """Crop all images to the same bounding box."""
    cropped_paths = []
    
    for img_path in image_paths:
        if not img_path.exists():
            continue
        
        img = Image.open(img_path)
        w, h = img.size
        
        box = (
            max(0, min(crop_box[0], w)),
            max(0, min(crop_box[1], h)),
            max(0, min(crop_box[2], w)),
            max(0, min(crop_box[3], h))
        )
        
        cropped = img.crop(box)
        cropped_path = img_path.parent / f"{img_path.stem}_cropped{img_path.suffix}"
        cropped.save(cropped_path)
        cropped_paths.append(cropped_path)
        
        print(f"  Cropped {img_path.name}: {w}x{h} -> {box[2]-box[0]}x{box[3]-box[1]}")
    
    return cropped_paths


def main():
    args = parse_arguments()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    if not input_folder.exists():
        print(f"ERROR: Input folder not found: {input_folder}")
        sys.exit(1)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load config or use args
    config = load_config(args.config) if args.config else None
    
    if config:
        settings = {
            'resolution_x': config.get('render', {}).get('resolution_x', args.resolution_x),
            'resolution_y': config.get('render', {}).get('resolution_y', args.resolution_y),
            'samples': config.get('render', {}).get('samples', args.samples),
            'exposure': config.get('render', {}).get('exposure', args.exposure),
            'focal_length': config.get('camera', {}).get('focal_length', args.focal_length),
            'camera_location': config.get('camera', {}).get('location'),
            'look_at': config.get('camera', {}).get('look_at'),
            'front_color': tuple(config.get('material', {}).get('front_color', [0.67, 0.4, 0.95])),
            'back_color': tuple(config.get('material', {}).get('back_color', [0.6, 0.6, 0.7])),
            'ao_strength': config.get('material', {}).get('ao_strength', 0.5),
            'flat_shading': config.get('options', {}).get('flat_shading', args.flat_shading),
            'do_crop': config.get('options', {}).get('crop_images', args.crop),
        }
    else:
        try:
            front_color = tuple(float(x) for x in args.front_color.split(','))
        except:
            front_color = (0.67, 0.4, 0.95)
        
        try:
            back_color = tuple(float(x) for x in args.back_color.split(','))
        except:
            back_color = (0.6, 0.6, 0.7)
        
        settings = {
            'resolution_x': args.resolution_x,
            'resolution_y': args.resolution_y,
            'samples': args.samples,
            'exposure': args.exposure,
            'focal_length': args.focal_length,
            'camera_location': None,
            'look_at': None,
            'front_color': front_color,
            'back_color': back_color,
            'ao_strength': 0.5,
            'flat_shading': args.flat_shading,
            'do_crop': args.crop,
        }
    
    # Print settings
    print("\n" + "=" * 50)
    print("BlenderToolbox Mesh Renderer")
    print("=" * 50)
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Resolution: {settings['resolution_x']}x{settings['resolution_y']}")
    print(f"Samples: {settings['samples']}, Exposure: {settings['exposure']}")
    print(f"Front color: {settings['front_color']}")
    print(f"Back color: {settings['back_color']}")
    print(f"Shading: {'flat' if settings['flat_shading'] else 'smooth'}")
    
    # Find mesh files
    mesh_files = []
    for ext in ['.ply', '.obj', '.PLY', '.OBJ']:
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
    # PASS 1: Load all meshes to compute combined bounding box
    # ========================================
    print(f"\n{'='*50}")
    print("Pass 1: Computing combined bounding box")
    print(f"{'='*50}")
    
    # Initialize Blender for bounds computation
    bt.blenderInit(
        settings['resolution_x'],
        settings['resolution_y'],
        settings['samples'],
        settings['exposure']
    )
    
    mesh_location = (0, 0, 0)
    mesh_rotation = (90, 0, 0)  # Y-up to Z-up
    mesh_scale = (1, 1, 1)
    
    # Track global bounds across all meshes
    global_min = [float('inf')] * 3
    global_max = [float('-inf')] * 3
    
    for i, mesh_file in enumerate(mesh_files):
        print(f"  [{i+1}/{len(mesh_files)}] Loading: {mesh_file.name}")
        mesh = bt.readMesh(str(mesh_file), mesh_location, mesh_rotation, mesh_scale)
        
        # Get this mesh's bounds
        bbox_center, bbox_size = get_mesh_bounds(mesh)
        
        # Update global bounds
        for j in range(3):
            mesh_min = bbox_center[j] - bbox_size[j] / 2
            mesh_max = bbox_center[j] + bbox_size[j] / 2
            global_min[j] = min(global_min[j], mesh_min)
            global_max[j] = max(global_max[j], mesh_max)
        
        print(f"    Bounds: center={tuple(f'{x:.3f}' for x in bbox_center)}, size={tuple(f'{x:.3f}' for x in bbox_size)}")
    
    # Compute combined bounds
    combined_center = tuple((global_min[i] + global_max[i]) / 2 for i in range(3))
    combined_size = tuple(global_max[i] - global_min[i] for i in range(3))
    max_dim = max(combined_size)
    
    # Ground position at the bottom of all meshes
    ground_z = global_min[2]
    
    print(f"\n  Combined bounding box:")
    print(f"    Center: {tuple(f'{x:.3f}' for x in combined_center)}")
    print(f"    Size: {tuple(f'{x:.3f}' for x in combined_size)}")
    print(f"    Max dimension: {max_dim:.3f}")
    print(f"    Ground Z position: {ground_z:.3f}")
    
    # ========================================
    # Compute camera settings based on combined bounds
    # ========================================
    print(f"\n{'='*50}")
    print("Computing camera settings")
    print(f"{'='*50}")
    
    if settings.get('camera_location'):
        cam_location = tuple(settings['camera_location'])
        look_at = tuple(settings.get('look_at', combined_center))
        print("  Using fixed camera from config")
    else:
        # Auto camera position (corner view)
        distance = max_dim * 1.8
        cam_location = (
            combined_center[0] + distance,
            combined_center[1] + distance,
            combined_center[2] + distance * 2
        )
        look_at = combined_center
        print("  Auto-computing camera from combined bounds")
    
    print(f"  Camera location: {tuple(f'{x:.3f}' for x in cam_location)}")
    print(f"  Look at: {tuple(f'{x:.3f}' for x in look_at)}")
    print(f"  Focal length: {settings['focal_length']}mm")
    
    # Light settings
    light_angle = (6, -30, -155)
    light_strength = 2.0
    shadow_softness = 0.3
    ambient_color = (0.1, 0.1, 0.1, 1)
    
    print(f"\n  Sun light: angle={light_angle}, strength={light_strength}")
    print(f"  Ambient light: {ambient_color[:3]}")
    
    # Material settings
    front_rgba = (*settings['front_color'], 1.0)
    back_rgba = (*settings['back_color'], 1.0)
    meshColor_top = bt.colorObj(front_rgba, 0.5, 1.0, 1.0, 0.0, 0.0)
    meshColor_bottom = bt.colorObj(back_rgba, 0.5, 1.0, 1.0, 0.0, 0.0)
    ao_strength = settings.get('ao_strength', 0.5)
    
    print(f"\n  Material front: {settings['front_color']}")
    print(f"  Material back: {settings['back_color']}")
    print(f"  AO strength: {ao_strength}")
    
    # ========================================
    # PASS 2: Render each mesh with same settings
    # ========================================
    print(f"\n{'='*50}")
    print("Pass 2: Rendering each mesh")
    print(f"{'='*50}")
    
    rendered_paths = []
    
    for i, mesh_file in enumerate(mesh_files):
        print(f"\n  [{i+1}/{len(mesh_files)}] Rendering: {mesh_file.name}")
        
        # Initialize fresh scene
        bt.blenderInit(
            settings['resolution_x'],
            settings['resolution_y'],
            settings['samples'],
            settings['exposure']
        )
        
        # Load mesh
        mesh = bt.readMesh(str(mesh_file), mesh_location, mesh_rotation, mesh_scale)
        
        # Apply shading
        if settings['flat_shading']:
            bpy.ops.object.shade_flat()
        else:
            bpy.ops.object.shade_smooth()
        
        # Apply material
        setMat_doubleColorWire(mesh, meshColor_top, meshColor_bottom, ao_strength)
        
        # Set camera (same settings for all)
        cam = bt.setCamera(cam_location, look_at, settings['focal_length'])
        
        # Set lighting (same for all)
        bt.invisibleGround(shadowBrightness=0.9, location=(0, 0, ground_z))
        
        # Move ground plane to correct Z position (bottom of mesh bounds)
        ground = bpy.context.object  # invisibleGround leaves ground as active object
        if ground and 'Ground' in ground.name:
            ground.location.z = ground_z
        
        bt.setLight_sun(light_angle, light_strength, shadow_softness)
        bt.setLight_ambient(color=ambient_color)
        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')
        
        # Save blend file for first mesh only
        if i == 0:
            blend_path = output_folder / "scene.blend"
            bpy.ops.wm.save_mainfile(filepath=str(blend_path))
            print(f"    Saved blend: {blend_path}")
        
        # Output path
        suffix = "_flat" if settings['flat_shading'] else ""
        output_path = output_folder / f"{mesh_file.stem}{suffix}.png"
        
        # Render
        print(f"    Rendering to: {output_path}")
        bt.renderImage(str(output_path), cam)
        rendered_paths.append(output_path)
        
        print(f"    ✓ Complete")
    
    # ========================================
    # Crop images (optional)
    # ========================================
    if settings['do_crop'] and rendered_paths:
        print(f"\n{'='*50}")
        print("Cropping images")
        print(f"{'='*50}")
        
        crop_box = find_crop_box(rendered_paths)
        if crop_box:
            crop_images(rendered_paths, crop_box)
    
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
    
    print("\nConsistent settings applied to all meshes:")
    print(f"  ✓ Camera: {tuple(f'{x:.2f}' for x in cam_location)} (from combined bounds)")
    print(f"  ✓ Material: front={settings['front_color']}, back={settings['back_color']}")
    print(f"  ✓ Shading: {'flat' if settings['flat_shading'] else 'smooth'}")
    print(f"  ✓ Lighting: sun={light_angle}, ambient={ambient_color[:3]}")


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
