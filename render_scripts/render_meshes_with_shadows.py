"""
Blender Script to Render All PLY/OBJ Files with Consistent Camera, Light, and Shadow Settings

This script renders all mesh files in a given folder with:
- Mesh rotation (90, 0, 0) to convert Y-up to Z-up coordinate system
- Two-sided material (different colors for front/back faces, great for cloth)
- Consistent camera positioning (corner view)
- Shadow-casting sun light
- Ground plane shadow catcher
- Automatic cropping to remove white space (common bounding box for all images)
- High-quality rendering settings with GPU acceleration

Usage:
    # Basic usage (auto camera, 1080x1080, smooth shading)
    blender --background --python RenderMeshesWithShadows.py -- \
        -i /path/to/meshes \
        -o /path/to/output
    
    # With flat shading to show sharp creases
    blender --background --python RenderMeshesWithShadows.py -- \
        -i /path/to/meshes \
        -o /path/to/output \
        --flat-shading
    
    # With JSON config for fixed camera and custom settings
    blender --background --python RenderMeshesWithShadows.py -- \
        -i /path/to/meshes \
        -o /path/to/output \
        --config render_config.json
"""

import bpy
import mathutils
import sys
import argparse
import math
import json
import numpy as np
from pathlib import Path
from PIL import Image
from setMat_doubleColorWire import setMat_doubleColorWire
import BlenderToolBox as bt


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Render all PLY/OBJ files with shadows')
    parser.add_argument('-i', '--input-folder', type=str, required=True,
                        help='Folder containing PLY/OBJ files to render')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Folder to save rendered images')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file with camera and render settings')
    parser.add_argument('--samples', type=int, default=128,
                        help='Render samples for quality (default: 128)')
    parser.add_argument('--resolution-x', type=int, default=1080,
                        help='Render resolution width (default: 1080)')
    parser.add_argument('--resolution-y', type=int, default=1080,
                        help='Render resolution height (default: 1080)')
    parser.add_argument('--focal-length', type=float, default=45.0,
                        help='Camera focal length in mm (default: 45.0)')
    parser.add_argument('--material-color', type=str, default='0.6,0.6,0.7',
                        help='Material color as R,G,B (default: 0.6,0.6,0.7)')
    parser.add_argument('--material-color-back', type=str, default='0.67,0.4,0.95',
                        help='Back face material color as R,G,B (default: 0.67,0.4,0.95)')
    parser.add_argument('--background-transparent', action='store_true', default=True,
                        help='Use transparent background instead of white')
    parser.add_argument('--crop-images', action='store_true', default=True,
                        help='Crop all images to common bounding box to remove white space')
    parser.add_argument('--flat-shading', action='store_true', default=False,
                        help='Use flat shading (face normals) to show sharp creases instead of smooth shading')
    
    # Parse args, handling Blender's '--' separator
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args([])
    
    return args


def load_config(config_path):
    """
    Load render and camera settings from JSON config file.
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        dict: Configuration dictionary
    
    Example JSON format:
    {
        "camera": {
            "focal_length": 90.0,
            "location": [1.9318, 3.27018, 1.95009],
            "rotation": [53.2644, 0, 150]
        },
        "render": {
            "samples": 128,
            "resolution_x": 1080,
            "resolution_y": 1080
        },
        "material": {
            "front_color": [0.6, 0.6, 0.7],
            "back_color": [0.67, 0.4, 0.95]
        },
        "options": {
            "flat_shading": false,
            "crop_images": true,
            "transparent_background": true
        }
    }
    """
    print(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("  ✓ Config loaded successfully")
        return config
    except FileNotFoundError:
        print(f"  ERROR: Config file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in config file: {e}")
        return None


def clear_scene():
    """Clear all objects from the scene."""
    print("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def load_mesh(mesh_path, name="Mesh"):
    """
    Load PLY or OBJ mesh into Blender.
    
    Args:
        mesh_path: Path to mesh file
        name: Name for the object
    
    Returns:
        Imported mesh object
    """
    print(f"  Loading {name} from: {mesh_path.name}")
    
    file_ext = mesh_path.suffix.lower()
    
    if file_ext == '.obj':
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
    elif file_ext == '.ply':
        bpy.ops.wm.ply_import(filepath=str(mesh_path))
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    obj = bpy.context.selected_objects[0]
    obj.name = name
    
    print(f"    Loaded: {len(obj.data.vertices)} vertices")
    return obj


def apply_rotation_to_mesh(obj):
    """
    Apply (90, 0, 0) rotation to convert Y-up to Z-up coordinate system.
    
    Args:
        obj: Mesh object to rotate
    """
    print(f"  Applying rotation (90, 0, 0) to convert Y-up to Z-up")
    
    # Make mesh data single-user to avoid multi-user error
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    
    obj.rotation_euler = (math.radians(90), 0, 0)
    
    # Apply the rotation so it becomes permanent
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    
    print(f"    Rotation applied and baked into mesh")


def apply_two_sided_material(obj, front_color=(0.6, 0.6, 0.7), back_color=(0.67, 0.4, 0.95), name="TwoSidedMaterial"):
    """
    Apply a two-sided plastic material to the object.
    Different colors for front and back faces (useful for cloth).
    
    Args:
        obj: Mesh object
        front_color: RGB color tuple for front faces (0-1 range)
        back_color: RGB color tuple for back faces (0-1 range)
        name: Material name
    """
    # Create a new material
    mat = bpy.data.materials.new(name)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.active_material = mat
    mat.use_nodes = True
    mat.use_backface_culling = False
    tree = mat.node_tree

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Add nodes
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    mix_shader = tree.nodes.new('ShaderNodeMixShader')
    geometry = tree.nodes.new('ShaderNodeNewGeometry')

    # Create Principled BSDF for front and back
    front_bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
    back_bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')

    # Configure front BSDF
    front_bsdf.inputs['Roughness'].default_value = 0.3
    front_bsdf.inputs['Base Color'].default_value = (*front_color, 1.0)

    # Configure back BSDF
    back_bsdf.inputs['Roughness'].default_value = 0.3
    back_bsdf.inputs['Base Color'].default_value = (*back_color, 1.0)

    # Connect Backfacing to Mix Shader
    tree.links.new(geometry.outputs['Backfacing'], mix_shader.inputs['Fac'])
    tree.links.new(back_bsdf.outputs['BSDF'], mix_shader.inputs[1])  # Back face
    tree.links.new(front_bsdf.outputs['BSDF'], mix_shader.inputs[2])  # Front face

    # Connect Mix Shader to Output
    tree.links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])
    
    print(f"    Two-sided material applied: front={front_color}, back={back_color}")


def calculate_bounding_box(obj):
    """Calculate bounding box of object."""
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    
    min_corner = [float('inf')] * 3
    max_corner = [float('-inf')] * 3
    
    for corner in bbox_corners:
        for i in range(3):
            min_corner[i] = min(min_corner[i], corner[i])
            max_corner[i] = max(max_corner[i], corner[i])
    
    bbox_center = tuple((min_corner[i] + max_corner[i]) / 2 for i in range(3))
    bbox_size = tuple(max_corner[i] - min_corner[i] for i in range(3))
    
    return bbox_center, bbox_size


def setup_camera_and_lighting(obj, focal_length=45.0, camera_location=None, camera_rotation=None):
    """
    Set up camera and lighting.
    If camera_location/rotation are None, auto-calculate based on mesh bounding box.
    
    Args:
        obj: Object to frame
        focal_length: Camera focal length in mm
        camera_location: Camera location (X, Y, Z) in meters (None = auto-calculate)
        camera_rotation: Camera rotation (X, Y, Z) in degrees (None = auto-calculate)
    
    Returns:
        camera: Camera object
    """
    print("Setting up camera and lighting...")
    
    bbox_center, bbox_size = calculate_bounding_box(obj)
    max_dimension = max(bbox_size)
    
    print(f"  Bounding box: center={bbox_center}, size={bbox_size}")
    print(f"  Max dimension: {max_dimension:.2f}")
    
    if camera_location is None or camera_rotation is None:
        # Auto-calculate camera position (corner view)
        distance = max_dimension * 1.8
        camera_location = (
            bbox_center[0] + distance * 1.0,
            bbox_center[1] + distance * 1.0,
            bbox_center[2] + distance * 2.0
        )
        
        bpy.ops.object.camera_add(location=camera_location)
        camera = bpy.context.object
        camera.name = "Camera"
        
        # Camera target at mesh center
        bpy.ops.object.empty_add(location=bbox_center)
        empty = bpy.context.object
        empty.name = "Camera_Target"
        empty.empty_display_size = 0.1
        
        # Track to constraint
        track_to = camera.constraints.new(type='TRACK_TO')
        track_to.target = empty
        track_to.track_axis = 'TRACK_NEGATIVE_Z'
        track_to.up_axis = 'UP_Y'
        
        bpy.context.view_layer.objects.active = camera
        bpy.ops.constraint.apply(constraint=track_to.name, owner='OBJECT')
        
        print(f"  Camera auto-positioned at: {camera_location}")
        print(f"  Looking at: {bbox_center}")
    else:
        # Use fixed camera position and rotation
        bpy.ops.object.camera_add(location=camera_location)
        camera = bpy.context.object
        camera.name = "Camera"
        
        # Set camera rotation (convert degrees to radians)
        import math
        camera.rotation_euler = (
            math.radians(camera_rotation[0]),
            math.radians(camera_rotation[1]),
            math.radians(camera_rotation[2])
        )
        
        print(f"  Camera at fixed location: {camera_location}")
        print(f"  Camera rotation: {camera_rotation} degrees")
    
    bpy.context.scene.camera = camera
    camera.data.lens = focal_length
    camera.data.clip_end = 1000
    camera.data.sensor_width = 36
    
    print(f"  Focal length: {focal_length} mm")
    
    # Lighting setup - Sun light with shadows (from reference script)
    bpy.ops.object.light_add(type='SUN')
    sun_light = bpy.context.object
    sun_light.name = "SunLight"
    
    # Set rotation (6, -30, -155) from reference
    sun_light.rotation_euler = (
        math.radians(6),
        math.radians(-30),
        math.radians(-155)
    )
    
    # Sun light properties
    sun_light.data.energy = 2.0
    sun_light.data.angle = math.radians(0.3)  # Shadow softness
    sun_light.data.use_shadow = True
    
    print(f"  Sun light: rotation=(6°, -30°, -155°), energy=2.0, shadows enabled")
    
    # Add ambient light
    world = bpy.context.scene.world
    if not world.use_nodes:
        world.use_nodes = True
    
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0)
        bg_node.inputs['Strength'].default_value = 1.0
    
    print(f"  Ambient light: RGB=(0.1, 0.1, 0.1)")
    
    return camera


def create_ground_plane(bbox_center, bbox_size):
    """
    Create shadow catcher ground plane.
    Positioned horizontally at the bottom (lowest Z).
    
    Args:
        bbox_center: Center of the bounding box
        bbox_size: Size of the bounding box
    
    Returns:
        Ground plane object
    """
    print("Creating ground plane (shadow catcher)...")
    
    # Position plane at bottom (Z-up system after rotation)
    # Center it under the mesh, not at world origin
    min_z = bbox_center[2] - bbox_size[2] / 2
    plane_location = (bbox_center[0], bbox_center[1], min_z)
    
    # Large plane to catch all shadows
    max_dimension = max(bbox_size)
    plane_size = max(max_dimension * 30.0, 100.0)
    
    print(f"  Ground plane location: {plane_location}")
    print(f"  Ground plane size: {plane_size:.2f}")
    
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=plane_location)
    plane = bpy.context.object
    plane.name = "GroundPlane"
    
    # Horizontal orientation
    plane.rotation_euler = (0, 0, 0)
    
    # Enable shadow catcher
    plane.is_shadow_catcher = True
    
    # White material
    mat = bpy.data.materials.new(name="GroundPlane_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.2
        bsdf.inputs['Metallic'].default_value = 0.0
    
    plane.data.materials.append(mat)
    
    return plane


def find_common_crop_box(image_paths, padding=20):
    """
    Find the common bounding box for all images to remove white space.
    
    Args:
        image_paths: List of paths to images
        padding: Padding to add around content
    
    Returns:
        Tuple of (min_x, min_y, max_x, max_y) for cropping
    """
    print("\nFinding common crop box for all images...")
    
    if not image_paths:
        return None
    
    # Initialize with extreme values
    global_min_x = float('inf')
    global_min_y = float('inf')
    global_max_x = 0
    global_max_y = 0
    
    for img_path in image_paths:
        if not img_path.exists():
            continue
            
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Handle different image formats
        if len(img_array.shape) < 2:
            print(f"  WARNING: Skipping {img_path.name} - invalid image shape")
            continue
        
        # Detect content based on image format
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA image with transparency
            alpha = img_array[:, :, 3]
            
            # For transparent renders: content is where alpha is significantly non-zero
            # Use a much higher threshold to exclude faint shadows/edges
            # Only keep truly opaque pixels (alpha > 200)
            content_mask = alpha > 200
            
            # Debug statistics
            alpha_min, alpha_max = alpha.min(), alpha.max()
            alpha_mean = alpha.mean()
            content_pixels = np.sum(content_mask)
            total_pixels = content_mask.shape[0] * content_mask.shape[1]
            content_pct = 100 * content_pixels / total_pixels
            
            print(f"  {img_path.name}:")
            print(f"    Alpha: min={alpha_min}, max={alpha_max}, mean={alpha_mean:.1f}")
            print(f"    Content pixels: {content_pixels}/{total_pixels} ({content_pct:.1f}%)")
            
        elif len(img_array.shape) == 3:
            # RGB image - detect non-white pixels
            # White is (255, 255, 255), so sum should be close to 765
            pixel_sum = np.sum(img_array, axis=2)
            content_mask = pixel_sum < 750  # Not perfectly white
            
        else:
            # Grayscale
            content_mask = img_array < 250
        
        # Find content bounding box
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            
            min_x = col_indices[0]
            min_y = row_indices[0]
            max_x = col_indices[-1] + 1
            max_y = row_indices[-1] + 1
            
            # Update global bounds
            global_min_x = min(global_min_x, min_x)
            global_min_y = min(global_min_y, min_y)
            global_max_x = max(global_max_x, max_x)
            global_max_y = max(global_max_y, max_y)
            
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            print(f"    bbox=({min_x}, {min_y}, {max_x}, {max_y}), size={bbox_width}x{bbox_height}")
    
    if global_min_x == float('inf'):
        print("  No content found in images")
        return None
    
    # Add padding (but don't exceed image bounds - we'll handle that per image)
    global_min_x = max(0, global_min_x - padding)
    global_min_y = max(0, global_min_y - padding)
    global_max_x = global_max_x + padding
    global_max_y = global_max_y + padding
    
    crop_box = (int(global_min_x), int(global_min_y), int(global_max_x), int(global_max_y))
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    
    print(f"  Common crop box: {crop_box}")
    print(f"  Cropped size: {crop_width}x{crop_height}")
    
    return crop_box


def crop_images(image_paths, crop_box, suffix="_cropped"):
    """
    Crop all images to the same bounding box.
    
    Args:
        image_paths: List of paths to images
        crop_box: Tuple of (min_x, min_y, max_x, max_y)
        suffix: Suffix to add to cropped filenames
    
    Returns:
        List of paths to cropped images
    """
    print("\nCropping images to common bounding box...")
    
    cropped_paths = []
    total_size_reduction = 0
    
    for img_path in image_paths:
        if not img_path.exists():
            continue
        
        img = Image.open(img_path)
        
        # Ensure crop box is within image bounds
        width, height = img.size
        min_x, min_y, max_x, max_y = crop_box
        min_x = max(0, min(min_x, width))
        min_y = max(0, min(min_y, height))
        max_x = max(0, min(max_x, width))
        max_y = max(0, min(max_y, height))
        
        # Crop image
        cropped = img.crop((min_x, min_y, max_x, max_y))
        
        # Calculate size reduction
        original_pixels = width * height
        cropped_pixels = (max_x - min_x) * (max_y - min_y)
        size_reduction = 100 * (1 - cropped_pixels / original_pixels)
        total_size_reduction += size_reduction
        
        # Save with suffix
        cropped_path = img_path.parent / f"{img_path.stem}{suffix}{img_path.suffix}"
        cropped.save(cropped_path)
        cropped_paths.append(cropped_path)
        
        print(f"  {img_path.name}: {width}x{height} -> {max_x-min_x}x{max_y-min_y} (reduced {size_reduction:.1f}%)")
    
    if cropped_paths:
        avg_reduction = total_size_reduction / len(cropped_paths)
        print(f"\n  Average size reduction: {avg_reduction:.1f}%")
    
    return cropped_paths


def setup_render_settings(samples=128, resolution_x=1920, resolution_y=1080, transparent_bg=False):
    """Configure render settings."""
    print(f"Setting up render: {resolution_x}x{resolution_y}, {samples} samples")
    
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.film_transparent = transparent_bg
    scene.render.filter_size = 1.5
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA' if transparent_bg else 'RGB'
    
    # Use GPU if available - automatically detect the right compute device type
    try:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        
        # Try to detect available compute device types
        # Common types: METAL (macOS), CUDA (NVIDIA), OPTIX (NVIDIA RTX), HIP (AMD), OPENCL
        available_types = []
        for compute_type in ['METAL', 'CUDA', 'OPTIX', 'HIP', 'OPENCL']:
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


def render_mesh_with_shadows(mesh_path, output_path, front_color, back_color, transparent_bg=True, save_blend=None, 
                             flat_shading=False, focal_length=45.0, camera_location=None, 
                             camera_rotation=None):
    """
    Render a single mesh with shadows.
    
    Args:
        mesh_path: Path to mesh file
        output_path: Output image path
        front_color: RGB color tuple for front faces
        back_color: RGB color tuple for back faces
        transparent_bg: Whether to use transparent background
        save_blend: Optional path to save the Blender file before rendering
        flat_shading: Use flat shading (face normals) instead of smooth shading
        focal_length: Camera focal length in mm
        camera_location: Camera location tuple (X, Y, Z) or None for auto
        camera_rotation: Camera rotation tuple (X, Y, Z) in degrees or None for auto
    """
    print(f"\n{'='*60}")
    print(f"Rendering: {mesh_path.name}")
    print(f"{'='*60}")
    
    # Clear scene
    clear_scene()
    
    # Load mesh
    mesh_obj = load_mesh(mesh_path, "Mesh")
    
    # Apply rotation (90, 0, 0) to convert Y-up to Z-up
    apply_rotation_to_mesh(mesh_obj)
    
    # Apply two-sided material
    apply_two_sided_material(mesh_obj, front_color, back_color, "Mesh_Material")
    
    # Setup camera and lighting
    camera = setup_camera_and_lighting(mesh_obj, focal_length, camera_location, camera_rotation)
    
    # Create ground plane for shadows
    bbox_center, bbox_size = calculate_bounding_box(mesh_obj)
    ground = create_ground_plane(bbox_center, bbox_size)
    
    # Apply shading (flat or smooth)
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    
    if flat_shading:
        bpy.ops.object.shade_flat()
        print("  Using flat shading (face normals) to show sharp creases")
    else:
        bpy.ops.object.shade_smooth()
        print("  Using smooth shading (vertex normals)")
    
    # Save blend file if requested
    if save_blend:
        print(f"Saving Blender file to: {save_blend}")
        bpy.ops.wm.save_as_mainfile(filepath=str(save_blend))
        print(f"✓ Blend file saved: {save_blend.name}")
    
    # Render
    print(f"Rendering to: {output_path}")
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    
    print(f"✓ Render complete: {output_path.name}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup paths
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    if not input_folder.exists():
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load config file if provided
    config = None
    if args.config:
        config = load_config(Path(args.config))
        if config is None:
            print("ERROR: Failed to load config file, exiting")
            sys.exit(1)
    
    # Get settings from config or command-line args
    # Command-line args take priority over config file
    if config:
        # Camera settings (from config)
        camera_config = config.get('camera', {})
        focal_length = camera_config.get('focal_length', 45.0)
        camera_location = tuple(camera_config.get('location')) if 'location' in camera_config else None
        camera_rotation = tuple(camera_config.get('rotation')) if 'rotation' in camera_config else None
        
        # Render settings (config as default, command-line overrides if different from default)
        samples = config.get('render', {}).get('samples', args.samples)
        resolution_x = config.get('render', {}).get('resolution_x', args.resolution_x)
        resolution_y = config.get('render', {}).get('resolution_y', args.resolution_y)
        
        # Material colors
        front_color = tuple(config.get('material', {}).get('front_color', [0.6, 0.6, 0.7]))
        back_color = tuple(config.get('material', {}).get('back_color', [0.67, 0.4, 0.95]))
        
        # Options - command-line flags override config if provided
        # If command-line flag is True (explicitly provided), it takes priority
        flat_shading = args.flat_shading or config.get('options', {}).get('flat_shading', False)
        # crop_images and transparent_bg default to True, so we can't easily detect if user set them
        # Just use config values for these
        do_crop = config.get('options', {}).get('crop_images', True)
        transparent_bg = config.get('options', {}).get('transparent_background', True)
    else:
        # Use command-line arguments - auto camera by default
        focal_length = args.focal_length
        camera_location = None  # Auto-calculate based on mesh
        camera_rotation = None  # Auto-calculate based on mesh
        
        samples = args.samples
        resolution_x = args.resolution_x
        resolution_y = args.resolution_y
        flat_shading = args.flat_shading
        do_crop = args.crop_images
        transparent_bg = args.background_transparent
        
        # Parse material colors
        try:
            color_parts = [float(x) for x in args.material_color.split(',')]
            if len(color_parts) != 3:
                raise ValueError()
            front_color = tuple(color_parts)
        except:
            print(f"WARNING: Invalid front color format '{args.material_color}', using default (0.6, 0.6, 0.7)")
            front_color = (0.6, 0.6, 0.7)
        
        try:
            color_parts = [float(x) for x in args.material_color_back.split(',')]
            if len(color_parts) != 3:
                raise ValueError()
            back_color = tuple(color_parts)
        except:
            print(f"WARNING: Invalid back color format '{args.material_color_back}', using default (0.67, 0.4, 0.95)")
            back_color = (0.67, 0.4, 0.95)
    
    print("\n" + "=" * 60)
    print("Blender Mesh Rendering with Shadows")
    print("=" * 60)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    if args.config:
        print(f"Config file: {args.config}")
    print(f"\nRender Settings:")
    print(f"  Resolution: {resolution_x}x{resolution_y}")
    print(f"  Samples: {samples}")
    print(f"  Transparent background: {transparent_bg}")
    print(f"  Crop images: {do_crop}")
    print(f"\nCamera Settings:")
    print(f"  Focal length: {focal_length} mm")
    if camera_location is not None and camera_rotation is not None:
        print(f"  Location: {camera_location} (fixed)")
        print(f"  Rotation: {camera_rotation} degrees (fixed)")
    else:
        print(f"  Position: Auto-calculated (corner view based on mesh bounds)")
    print(f"\nMaterial Settings:")
    print(f"  Front color: RGB{front_color}")
    print(f"  Back color: RGB{back_color}")
    print(f"  Shading mode: {'flat (face normals)' if flat_shading else 'smooth (vertex normals)'}")
    print(f"\nMesh Processing:")
    print(f"  Rotation: (90, 0, 0) to convert Y-up to Z-up")
    
    # Find all mesh files
    mesh_files = []
    for ext in ['.ply', '.obj', '.PLY', '.OBJ']:
        mesh_files.extend(list(input_folder.glob(f'*{ext}')))
    
    mesh_files = sorted(mesh_files)
    
    if not mesh_files:
        print(f"\nERROR: No PLY or OBJ files found in {input_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(mesh_files)} mesh files to render:")
    for mesh_file in mesh_files:
        print(f"  - {mesh_file.name}")
    
    # Setup render settings (once for all renders)
    setup_render_settings(
        samples,
        resolution_x,
        resolution_y,
        transparent_bg
    )
    
    # Render each mesh
    print(f"\n{'='*60}")
    print("Starting batch rendering...")
    print(f"{'='*60}")
    
    rendered_paths = []
    
    for i, mesh_file in enumerate(mesh_files, 1):
        print(f"\n[{i}/{len(mesh_files)}]")
        
        # Generate output filename with suffix for flat shading
        if flat_shading:
            output_filename = mesh_file.stem + "_render_flat.png"
        else:
            output_filename = mesh_file.stem + "_render.png"
        output_path = output_folder / output_filename
        
        # Save blend file for first mesh
        blend_path = None
        if i == 1:
            blend_path = output_folder / "first_mesh_setup.blend"
        
        try:
            render_mesh_with_shadows(
                mesh_file,
                output_path,
                front_color,
                back_color,
                transparent_bg,
                save_blend=blend_path,
                flat_shading=flat_shading,
                focal_length=focal_length,
                camera_location=camera_location,
                camera_rotation=camera_rotation
            )
            rendered_paths.append(output_path)
        except Exception as e:
            print(f"ERROR rendering {mesh_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Crop images if requested
    cropped_paths = []
    if do_crop and rendered_paths:
        crop_box = find_common_crop_box(rendered_paths, padding=20)
        if crop_box:
            cropped_paths = crop_images(rendered_paths, crop_box)
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH RENDERING COMPLETE!")
    print("=" * 60)
    print(f"Rendered {len(rendered_paths)} meshes")
    print(f"Output saved to: {output_folder}")
    
    print("\nRendered images:")
    for img_path in rendered_paths:
        print(f"  - {img_path.name}")
    
    print("\nAll renders include:")
    print("  ✓ Mesh rotated (90, 0, 0) to Z-up coordinate system")
    print("  ✓ Two-sided material (different colors for front/back)")
    print(f"    Front: RGB{front_color}, Back: RGB{back_color}")
    print(f"  ✓ Shading: {'flat (face normals for sharp creases)' if flat_shading else 'smooth (vertex normals)'}")
    print(f"    Filename suffix: {'_render_flat.png' if flat_shading else '_render.png'}")
    print("  ✓ Sun light with shadows")
    print("  ✓ Ambient lighting (RGB 0.1, 0.1, 0.1)")
    print("  ✓ Shadow catcher ground plane")
    if camera_location is not None:
        print("  ✓ Fixed camera position from config")
    else:
        print("  ✓ Auto camera (corner view 1:1:2 ratio)")
    
    if cropped_paths:
        print(f"\n✓ Cropped {len(cropped_paths)} images to common bounding box")
        print("  Cropped images:")
        for img_path in cropped_paths:
            print(f"    - {img_path.name}")
    
    print(f"\nBlender file saved: {output_folder / 'first_mesh_setup.blend'}")
    print("  (Open this file in Blender to adjust settings)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
