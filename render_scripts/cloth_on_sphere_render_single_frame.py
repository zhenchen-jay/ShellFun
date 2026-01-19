"""
Custom render script for cloth on sphere simulation using BlenderToolbox (Single Frame).

Uses utility functions from render_meshes_bt.py

Usage:
    python cloth_on_sphere_render_single_frame.py -- -i /path/to/mesh -o /path/to/output
    python cloth_on_sphere_render_single_frame.py -- -i /path/to/mesh -o /path/to/output --flat-shading
"""

import bpy
import sys
import math
import argparse
from pathlib import Path

import blendertoolbox as bt

# Import utility functions from render_meshes_bt
from render_meshes_bt import (
    get_color_for_method,
    load_mesh_with_fallback,
    setup_gpu_rendering,
    save_largest_and_minz_of_smallest,
)
from setMat_doubleColor_with_wireframe_modifier import setMat_doubleColor_with_wireframe_modifier
from setLight_sun_with_strength import setLight_sun_with_strength
from setMat_metal_wrapper import setMat_metal_wrapper

def parse_cloth_on_sphere_single_frame_arguments():
    """Parse command line arguments for single frame rendering."""
    parser = argparse.ArgumentParser(description='Render cloth on sphere mesh using BlenderToolbox')
    parser.add_argument('-i', '--input-mesh', type=str, required=True,
                        help='Input mesh file')
    parser.add_argument('-o', '--output-image', type=str, default="output.png",
                        help='Output image file (default: output.png)')

    parser.add_argument('--samples', type=int, default=300,
                        help='Render samples (default: 300)')
    parser.add_argument('--resolution-x', type=int, default=2160,
                        help='Render width (default: 2160)')
    parser.add_argument('--resolution-y', type=int, default=2160,
                        help='Render height (default: 2160)')
    parser.add_argument('--exposure', type=float, default=1.5,
                        help='Exposure (default: 1.5)')
    parser.add_argument('--focal-length', type=float, default=45.0,
                        help='Camera focal length in mm (default: 45.0)')
    parser.add_argument('--flat-shading', action='store_true', help="Whether to use flat shading", default=False)
    parser.add_argument('--edge-thickness', type=float, help="Edge thickness (default: 0)", default=0)
    
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args([])
    
    return args

########################################################
# camera settings (fixed)
# location: (-3.04597, -1.52292, 1.31971)
# rotation: (63.0665, 0, -60.8)

# Sun light: 
# location: (-2.83687, -1.34623, 1.6832)
# rotation: (34.9381, -54.592, -11.2248)

########################################################

def main():
    """Custom main function for cloth on sphere rendering."""
    
    # ========================================
    # Parse arguments
    # ========================================
    args = parse_cloth_on_sphere_single_frame_arguments()
    
    input_mesh = Path(args.input_mesh)
    output_image = Path(args.output_image)
    samples = args.samples
    resolution_x = args.resolution_x
    resolution_y = args.resolution_y
    exposure = args.exposure
    focal_length = args.focal_length
    flat_shading = args.flat_shading
    edge_thickness = args.edge_thickness
    
    if not input_mesh.exists():
        print(f"ERROR: Input mesh not found: {input_mesh}")
        sys.exit(1)
    
    
    # ========================================
    # Object color
    # ========================================
    # rgba
    # get the method name from the input folder name
    method_name = input_mesh.parent.stem
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
    print("Cloth on Sphere Renderer (Single Frame)")
    print("=" * 50)
    print(f"Input mesh: {input_mesh}")
    print(f"Output image: {output_image}")
    print(f"Samples: {samples}")
    print(f"Resolution: {resolution_x}x{resolution_y}")
    print(f"Exposure: {exposure}")
    print(f"Focal length: {focal_length}mm")
    print(f"Flat shading: {flat_shading}")
    print(f"Edge thickness: {edge_thickness}")
    
    # ========================================
    # Preprocess mesh: Split into cloth and sphere
    # ========================================
    print(f"\n{'='*50}")
    print("Preprocessing: Splitting mesh into cloth and sphere")
    print(f"{'='*50}")
    
    # Create temporary folder for preprocessed meshes
    preprocessed_folder = input_mesh.parent / "preprocessed_temp"
    preprocessed_folder.mkdir(parents=True, exist_ok=True)
    
    cloth_mesh_file = preprocessed_folder / f"{input_mesh.stem}_cloth.ply"
    sphere_mesh_file = preprocessed_folder / f"{input_mesh.stem}_sphere.ply"
    
    print(f"  Splitting components...")
    result = save_largest_and_minz_of_smallest(str(input_mesh), str(cloth_mesh_file), str(sphere_mesh_file))
    print(f"  Result: {result}")
    print(f"  Cloth mesh: {cloth_mesh_file}")
    print(f"  Sphere mesh: {sphere_mesh_file}")
    
    # ========================================
    # Camera settings (fixed)
    # ========================================
    print(f"\n{'='*50}")
    print("Camera settings (fixed)")
    print(f"{'='*50}")
    
    # Fixed camera position and rotation
    camera_location = (-3.04597, -1.52292, 1.31971)
    camera_rotation = (63.0665, 0, -60.8)  # Euler rotation in degrees
    
    print(f"  Camera location: {camera_location}")
    print(f"  Camera rotation: {camera_rotation}")
    print(f"  Focal length: {focal_length}mm")
    
    # ========================================
    # Light settings (fixed)
    # ========================================
    # Sun light with fixed rotation
    light_rotation = (34.9381, -54.592, -11.2248)  # Euler rotation in degrees
    light_location = (-2.83687, -1.34623, 1.6832)
    light_strength = 2.0
    shadow_softness = 0.3
    
    print(f"\n  Sun light location: {light_location}")
    print(f"  Sun light rotation: {light_rotation}")
    print(f"  Sun light strength: {light_strength}")
    
    # Material color (custom, not from args)
    print(f"  Material color: {obj_color}")
    meshColor_top = bt.colorObj(obj_color, 0.5, 1.0, 1.0, 0.0, 0.0)
    meshColor_bottom = bt.colorObj((0.3, 0.3, 0.3, 1.0), 0.5, 1.0, 1.0, 0.0, 0.0)
    ao_strength = 0.5
    
    # ========================================
    # Render the mesh
    # ========================================
    print(f"\n{'='*50}")
    print("Rendering mesh")
    print(f"{'='*50}")
    
    # Fresh scene
    bt.blenderInit(resolution_x, resolution_y, samples, exposure)
    setup_gpu_rendering()
    
    # Load cloth mesh
    print(f"  Loading cloth mesh...")
    mesh = load_mesh_with_fallback(bt, cloth_mesh_file, mesh_location, mesh_rotation, mesh_scale)

    if mesh is None:
        print(f"    ERROR: Failed to load cloth mesh {cloth_mesh_file}, exiting...")
        sys.exit(1)

    # Load sphere mesh
    print(f"  Loading sphere mesh...")
    if not sphere_mesh_file.exists():
        print(f"    ERROR: Sphere mesh file not found: {sphere_mesh_file}")
        sys.exit(1)
    
    sphere_mesh = load_mesh_with_fallback(bt, sphere_mesh_file, mesh_location, mesh_rotation, mesh_scale)
    
    if sphere_mesh is None:
        print(f"    ERROR: Failed to load sphere mesh {sphere_mesh_file}, exiting...")
        sys.exit(1)

    # Subdivide the sphere mesh
    print(f"  Subdividing sphere mesh...")
    bt.subdivision(sphere_mesh, level=2)
    sphere_color = (0.2, 0.2, 0.2, 1.0)  # grey
    setMat_metal_wrapper(sphere_mesh, sphere_color, metalVal=0.1, roughnessVal=0.05)
    
    # Shading for cloth
    print(f"  Setting cloth shading...")
    print(f"  Flat shading: {flat_shading}")
    if flat_shading:
        bpy.ops.object.shade_flat()
    else:
        bpy.ops.object.shade_smooth()
    
    # Material for cloth
    print(f"  Applying cloth material...")
    setMat_doubleColor_with_wireframe_modifier(mesh, meshColor_top, meshColor_bottom, AOStrength=ao_strength, edgeThickness=edge_thickness)
    
    # Camera (fixed position and rotation using direct Blender API)
    print(f"  Setting up camera...")
    bpy.ops.object.camera_add(location=camera_location)
    cam = bpy.context.object
    cam.rotation_euler = tuple(math.radians(a) for a in camera_rotation)
    cam.data.lens = focal_length
    bpy.context.scene.camera = cam
    
    # Lighting (fixed rotation) - using direct Blender API for Blender 4.x compatibility
    print(f"  Setting up lighting...")
    sun_light = setLight_sun_with_strength(location=light_location, rotation_euler=light_rotation, strength=light_strength, shadow_soft_size=shadow_softness)

    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')
    
    # Ensure output directory exists
    output_image.parent.mkdir(parents=True, exist_ok=True)
    
    # Save blend file
    blend_path = output_image.parent / "scene.blend"
    bpy.ops.wm.save_mainfile(filepath=str(blend_path))
    print(f"    Saved blend: {blend_path}")
    
    # Render
    print(f"  Rendering to: {output_image}")
    bt.renderImage(str(output_image), cam)
    
    print(f"\n{'='*50}")
    print("COMPLETE!")
    print(f"{'='*50}")
    print(f"Output: {output_image}")


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
