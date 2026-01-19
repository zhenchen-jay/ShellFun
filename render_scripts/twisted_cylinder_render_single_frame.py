"""
Custom render script for twisted cloth simulation using BlenderToolbox.

Uses utility functions from render_meshes_bt.py

Usage:
    python twisted_clynder_render_sliced.py -- -i /path/to/mesh -o /path/to/output
    python twisted_clynder_render_sliced.py -- -i /path/to/mesh -o /path/to/output --flat-shading
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
)
from setMat_doubleColor_with_wireframe_modifier import setMat_doubleColor_with_wireframe_modifier
from setLight_sun_with_strength import setLight_sun_with_strength
from bisect_mesh import bisect_mesh

def parse_twisted_cylinder_render_single_frame_arguments():
    """Parse command line arguments with video export option."""
    parser = argparse.ArgumentParser(description='Render twisted cloth meshes using BlenderToolbox')
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
# camera settings
# locattion: (0, 0, 0.18)
# rotation: (0, 0, 0)

# Sun light: 
# location: (0, 0, 0.2)
# rotation: (0, 0, 0)

########################################################

def main():
    """Custom main function for twisted cloth rendering."""
    
    # ========================================
    # Parse arguments
    # ========================================
    args = parse_twisted_cylinder_render_single_frame_arguments()
    
    input_mesh = Path(args.input_mesh)
    output_image = Path(args.output_image)
    samples = args.samples
    resolution_x = args.resolution_x
    resolution_y = args.resolution_y
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
    print("Twisted Cylinder Renderer (Single Frame)")
    print("=" * 50)
    print(f"Input mesh: {input_mesh}")
    print(f"Output image: {output_image}")
    print(f"Samples: {samples}")
    print(f"Resolution: {resolution_x}x{resolution_y}")
    print(f"Focal length: {focal_length}mm")
    print(f"Flat shading: {flat_shading}")
    print(f"Edge thickness: {edge_thickness}")
    
    # ========================================
    # Camera settings (fixed)
    # ========================================
    print(f"\n{'='*50}")
    print("Camera settings (fixed)")
    print(f"{'='*50}")
    
     # Fixed camera position and rotation
    camera_location = (0, 0, 0.18)
    camera_rotation = (0, 0, 0)  # Euler rotation in degrees
    
    print(f"  Camera location: {camera_location}")
    print(f"  Camera rotation: {camera_rotation}")
    print(f"  Focal length: {focal_length}mm")
    
    # ========================================
    # Light settings (fixed)
    # ========================================
    # Sun light with fixed rotation
    light_rotation = (0, 0, 0)  # Euler rotation in degrees
    light_location = (0, 0, 0)
    light_strength = 4.0
    shadow_softness = 0.3
    
    print(f"\n  Sun light rotation: {light_rotation}")
    print(f"  Sun light strength: {light_strength}")
    
    # Material color (custom, not from args)
    print(f"  Material color: {obj_color}")
    meshColor_top = bt.colorObj(obj_color, 0.5, 1.0, 1.0, 0.0, 0.0)
    meshColor_bottom = bt.colorObj(obj_color, 0.5, 1.0, 1.0, 0.0, 0.0)
    ao_strength = 0.5
    
    # ========================================
    # PASS 2: Render each mesh
    # ========================================
    print(f"\n{'='*50}")
    print("Pass 2: Rendering each mesh")
    print(f"{'='*50}")
    
    # Fresh scene
    bt.blenderInit(resolution_x, resolution_y, samples)
    setup_gpu_rendering()
    
    # Load mesh (with fallback to PyMeshLab conversion for problematic PLY files)
    mesh = load_mesh_with_fallback(bt, input_mesh, mesh_location, mesh_rotation, mesh_scale)

    if mesh is None:
        print(f"    ERROR: Failed to load {input_mesh}, skipping...")
        sys.exit(1)

    # Shading
    if flat_shading:
        bpy.ops.object.shade_flat()
    else:
        bpy.ops.object.shade_smooth()
    
    # subdivide the mesh
    # bt.subdivision(mesh, level = 2)
    
    # Material
    setMat_doubleColor_with_wireframe_modifier(mesh, meshColor_top, meshColor_bottom, AOStrength=ao_strength, edgeThickness=edge_thickness)
    
    # Camera (fixed position and rotation using direct Blender API)
    bpy.ops.object.camera_add(location=camera_location)
    cam = bpy.context.object
    cam.rotation_euler = tuple(math.radians(a) for a in camera_rotation)
    cam.data.lens = focal_length
    bpy.context.scene.camera = cam
    
    # Lighting (fixed rotation) - using direct Blender API for Blender 4.x compatibility
    # bt.invisibleGround(shadowBrightness=0.9, location=(0, 0, ground_z))

    # Sun light
    sun_light = setLight_sun_with_strength(location=light_location, rotation_euler=light_rotation, strength=light_strength, shadow_soft_size=shadow_softness)

    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')
    
    # Ensure output directory exists
    output_image.parent.mkdir(parents=True, exist_ok=True)
    
    # Save blend for first mesh
    blend_path = output_image.parent / "scene.blend"
    bpy.ops.wm.save_mainfile(filepath=str(blend_path))
    print(f"    Saved blend: {blend_path}")
    
    # Render
    print(f"    Rendering to: {output_image}")
    bt.renderImage(str(output_image), cam)
    


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
