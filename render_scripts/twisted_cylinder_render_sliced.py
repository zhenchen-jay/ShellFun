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
from setMat_doubleColorWire import setMat_doubleColorWire
from setLight_sun_with_strength import setLight_sun_with_strength
from bisect_mesh import bisect_mesh

def parse_twisted_clynder_render_sliced_arguments():
    """Parse command line arguments with video export option."""
    parser = argparse.ArgumentParser(description='Render twisted cloth meshes using BlenderToolbox')
    parser.add_argument('-i', '--input-mesh', type=str, required=True,
                        help='Input mesh file')
    parser.add_argument('-o', '--output-image', type=str, default="output.png",
                        help='Output image file (default: output.png)')
    parser.add_argument('--nx', type=float, help="The x coordinate of the normal of the slicing plane", default=0.0)
    parser.add_argument('--ny', type=float, help="The y coordinate of the normal of the slicing plane", default=0.0)
    parser.add_argument('--nz', type=float, help="The z coordinate of the normal of the slicing plane", default=1.0)
    parser.add_argument('--cx', type=float, help="The x coordinate of the center of the slicing plane", default=0.0)
    parser.add_argument('--cy', type=float, help="The y coordinate of the center of the slicing plane", default=0.0)
    parser.add_argument('--cz', type=float, help="The z coordinate of the center of the slicing plane", default=0.0)
    parser.add_argument('--clear-outer', action='store_false', help="Whether to clear the outer part of the mesh", default=True)
    parser.add_argument('--clear-inner', action='store_true', help="Whether to clear the inner part of the mesh", default=False)

    parser.add_argument('--samples', type=int, default=300,
                        help='Render samples (default: 300)')
    parser.add_argument('--resolution-x', type=int, default=2160,
                        help='Render width (default: 2160)')
    parser.add_argument('--resolution-y', type=int, default=2160,
                        help='Render height (default: 2160)')
    parser.add_argument('--focal-length', type=float, default=45.0,
                        help='Camera focal length in mm (default: 45.0)')
    parser.add_argument('--flat-shading', action='store_true', help="Whether to use flat shading", default=False)
    
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
    args = parse_twisted_clynder_render_sliced_arguments()
    
    input_mesh = Path(args.input_mesh)
    output_image = Path(args.output_image)
    samples = args.samples
    resolution_x = args.resolution_x
    resolution_y = args.resolution_y
    focal_length = args.focal_length
    flat_shading = args.flat_shading
    
    if not input_mesh.exists():
        print(f"ERROR: Input mesh not found: {input_mesh}")
        sys.exit(1)
    
    # Use args for most settings
    nx = args.nx
    ny = args.ny
    nz = args.nz
    cx = args.cx
    cy = args.cy
    cz = args.cz
    clear_outer = args.clear_outer
    clear_inner = args.clear_inner
    
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
    print("Twisted Cylinder Renderer (Sliced)")
    print("=" * 50)
    print(f"Input mesh: {input_mesh}")
    print(f"Output image: {output_image}")
    print(f"Slicing plane: {nx}, {ny}, {nz}, {cx}, {cy}, {cz}")
    print(f"Clear outer: {clear_outer}")
    print(f"Clear inner: {clear_inner}")
    print(f"Samples: {samples}")
    print(f"Resolution: {resolution_x}x{resolution_y}")
    print(f"Focal length: {focal_length}mm")
    print(f"Flat shading: {flat_shading}")
    
    
    # ========================================
    # Camera settings (fixed)
    # ========================================
    print(f"\n{'='*50}")
    print("Camera settings (fixed)")
    print(f"{'='*50}")
    
    # Fixed camera position and rotation
    camera_location = (0, -0.12, 0)
    camera_rotation = (90, 0, 0)  # Euler rotation in degrees
    
    # ========================================
    # Light settings (fixed)
    # ========================================
    # Sun light with fixed rotation
    light_location = (0, 0, 0)
    light_rotation = (90, 0, 0)  # Euler rotation in degrees
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
    
    # bisect the mesh
    mesh = bisect_mesh(
        mesh,
        plane_co=(cx, cy, cz),
        plane_no=(nx, ny, nz),
        clear_outer=clear_outer,
        clear_inner=clear_inner,
    )
    if mesh is None:
        print("    ERROR: Failed to bisect mesh")
        sys.exit(1)

    # Shading
    if flat_shading:
        bpy.ops.object.shade_flat()
    else:
        bpy.ops.object.shade_smooth()
    
    # subdivide the mesh
    # bt.subdivision(mesh, level = 2)
    
    # Material
    setMat_doubleColorWire(mesh, meshColor_top, meshColor_bottom, AOStrength=ao_strength)
    
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
