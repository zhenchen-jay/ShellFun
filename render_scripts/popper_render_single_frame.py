"""
Custom render script for popper simulation using BlenderToolbox (Single Frame).

Uses utility functions from render_meshes_bt.py

Usage:
    python popper_render_single_frame.py -- -i /path/to/mesh -o /path/to/output.png
    python popper_render_single_frame.py -- -i /path/to/mesh -o /path/to/output.png --flat-shading
    python popper_render_single_frame.py -- -i /path/to/mesh -o /path/to/output.png --vertex-file vertices.txt --point-size 0.005
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

def parse_vertex_file(vertex_file_path):
    """
    Parse vertex file to extract pinned and scripted vertex indices.
    Converts global indices to local indices using the Index Offset.
    
    Returns:
        dict with keys 'pinned' and 'scripted', each containing a list of local vertex indices
    """
    result = {'pinned': [], 'scripted': [], 'index_offset': 0}
    
    if not vertex_file_path or not Path(vertex_file_path).exists():
        return result
    
    with open(vertex_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the file
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Extract index offset
        if 'Index Offset:' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                result['index_offset'] = int(parts[1].strip())
        elif 'Pinned Vertex Global Indices' in line:
            current_section = 'pinned'
        elif 'Scripted Vertex Global Indices' in line:
            current_section = 'scripted'
        elif current_section and line and line[0].isdigit():
            # Parse the global indices
            global_indices = [int(x) for x in line.split()]
            result[current_section].extend(global_indices)
    
    # Convert global indices to local indices by subtracting the offset
    offset = result['index_offset']
    result['pinned'] = [idx - offset for idx in result['pinned'] if idx >= offset]
    result['scripted'] = [idx - offset for idx in result['scripted'] if idx >= offset]
    
    # Remove duplicates while preserving order
    result['pinned'] = list(dict.fromkeys(result['pinned']))
    result['scripted'] = list(dict.fromkeys(result['scripted']))
    
    return result

def draw_vertex_points(mesh_obj, vertex_indices, color, point_size=0.0002, name_prefix="vertex_points", is_first_frame=False):
    """
    Draw spheres at specified vertex positions.
    
    Args:
        mesh_obj: Blender mesh object
        vertex_indices: List of local vertex indices to draw
        color: RGB or RGBA tuple for the sphere color
        point_size: Radius of the spheres
        name_prefix: Prefix for the sphere object names
        is_first_frame: Whether this is the first frame of the animation
    
    Returns:
        List of sphere objects created
    """
    if not vertex_indices:
        return []
    
    # Get mesh data
    mesh = mesh_obj.data
    
    # Create spheres at vertex positions
    spheres = []
    for idx in vertex_indices:
        # Validate index
        if idx < 0 or idx >= len(mesh.vertices):
            print(f"    Warning: Vertex index {idx} out of range (mesh has {len(mesh.vertices)} vertices)")
            continue
        
        # Get vertex position in world space
        vert_local = mesh.vertices[idx].co
        vert_world = mesh_obj.matrix_world @ vert_local
        
        # Create sphere
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=point_size,
            location=vert_world,
            segments=16,
            ring_count=8
        )
        sphere = bpy.context.active_object
        sphere.name = f"{name_prefix}_{idx}"
        
        # CRITICAL: Make mesh data unique for each sphere
        sphere.data = sphere.data.copy()
        
        # Create a UNIQUE material for THIS sphere (no sharing!)
        mat_name = f"Material_{name_prefix}_{idx}"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        mat_nodes = mat.node_tree.nodes
        mat_nodes.clear()
        
        # Use Principled BSDF without emission for clean colors that respond to lighting
        node_bsdf = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        node_bsdf.inputs['Base Color'].default_value = (*color[:3], 1.0)
        node_bsdf.inputs['Metallic'].default_value = 0.0
        node_bsdf.inputs['Roughness'].default_value = 0.4
        node_bsdf.location = (0, 0)

        # only for the first frame of the animation
        if is_first_frame and name_prefix == "scripted":
            node_bsdf.inputs['Emission Color'].default_value = (*color[:3], 1.0)
            node_bsdf.inputs['Emission Strength'].default_value = 1.0  # make scripted points brighter
        
        node_output = mat_nodes.new(type='ShaderNodeOutputMaterial')
        node_output.location = (300, 0)
        mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
        
        # Clear any existing materials and apply the new unique material
        sphere.data.materials.clear()
        sphere.data.materials.append(mat)
        sphere.active_material_index = 0
        
        # Force Blender to update the scene
        bpy.context.view_layer.update()
        
        # Verify the material was applied correctly
        if len(sphere.data.materials) > 0 and sphere.data.materials[0] == mat:
            spheres.append(sphere)
        else:
            print(f"    ERROR: Failed to apply material to sphere {sphere.name}")
    
    print(f"    Created {len(spheres)} spheres with unique materials, color {color[:3]}")
    
    # Final scene update after all spheres created
    bpy.context.view_layer.update()
    
    return spheres

def parse_popper_render_single_frame_arguments():
    """Parse command line arguments for single frame popper rendering."""
    parser = argparse.ArgumentParser(description='Render popper mesh (single frame) using BlenderToolbox')
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
    parser.add_argument('--edge-thickness', type=float, help="Edge thickness (default: 0.0001)", default=0.0001)
    parser.add_argument('--vertex-file', type=str, default=None,
                        help='Path to vertex file with pinned/scripted vertex indices')
    parser.add_argument('--point-size', type=float, default=0.0002,
                        help='Size of vertex points (default: 0.0002)')
    parser.add_argument('--is-first-frame', action='store_true', default=False,
                        help='Whether this is the first frame of the animation')
    
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args([])
    
    return args

########################################################
# camera settings (from popper_render.py)
# Camera location (after transition): (-0.006675, 0.318642, 0.385257)
# Camera rotation (after transition): (-39.6, 0, 1.2)

# Sun light: 
# location: (0, 0.17, 0.20)
# rotation: (-30, 0, 0)
# strength: 4.0

########################################################

def main():
    """Custom main function for popper single frame rendering."""
    
    # ========================================
    # Parse arguments
    # ========================================
    args = parse_popper_render_single_frame_arguments()
    
    input_mesh = Path(args.input_mesh)
    output_image = Path(args.output_image)
    samples = args.samples
    resolution_x = args.resolution_x
    resolution_y = args.resolution_y
    exposure = args.exposure
    focal_length = args.focal_length
    flat_shading = args.flat_shading
    edge_thickness = args.edge_thickness
    vertex_file = args.vertex_file
    point_size = args.point_size
    is_first_frame = args.is_first_frame
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
    print("Popper Renderer (Single Frame)")
    print("=" * 50)
    print(f"Input mesh: {input_mesh}")
    print(f"Output image: {output_image}")
    print(f"Samples: {samples}")
    print(f"Resolution: {resolution_x}x{resolution_y}")
    print(f"Exposure: {exposure}")
    print(f"Focal length: {focal_length}mm")
    print(f"Flat shading: {flat_shading}")
    print(f"Edge thickness: {edge_thickness}")
    print(f"Vertex file: {vertex_file if vertex_file else 'None'}")
    print(f"Point size: {point_size}")
    print(f"Is first frame: {is_first_frame}")
    # ========================================
    # Parse vertex file if provided
    # ========================================
    vertices_data = None
    if vertex_file:
        vertex_file_path = Path(vertex_file)
        if vertex_file_path.exists():
            print(f"\n{'='*50}")
            print("Parsing vertex file")
            print(f"{'='*50}")
            vertices_data = parse_vertex_file(vertex_file)
            print(f"  Index offset: {vertices_data['index_offset']}")
            print(f"  Pinned vertices (local): {len(vertices_data['pinned'])}")
            print(f"  Scripted vertices (local): {len(vertices_data['scripted'])}")
        else:
            print(f"\nWarning: Vertex file not found: {vertex_file}")
    
    # ========================================
    # Camera settings (fixed)
    # ========================================
    print(f"\n{'='*50}")
    print("Camera settings (fixed)")
    print(f"{'='*50}")
    
    # Fixed camera position and rotation (after transition view from popper_render.py)
    camera_location = (-0.006675, 0.318642, 0.385257)
    camera_rotation = (-39.6, 0, 1.2)  # Euler rotation in degrees
    
    print(f"  Camera location: {camera_location}")
    print(f"  Camera rotation: {camera_rotation}")
    print(f"  Focal length: {focal_length}mm")
    
    # ========================================
    # Light settings (fixed)
    # ========================================
    # Sun light with fixed rotation (from popper_render.py)
    light_rotation = (-30, 0, 0)  # Euler rotation in degrees
    light_location = (0, 0.17, 0.20)
    light_strength = 4.0
    shadow_softness = 0.3
    
    print(f"\n  Sun light location: {light_location}")
    print(f"  Sun light rotation: {light_rotation}")
    print(f"  Sun light strength: {light_strength}")
    
    # Material color (custom, not from args)
    print(f"  Material color: {obj_color}")
    meshColor_top = bt.colorObj(obj_color, 0.5, 1.0, 1.0, 0.0, 0.0)
    meshColor_bottom = bt.colorObj(obj_color, 0.5, 1.0, 1.0, 0.0, 0.0)
    ao_strength = 0.5
    
    # ========================================
    # Render mesh
    # ========================================
    print(f"\n{'='*50}")
    print("Rendering mesh")
    print(f"{'='*50}")
    
    # Fresh scene
    bt.blenderInit(resolution_x, resolution_y, samples, exposure)
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
    
    # Draw vertex points if vertex file was provided
    if vertices_data:
        print(f"\n  Drawing vertex points...")

        pinned_set = set(vertices_data["pinned"])
        scripted_set = set(vertices_data["scripted"])
        overlap = pinned_set & scripted_set

        if overlap:
            print(f"  Removing {len(overlap)} overlapped vertices from PINNED "
                f"(scripted takes priority)")

        # Keep order, remove overlaps from pinned
        vertices_data["pinned"] = [i for i in vertices_data["pinned"] if i not in scripted_set]

        
        # Draw scripted vertices in blue
        if vertices_data['scripted']:
            scripted_color = (1.0, 0.0, 0.0, 1.0)  # Pure blue (R=0.0, G=0.0, B=1.0)
            print(f"    Scripted color (RGBA): {scripted_color}")
            scripted_spheres = draw_vertex_points(
                mesh, 
                vertices_data['scripted'], 
                scripted_color, 
                point_size=point_size,
                name_prefix="scripted",
                is_first_frame=is_first_frame
            )
            print(f"    Drew {len(scripted_spheres)} scripted sphere(s) in pure blue")
        
        # Draw pinned vertices in red
        if vertices_data['pinned']:
            pinned_color = (1.0, 1.0, 0.0, 1.0)  # Pure red (R=1.0, G=0.0, B=0.0)
            print(f"    Pinned color (RGBA): {pinned_color}")
            pinned_spheres = draw_vertex_points(
                mesh, 
                vertices_data['pinned'], 
                pinned_color, 
                point_size=2 * point_size,
                name_prefix="pinned",
                is_first_frame=is_first_frame
            )
            print(f"    Drew {len(pinned_spheres)} pinned sphere(s) in pure red")
    
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
