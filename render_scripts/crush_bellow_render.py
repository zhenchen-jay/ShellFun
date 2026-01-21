"""
Custom render script for crush bellows simulation using BlenderToolbox.

Uses utility functions from render_meshes_bt.py

Usage:
    python crush_bellow_render.py -- -i /path/to/meshes -o /path/to/output
    python crush_bellow_render.py -- -i /path/to/meshes -o /path/to/output --flat-shading
    python crush_bellow_render.py -- -i /path/to/meshes -o /path/to/output --video --fps 30
"""

import bpy
import sys
import math
import argparse
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
    split_bellows_cone_sphere,
    setup_gpu_rendering,
)
from setMat_metal_wrapper import setMat_metal_wrapper
from setMat_doubleColor_with_wireframe_modifier import setMat_doubleColor_with_wireframe_modifier
from setLight_sun_with_strength import setLight_sun_with_strength
from setup_world_background import setup_world_background
from setup_world import setup_world, get_blender_hdri

def parse_crush_cone_arguments():
    """Parse command line arguments with video export option."""
    parser = argparse.ArgumentParser(description='Render crush bellows meshes using BlenderToolbox')
    parser.add_argument('-i', '--input-folder', type=str, required=True,
                        help='Folder containing PLY/OBJ files')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Folder to save rendered images')
    parser.add_argument('--samples', type=int, default=300,
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
    parser.add_argument('--crop', action='store_true', default=False,
                        help='Crop images to content')
    parser.add_argument('--edge-thickness', type=float, default=0,
                        help='Edge thickness (default: 0)')
    # Video export options
    parser.add_argument('--video', action='store_true', default=True,
                        help='Export video from rendered images (default: enabled)')
    parser.add_argument('--no-video', action='store_true', default=False,
                        help='Disable video export')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frame rate (default: 30)')
    parser.add_argument('--video-name', type=str, default='animation.mp4',
                        help='Output video filename (default: animation.mp4)')
    parser.add_argument('--preprocess-done', action='store_true', default=False,
                        help='Preprocess done flag (default: False)')
    
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
    """Custom main function for crush bellows rendering."""
    
    # ========================================
    # Parse arguments
    # ========================================
    args = parse_crush_cone_arguments()
    
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
    method_name = input_folder.parent.stem
    obj_color = get_color_for_method(method_name)
    cone_color = (0.5, 0.5, 0.5, 1.0)
    sphere_color = (199/255.0,175/255.0,147/255.0,255/255.0) 
    
    print(f"Method name: {method_name}, Object color: {obj_color}, Cone color: {cone_color}, Sphere color: {sphere_color}")

    # ========================================
    # World settings (auto-detect Blender HDRI)
    # ========================================
    # world_path = get_blender_hdri("forest")  # Options: forest, city, courtyard, interior, night, studio, sunrise, sunset
    

   
    # Mesh transform
    mesh_location = (0, 0, 0)
    mesh_rotation = (90, 0, 0)  # Y-up to Z-up
    if method_name in ["Discrete_Hinge_Bending_Tan"]:
        mesh_rotation = (90, 180, 0)
    mesh_scale = (1, 1, 1)
    
    # ========================================
    # Print settings
    # ========================================
    print("\n" + "=" * 50)
    print("Crush Bellows Renderer")
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
    for ext in ['.ply', '.obj']:
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
    # Preprocess meshes:
    # 1. For each frame, load the mesh, then split the mesh into connected components with meshlab
    # 2. It will have 3 connected components: 
    #    - the bellow mesh, which has the largest vertex count
    #    - the cone mesh, 
    #    - the sphere mesh
    # 3. The cone mesh's bbox is larger than the sphere mesh's. We should use this to distinguish the two meshes.
    # 4. Save all the meshes as new mesh files with the name {mesh_file.stem}_bellows.ply, {mesh_file.stem}_cone.ply, {mesh_file.stem}_sphere.ply
    # ========================================
    # create a new folder to save the preprocessed meshes (with the same name as the input folder)
    preprocessed_folder = input_folder / "preprocessed"
    preprocessed_folder.mkdir(parents=True, exist_ok=True)
    
    bellows_mesh_files = []
    cone_mesh_files = []
    sphere_mesh_files = []
    
    for i, mesh_file in enumerate(mesh_files):
        print(f"  [{i+1}/{len(mesh_files)}] Preprocessing: {mesh_file.name}")
        
        # Define output paths
        bellows_path = preprocessed_folder / f"{mesh_file.stem}_bellows.ply"
        cone_path = preprocessed_folder / f"{mesh_file.stem}_cone.ply"
        sphere_path = preprocessed_folder / f"{mesh_file.stem}_sphere.ply"

        if args.preprocess_done:
            bellows_mesh_files.append(preprocessed_folder / f"{mesh_file.stem}_bellows.ply")
            cone_mesh_files.append(preprocessed_folder / f"{mesh_file.stem}_cone.ply")
            sphere_mesh_files.append(preprocessed_folder / f"{mesh_file.stem}_sphere.ply")
            continue
        
        # Split mesh into 3 components
        result = split_bellows_cone_sphere(
            str(mesh_file), 
            str(bellows_path),
            str(cone_path),
            str(sphere_path)
        )
        
        if result:
            print(f"    Bellows: {result['bellows']['vertices']} vertices, bbox={result['bellows']['bbox_size']:.4f}")
            print(f"    Cone:    {result['cone']['vertices']} vertices, bbox={result['cone']['bbox_size']:.4f}")
            print(f"    Sphere:  {result['sphere']['vertices']} vertices, bbox={result['sphere']['bbox_size']:.4f}")
            
            bellows_mesh_files.append(bellows_path)
            cone_mesh_files.append(cone_path)
            sphere_mesh_files.append(sphere_path)
        else:
            print(f"    ERROR: Failed to split mesh")

    tmp_dir = output_folder / '_tmp_converted'
    
    # ========================================
    # Camera settings (fixed)
    # ========================================
    print(f"\n{'='*50}")
    print("Camera settings (fixed)")
    print(f"{'='*50}")
    
    # Fixed camera position and rotation
    camera_location = (0.624864, -0.813431, 0.436298)
    camera_rotation = (-90.873, 180, 203.77)  # Euler rotation in degrees
    
    print(f"  Camera location: {camera_location}")
    print(f"  Camera rotation: {camera_rotation}")
    print(f"  Focal length: {focal_length}mm")
    
    # ========================================
    # Light settings (fixed)
    # ========================================
    # Sun light with fixed rotation
    # light_location = (-0.002021, -0.070422, 0.139655)  # Euler rotation in degrees
    # light_rotation = (14.415, 3.93116, -31.1833)
    light_location = (0.156284, -0.202269, 0.456346)  # Euler rotation in degrees
    light_rotation = (-87.4471, 30, -165.375)
    light_strength = 3.0
    shadow_softness = 0.3
    
    print(f"\n  Sun light rotation: {light_rotation}")
    print(f"  Sun light strength: {light_strength}")
    
    # Material color (custom, not from args)
    print(f"  Material color: {obj_color}")
    
    # ========================================
    # PASS 2: Render each mesh
    # ========================================
    print(f"\n{'='*50}")
    print("Pass 2: Rendering each mesh")
    print(f"{'='*50}")
    
    rendered_paths = []
    
    for i, bellows_mesh_file in enumerate(bellows_mesh_files):
        print(f"\n  [{i+1}/{len(bellows_mesh_files)}] Rendering: {bellows_mesh_file.name}")
        
        # Fresh scene
        bt.blenderInit(resolution_x, resolution_y, samples, exposure)
        setup_gpu_rendering()
        
        # Load mesh (with fallback to PyMeshLab conversion for problematic PLY files)
        mesh = load_mesh_with_fallback(bt, bellows_mesh_file, mesh_location, mesh_rotation, mesh_scale, tmp_dir)
        if mesh is None:
            print(f"    ERROR: Failed to load {bellows_mesh_file.name}, skipping...")
            continue
        
        # Shading
        if flat_shading:
            bpy.ops.object.shade_flat()
        else:
            bpy.ops.object.shade_smooth()
        
        # Material
        # setMat_metal_wrapper(mesh, obj_color, metal_val, roughness_val)
        meshColor_top = bt.colorObj(obj_color, 0.5, 1.0, 1.0, 0.0, 0.0)
        meshColor_bottom = bt.colorObj(obj_color, 0.5, 1.0, 1.0, 0.0, 0.0)
        setMat_doubleColor_with_wireframe_modifier(mesh, meshColor_top, meshColor_bottom, AOStrength=1.0, edgeThickness=args.edge_thickness)

        # setup_world(world_path=world_path, world_name="World", strength=1.0, make_film_transparent=True, use_existing_world=True, set_as_scene_world=True)

        ########################################################
        # Cone mesh
        ########################################################
        cone_mesh_file = cone_mesh_files[i]
        cone_mesh = load_mesh_with_fallback(bt, cone_mesh_file, mesh_location, mesh_rotation, mesh_scale, tmp_dir)
        if cone_mesh is None:
            print(f"    ERROR: Failed to load {cone_mesh_file.name}, skipping...")
            continue
        # bt.subdivision(cone_mesh, level = 2)

        # Material
        metal_val = 1.0
        roughness_val = 0.3
        # setMat_metal_wrapper(cone_mesh, cone_color, metal_val, roughness_val)
        cone_meshColor_top = bt.colorObj(cone_color, 0.5, 1.0, 1.0, 0.0, 0.0)
        cone_meshColor_bottom = bt.colorObj(cone_color, 0.5, 1.0, 1.0, 0.0, 0.0)
        setMat_doubleColor_with_wireframe_modifier(cone_mesh, cone_meshColor_top, cone_meshColor_bottom, AOStrength=1.0, edgeThickness=0)
        # roughness = 0.2


        ########################################################
        # Sphere mesh
        ########################################################
        sphere_mesh_file = sphere_mesh_files[i]
        sphere_mesh = load_mesh_with_fallback(bt, sphere_mesh_file, mesh_location, mesh_rotation, mesh_scale, tmp_dir)
        if sphere_mesh is None:
            print(f"    ERROR: Failed to load {sphere_mesh_file.name}, skipping...")
            continue
        bt.subdivision(sphere_mesh, level = 2)
        
        # Material
        # setMat_metal_wrapper(sphere_mesh, sphere_color, metal_val, roughness_val)
        # sphere_meshColor_top = bt.colorObj(sphere_color, 0.5, 1.0, 1.0, 0.0, 0.0)
        # sphere_meshColor_bottom = bt.colorObj(sphere_color, 0.5, 1.0, 1.0, 0.0, 0.0)
        # setMat_doubleColor_with_wireframe_modifier(sphere_mesh, sphere_meshColor_top, sphere_meshColor_bottom, AOStrength=1.0, edgeThickness=args.edge_thickness)
        noiseScale = 80
        meshColor = bt.colorObj(sphere_color, 0.5, 1.4, 1.0, 0.0, 1.0)
        AOStrength = 1.0
        distortion = 0
        bt.setMat_stone(sphere_mesh, meshColor, noiseScale, distortion, AOStrength)


        # Camera (fixed position and rotation using direct Blender API)
        bpy.ops.object.camera_add(location=camera_location)
        cam = bpy.context.object
        cam.rotation_euler = tuple(math.radians(a) for a in camera_rotation)
        cam.data.lens = focal_length # for zoom out view, we need to set this to 25
        bpy.context.scene.camera = cam
        
        # Lighting (fixed rotation) - using direct Blender API for Blender 4.x compatibility
        # bt.invisibleGround(shadowBrightness=0.9, location=(0, 0, ground_z))

        # Sun light
        sun_light = setLight_sun_with_strength(location=light_location, rotation_euler=light_rotation, strength=light_strength, shadow_soft_size=shadow_softness)

        # setup_world(world_path=world_path, world_name="World", strength=1.0, make_film_transparent=True, use_existing_world=True, set_as_scene_world=True)
        bt.setLight_ambient(color=(0.2, 0.2, 0.2, 1))
        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')
        
        # Save blend for first mesh
        if i == 100:
            blend_path = output_folder / "scene.blend"
            bpy.ops.wm.save_mainfile(filepath=str(blend_path))
            print(f"    Saved blend: {blend_path}")
        
        # Render
        suffix = "_flat" if flat_shading else ""
        output_path = output_folder / f"{bellows_mesh_file.stem}{suffix}.png"
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
        import shutil
        shutil.rmtree(tmp_dir)
        print(f"\n  Cleaned up temp folder: {tmp_dir}")
    
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
