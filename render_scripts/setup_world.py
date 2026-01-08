import bpy
from pathlib import Path


def get_blender_hdri_folder():
    """
    Automatically find Blender's built-in HDRI/world folder.
    Works across different Blender versions and OS.
    
    Returns:
        Path object to the studiolights/world folder, or None if not found.
    """
    possible_paths = []
    version = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
    
    # Method 1: Use bpy.utils.resource_path (most reliable)
    try:
        local_path = Path(bpy.utils.resource_path('LOCAL'))
        possible_paths.append(local_path / "datafiles" / "studiolights" / "world")
    except:
        pass
    
    # Method 2: From binary path
    blender_path = Path(bpy.app.binary_path)
    
    # macOS pattern: /Applications/Blender.app/Contents/MacOS/Blender
    # Need to go to: /Applications/Blender.app/Contents/Resources/X.X/datafiles/studiolights/world
    if "Blender.app" in str(blender_path):
        app_contents = blender_path.parent.parent  # Go up from MacOS to Contents
        resources = app_contents / "Resources"
        # Find version folder (e.g., 4.1, 4.2)
        if resources.exists():
            for item in resources.iterdir():
                if item.is_dir() and item.name[0].isdigit():
                    world_path = item / "datafiles" / "studiolights" / "world"
                    possible_paths.append(world_path)
    
    # Linux/Windows pattern - look relative to binary
    blender_dir = blender_path.parent
    
    # Try version folder in same directory
    possible_paths.append(blender_dir / version / "datafiles" / "studiolights" / "world")
    possible_paths.append(blender_dir / "datafiles" / "studiolights" / "world")
    
    # Try share folder (Linux)
    possible_paths.append(Path(f"/usr/share/blender/{version}/datafiles/studiolights/world"))
    
    # Debug: print what we're checking
    for path in possible_paths:
        if path.exists():
            return path
    
    # If nothing found, print debug info
    print(f"Warning: Could not find Blender HDRI folder")
    print(f"  Blender binary: {blender_path}")
    print(f"  Version: {version}")
    print(f"  Checked paths:")
    for p in possible_paths:
        print(f"    - {p} (exists: {p.exists()})")
    
    return None


def list_blender_hdris():
    """
    List all available built-in HDRI files in Blender.
    
    Returns:
        List of (name, full_path) tuples for available HDRIs.
    """
    folder = get_blender_hdri_folder()
    if folder is None:
        return []
    
    hdris = []
    for ext in ['*.exr', '*.hdr', '*.png', '*.jpg']:
        for f in folder.glob(ext):
            hdris.append((f.stem, str(f)))
    
    return sorted(hdris, key=lambda x: x[0])


def get_blender_hdri(name=None):
    """
    Get path to a Blender built-in HDRI by name.
    
    Args:
        name: HDRI name (without extension), e.g., 'forest', 'city', 'courtyard'.
              If None, returns the first available HDRI.
    
    Returns:
        Full path to the HDRI file, or None if not found.
    
    Available HDRIs (may vary by Blender version):
        - forest
        - city
        - courtyard
        - interior
        - night
        - studio
        - sunrise
        - sunset
    """
    folder = get_blender_hdri_folder()
    if folder is None:
        print("Warning: Could not find Blender HDRI folder")
        return None
    
    if name is None:
        # Return first available
        hdris = list_blender_hdris()
        if hdris:
            return hdris[0][1]
        return None
    
    # Search for the named HDRI
    for ext in ['.exr', '.hdr', '.png', '.jpg']:
        path = folder / f"{name}{ext}"
        if path.exists():
            return str(path)
    
    # Try case-insensitive search
    for f in folder.iterdir():
        if f.stem.lower() == name.lower():
            return str(f)
    
    print(f"Warning: HDRI '{name}' not found in {folder}")
    print(f"Available: {[h[0] for h in list_blender_hdris()]}")
    return None


def setup_world(
    world_path: str,
    world_name: str = "World",
    strength: float = 1.0,
    make_film_transparent: bool = True,
    use_existing_world: bool = True,
    set_as_scene_world: bool = True,
):
    """
    Create/modify a World with:
        Environment Texture -> Background -> World Output

    Parameters
    ----------
    world_path : str
        Path to .hdr/.exr/.png/.jpg file.
    world_name : str
        Name of the World datablock to create/use.
    strength : float
        Background strength.
    make_film_transparent : bool
        If True, sets Render > Film > Transparent (alpha background).
    use_existing_world : bool
        If True and scene already has a world, modify it instead of creating a new one.
    set_as_scene_world : bool
        If True, assigns the world to bpy.context.scene.world.
    """
    # Check for None world_path
    if world_path is None:
        raise ValueError(
            "world_path is None. Could not find HDRI file.\n"
            "Try specifying the full path manually, or check available HDRIs with list_blender_hdris()"
        )
    
    scene = bpy.context.scene

    # Choose or create world
    if use_existing_world and scene.world is not None:
        world = scene.world
    else:
        world = bpy.data.worlds.get(world_name) or bpy.data.worlds.new(world_name)

    if set_as_scene_world:
        scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    # Clear existing nodes (optional; matches your simple graph)
    nodes.clear()

    # Create nodes
    node_env = nodes.new(type="ShaderNodeTexEnvironment")
    node_env.location = (-500, 0)

    node_bg = nodes.new(type="ShaderNodeBackground")
    node_bg.location = (-200, 0)
    node_bg.inputs["Strength"].default_value = strength

    node_out = nodes.new(type="ShaderNodeOutputWorld")
    node_out.location = (200, 0)

    # Load image
    try:
        img = bpy.data.images.load(world_path, check_existing=True)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load world: {world_path}\n{e}")

    node_env.image = img

    # Match your screenshot settings
    # (Good defaults for HDRIs)
    node_env.interpolation = 'Linear'
    node_env.projection = 'EQUIRECTANGULAR'

    # Link: Env Color -> Background Color -> World Surface
    links.new(node_env.outputs["Color"], node_bg.inputs["Color"])
    links.new(node_bg.outputs["Background"], node_out.inputs["Surface"])

    # Optional: transparent film (alpha background)
    if make_film_transparent:
        scene.render.film_transparent = True

    return world