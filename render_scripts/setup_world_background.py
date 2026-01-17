import bpy

def setup_world_background(color=(1.0, 1.0, 1.0, 1.0), strength=0.2, world_name="World"):
    scene = bpy.context.scene

    # Get or create the world
    world = bpy.data.worlds.get(world_name)
    if world is None:
        world = bpy.data.worlds.new(world_name)
    scene.world = world

    # Ensure nodes are enabled
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    # Find or create World Output
    out = nodes.get("World Output")
    if out is None:
        out = nodes.new(type="ShaderNodeOutputWorld")
        out.name = "World Output"
        out.location = (300, 0)

    # Find or create Background node
    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new(type="ShaderNodeBackground")
        bg.name = "Background"
        bg.location = (0, 0)

    # Set color & strength
    bg.inputs["Color"].default_value = color
    bg.inputs["Strength"].default_value = strength

    # Ensure Background -> World Output link
    # Remove existing links into World Output Surface
    for l in list(out.inputs["Surface"].links):
        links.remove(l)
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    return world