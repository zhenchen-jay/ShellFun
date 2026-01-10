"""
Two-sided material with optional wireframe for BlenderToolbox.
Compatible with Blender 4.x

This material shows different colors for front and back faces,
which is useful for cloth rendering where you want to distinguish
the two sides.

Wireframe overlay is done via Wireframe modifier + 2nd material slot.

Usage:
    import blendertoolbox as bt
    from setMat_doubleColor_with_wireframe_modifier import setMat_doubleColor_with_wireframe_modifier

    meshColor_top = bt.colorObj((0.6, 0.6, 0.7, 1.0), 0.5, 1.0, 1.0, 0.0, 0.0)
    meshColor_bottom = bt.colorObj((0.67, 0.4, 0.95, 1.0), 0.5, 1.0, 1.0, 0.0, 0.0)

    setMat_doubleColor_with_wireframe_modifier(mesh, meshColor_top, meshColor_bottom, AOStrength=0.5, edgeThickness=0.001)
"""

import bpy


def _ensure_black_wire_material(name="WireframeBlack"):
    """
    Create (or reuse) a black Principled BSDF material for wireframe geometry.
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)

    mat.use_nodes = True
    mat.use_backface_culling = False

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    out = nodes.get("Material Output")
    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.name = "Material Output"

    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.name = "Principled BSDF"

    # Ensure connected
    if not out.inputs["Surface"].is_linked:
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Black, low specular
    bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.0

    return mat


def setMat_doubleColor_with_wireframe_modifier(mesh, meshColor_top, meshColor_bottom, AOStrength, edgeThickness=0):
    """
    Apply a two-sided material with optional wireframe overlay.

    Args:
        mesh: Blender mesh object
        meshColor_top: BlenderToolbox colorObj for front faces
        meshColor_bottom: BlenderToolbox colorObj for back faces
        AOStrength: Ambient occlusion strength (gamma value)
        edgeThickness: Wireframe edge thickness (0 = no wireframe), via Wireframe modifier
    """
    if mesh is None or mesh.type != "MESH":
        raise RuntimeError("setMat_doubleColor_with_wireframe_modifier: 'mesh' must be a mesh object")

    # Clear existing materials
    mesh.data.materials.clear()

    # Create base (surface) material
    mat = bpy.data.materials.new("MeshMaterial")
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    mat.use_backface_culling = False
    tree = mat.node_tree

    # Ensure nodes exist
    out = tree.nodes.get("Material Output")
    if out is None:
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        out.name = "Material Output"
        out.location = (650, 0)

    PRIN1 = tree.nodes.get("Principled BSDF")
    if PRIN1 is None:
        PRIN1 = tree.nodes.new("ShaderNodeBsdfPrincipled")
        PRIN1.name = "Principled BSDF"
        PRIN1.location = (0, 0)

    # Front face properties
    PRIN1.inputs["Roughness"].default_value = 0.7

    # Ambient Occlusion chain (front face)
    AO = tree.nodes.new("ShaderNodeAmbientOcclusion")
    AO.inputs["Distance"].default_value = 10.0
    AO.location = (-600, 300)

    GAMMA = tree.nodes.new("ShaderNodeGamma")
    GAMMA.inputs["Gamma"].default_value = AOStrength
    GAMMA.location = (-400, 200)

    MIXRGB = tree.nodes.new("ShaderNodeMixRGB")
    MIXRGB.blend_type = "MULTIPLY"
    MIXRGB.location = (-200, 300)

    HSVNode = tree.nodes.new("ShaderNodeHueSaturation")
    HSVNode.inputs["Color"].default_value = meshColor_top.RGBA
    HSVNode.inputs["Saturation"].default_value = meshColor_top.S
    HSVNode.inputs["Value"].default_value = meshColor_top.V
    HSVNode.inputs["Hue"].default_value = meshColor_top.H
    HSVNode.location = (-600, 100)

    BCNode = tree.nodes.new("ShaderNodeBrightContrast")
    BCNode.inputs["Bright"].default_value = meshColor_top.B
    BCNode.inputs["Contrast"].default_value = meshColor_top.C
    BCNode.location = (-400, 100)

    tree.links.new(HSVNode.outputs["Color"], BCNode.inputs["Color"])
    tree.links.new(BCNode.outputs["Color"], AO.inputs["Color"])
    tree.links.new(AO.outputs["Color"], MIXRGB.inputs["Color1"])
    tree.links.new(AO.outputs["AO"], GAMMA.inputs["Color"])
    tree.links.new(GAMMA.outputs["Color"], MIXRGB.inputs["Color2"])
    tree.links.new(MIXRGB.outputs["Color"], PRIN1.inputs["Base Color"])

    # Back face Principled
    PRIN2 = tree.nodes.new("ShaderNodeBsdfPrincipled")
    PRIN2.location = (0, -200)
    PRIN2.inputs["Base Color"].default_value = meshColor_bottom.RGBA
    PRIN2.inputs["Roughness"].default_value = 0.9
    if "Specular IOR Level" in PRIN2.inputs:
        PRIN2.inputs["Specular IOR Level"].default_value = 0.1
    elif "Specular" in PRIN2.inputs:
        PRIN2.inputs["Specular"].default_value = 0.1

    # Backfacing mix
    GEOM = tree.nodes.new("ShaderNodeNewGeometry")
    GEOM.location = (-200, -100)

    MIX1 = tree.nodes.new("ShaderNodeMixShader")
    MIX1.location = (400, 0)

    tree.links.new(GEOM.outputs["Backfacing"], MIX1.inputs[0])
    tree.links.new(PRIN1.outputs["BSDF"], MIX1.inputs[1])  # front
    tree.links.new(PRIN2.outputs["BSDF"], MIX1.inputs[2])  # back
    tree.links.new(MIX1.outputs["Shader"], out.inputs["Surface"])

    # ---- Optional wireframe overlay (same object, no solidify, offset=0) ----
    if edgeThickness and edgeThickness > 0:
        wire_mat = _ensure_black_wire_material("WireframeBlack")
        mesh.data.materials.append(wire_mat)  # slot 1

        wf = mesh.modifiers.get("Wireframe")
        if wf is None:
            wf = mesh.modifiers.new(name="Wireframe", type="WIREFRAME")

        wf.thickness = float(edgeThickness)
        wf.use_replace = False          # overlay on surface
        wf.use_even_offset = True
        wf.use_boundary = True
        wf.use_relative_offset = False
        wf.offset = 0.0                 # as requested
        wf.material_offset = 1          # use material slot 1 for wire geometry

    return mat
