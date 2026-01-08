"""
Two-sided material with optional wireframe for BlenderToolbox.
Compatible with Blender 4.x

This material shows different colors for front and back faces,
which is useful for cloth rendering where you want to distinguish
the two sides.

Usage:
    import blendertoolbox as bt
    from setMat_doubleColorWire import setMat_doubleColorWire
    
    # Create color objects
    meshColor_top = bt.colorObj((0.6, 0.6, 0.7, 1.0), 0.5, 1.0, 1.0, 0.0, 0.0)
    meshColor_bottom = bt.colorObj((0.67, 0.4, 0.95, 1.0), 0.5, 1.0, 1.0, 0.0, 0.0)
    
    # Apply material
    setMat_doubleColorWire(mesh, meshColor_top, meshColor_bottom, AOStrength=0.5)
"""

import bpy


def setMat_doubleColorWire(mesh, meshColor_top, meshColor_bottom, AOStrength, edgeThickness=0):
    """
    Apply a two-sided material with optional wireframe overlay.
    
    Args:
        mesh: Blender mesh object
        meshColor_top: BlenderToolbox colorObj for front faces
        meshColor_bottom: BlenderToolbox colorObj for back faces  
        AOStrength: Ambient occlusion strength (gamma value)
        edgeThickness: Wireframe edge thickness (0 = no wireframe)
    """
    # Clear existing materials
    mesh.data.materials.clear()
    
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    mat.use_backface_culling = False
    tree = mat.node_tree

    # Get Principled BSDF node
    PRIN1 = tree.nodes["Principled BSDF"]
    
    # Set Principled BSDF properties (Blender 4.x compatible)
    PRIN1.inputs['Roughness'].default_value = 0.7
    # Sheen Tint is now a color in Blender 4.x, so we skip it or set to white
    # PRIN1.inputs['Sheen Tint'].default_value = (1.0, 1.0, 1.0, 1.0)  # Skip for compatibility

    # Add Ambient Occlusion nodes
    AO = tree.nodes.new('ShaderNodeAmbientOcclusion')
    AO.inputs["Distance"].default_value = 10.0
    AO.location = (-600, 300)
    
    GAMMA = tree.nodes.new('ShaderNodeGamma')
    GAMMA.inputs["Gamma"].default_value = AOStrength
    GAMMA.location = (-400, 200)
    
    MIXRGB = tree.nodes.new('ShaderNodeMixRGB')
    MIXRGB.blend_type = 'MULTIPLY'
    MIXRGB.location = (-200, 300)

    # Set color using Hue/Saturation node
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    HSVNode.inputs['Color'].default_value = meshColor_top.RGBA
    HSVNode.inputs['Saturation'].default_value = meshColor_top.S
    HSVNode.inputs['Value'].default_value = meshColor_top.V
    HSVNode.inputs['Hue'].default_value = meshColor_top.H
    HSVNode.location = (-600, 100)

    # Set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = meshColor_top.B
    BCNode.inputs['Contrast'].default_value = meshColor_top.C
    BCNode.location = (-400, 100)

    # Link color nodes for front face
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'], AO.inputs['Color'])
    tree.links.new(AO.outputs['Color'], MIXRGB.inputs['Color1'])
    tree.links.new(AO.outputs['AO'], GAMMA.inputs['Color'])
    tree.links.new(GAMMA.outputs['Color'], MIXRGB.inputs['Color2'])
    tree.links.new(MIXRGB.outputs['Color'], PRIN1.inputs['Base Color'])

    # Create backface material
    MIX1 = tree.nodes.new('ShaderNodeMixShader')
    MIX1.location = (400, 0)
    
    PRIN2 = tree.nodes.new('ShaderNodeBsdfPrincipled')
    PRIN2.location = (0, -200)
    PRIN2.inputs['Base Color'].default_value = meshColor_bottom.RGBA
    PRIN2.inputs['Roughness'].default_value = 0.9
    # Skip Sheen Tint (color in 4.x) and use Specular IOR Level instead of Specular
    if 'Specular IOR Level' in PRIN2.inputs:
        PRIN2.inputs['Specular IOR Level'].default_value = 0.1
    elif 'Specular' in PRIN2.inputs:
        PRIN2.inputs['Specular'].default_value = 0.1

    # Create geometry node for backface detection
    GEOM = tree.nodes.new('ShaderNodeNewGeometry')
    GEOM.location = (-200, -100)

    if edgeThickness > 0:
        # With wireframe overlay
        MIX2 = tree.nodes.new('ShaderNodeMixShader')
        MIX2.location = (200, 200)
        
        WIRE = tree.nodes.new('ShaderNodeWireframe')
        WIRE.inputs['Size'].default_value = edgeThickness
        WIRE.location = (0, 300)

        MIX3 = tree.nodes.new('ShaderNodeMixShader')
        MIX3.location = (200, -200)
        
        WIRE1 = tree.nodes.new('ShaderNodeWireframe')
        WIRE1.inputs['Size'].default_value = edgeThickness
        WIRE1.location = (0, -300)

        # Back face with wireframe
        tree.links.new(PRIN2.outputs[0], MIX3.inputs[1])
        tree.links.new(WIRE1.outputs[0], MIX3.inputs[0])
        tree.links.new(MIX3.outputs[0], MIX1.inputs[2])

        # Front face with wireframe
        tree.links.new(PRIN1.outputs[0], MIX2.inputs[1])
        tree.links.new(WIRE.outputs[0], MIX2.inputs[0])
        tree.links.new(MIX2.outputs[0], MIX1.inputs[1])

        # Mix based on backfacing
        tree.links.new(GEOM.outputs['Backfacing'], MIX1.inputs[0])
        tree.links.new(MIX1.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
    else:
        # Without wireframe - simple front/back mixing
        tree.links.new(GEOM.outputs['Backfacing'], MIX1.inputs[0])
        tree.links.new(PRIN1.outputs[0], MIX1.inputs[1])  # Front face
        tree.links.new(PRIN2.outputs[0], MIX1.inputs[2])  # Back face
        tree.links.new(MIX1.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
