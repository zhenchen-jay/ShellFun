import bpy
import bmesh
from mathutils import Matrix

def create_boundary_wire_object(
    mesh_obj: bpy.types.Object,
    name_suffix: str = "_boundary",
    radius: float = 0.0001,
    wire_material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """
    Create a new object containing only boundary edges of mesh_obj,
    convert it to a Curve, and give it thickness via bevel radius.

    Boundary edges are edges with exactly one adjacent face.
    """
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError("create_boundary_wire_object: mesh_obj must be a mesh object")

    # Build a new mesh that contains only boundary edges
    src_me = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(src_me)

    # Identify boundary edges (exactly 1 linked face)
    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]

    # Mark keep set
    keep_edges = set(boundary_edges)
    keep_verts = set()
    for e in keep_edges:
        keep_verts.add(e.verts[0])
        keep_verts.add(e.verts[1])

    # Delete all faces
    bmesh.ops.delete(bm, geom=bm.faces[:], context='FACES_ONLY')

    # Delete edges that are not boundary
    non_boundary_edges = [e for e in bm.edges if e not in keep_edges]
    if non_boundary_edges:
        bmesh.ops.delete(bm, geom=non_boundary_edges, context='EDGES')

    # Delete verts not used
    unused_verts = [v for v in bm.verts if v not in keep_verts]
    if unused_verts:
        bmesh.ops.delete(bm, geom=unused_verts, context='VERTS')

    # Write new mesh datablock
    new_me = bpy.data.meshes.new(mesh_obj.name + name_suffix + "_mesh")
    bm.to_mesh(new_me)
    bm.free()

    # Create new object
    new_obj = bpy.data.objects.new(mesh_obj.name + name_suffix, new_me)
    bpy.context.scene.collection.objects.link(new_obj)

    # Match transform
    new_obj.matrix_world = mesh_obj.matrix_world.copy()

    # Convert edge mesh to curve so we can bevel for thickness
    bpy.ops.object.select_all(action='DESELECT')
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj
    bpy.ops.object.convert(target='CURVE')

    # Set curve thickness
    curve = new_obj.data
    curve.dimensions = '3D'
    curve.fill_mode = 'FULL'
    curve.bevel_depth = float(radius)  # radius in Blender units
    curve.bevel_resolution = 2

    # Assign material
    if wire_material is not None:
        new_obj.data.materials.clear()
        new_obj.data.materials.append(wire_material)

    return new_obj
