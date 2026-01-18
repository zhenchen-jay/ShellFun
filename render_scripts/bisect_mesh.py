import bpy
import bmesh
from mathutils import Vector

def bisect_mesh(obj, plane_co=(0, 0, 0), plane_no=(1, 0, 0), clear_outer=False, clear_inner=False):
    """
    Bisect a mesh object with a plane.
    
    Args:
        obj: The mesh object to bisect
        plane_co: Point on the bisect plane (x, y, z)
        plane_no: Normal vector of the plane (x, y, z)
        clear_outer: If True, delete geometry on the positive side of the plane
        clear_inner: If True, delete geometry on the negative side of the plane
    
    Returns:
        bpy.types.Object | None: The original object if successful, otherwise None
    """
    # Validate input
    if not obj or obj.type != 'MESH':
        print("Error: Object must be a mesh")
        return None
    
    # Store original mode
    original_mode = obj.mode
    
    # Enter edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Create bmesh from the mesh
    bm = bmesh.from_edit_mesh(obj.data)
    
    # Convert tuples to Vectors
    plane_co = Vector(plane_co)
    plane_no = Vector(plane_no)
    
    # Perform bisect operation
    bmesh.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        plane_co=plane_co,
        plane_no=plane_no,
        clear_outer=clear_outer,
        clear_inner=clear_inner
    )
    
    # Update the mesh
    bmesh.update_edit_mesh(obj.data)
    
    # Return to original mode
    bpy.ops.object.mode_set(mode=original_mode)
    
    return obj