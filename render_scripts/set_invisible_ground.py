# Copyright 2020 Hsueh-Ti Derek Liu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bpy
import numpy as np


def set_invisible_ground(location=(0, 0, 0), rotation_euler=(0, 0, 0), groundSize=100, shadowBrightness=0.7):
    # initialize a ground for shadow
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
    angle = (x, y, z)
    bpy.context.scene.cycles.film_transparent = True
    bpy.ops.mesh.primitive_plane_add(location=location, rotation=angle, size=groundSize)
    try:
        bpy.context.object.is_shadow_catcher = True  # for blender 3.X
    except:
        bpy.context.object.cycles.is_shadow_catcher = True  # for blender 2.X

    # set material
    ground = bpy.context.object
    mat = bpy.data.materials.new('MeshMaterial')
    ground.data.materials.append(mat)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes["Principled BSDF"].inputs['Transmission Weight'].default_value = shadowBrightness
