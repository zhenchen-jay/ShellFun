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

def setMat_metal_wrapper(mesh, color, metalVal=1.0, roughnessVal=0.25):
	"""
	Simple metal material.
	
	Args:
		mesh: The mesh object to apply material to
		color: RGBA tuple, e.g. (0.2, 0.5, 0.9, 1.0)
		metalVal: Metallic value (0-1)
		roughnessVal: Surface roughness (0=mirror, 1=matte)
	"""
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.clear()
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree
	
	# Get Principled BSDF
	principled = tree.nodes["Principled BSDF"]
	
	# Set properties directly
	principled.inputs['Base Color'].default_value = color
	principled.inputs['Metallic'].default_value = metalVal
	principled.inputs['Roughness'].default_value = roughnessVal
	
	# Specular settings
	principled.inputs['Specular IOR Level'].default_value = 0.5
	principled.inputs['Specular Tint'].default_value = (1.0, 1.0, 0.5, 1.0)  # Yellow tint
	principled.inputs['Anisotropic'].default_value = 1.0
	principled.inputs['Anisotropic Rotation'].default_value = 0.458
	
	return mat
