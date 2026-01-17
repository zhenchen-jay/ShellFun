#!/usr/bin/env python3
"""
Compute number of vertices (#V) and edges (#E) of a triangle mesh.

Usage:
  python mesh_stats.py path/to/mesh.obj
"""

import argparse
import trimesh


def main():
    parser = argparse.ArgumentParser(description="Compute #Vertices and #Edges of a triangle mesh")
    parser.add_argument("mesh", help="Path to mesh file (obj, ply, stl, etc.)")
    args = parser.parse_args()

    # Load mesh
    mesh = trimesh.load(args.mesh, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Input file does not contain a triangle mesh")

    num_vertices = mesh.vertices.shape[0]
    num_edges = mesh.edges_unique.shape[0]

    print(f"#Vertices: {num_vertices}")
    print(f"#Edges:    {num_edges}")
    print(f"#DOFs:     {3 * num_vertices + num_edges}")
    print(f"#VDOFs:    {3 * num_vertices}")


if __name__ == "__main__":
    main()