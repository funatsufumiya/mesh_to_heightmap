import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import os
from numba import jit, prange

@jit(nopython=True, parallel=True)
def generate_mesh_data(heightmap, scale_z, height_axis, plane_axes):
    height, width = heightmap.shape
    vertices = np.zeros((height * width, 3))
    triangles = np.zeros(((height-1) * (width-1) * 2, 3), dtype=np.int32)
    
    for y in prange(height):
        for x in range(width):
            idx = y * width + x
            point = np.zeros(3)
            point[plane_axes[0]] = x / (width - 1)
            point[plane_axes[1]] = y / (height - 1)
            point[height_axis] = heightmap[y, x] * scale_z
            vertices[idx] = point
    
    triangle_idx = 0
    for y in prange(height - 1):
        for x in range(width - 1):
            v0 = y * width + x
            v1 = v0 + 1
            v2 = (y + 1) * width + x
            v3 = v2 + 1
            
            triangles[triangle_idx] = [v0, v2, v1]
            triangles[triangle_idx + 1] = [v1, v2, v3]
            triangle_idx += 2
    
    return vertices, triangles

def heightmap_to_mesh(heightmap, scale_z=1.0, plane_axis='z'):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    height_axis = axis_map[plane_axis.lower()]
    plane_axes = np.array([i for i in range(3) if i != height_axis])
    
    vertices, triangles = generate_mesh_data(heightmap, scale_z, height_axis, plane_axes)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    return mesh

def main():
    parser = argparse.ArgumentParser(description='Convert heightmap to 3D mesh')
    parser.add_argument('input_heightmap', help='Input heightmap image path')
    parser.add_argument('output_mesh', help='Output mesh path')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Height scale factor (default: 1.0)')
    parser.add_argument('--format', choices=['obj', 'ply', 'stl'], default='obj',
                        help='Output format (default: ply)')
    parser.add_argument('--axis', choices=['x', 'y', 'z', '-x', '-y', '-z'], 
                        default='z', help='Height axis (default: z)')
    parser.add_argument('--invert', action='store_true',
                        help='Invert height values')
    parser.add_argument('--resolution', type=int, default=2048,
                        help='Output mesh resolution (default: 2048)')
    parser.add_argument('--max-vertices', type=int, default=100000,
                        help='Maximum number of vertices (default: 100000)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify mesh to reduce vertex count')

    args = parser.parse_args()

    img = Image.open(args.input_heightmap).convert('L')
    img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
    heightmap = np.array(img) / 255.0
    
    if args.invert:
        heightmap = 1.0 - heightmap

    plane_axis = args.axis.lower().replace('-', '')
    reverse = args.axis.startswith('-')
    if reverse:
        heightmap = 1.0 - heightmap

    mesh = heightmap_to_mesh(heightmap, args.scale, plane_axis)

    if args.simplify:
        mesh = mesh.simplify_quadric_decimation(args.max_vertices)
        mesh.compute_vertex_normals()

    output_file = f"{args.output_mesh}.{args.format}"
    o3d.io.write_triangle_mesh(output_file, mesh)
    print(f"Mesh saved to: {output_file}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")

if __name__ == "__main__":
    main()
