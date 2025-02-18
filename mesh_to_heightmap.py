import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
from stl import mesh as stl_mesh
import os

import numba
from numba import jit

@jit(nopython=True, parallel=True)
def process_triangles(vertices, triangles, min_bound, max_bound, resolution, height_axis, plane_axes, reverse):
    pixel_size_x = (max_bound[plane_axes[0]] - min_bound[plane_axes[0]]) / resolution
    pixel_size_y = (max_bound[plane_axes[1]] - min_bound[plane_axes[1]]) / resolution
    
    heightmap = np.zeros((resolution, resolution))
    
    for i in numba.prange(len(triangles)):
        triangle = triangles[i]
        triangle_vertices = vertices[triangle]
        
        tri_min = np.array([
            min(triangle_vertices[0][j], triangle_vertices[1][j], triangle_vertices[2][j]) 
            for j in range(3)
        ])
        tri_max = np.array([
            max(triangle_vertices[0][j], triangle_vertices[1][j], triangle_vertices[2][j]) 
            for j in range(3)
        ])
        
        x_start = max(0, int((tri_min[plane_axes[0]] - min_bound[plane_axes[0]]) / pixel_size_x))
        x_end = min(resolution-1, int((tri_max[plane_axes[0]] - min_bound[plane_axes[0]]) / pixel_size_x) + 1)
        y_start = max(0, int((tri_min[plane_axes[1]] - min_bound[plane_axes[1]]) / pixel_size_y))
        y_end = min(resolution-1, int((tri_max[plane_axes[1]] - min_bound[plane_axes[1]]) / pixel_size_y) + 1)
        
        v1 = triangle_vertices[1] - triangle_vertices[0]
        v2 = triangle_vertices[2] - triangle_vertices[0]
        normal = np.cross(v1, v2)
        
        if abs(normal[height_axis]) > 1e-6:
            for y in range(y_start, y_end + 1):
                for x in range(x_start, x_end + 1):
                    px = min_bound[plane_axes[0]] + (x + 0.5) * pixel_size_x
                    py = min_bound[plane_axes[1]] + (y + 0.5) * pixel_size_y
                    
                    t = -(normal[plane_axes[0]] * (px - triangle_vertices[0][plane_axes[0]]) +
                         normal[plane_axes[1]] * (py - triangle_vertices[0][plane_axes[1]])) / normal[height_axis]
                    height = triangle_vertices[0][height_axis] + t
                    
                    if reverse:
                        height = max_bound[height_axis] - height + min_bound[height_axis]
                    
                    heightmap[y, x] = max(heightmap[y, x], height)
    
    return heightmap


def mesh_to_heightmap(mesh, resolution=2048, plane_axis='z', reverse=False):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    height_axis = axis_map[plane_axis.lower()]
    plane_axes = np.array([i for i in range(3) if i != height_axis])
    
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    
    return process_triangles(vertices, triangles, min_bound, max_bound, resolution, height_axis, plane_axes, reverse)

def main():
    parser = argparse.ArgumentParser(description='Convert 3D mesh to heightmap')
    parser.add_argument('input_mesh', help='Input mesh file path (obj, ply, etc.)')
    parser.add_argument('output_path', help='Output heightmap path')
    parser.add_argument('--resolution', type=int, default=4096,
                        help='Output resolution (default: 4096)')
    parser.add_argument('--format', choices=['png', 'jpg', 'tiff'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--colormap', default='gray',
                        help='Matplotlib colormap (default: gray)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize height values to 0-1 range')
    parser.add_argument('--axis', choices=['x', 'y', 'z', '-x', '-y', '-z'], 
                        default='z', help='Height axis (default: z)')

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_mesh)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input mesh file not found: {input_path}")

    mesh = o3d.io.read_triangle_mesh(args.input_mesh)

     # Validate mesh data
    if not mesh.has_vertices():
        # If Open3D fails, try using numpy-stl as fallback for STL files
        if args.input_mesh.lower().endswith('.stl'):
            stl_data = stl_mesh.Mesh.from_file(args.input_mesh)
            vertices = stl_data.vectors.reshape(-1, 3)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
        else:
            raise ValueError("Failed to load mesh or mesh has no vertices")

    
    # 軸の向きを処理
    plane_axis = args.axis.lower().replace('-', '')
    reverse = args.axis.startswith('-')
    
    heightmap = mesh_to_heightmap(mesh, 
                                resolution=args.resolution,
                                plane_axis=plane_axis,
                                reverse=reverse)

    if args.normalize:
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

    output_file = f"{args.output_path}.{args.format}"
    
    plt.imsave(output_file, heightmap, cmap=args.colormap)
    print(f"Heightmap saved to: {output_file}")

if __name__ == '__main__':
    main()