import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse

def mesh_to_heightmap(mesh, resolution=2048):
    vertices = np.asarray(mesh.vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    
    x = np.linspace(min_bound[0], max_bound[0], resolution)
    y = np.linspace(min_bound[1], max_bound[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    heightmap = np.zeros((resolution, resolution))
    
    for i in range(len(vertices)):
        x_idx = int((vertices[i][0] - min_bound[0]) / (max_bound[0] - min_bound[0]) * (resolution-1))
        y_idx = int((vertices[i][1] - min_bound[1]) / (max_bound[1] - min_bound[1]) * (resolution-1))
        if 0 <= x_idx < resolution and 0 <= y_idx < resolution:
            heightmap[y_idx, x_idx] = max(heightmap[y_idx, x_idx], vertices[i][2])
    
    return heightmap

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

    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.input_mesh)

    heightmap = mesh_to_heightmap(mesh, resolution=args.resolution)

    if args.normalize:
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

    output_file = f"{args.output_path}.{args.format}"
    
    plt.imsave(output_file, heightmap, cmap=args.colormap)
    print(f"Heightmap saved to: {output_file}")

if __name__ == "__main__":
    main()
