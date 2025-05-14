#!/usr/bin/env python3
"""
mesh_domain.py

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-05-13

Reads wave data from CSV (x, eta, phi, L, h0), extrapolates endpoints via CubicSpline,
generates a triangular mesh of the fluid domain (for FEniCS), and
exports to Gmsh (.msh) and XDMF files, plus plots the domain and mesh.

Usage:
    python mesh_domain.py -i data/data.csv -m mesh/mesh.msh -x mesh/mesh.xdmf --mesh-size 0.1
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
import argparse
import gmsh
import meshio
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def read_data(path: str):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    x = data[:, 0]
    eta = data[:, 1]
    L = float(data[0, 3])
    h0 = float(data[0, 4])
    return x, eta, L, h0

def extrapolate_eta(x: np.ndarray, eta: np.ndarray, L: float):
    """Use a CubicSpline interpolant to extrapolate η at x=0 and x=L, enforcing periodicity."""
    idx = np.argsort(x)
    xs = x[idx]
    ys = eta[idx]
    spline = CubicSpline(xs, ys, extrapolate=True)
    # Ensure x=0 and x=L are present
    if xs[0] > 1e-12:
        xs = np.insert(xs, 0, 0.0)
        ys = np.insert(ys, 0, spline(0.0))
    if abs(xs[-1] - L) > 1e-12:
        xs = np.append(xs, L)
        ys = np.append(ys, spline(L))
    # Enforce periodicity: eta[0] = eta[-1] = avg
    ys[0] = ys[-1] = 0.5 * (ys[0] + ys[-1])
    return xs, ys

def generate_mesh(x: np.ndarray, eta: np.ndarray, L: float, h0: float, mesh_size: float):
    """Use Gmsh to create a 2D mesh of the fluid domain, with smoothing and optimization for FEM."""
    gmsh.initialize()
    gmsh.model.add("wave_domain")
    # Free surface points and lines
    pts = [gmsh.model.geo.addPoint(xi, yi, 0, mesh_size) for xi, yi in zip(x, eta)]
    lines = [gmsh.model.geo.addLine(pts[i], pts[i+1]) for i in range(len(pts)-1)]
    # Bottom closure
    pL = gmsh.model.geo.addPoint(L, -h0, 0, mesh_size)
    p0 = gmsh.model.geo.addPoint(0.0, -h0, 0, mesh_size)
    lines += [gmsh.model.geo.addLine(pts[-1], pL),
              gmsh.model.geo.addLine(pL, p0),
              gmsh.model.geo.addLine(p0, pts[0])]
    loop = gmsh.model.geo.addCurveLoop(lines)
    gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    # Mesh generation
    gmsh.model.mesh.generate(2)
    # Mesh optimization (smoothing via optimize)
    # Laplacian smoothing is not available in the gmsh Python API directly
    # Use built-in mesh optimization instead
    gmsh.option.setNumber('Mesh.Optimize', 1)
    gmsh.model.mesh.optimize('Netgen')  # Netgen optimizer is good for FEM
    # Optionally, you can use 'Mesh.OptimizeNetgen' for further improvement
    gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
    gmsh.model.mesh.optimize('Netgen')

def save_mesh(msh_file: str, xdmf_file: str):
    """Write Gmsh .msh and convert to XDMF for FEniCS."""
    gmsh.write(msh_file)
    gmsh.finalize()
    
    # Read the mesh
    mesh = meshio.read(msh_file)
    
    # Extract triangular cells
    triangle_cells = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
            break
    
    if triangle_cells is None:
        raise ValueError("No triangle cells found in the mesh!")
    
    # Create a new mesh with only triangle cells
    triangle_mesh = meshio.Mesh(
        points=mesh.points,
        cells={"triangle": triangle_cells}
    )
    
    # Write the XDMF file
    meshio.write(xdmf_file, triangle_mesh)

def plot_domain_and_mesh(msh_file: str, x_full: np.ndarray, eta_full: np.ndarray, h0: float):
    """Plot the fluid domain boundary and the mesh triangles."""
    # Read mesh
    mesh = meshio.read(msh_file)
    points = mesh.points[:, :2]
    # Extract triangle connectivity
    triangles = None
    for cell in mesh.cells:
        if cell.type == 'triangle':
            triangles = cell.data
            break
    if triangles is None:
        raise ValueError("No triangular cells found in mesh.")
    # Triangulation for plotting
    tri = mtri.Triangulation(points[:,0], points[:,1], triangles)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.triplot(tri, linewidth=0.5, color='gray')
    # Plot free surface and bottom
    ax.plot(x_full, eta_full, 'b-', linewidth=2, label='Free surface')
    ax.plot([x_full[0], x_full[-1]], [-h0, -h0], 'k-', linewidth=2, label='Bottom')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title('Fluid Domain Mesh')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize='small')
    plt.tight_layout()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate, save, and plot 2D wave-tank mesh from CSV data."
    )
    parser.add_argument('-i', '--input', default='data/data.csv',
                        help='Path to input CSV file')
    parser.add_argument('-m', '--msh', default='mesh/mesh.msh',
                        help='Output Gmsh .msh file (default: mesh/mesh.msh)')
    parser.add_argument('-x', '--xdmf', default='mesh/mesh.xdmf',
                        help='Output XDMF file for FEniCS (default: mesh/mesh.xdmf)')
    parser.add_argument('--mesh-size', type=float, default=0.1,
                        help='Target mesh element size (default: 0.1)')
    return parser.parse_args()

def print_mesh_statistics(msh_file: str):
    """Print detailed statistics about the mesh: node, edge, triangle counts, edge lengths, triangle areas, quality."""
    import meshio
    import numpy as np
    import itertools

    mesh = meshio.read(msh_file)
    points = mesh.points[:, :2]
    triangles = None
    for cell in mesh.cells:
        if cell.type == 'triangle':
            triangles = cell.data
            break
    if triangles is None:
        print("No triangles found in mesh.")
        return

    n_nodes = points.shape[0]
    n_triangles = triangles.shape[0]
    # Build edge set
    edges = set()
    for tri in triangles:
        for i, j in [(0,1), (1,2), (2,0)]:
            edge = tuple(sorted((tri[i], tri[j])))
            edges.add(edge)
    edge_list = np.array(list(edges))
    n_edges = edge_list.shape[0]

    # Edge lengths
    edge_lengths = np.linalg.norm(points[edge_list[:,0]] - points[edge_list[:,1]], axis=1)
    min_edge = edge_lengths.min()
    max_edge = edge_lengths.max()
    mean_edge = edge_lengths.mean()
    std_edge = edge_lengths.std()

    # Triangle areas
    def triangle_area(p0, p1, p2):
        return 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
    areas = np.array([
        triangle_area(points[tri[0]], points[tri[1]], points[tri[2]])
        for tri in triangles
    ])
    min_area = areas.min()
    max_area = areas.max()
    mean_area = areas.mean()
    std_area = areas.std()

    # Mesh quality: aspect ratio and min angle
    def triangle_aspect_ratio(p0, p1, p2):
        a = np.linalg.norm(p1-p0)
        b = np.linalg.norm(p2-p1)
        c = np.linalg.norm(p0-p2)
        s = 0.5*(a+b+c)
        area = triangle_area(p0, p1, p2)
        if area == 0:
            return np.inf
        R = (a*b*c)/(4*area)
        r = area/s if s > 0 else 0
        return R/r if r > 0 else np.inf
    def triangle_angles(p0, p1, p2):
        a = np.linalg.norm(p2-p1)
        b = np.linalg.norm(p2-p0)
        c = np.linalg.norm(p1-p0)
        angles = []
        # Law of cosines
        for x, y, z in [(a,b,c),(b,c,a),(c,a,b)]:
            cos_theta = (y**2 + z**2 - x**2)/(2*y*z) if y > 0 and z > 0 else -1
            angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180/np.pi
            angles.append(angle)
        return angles
    aspect_ratios = np.array([
        triangle_aspect_ratio(points[tri[0]], points[tri[1]], points[tri[2]])
        for tri in triangles
    ])
    min_angles = np.array([
        min(triangle_angles(points[tri[0]], points[tri[1]], points[tri[2]]))
        for tri in triangles
    ])
    max_angles = np.array([
        max(triangle_angles(points[tri[0]], points[tri[1]], points[tri[2]]))
        for tri in triangles
    ])

    print("\n===== Mesh Statistics Report =====")
    print(f"Number of nodes:       {n_nodes}")
    print(f"Number of edges:       {n_edges}")
    print(f"Number of triangles:   {n_triangles}")
    print("\nEdge length statistics:")
    print(f"  Min:    {min_edge:.6g}")
    print(f"  Max:    {max_edge:.6g}")
    print(f"  Mean:   {mean_edge:.6g}")
    print(f"  Stddev: {std_edge:.6g}")
    print("\nTriangle area statistics:")
    print(f"  Min:    {min_area:.6g}")
    print(f"  Max:    {max_area:.6g}")
    print(f"  Mean:   {mean_area:.6g}")
    print(f"  Stddev: {std_area:.6g}")
    print("\nMesh quality metrics:")
    print(f"  Aspect ratio (R/r): min {aspect_ratios.min():.4g}, max {aspect_ratios.max():.4g}, mean {aspect_ratios.mean():.4g}, std {aspect_ratios.std():.4g}")
    print(f"  Min angle: min {min_angles.min():.4g}°, max {min_angles.max():.4g}°, mean {min_angles.mean():.4g}°, std {min_angles.std():.4g}°")
    print(f"  Max angle: min {max_angles.min():.4g}°, max {max_angles.max():.4g}°, mean {max_angles.mean():.4g}°, std {max_angles.std():.4g}°")
    print("=================================\n")


def main():
    args = parse_args()
    # Ensure mesh directory exists
    mesh_dir = os.path.dirname(args.msh) or 'mesh'
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir, exist_ok=True)
    x_raw, eta_raw, L, h0 = read_data(args.input)
    x_full, eta_full = extrapolate_eta(x_raw, eta_raw, L)
    generate_mesh(x_full, eta_full, L, h0, args.mesh_size)
    save_mesh(args.msh, args.xdmf)
    print(f"Mesh written to '{args.msh}' and '{args.xdmf}'")
    plot_domain_and_mesh(args.msh, x_full, eta_full, h0)
    print_mesh_statistics(args.msh)

if __name__ == '__main__':
    main()
