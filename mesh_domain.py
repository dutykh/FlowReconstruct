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

if __name__ == '__main__':
    main()
