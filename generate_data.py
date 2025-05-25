#!/usr/bin/env python3
"""
generate_data.py

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-05-13

Generates random samples of x in [0, L], computes η(x) and φ(x), and saves the results to a CSV file.
Usage:
    python generate_data.py [-a AMPLITUDE] [-L LENGTH] [-N NUM_SAMPLES]
                             [-o OUTPUT_DIR] [--seed SEED]
"""

import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
import argparse
import matplotlib.pyplot as plt


def generate_data(
    a: float = 0.1, L: float = 4 * np.pi, N: int = 100, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate N random sample points x in [0, L], compute
    η(x) = a * sin(x) and φ(x) = sin^2(x) - cos^4(x).

    Returns
    -------
    x : ndarray
        Sorted sample coordinates in [0, L].
    eta : ndarray
        Free-surface elevation values η(x).
    phi : ndarray
        Velocity potential trace φ(x) at the free surface.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(0, L, size=N)
    x.sort()

    eta = a * np.sin(x)
    phi = np.sin(x) ** 2 - np.cos(x) ** 4

    return x, eta, phi


def save_data(
    x: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    L: float,
    h0: float,
    output_dir: str = "data",
) -> str:
    """
    Save x, eta, phi arrays and parameters L, h0 to CSV in the specified directory.

    Returns
    -------
    output_path : str
        Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    # repeat L and h0 for each sample
    L_arr = np.full_like(x, L)
    h0_arr = np.full_like(x, h0)
    data = np.column_stack((x, eta, phi, L_arr, h0_arr))
    output_path = os.path.join(output_dir, "data.csv")
    header = "x,eta,phi,L,h0"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")
    return output_path


def plot_wave(
    x: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    h0: float = 1.0,
    L: float = 4 * np.pi,
) -> None:
    """
    Plot the wave tank cross-section with water below η(x) and
    a separate panel for φ(x) over x.

    Parameters
    ----------
    x : ndarray
        Sample coordinates.
    eta : ndarray
        Free-surface elevations.
    phi : ndarray
        Velocity potential at the free surface.
    h0 : float
        Undisturbed water depth (bottom at y = -h0).
    L : float
        Length of the tank domain.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Wave tank cross-section
    bottom = -h0
    axes[0].fill_between(
        x, bottom, eta, where=eta >= bottom, facecolor="skyblue", alpha=0.7
    )
    axes[0].plot(x, eta, "k-", linewidth=1.5, label="Free surface (η)")
    axes[0].scatter(x, eta, s=20, c="red", marker="o", label="Sample points")
    axes[0].set_xlim(0, L)
    axes[0].set_ylim(bottom - 0.1, eta.max() + 0.1)
    axes[0].set_ylabel("y")
    axes[0].set_title("Wave tank cross-section")
    # Move legend closer to the bottom (lower right inside axes)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 0.01), fontsize="small")

    # Velocity potential
    axes[1].plot(x, phi, "b-", linewidth=1.5, label="φ(x)")
    axes[1].scatter(x, phi, s=20, c="orange", marker="o", label="Sample points")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("φ(x)")
    axes[1].set_title("Velocity potential at free surface")
    axes[1].legend(loc="lower right", bbox_to_anchor=(1, 0.01), fontsize="small")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate, save, and visualise wave data over [0, L]."
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        type=float,
        default=0.1,
        help="Amplitude a for η(x) = a*sin(x)",
    )
    parser.add_argument(
        "-L", "--length", type=float, default=4 * np.pi, help="Tank length L"
    )
    parser.add_argument(
        "-N", "--num-samples", type=int, default=100, help="Number of random samples"
    )
    parser.add_argument(
        "--h0", type=float, default=1.0, help="Undisturbed water depth (default: 1.0)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save data.csv",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x, eta, phi = generate_data(
        a=args.amplitude, L=args.length, N=args.num_samples, seed=args.seed
    )
    path = save_data(x, eta, phi, L=args.length, h0=args.h0, output_dir=args.output_dir)
    print(f"Data successfully saved to: {path}")
    plot_wave(x, eta, phi, h0=args.h0, L=args.length)
