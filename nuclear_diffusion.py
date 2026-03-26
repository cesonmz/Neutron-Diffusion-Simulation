"""
=============================================================
  NEUTRON DIFFUSION SIMULATION  —  Starter Code
=============================================================
  Covers:
    Part 1 — 1D Steady-State (eigenvalue / criticality solve)
    Part 2 — 1D Time-Dependent (watch flux grow or decay)
    Part 3 — 2D Steady-State Flux Map

  Physics model: one-speed (mono-energetic) diffusion theory
  Numerics:      finite difference on a uniform mesh
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh  # for eigenvalue solve

# ─────────────────────────────────────────────────────────────
#  MATERIAL PROPERTIES  (all in CGS units: cm, cm⁻¹)
# ─────────────────────────────────────────────────────────────

# Uranium-235 (thermal neutron group, homogeneous fuel mix)
U235 = {
    "D":       1.13,    # diffusion coefficient  [cm]
    "Sig_a":   0.106,   # absorption cross-section [cm⁻¹]
    "Sig_f":   0.0978,  # fission cross-section  [cm⁻¹]
    "nu":      2.43,    # neutrons released per fission
}

# Light water moderator
WATER = {
    "D":       0.16,
    "Sig_a":   0.022,
    "Sig_f":   0.0,
    "nu":      0.0,
}

# Reflector (graphite-like, low absorption)
REFLECTOR = {
    "D":       0.84,
    "Sig_a":   0.0005,
    "Sig_f":   0.0,
    "nu":      0.0,
}


# ─────────────────────────────────────────────────────────────
#  BUILD THE FINITE-DIFFERENCE LAPLACIAN MATRIX (1D)
# ─────────────────────────────────────────────────────────────

def build_laplacian_1d(N, dx):
    """
    Returns the N×N tri-diagonal matrix for d²/dx²
    with zero-flux boundary conditions (φ = 0 at edges).
    """
    diag  = -2.0 * np.ones(N)
    upper =  1.0 * np.ones(N - 1)
    lower =  1.0 * np.ones(N - 1)
    L = (np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)) / dx**2
    return L


# ─────────────────────────────────────────────────────────────
#  PART 1  —  1D Steady-State Eigenvalue Problem
#
#  Solve:  D ∇²φ - Σ_a φ + (1/k) ν Σ_f φ = 0
#  as a generalised eigenvalue problem: A φ = (1/k) B φ
#  The dominant eigenvalue gives k_eff and the corresponding
#  eigenvector is the fundamental flux mode.
# ─────────────────────────────────────────────────────────────

def solve_1d_eigenvalue(mat, slab_length_cm=60.0, N=200):
    """
    Parameters
    ----------
    mat            : material dict (D, Sig_a, Sig_f, nu)
    slab_length_cm : full slab thickness [cm]
    N              : number of spatial cells

    Returns
    -------
    k_eff  : effective multiplication factor
    x      : spatial grid [cm]
    phi    : normalised neutron flux
    """
    dx = slab_length_cm / N
    x  = np.linspace(dx / 2, slab_length_cm - dx / 2, N)  # cell centres

    D     = mat["D"]
    Sig_a = mat["Sig_a"]
    nu_Sf = mat["nu"] * mat["Sig_f"]

    L = build_laplacian_1d(N, dx)

    # Diffusion + absorption operator  (A φ = loss terms)
    A = -D * L + Sig_a * np.eye(N)

    # Fission production operator  (B φ = gain terms)
    B = nu_Sf * np.eye(N)

    # Solve generalised eigenvalue:  A φ = (1/k) B φ
    # scipy.linalg.eigh returns eigenvalues in ascending order
    # We want the LARGEST k, i.e. the SMALLEST eigenvalue of A·B⁻¹
    eigenvalues, eigenvectors = eigh(B, A)  # solves B v = λ A v  →  k = λ
    idx   = np.argmax(eigenvalues)
    k_eff = eigenvalues[idx]
    phi   = eigenvectors[:, idx]

    # Normalise so peak flux = 1
    phi = np.abs(phi) / np.max(np.abs(phi))

    return k_eff, x, phi


def analytical_flux_1d(slab_length_cm, N=200):
    """
    Analytical solution for a bare homogeneous slab:
      φ(x) = cos(π x / a)   with x in [-a/2, a/2]
    Returned on the same cell-centre grid as the numerical solve.
    """
    dx  = slab_length_cm / N
    x   = np.linspace(dx / 2, slab_length_cm - dx / 2, N)
    x_c = x - slab_length_cm / 2          # shift to [-a/2, a/2]
    phi = np.cos(np.pi * x_c / slab_length_cm)
    phi = np.clip(phi, 0, None)            # zero outside the slab
    return x, phi / phi.max()

# ─────────────────────────────────────────────────────────────
#  PART 3  —  2D Steady-State Flux Map
#
#  Same eigenvalue approach but on a 2D grid.
#  Geometry: fuel core (inner square) + reflector (outer ring)
# ─────────────────────────────────────────────────────────────
 
import numpy as np
from scipy.linalg import eigh

def harmonic_mean(a, b):
    return 2*a*b/(a+b) if (a+b) > 0 else 0.0

def solve_2d_flux_map(core_size=40.0, total_size=60.0, N=60):
    dx = total_size / N
    x = np.linspace(dx/2, total_size - dx/2, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    # Fuel mask
    x0 = (total_size - core_size) / 2
    x1 = (total_size + core_size) / 2
    fuel = ((X >= x0) & (X <= x1) & (Y >= x0) & (Y <= x1))

    D_map = np.where(fuel, U235["D"], REFLECTOR["D"])
    Sig_a_map = np.where(fuel, U235["Sig_a"], REFLECTOR["Sig_a"])
    nu_Sf_map = np.where(fuel, U235["nu"] * U235["Sig_f"], 0.0)

    n = N * N
    A = np.zeros((n, n))
    B = np.zeros((n, n))

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            p = idx(i, j)

            Dc = D_map[i, j]
            Sa = Sig_a_map[i, j]
            Sf = nu_Sf_map[i, j]

            diag = Sa

            # East face
            if i < N - 1:
                De = harmonic_mean(Dc, D_map[i+1, j])
                q = idx(i+1, j)
                A[p, q] -= De / dx**2
                diag += De / dx**2
            # West face
            if i > 0:
                Dw = harmonic_mean(Dc, D_map[i-1, j])
                q = idx(i-1, j)
                A[p, q] -= Dw / dx**2
                diag += Dw / dx**2
            # North face
            if j < N - 1:
                Dn = harmonic_mean(Dc, D_map[i, j+1])
                q = idx(i, j+1)
                A[p, q] -= Dn / dx**2
                diag += Dn / dx**2
            # South face
            if j > 0:
                Ds = harmonic_mean(Dc, D_map[i, j-1])
                q = idx(i, j-1)
                A[p, q] -= Ds / dx**2
                diag += Ds / dx**2

            A[p, p] += diag
            B[p, p] = Sf

    print("Solving corrected 2D eigenvalue problem...")
    eigvals, eigvecs = eigh(B, A)   # B v = k A v
    k_eff = eigvals[-1]
    phi = np.abs(eigvecs[:, -1])
    phi /= phi.max()

    return k_eff, phi.reshape((N, N))
 

# ─────────────────────────────────────────────────────────────
#  PLOTTING UTILITIES
# ─────────────────────────────────────────────────────────────

def plot_1d_eigenvalue(mat, slab_length_cm=60.0):
    k_eff, x_num, phi_num = solve_1d_eigenvalue(mat, slab_length_cm)
    x_ana, phi_ana        = analytical_flux_1d(slab_length_cm)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_num, phi_num, "b-",  lw=2, label="Numerical (FD)")
    ax.plot(x_ana, phi_ana, "r--", lw=2, label="Analytical (cosine)")
    ax.set_xlabel("Position  x  [cm]", fontsize=12)
    ax.set_ylabel("Normalised Neutron Flux  φ(x)", fontsize=12)
    criticality = ("CRITICAL" if abs(k_eff - 1) < 0.02
                   else "SUPERCRITICAL" if k_eff > 1 else "SUBCRITICAL")
    ax.set_title(f"1D Bare Slab — k_eff = {k_eff:.4f}  →  {criticality}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, slab_length_cm)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/part1_eigenvalue.png", dpi=150)
    plt.show()
    print(f"\n[Part 1]  k_eff = {k_eff:.5f}  ({criticality})")

def plot_2d_flux_map():
    k_eff, phi2d = solve_2d_flux_map()
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(phi2d, origin="lower", cmap="inferno",
                    extent=[0, 60, 0, 60])
    plt.colorbar(img, ax=ax, label="Normalised Flux")
    # Draw core boundary
    from matplotlib.patches import Rectangle
    core_offset = (60 - 40) / 2
    rect = Rectangle((core_offset, core_offset), 40, 40,
                      linewidth=2, edgecolor="white",
                      facecolor="none", linestyle="--")
    ax.add_patch(rect)
    ax.set_title("2D Flux Map — Fuel Core (dashed) + Reflector",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x  [cm]", fontsize=12)
    ax.set_ylabel("y  [cm]", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/part3_flux_map.png", dpi=150)
    plt.show()
    print(f"\n[Part 3]  2D flux map saved (k_eff = {k_eff:.4f})")

def plot_2d_flux_map_bare():
    """2D flux map with bare core (no reflector)"""
    core_size = 40.0
    total_size = 40.0  # Same as core (no reflector)
    N = 60
    dx = total_size / N
    x = np.linspace(dx/2, total_size - dx/2, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    # All fuel (no reflector)
    D_map = np.full((N, N), U235["D"])
    Sig_a_map = np.full((N, N), U235["Sig_a"])
    nu_Sf_map = np.full((N, N), U235["nu"] * U235["Sig_f"])

    n = N * N
    A = np.zeros((n, n))
    B = np.zeros((n, n))

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            p = idx(i, j)

            Dc = D_map[i, j]
            Sa = Sig_a_map[i, j]
            Sf = nu_Sf_map[i, j]

            diag = Sa

            # East
            if i < N - 1:
                De = harmonic_mean(Dc, D_map[i+1, j])
                q = idx(i+1, j)
                A[p, q] -= De / dx**2
                diag += De / dx**2
            else:
                diag += Dc / dx**2

            # West
            if i > 0:
                Dw = harmonic_mean(Dc, D_map[i-1, j])
                q = idx(i-1, j)
                A[p, q] -= Dw / dx**2
                diag += Dw / dx**2
            else:
                diag += Dc / dx**2

            # North
            if j < N - 1:
                Dn = harmonic_mean(Dc, D_map[i, j+1])
                q = idx(i, j+1)
                A[p, q] -= Dn / dx**2
                diag += Dn / dx**2
            else:
                diag += Dc / dx**2

            # South
            if j > 0:
                Ds = harmonic_mean(Dc, D_map[i, j-1])
                q = idx(i, j-1)
                A[p, q] -= Ds / dx**2
                diag += Ds / dx**2
            else:
                diag += Dc / dx**2

            A[p, p] = diag
            B[p, p] = Sf

    eigvals, eigvecs = eigh(B, A)
    k_eff = eigvals[-1]
    phi = np.abs(eigvecs[:, -1])
    phi /= phi.max()
    phi2d = phi.reshape((N, N))

    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(phi2d, origin="lower", cmap="inferno",
                    extent=[0, total_size, 0, total_size])
    plt.colorbar(img, ax=ax, label="Normalised Flux")
    ax.set_title(f"2D Flux Map — Bare Fuel Core (no reflector)\nk_eff = {k_eff:.4f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x  [cm]", fontsize=12)
    ax.set_ylabel("y  [cm]", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/part3_flux_map_bare.png", dpi=150)
    plt.show()
    print(f"\n[Part 3b] 2D bare core flux map saved (k_eff = {k_eff:.4f})")

# ─────────────────────────────────────────────────────────────
#  CRITICALITY SWEEP  —  k_eff vs slab thickness
# ─────────────────────────────────────────────────────────────

def plot_criticality_sweep(mat, lengths=None):
    if lengths is None:
        lengths = np.linspace(2, 30, 40)
    k_vals = []
    for L in lengths:
        k, _, _ = solve_1d_eigenvalue(mat, slab_length_cm=L, N=150)
        k_vals.append(k)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lengths, k_vals, "g-o", lw=2, markersize=4)
    ax.axhline(1.0, color="red", linestyle="--", lw=1.5, label="Critical  k = 1")
    ax.fill_between(lengths, k_vals, 1.0,
                    where=np.array(k_vals) >= 1.0,
                    alpha=0.15, color="red", label="Supercritical region")
    ax.set_xlabel("Slab Thickness  [cm]", fontsize=12)
    ax.set_ylabel("k_eff", fontsize=12)
    ax.set_title("Criticality Sweep  —  k_eff vs. Slab Thickness",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/criticality_sweep.png", dpi=150)
    plt.show()

    # Find approximate critical thickness
    k_arr = np.array(k_vals)
    idx   = np.argmin(np.abs(k_arr - 1.0))
    print(f"\n[Sweep]  Approximate critical thickness: {lengths[idx]:.1f} cm  "
          f"(k_eff = {k_arr[idx]:.4f})")
    print("\n All plots saved to outputs/")


# ─────────────────────────────────────────────────────────────
#  MAIN  —  run all four parts
# ─────────────────────────────────────────────────────────────

import os

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  NEUTRON DIFFUSION SIMULATION")
    print("=" * 60)

    # ── Part 1: Eigenvalue / flux shape vs analytical solution
    print("\n[Part 1]  1D steady-state eigenvalue solve...")
    plot_1d_eigenvalue(U235, slab_length_cm=150.0)

    # ── Criticality sweep: how does k_eff change with geometry?
    print("\n[Sweep]   k_eff vs slab thickness...")
    plot_criticality_sweep(U235)

    # ── Part 3: 2D flux map with reflector
    print("\n[Part 3]  2D flux map (fuel + reflector)...")
    plot_2d_flux_map()

    # ── Part 3b: 2D flux map without reflector
    print("\n[Part 3b] 2D flux map (bare core)...")
    plot_2d_flux_map_bare()

    print("\n All plots saved to outputs/")