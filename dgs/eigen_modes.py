"""Engineering eigenvalues -- natural frequencies and mode shapes, via torch.

A vibrating structure's resonances are an EIGENVALUE problem. For a chain of N equal
masses m joined by springs k (and to fixed walls), Newton's law m x'' = -K x has the
dynamical matrix
    D = (k/m) * tridiag(-1, 2, -1),
and torch.linalg.eigh(D) returns the squared natural frequencies omega_j^2
(eigenvalues) and the mode shapes (eigenvectors). The closed form is
    omega_j = 2 sqrt(k/m) sin( j pi / (2(N+1)) ),  j = 1..N,
and the j-th mode is a standing sine with j-1 interior nodes -- the SAME eigenvalue
structure as the Bessel drumhead (dgs.bessel_linalg) and the quantum particle in a
box. Eigenvalues are how engineers find resonances, buckling loads, and PCA axes.
Torch (py-3.12); the eigensolve is torch.linalg.eigh.
"""

import numpy as np
import torch


def chain_dynamical_matrix(N, k=1.0, m=1.0):
    """Tridiagonal dynamical matrix (k/m)*(-1, 2, -1) for a fixed-fixed mass-spring
    chain of N masses. Symmetric and positive-definite -> real positive eigenvalues."""
    d = torch.full((N,), 2.0, dtype=torch.float64)
    o = torch.full((N - 1,), -1.0, dtype=torch.float64)
    return (k / m) * (torch.diag(d) + torch.diag(o, 1) + torch.diag(o, -1))


def normal_modes(D):
    """Natural frequencies omega (= sqrt of the ascending eigenvalues) and mode shapes
    (eigenvectors), from torch.linalg.eigh on the symmetric dynamical matrix D."""
    eig, vec = torch.linalg.eigh(D)
    omega = torch.sqrt(torch.clamp(eig, min=0.0))
    return omega, vec


def chain_frequencies_analytic(N, k=1.0, m=1.0):
    """Closed-form natural frequencies omega_j = 2 sqrt(k/m) sin(j pi/(2(N+1)))."""
    j = np.arange(1, N + 1)
    return 2 * np.sqrt(k / m) * np.sin(j * np.pi / (2 * (N + 1)))


def chain_mode_shape(N, j):
    """Closed-form j-th mode shape: component i is sin(j i pi/(N+1)), i=1..N (j-1
    interior nodes). Normalized to unit peak."""
    i = np.arange(1, N + 1)
    s = np.sin(j * i * np.pi / (N + 1))
    return s / np.max(np.abs(s))


if __name__ == "__main__":
    N, k, m = 12, 4.0, 0.5
    D = chain_dynamical_matrix(N, k, m)
    omega, modes = normal_modes(D)
    ref = chain_frequencies_analytic(N, k, m)
    print("torch eigenvalue frequencies vs analytic (first 5):")
    for j in range(5):
        print(f"  mode {j+1}: omega = {omega[j].item():.5f}   analytic {ref[j]:.5f}")
    print(f"\nmax |torch - analytic| = {np.max(np.abs(omega.numpy() - ref)):.2e}")
    print(f"fundamental period T1 = 2 pi / omega_1 = {2*np.pi/omega[0].item():.4f} s")
