"""Multipole expansion: monopole, dipole, quadrupole -- the same expansion
that turns a messy charge distribution into a few numbers, in increasing
order of how fast they fall off with distance.

The potential of any localized charge distribution, far away, is

    V(r) = (1/4pi eps0) [ Q/r + p.rhat/r^2 + (1/2) rhat^T Theta rhat / r^3 + ... ]

monopole (total charge Q) ~ 1/r, dipole (p) ~ 1/r^2, quadrupole (traceless
tensor Theta) ~ 1/r^3 -- each successive term falls off one power of r faster
because it's sensitive to a finer feature of the charge arrangement. A point
charge has only a monopole; two opposite point charges have a dipole but no
monopole; four alternating charges on a square (the classic textbook
quadrupole) have neither monopole nor dipole, only a quadrupole.
"""

import numpy as np

_K_COULOMB = 8.9875517923e9  # 1/(4 pi eps0), SI [N m^2 / C^2]


def monopole_moment(charges):
    """Q = sum q_i -- the total charge (the n=0 multipole)."""
    return float(np.sum(charges))


def dipole_moment(charges, positions):
    """p = sum q_i * r_i -- the dipole moment vector (n=1 multipole). Vanishes
    for any charge distribution with a center of symmetry through which every
    charge has an equal-and-opposite partner."""
    charges = np.asarray(charges, dtype=float)
    positions = np.asarray(positions, dtype=float)
    return charges @ positions


def quadrupole_moment(charges, positions):
    """Theta_ij = sum q_k * (3 x_i x_j - r^2 delta_ij) -- the traceless
    quadrupole tensor (n=2 multipole). Built explicitly with a loop over
    charges and over the 3x3 tensor entries so every term is visible (no
    hidden einsum magic), then checked to be symmetric and traceless, which
    any correct quadrupole tensor must be by construction."""
    charges = np.asarray(charges, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n_dim = positions.shape[1]
    Theta = np.zeros((n_dim, n_dim))
    for q_k, r_k in zip(charges, positions):
        r2 = np.dot(r_k, r_k)
        for i in range(n_dim):
            for j in range(n_dim):
                Theta[i, j] += q_k * (3 * r_k[i] * r_k[j] - r2 * (1.0 if i == j else 0.0))
    return Theta


def multipole_potential(charges, positions, field_point, k=_K_COULOMB):
    """Approximate V at `field_point` (far from the charges) by summing the
    monopole + dipole + quadrupole terms, each one order higher in 1/r than
    the last."""
    field_point = np.asarray(field_point, dtype=float)
    r = np.linalg.norm(field_point)
    r_hat = field_point / r

    Q = monopole_moment(charges)
    p = dipole_moment(charges, positions)
    Theta = quadrupole_moment(charges, positions)

    V_monopole = k * Q / r
    V_dipole = k * np.dot(p, r_hat) / r ** 2
    V_quadrupole = k * 0.5 * (r_hat @ Theta @ r_hat) / r ** 3

    return {
        "V_monopole": V_monopole, "V_dipole": V_dipole, "V_quadrupole": V_quadrupole,
        "V_total": V_monopole + V_dipole + V_quadrupole,
        "Q": Q, "p": p, "Theta": Theta,
    }


def exact_potential(charges, positions, field_point, k=_K_COULOMB):
    """Exact V at field_point = sum_i k*q_i / |field_point - r_i| -- no
    expansion, the multipole series' answer should converge to this as the
    field point moves farther from the charge distribution."""
    charges = np.asarray(charges, dtype=float)
    positions = np.asarray(positions, dtype=float)
    field_point = np.asarray(field_point, dtype=float)
    distances = np.linalg.norm(field_point[None, :] - positions, axis=1)
    return float(np.sum(k * charges / distances))


def quadrupole_square_example(q=1.0, a=1.0):
    """The classic textbook quadrupole: charges +q,-q,+q,-q at the corners of
    a square of half-width a, alternating sign. Has ZERO monopole and ZERO
    dipole by symmetry -- only the quadrupole term survives, which this
    function lets you verify directly."""
    positions = np.array([[a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0]], dtype=float)
    charges = np.array([q, -q, q, -q], dtype=float)
    return charges, positions
