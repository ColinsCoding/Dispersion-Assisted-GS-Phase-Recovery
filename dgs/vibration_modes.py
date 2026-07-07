"""Modes of vibration = quantum states = eigenvectors = discretized calculus.

Four subjects, one equation. A vibrating system and a quantum particle are BOTH
Hermitian eigenvalue problems, and both come from the same differential operator:

  VIBRATION (classical):   K v = omega^2 M v
      masses on springs; eigenvalues are squared normal-mode frequencies,
      eigenvectors are the mode SHAPES (which masses move together).

  QUANTUM (Schrodinger):   H psi = E psi
      a particle in a potential; eigenvalues are ENERGY levels, eigenvectors are
      the wavefunctions. H = -(hbar^2/2m) d^2/dx^2 + V is Hermitian.

  LINEAR ALGEBRA:  both are "find eigenvectors of a symmetric operator." The
      eigenvectors are orthogonal (mass-orthonormal for vibration, orthonormal
      for QM) and form a basis -- exactly the eigenvector->PCA, Hermitian->
      attention structure this repo keeps meeting.

  CALCULUS: the operator in the middle is the second derivative d^2/dx^2. Discretize
      it on a grid and it becomes the tridiagonal SECOND-DIFFERENCE matrix
      [1, -2, 1]/dx^2 -- and that SAME matrix is the stiffness of a uniform spring
      chain AND the kinetic-energy operator of the Schrodinger equation. Modes of
      vibration and quantum states are the eigenvectors of one discretized
      derivative.

Verified against closed forms both ways: the fixed-fixed spring chain's
omega_n = 2 sqrt(k/m) sin(n*pi/(2(N+1))), and the particle-in-a-box
E_n = n^2 pi^2 hbar^2 / (2 m L^2).

The "64 vs 128 bit in Python" thread (continuing dgs.precision_phase_retrieval's
32-vs-64): two weakly coupled oscillators have two NEARLY DEGENERATE modes whose
tiny frequency splitting, computed naively as sqrt(l2) - sqrt(l1), catastrophically
cancels. float64 (binary64, ~16 digits) loses most of it; a true 128-bit compute
recovers it. On Windows numpy's longdouble is only float64, so genuine 128-bit uses
decimal.Decimal (decimal128 = 34 digits) -- the actual Python precision theory.
NumPy + stdlib decimal; py-3.13.
"""

import sys
import math
from decimal import Decimal, getcontext

import numpy as np
from dgs import lagrangian


# ----------------------------------------------------------------------
# The one operator: the second-difference matrix (discretized d^2/dx^2)
# ----------------------------------------------------------------------

def second_difference_matrix(n, dx=1.0):
    """The tridiagonal [1, -2, 1]/dx^2 operator: the finite-difference form of
    d^2/dx^2. It is the SAME matrix (up to a constant) that appears as spring-
    chain stiffness and as the Schrodinger kinetic term."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if dx <= 0:
        raise ValueError("dx must be positive")
    D2 = (np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1)
          + np.diag(np.ones(n - 1), -1)) / dx ** 2
    return D2


# ----------------------------------------------------------------------
# VIBRATION: normal modes of a mass-spring system
# ----------------------------------------------------------------------

def uniform_spring_chain(n, m=1.0, k=1.0):
    """Fixed-fixed chain of n equal masses m joined by equal springs k (both
    ends anchored to walls). Returns (M, K): M = m*I, K = k*[2,-1] tridiagonal.
    Note K = -k * second_difference_matrix(n, 1) -- the stiffness IS the
    discretized second derivative."""
    if n < 2 or m <= 0 or k <= 0:
        raise ValueError("need n >= 2 and positive m, k")
    K = k * (np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1)
             - np.diag(np.ones(n - 1), -1))
    M = m * np.eye(n)
    return M, K


def normal_modes(M, K):
    """Solve the generalized eigenproblem K v = omega^2 M v for a symmetric K
    and SPD diagonal M. Returns (omega, modes) with omega ascending and the
    mode shapes as columns, MASS-ORTHONORMAL (modes.T @ M @ modes = I). Uses
    the M^{-1/2} K M^{-1/2} whitening so a symmetric eigensolver applies --
    the same generalized problem dgs.lagrangian.normal_mode_frequencies poses."""
    M = np.asarray(M, float)
    K = np.asarray(K, float)
    d = np.diag(M)
    if np.any(d <= 0) or not np.allclose(M, np.diag(d)):
        raise ValueError("M must be diagonal with positive entries")
    inv_sqrt = 1.0 / np.sqrt(d)
    A = (inv_sqrt[:, None] * K) * inv_sqrt[None, :]      # M^-1/2 K M^-1/2
    A = (A + A.T) / 2                                    # symmetrize (roundoff)
    w, U = np.linalg.eigh(A)
    omega = np.sqrt(np.clip(w, 0, None))
    modes = inv_sqrt[:, None] * U                        # back to physical coords
    return omega, modes


def analytic_chain_frequencies(n, m=1.0, k=1.0):
    """Closed-form normal-mode frequencies of the fixed-fixed uniform chain:
    omega_j = 2 sqrt(k/m) sin(j*pi / (2(n+1))), j = 1..n. The reference the
    numerical solver must reproduce."""
    j = np.arange(1, n + 1)
    return 2 * np.sqrt(k / m) * np.sin(j * np.pi / (2 * (n + 1)))


# ----------------------------------------------------------------------
# QUANTUM: the same eigenproblem, now for energy levels
# ----------------------------------------------------------------------

def schrodinger_hamiltonian(V, dx, hbar=1.0, mass=1.0):
    """Finite-difference Hamiltonian H = -(hbar^2/2m) d^2/dx^2 + V on interior
    grid points (Dirichlet walls). V is the potential sampled on the grid.
    Returns the symmetric H whose eigenvalues are energies, eigenvectors
    wavefunctions -- structurally identical to the vibration stiffness."""
    V = np.asarray(V, float)
    n = len(V)
    T = -(hbar ** 2) / (2 * mass) * second_difference_matrix(n, dx)
    return T + np.diag(V)


def particle_in_box(n_grid=400, L=1.0, hbar=1.0, mass=1.0, n_levels=5):
    """Energies of a particle in an infinite square well of width L, by
    diagonalizing the finite-difference Hamiltonian (V=0 inside). Returns
    (numeric_energies, analytic_energies) for the lowest n_levels, where
    E_n = n^2 pi^2 hbar^2 / (2 m L^2)."""
    if n_grid < 10 or L <= 0:
        raise ValueError("need n_grid >= 10 and L > 0")
    dx = L / (n_grid + 1)                                # interior points only
    H = schrodinger_hamiltonian(np.zeros(n_grid), dx, hbar, mass)
    E = np.sort(np.linalg.eigvalsh(H))[:n_levels]
    n = np.arange(1, n_levels + 1)
    E_exact = n ** 2 * np.pi ** 2 * hbar ** 2 / (2 * mass * L ** 2)
    return E, E_exact


# ----------------------------------------------------------------------
# 64 vs 128 bit: near-degenerate mode splitting that cancels
# ----------------------------------------------------------------------

def precision_report():
    """The Python precision facts behind '64 vs 128 bit'. Python floats are
    always IEEE binary64 (~16 digits); numpy's longdouble is platform-dependent
    (80-bit on Linux/Mac, but only float64 on Windows/MSVC); genuine 128-bit
    needs decimal.Decimal (decimal128 = 34 significant digits)."""
    ld = np.finfo(np.longdouble)
    return {
        "python_float": {"bits": 64, "eps": sys.float_info.epsilon,
                         "decimal_digits": 53 * math.log10(2)},
        "numpy_longdouble": {"eps": float(ld.eps), "mantissa_bits": int(ld.nmant) + 1,
                             "extends_float64": bool(ld.eps < np.finfo(np.float64).eps)},
        "decimal128": {"significant_digits": 34, "eps": 1e-33},
    }


def _correct_digits(approx, truth):
    """Number of correct significant digits: -log10(relative error)."""
    truth = Decimal(truth)
    rel = abs(Decimal(approx) - truth) / abs(truth)
    if rel == 0:
        return 34.0
    return float(-rel.ln() / Decimal(10).ln())


def mode_splitting_precision(coupling_ratio=1e-13, m=1.0, k=1.0):
    """Two identical oscillators weakly coupled (k_c = coupling_ratio * k) have
    modes at omega1 = sqrt(k/m) and omega2 = sqrt((k+2k_c)/m). Their splitting
    omega2 - omega1 is tiny, and computing it NAIVELY as sqrt(l2) - sqrt(l1)
    cancels catastrophically. This computes it in float64 and in 128-bit
    Decimal(34), scoring each against a 60-digit stable truth
    (l2-l1)/(sqrt(l2)+sqrt(l1)). Returns the correct-digit counts -- 128-bit
    keeps far more. (l1,l2 are the eigenvalues of dgs.lagrangian's 2-DOF K,M.)"""
    if not 0 < coupling_ratio < 1:
        raise ValueError("coupling_ratio must be in (0, 1)")
    # exact inputs as Decimals (via str) so the high-precision paths are not
    # already spoiled by float64 before they start
    Kd, Md, Rd = Decimal(str(k)), Decimal(str(m)), Decimal(str(coupling_ratio))
    Kcd = Rd * Kd

    getcontext().prec = 60                               # the "truth" (stable form)
    L1, L2 = Kd / Md, (Kd + 2 * Kcd) / Md
    truth = (L2 - L1) / (L2.sqrt() + L1.sqrt())

    # float64 path: everything in binary64 -- the lossy k+2kc addition AND the
    # sqrt-difference cancellation both bite
    l1 = k / m
    l2 = (k + 2 * (coupling_ratio * k)) / m
    naive_f64 = math.sqrt(l2) - math.sqrt(l1)

    getcontext().prec = 34                               # decimal128 = 128-bit
    L1b, L2b = Kd / Md, (Kd + 2 * Kcd) / Md
    naive_d128 = L2b.sqrt() - L1b.sqrt()

    return {
        "splitting": float(truth),
        "float64_correct_digits": _correct_digits(naive_f64, truth),
        "decimal128_correct_digits": _correct_digits(naive_d128, truth),
    }


if __name__ == "__main__":
    n = 6
    M, K = uniform_spring_chain(n, m=1.0, k=1.0)
    # the unification, made literal: chain stiffness == -k * (d^2/dx^2)
    same = np.allclose(K, -1.0 * second_difference_matrix(n, 1.0))
    print(f"spring stiffness K == discretized d^2/dx^2 ? {same}")
    omega, modes = normal_modes(M, K)
    print("chain omegas :", np.round(omega, 4))
    print("closed form  :", np.round(analytic_chain_frequencies(n), 4),
          "  match:", np.allclose(omega, analytic_chain_frequencies(n)))
    print("modes mass-orthonormal? ",
          np.allclose(modes.T @ M @ modes, np.eye(n), atol=1e-9))

    E, E_exact = particle_in_box(n_grid=600, L=1.0, n_levels=5)
    print("\nparticle in a box (hbar=m=1, L=1):")
    print("  numeric E :", np.round(E, 3))
    print("  exact  E  :", np.round(E_exact, 3), " (E_n = n^2 pi^2 / 2)")

    print("\n64 vs 128 bit -- naive splitting sqrt(l2)-sqrt(l1) of near-degenerate modes:")
    rep = precision_report()
    print(f"  numpy longdouble extends float64 here? "
          f"{rep['numpy_longdouble']['extends_float64']} (Windows: it's just float64)")
    ps = mode_splitting_precision(coupling_ratio=1e-13)
    print(f"  splitting ~ {ps['splitting']:.3e}")
    print(f"  correct digits: float64 {ps['float64_correct_digits']:.1f}  vs  "
          f"decimal128 {ps['decimal128_correct_digits']:.1f}")
