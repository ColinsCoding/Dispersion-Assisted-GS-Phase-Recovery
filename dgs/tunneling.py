"""Quantum tunneling and the WKB approximation -- Griffiths QM Chapter 8.

Classically a particle with energy E cannot enter a region where the potential V > E.
Quantum-mechanically the wavefunction does not stop at the wall -- it DECAYS
exponentially inside the barrier, exp(-kappa x) with kappa = sqrt(2 m (V-E))/hbar, and
a bit leaks out the far side. The transmission probability through a thick barrier is

    WKB:  T ~ exp(-2 gamma),   gamma = integral over the forbidden region of kappa dx,

which the exact rectangular-barrier formula confirms (same exponential, plus a slowly
varying prefactor). That exponential sensitivity is why tunneling runs the scanning
tunneling microscope (atomic resolution from the gap dependence), alpha decay (the
Gamow factor), the tunnel diode, and flash-memory programming -- the quantum sequel to
the band theory in dgs.kronig_penney. Natural units (hbar = m = 1) by default. NumPy.
"""

import numpy as np


def barrier_decay_constant(E, V0, m=1.0, hbar=1.0):
    """kappa = sqrt(2 m (V0 - E)) / hbar, the exponential decay rate of the wavefunction
    inside a barrier of height V0 > E. The penetration depth is 1/kappa."""
    return np.sqrt(2 * m * (V0 - E)) / hbar


def rectangular_barrier_T(E, V0, width, m=1.0, hbar=1.0):
    """EXACT transmission through a rectangular barrier for E < V0:
        T = [1 + V0^2 sinh^2(kappa a) / (4 E (V0 - E))]^-1.
    Decreases exponentially with barrier width and height -- the textbook tunneling
    probability."""
    k = barrier_decay_constant(E, V0, m, hbar)
    return 1.0 / (1.0 + (V0 ** 2 * np.sinh(k * width) ** 2) / (4 * E * (V0 - E)))


def wkb_transmission(E, V, x, m=1.0, hbar=1.0):
    """WKB tunneling probability T ~ exp(-2 gamma), with
        gamma = integral over the classically forbidden region of sqrt(2 m (V-E))/hbar dx.
    V is the potential sampled on grid x. Works for an arbitrary barrier shape."""
    V = np.asarray(V, float)
    integrand = np.where(V > E, np.sqrt(2 * m * np.maximum(V - E, 0.0)) / hbar, 0.0)
    gamma = np.trapezoid(integrand, np.asarray(x, float))
    return float(np.exp(-2 * gamma))


def barrier_wavefunction(x, x0, E, V0, m=1.0, hbar=1.0):
    """Decaying amplitude exp(-kappa (x - x0)) inside a barrier starting at x0 -- the
    classically forbidden region the wavefunction penetrates (not a full scattering
    solution, just the evanescent decay)."""
    k = barrier_decay_constant(E, V0, m, hbar)
    return np.exp(-k * np.maximum(np.asarray(x, float) - x0, 0.0))


if __name__ == "__main__":
    E, V0, m, hbar = 1.0, 2.0, 1.0, 1.0
    k = barrier_decay_constant(E, V0)
    print(f"kappa = sqrt(2m(V0-E))/hbar = {k:.4f};  penetration depth 1/kappa = {1/k:.3f}")
    for a in (0.5, 1.0, 2.0, 3.0):
        Te = rectangular_barrier_T(E, V0, a)
        x = np.linspace(0, a, 400); V = np.full_like(x, V0)
        Tw = wkb_transmission(E, V, x)
        print(f"  width {a}: exact T = {Te:.3e},  WKB exp(-2 kappa a) = {Tw:.3e}")
    print("WKB captures the EXPONENT; exact has an O(1) prefactor. Both ~ exp(-2 kappa a).")
