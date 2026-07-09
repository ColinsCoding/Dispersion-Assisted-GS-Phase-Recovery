"""The harmonic approximation: near any minimum, everything is a spring.

Why is the simple harmonic oscillator everywhere -- molecules, crystals, pendulums, LC
circuits, bridges? Because ANY smooth potential, expanded about a stable minimum x0, is a
parabola to leading order:
        V(x) = V(x0) + V'(x0)(x-x0) + 1/2 V''(x0)(x-x0)^2 + ...
At a minimum V'(x0) = 0, so the first thing that survives is the QUADRATIC term -- a spring
of stiffness k = V''(x0). A mass m in it oscillates at
        omega = sqrt(V''(x0) / m),
and quantum-mechanically its energies are the ladder E_n = (n + 1/2) hbar*omega labeled by
the INTEGER n (dgs.quantum_oscillator). So the harmonic oscillator is not a special toy
potential; it is the universal SMALL-OSCILLATION limit of every real system, good as long as
the amplitude is small enough that the cubic term V'''(x0)(x-x0)^3 stays negligible. When it
does not, you get ANHARMONICITY -- amplitude-dependent frequency, the crowding of molecular
vibrational levels (dgs.schrodinger_lennard_jones), thermal expansion.

This module extracts the approximation from ANY potential you hand it: the local curvature
by finite differences, the frequency and period, the parabola itself, and a check of how far
you can stray before it breaks. Verified on an exact spring (approximation is exact), the
pendulum (omega -> sqrt(g/l)), and the Lennard-Jones well (matches its harmonic frequency).
NumPy + SciPy; py-3.13.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def derivatives(V, x0, h=1e-5):
    """Numerically evaluate (V, V', V'') of a 1-D potential at x0 by central
    differences -- the ingredients of the Taylor expansion."""
    V0 = V(x0)
    Vp = (V(x0 + h) - V(x0 - h)) / (2 * h)
    Vpp = (V(x0 + h) - 2 * V0 + V(x0 - h)) / h ** 2
    return float(V0), float(Vp), float(Vpp)


def is_stable_minimum(V, x0, h=1e-5, tol=1e-4):
    """True if x0 is a STABLE equilibrium: V'(x0) ~ 0 (stationary) and V''(x0) > 0
    (a well, not a hill). Only stable minima give real oscillations."""
    _, Vp, Vpp = derivatives(V, x0, h)
    return abs(Vp) < tol * max(1.0, abs(Vpp)) and Vpp > 0


def find_minimum(V, bracket=(-10.0, 10.0)):
    """Locate a potential minimum in the bracket (scipy bounded search). Returns x0."""
    res = minimize_scalar(V, bounds=bracket, method="bounded")
    return float(res.x)


def harmonic_frequency(V, x0, mass=1.0, h=1e-5):
    """Small-oscillation frequency omega = sqrt(V''(x0)/m) at a stable minimum.
    Raises if x0 is not a stable minimum (V'' <= 0)."""
    if mass <= 0:
        raise ValueError("mass must be positive")
    _, _, Vpp = derivatives(V, x0, h)
    if Vpp <= 0:
        raise ValueError("V''(x0) <= 0: not a stable minimum, no real oscillation")
    return np.sqrt(Vpp / mass)


def small_oscillation_period(V, x0, mass=1.0, h=1e-5):
    """Period T = 2 pi / omega of small oscillations about x0 (amplitude-independent
    in the harmonic approximation)."""
    return 2 * np.pi / harmonic_frequency(V, x0, mass, h)


def harmonic_potential(V, x0, x, h=1e-5):
    """The parabolic approximation V(x0) + 1/2 V''(x0)(x-x0)^2 (the V'(x0)=0 term
    dropped, as it vanishes at a minimum). Compare to V(x) to see where it breaks."""
    V0, _, Vpp = derivatives(V, x0, h)
    x = np.asarray(x, float)
    return V0 + 0.5 * Vpp * (x - x0) ** 2


def approximation_error(V, x0, amplitude, h=1e-5, n=200):
    """Max fractional deviation of the true potential from its harmonic approximation
    over [x0-amplitude, x0+amplitude], relative to the well depth 1/2 V'' amp^2.
    Small for a gentle well / small amplitude; grows with anharmonicity."""
    x = np.linspace(x0 - amplitude, x0 + amplitude, n)
    true = np.array([V(xi) for xi in x])
    approx = harmonic_potential(V, x0, x, h)
    _, _, Vpp = derivatives(V, x0, h)
    scale = 0.5 * Vpp * amplitude ** 2
    return float(np.max(np.abs(true - approx)) / scale) if scale > 0 else 0.0


if __name__ == "__main__":
    # 1. exact spring: the approximation is exact everywhere
    k = 3.0
    Vspring = lambda x: 0.5 * k * x ** 2
    print("spring V=1/2 k x^2 (k=3): omega =", round(harmonic_frequency(Vspring, 0.0), 4),
          "= sqrt(k/m) =", round(np.sqrt(3), 4),
          "; anharmonic error =", round(approximation_error(Vspring, 0.0, 5.0), 12), "(exact)")

    # 2. pendulum: V = g(1 - cos theta), omega -> sqrt(g/l) small-angle
    g = 9.81
    Vpend = lambda th: g * (1 - np.cos(th))
    print(f"\npendulum V=g(1-cos): omega = {harmonic_frequency(Vpend, 0.0):.4f} "
          f"= sqrt(g) = {np.sqrt(g):.4f}, period = {small_oscillation_period(Vpend, 0.0):.4f} s")
    print(f"  anharmonic error at 0.2 rad: {approximation_error(Vpend, 0.0, 0.2):.2e}, "
          f"at 1.5 rad: {approximation_error(Vpend, 0.0, 1.5):.2e}  (grows with amplitude)")

    # 3. Lennard-Jones well: matches its own harmonic frequency
    eps, sigma, mu = 1.0, 1.0, 200.0
    Vlj = lambda r: 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    r0 = find_minimum(Vlj, (0.9, 2.0))
    print(f"\nLennard-Jones: minimum at r0 = {r0:.4f} (= 2^(1/6) = {2**(1/6):.4f}), "
          f"stable? {is_stable_minimum(Vlj, r0)}")
    print(f"  omega = {harmonic_frequency(Vlj, r0, mu):.4f} "
          f"vs LJ formula sqrt(36*2^(2/3)/mu) = {np.sqrt(36*2**(2/3)/mu):.4f}")
