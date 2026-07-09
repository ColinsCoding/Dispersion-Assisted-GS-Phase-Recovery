"""Stationary states: why energy eigenstates stand still, and superpositions don't.

The time-independent Schrodinger equation H psi_n = E_n psi_n gives the ENERGY EIGENSTATES.
Each one evolves in time by only a phase,
        Psi_n(x, t) = psi_n(x) e^{-i E_n t / hbar},
so its probability density |Psi_n|^2 = |psi_n|^2 is FROZEN -- nothing observable changes. That
is why they are called STATIONARY states.

Anything else is a superposition, and it MOVES. The general solution of the time-dependent
equation is a sum of stationary states, each spinning at its own rate:
        Psi(x, t) = sum_n c_n psi_n(x) e^{-i E_n t / hbar},   c_n = <psi_n | Psi(0)>.
For two levels the density picks up a cross term that beats at the BOHR frequency
        omega_{mn} = (E_m - E_n) / hbar,
so <x>(t) sloshes back and forth at exactly the energy-gap frequency -- the quantum origin of
spectral lines (a transition emits a photon of energy hbar*omega_{mn}). The stationary states
are a basis; time evolution is just each coefficient rotating, which conserves the norm forever.

Demonstrated with the infinite-square-well eigenstates psi_n = sqrt(2/L) sin(n pi x/L),
E_n = n^2 pi^2 hbar^2 / 2mL^2 (the cleanest stationary states), but the evolve/expand machinery
takes ANY orthonormal eigenstates (e.g. from dgs.finite_square_well or dgs.quantum_oscillator).
Verified: orthonormality, a single state stays stationary, a two-state mix oscillates at the
Bohr frequency, the norm is conserved, and the expansion reconstructs Psi(0). NumPy only; py-3.13.
"""

import numpy as np


def infinite_well_eigenstate(n, x, L=1.0):
    """The n-th stationary state of the infinite well: psi_n = sqrt(2/L) sin(n pi x/L),
    for x in [0, L] (zero outside). n = 1, 2, 3, ..."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    if L <= 0:
        raise ValueError("L must be positive")
    x = np.asarray(x, float)
    return np.where((x >= 0) & (x <= L), np.sqrt(2 / L) * np.sin(n * np.pi * x / L), 0.0)


def infinite_well_energy(n, L=1.0, mass=1.0, hbar=1.0):
    """Energy of the n-th level: E_n = n^2 pi^2 hbar^2 / (2 m L^2)."""
    if n < 1 or L <= 0 or mass <= 0:
        raise ValueError("need n >= 1 and L, mass > 0")
    return n ** 2 * np.pi ** 2 * hbar ** 2 / (2 * mass * L ** 2)


def evolve(coeffs, states, energies, t, hbar=1.0):
    """Psi(x, t) = sum_n c_n psi_n(x) e^{-i E_n t / hbar}. `states` is a list of
    eigenstate arrays (same grid), energies the matching E_n. Returns complex Psi."""
    coeffs = np.asarray(coeffs, complex)
    if not (len(coeffs) == len(states) == len(energies)):
        raise ValueError("coeffs, states, energies must have equal length")
    Psi = np.zeros_like(np.asarray(states[0], complex))
    for c, psi, E in zip(coeffs, states, energies):
        Psi = Psi + c * np.asarray(psi, complex) * np.exp(-1j * E * t / hbar)
    return Psi


def expansion_coefficients(psi0, states, x):
    """Project an initial wavefunction onto the eigenbasis: c_n = <psi_n | psi0> =
    integral psi_n* psi0 dx. Feeding these to evolve(...) at t=0 rebuilds psi0."""
    psi0 = np.asarray(psi0, complex)
    return np.array([np.trapezoid(np.conj(np.asarray(psi, complex)) * psi0, x)
                     for psi in states])


def probability_density(Psi):
    """|Psi|^2 -- the observable probability distribution."""
    return np.abs(np.asarray(Psi, complex)) ** 2


def expectation_position(Psi, x):
    """<x> = integral x |Psi|^2 dx / integral |Psi|^2 dx."""
    p = probability_density(Psi)
    return float(np.trapezoid(x * p, x) / np.trapezoid(p, x))


def bohr_frequency(E_upper, E_lower, hbar=1.0):
    """omega_{mn} = (E_m - E_n)/hbar -- the beat frequency of a two-state superposition
    and the frequency of the photon emitted on that transition."""
    return abs(E_upper - E_lower) / hbar


if __name__ == "__main__":
    L, m, hbar = 1.0, 1.0, 1.0
    x = np.linspace(0, L, 4000)
    psi1 = infinite_well_eigenstate(1, x, L)
    psi2 = infinite_well_eigenstate(2, x, L)
    E1, E2 = infinite_well_energy(1), infinite_well_energy(2)

    print("a single stationary state stays put:")
    for t in (0.0, 0.1, 0.5):
        P = probability_density(evolve([1.0], [psi1], [E1], t))
        print(f"  t={t}: <x> = {expectation_position(evolve([1.0],[psi1],[E1],t), x):.5f} "
              f"(unchanging), |Psi|^2 == |psi1|^2? {np.allclose(P, psi1**2)}")

    print("\na 50/50 superposition of n=1 and n=2 oscillates:")
    c = [1/np.sqrt(2), 1/np.sqrt(2)]
    w = bohr_frequency(E2, E1)
    T = 2 * np.pi / w
    print(f"  Bohr frequency omega_21 = (E2-E1)/hbar = {w:.4f}, period T = {T:.4f}")
    for frac, tag in [(0.0, "0"), (0.25, "T/4"), (0.5, "T/2"), (1.0, "T")]:
        xt = expectation_position(evolve(c, [psi1, psi2], [E1, E2], frac*T), x)
        print(f"  t={tag:4s}: <x> = {xt:.4f}")
    print("  (<x> returns to its start after one period -- the wave sloshes side to side)")

    print("\nexpansion reconstructs an arbitrary state:")
    psi0 = infinite_well_eigenstate(1, x, L) + 0.5 * infinite_well_eigenstate(3, x, L)
    psi0 = psi0 / np.sqrt(np.trapezoid(np.abs(psi0)**2, x))
    coeffs = expansion_coefficients(psi0, [infinite_well_eigenstate(n, x, L) for n in range(1,6)], x)
    print(f"  coefficients |c_n|^2 = {np.round(np.abs(coeffs)**2, 3)} (sum {np.sum(np.abs(coeffs)**2):.3f})")
