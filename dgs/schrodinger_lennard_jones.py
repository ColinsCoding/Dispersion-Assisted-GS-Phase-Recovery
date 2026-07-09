"""Schrodinger in a Lennard-Jones well: the quantized vibrations of a molecule.

Two atoms attract weakly at long range and repel hard when they touch -- the Lennard-Jones
potential
        V(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ],
a bowl with minimum -eps at r0 = 2^(1/6) sigma. Classically the atoms can sit anywhere in
the bowl; QUANTUM mechanically the relative motion is a particle of reduced mass mu in that
bowl, and the time-independent Schrodinger equation
        -hbar^2/2mu  psi'' + V(r) psi = E psi
allows only DISCRETE bound energies E < 0 -- the VIBRATIONAL LEVELS of the molecule.

Two limits check the solution:
  * NEAR THE BOTTOM the bowl is a parabola, V ~ -eps + 1/2 k (r-r0)^2 with the LJ force
    constant k = 36 * 2^(2/3) eps/sigma^2, so the lowest levels are the harmonic ladder
    E_n ~ -eps + (n + 1/2) hbar omega, omega = sqrt(k/mu) -- exactly dgs.quantum_oscillator,
    including the zero-point 1/2 hbar omega that lifts the ground state OFF the bottom.
  * NEAR THE TOP the bowl flattens (ANHARMONICITY), so real levels crowd together as they
    approach dissociation at E = 0 -- unlike the evenly spaced oscillator. A finite well
    holds only finitely many.

The Hamiltonian is the finite-difference operator of dgs.stability_of_matter /
dgs.vibration_modes with the LJ potential of dgs.lennard_jones on the diagonal -- solving
it is one eigenproblem. Verified: the well geometry, the harmonic ground state (zero-point
above the floor), the anharmonic crowding, and more bound states for a heavier molecule.
NumPy only; py-3.13.
"""

import numpy as np
from dgs.lennard_jones import lj_potential
from dgs.vibration_modes import second_difference_matrix

R_MIN_FACTOR = 2 ** (1 / 6)          # r0 / sigma at the potential minimum
K_LJ = 36 * 2 ** (2 / 3)             # LJ force constant in units of eps/sigma^2 (~57.15)


def lj_minimum(eps=1.0, sigma=1.0):
    """The bottom of the well: (r0 = 2^(1/6) sigma, V_min = -eps)."""
    if eps <= 0 or sigma <= 0:
        raise ValueError("eps and sigma must be positive")
    return R_MIN_FACTOR * sigma, -eps


def harmonic_frequency(mu, eps=1.0, sigma=1.0, hbar=1.0):
    """The small-oscillation frequency at the well bottom, omega = sqrt(k/mu) with
    k = V''(r0) = 36*2^(2/3) eps/sigma^2 -- the harmonic approximation to the molecule."""
    if mu <= 0 or eps <= 0 or sigma <= 0:
        raise ValueError("mu, eps, sigma must be positive")
    k = K_LJ * eps / sigma ** 2
    return np.sqrt(k / mu)


def harmonic_levels(mu, n_levels=4, eps=1.0, sigma=1.0, hbar=1.0):
    """The harmonic-approximation vibrational levels E_n = -eps + (n+1/2) hbar omega
    -- the ladder the true levels follow near the bottom, off the floor by the
    zero-point 1/2 hbar omega."""
    w = harmonic_frequency(mu, eps, sigma, hbar)
    n = np.arange(n_levels)
    return -eps + (n + 0.5) * hbar * w


def solve_bound_states(mu, eps=1.0, sigma=1.0, hbar=1.0,
                       r_lo=0.85, r_hi=4.0, n_grid=3000):
    """Solve -hbar^2/2mu psi'' + V(r) psi = E psi on [r_lo, r_hi]*sigma by finite
    differences and return the BOUND energies E < 0 (the vibrational levels), sorted.
    r_lo starts inside the repulsive wall (where psi ~ 0), r_hi past the well."""
    if mu <= 0 or r_lo <= 0 or r_hi <= r_lo:
        raise ValueError("need mu > 0 and 0 < r_lo < r_hi")
    r = np.linspace(r_lo * sigma, r_hi * sigma, n_grid)
    dr = r[1] - r[0]
    T = -(hbar ** 2) / (2 * mu) * second_difference_matrix(n_grid, dr)
    V = np.diag(lj_potential(r, eps, sigma))
    E = np.linalg.eigvalsh(T + V)
    return np.sort(E[E < 0])


if __name__ == "__main__":
    mu = 200.0        # dimensionless reduced mass (eps=sigma=hbar=1)
    r0, Vmin = lj_minimum()
    print(f"LJ well: minimum at r0 = {r0:.4f} sigma, depth V_min = {Vmin} eps")
    w = harmonic_frequency(mu)
    print(f"harmonic omega = {w:.4f}, zero-point 1/2 hbar omega = {w/2:.4f} eps\n")

    E = solve_bound_states(mu)
    print(f"{len(E)} bound vibrational levels (heavier -> more):")
    harm = harmonic_levels(mu, min(4, len(E)))
    for n in range(min(6, len(E))):
        hstr = f", harmonic {harm[n]:+.4f}" if n < len(harm) else ""
        print(f"  E_{n} = {E[n]:+.4f} eps{hstr}")

    print(f"\nharmonic ground state (V_min + 1/2 hbar omega) = {Vmin + w/2:+.4f}, "
          f"exact E_0 = {E[0]:+.4f}")
    gaps = np.diff(E[:5])
    print(f"level gaps {np.round(gaps,4)} -- shrinking (anharmonic crowding toward E=0)")

    print(f"\nheavier molecule (mu=800): {len(solve_bound_states(800.0))} bound levels "
          f"(vs {len(E)} at mu=200)")
