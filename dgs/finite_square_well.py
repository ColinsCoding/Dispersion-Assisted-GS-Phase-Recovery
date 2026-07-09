"""The finite square well: bound-state energies and how far the particle leaks out.

A particle trapped in a well of depth V0 and half-width a is the first quantum system where
the walls are NOT infinite -- and two things change. First, the wavefunction does not stop
at the wall: it leaks into the classically forbidden region (E < V0) as a decaying
exponential e^{-x/delta}, with PENETRATION DEPTH
        delta = 1 / kappa = hbar / sqrt(2 m (V0 - E)),
deeper the closer E is to the top of the well. Second, because the wavefunction spreads a
little beyond the walls, the well behaves slightly WIDER, so the bound energies come out
LOWER than the infinite-well levels -- and there are only FINITELY many.

The bound energies are not a clean formula; they are the solutions of transcendental
matching conditions (continuity of psi and psi' at the walls). In the dimensionless
variables z = k a and z0 = a sqrt(2 m V0)/hbar (the "well strength"):
        even states:  z tan(z)  = sqrt(z0^2 - z^2),
        odd  states: -z cot(z)  = sqrt(z0^2 - z^2),
solved numerically here (their intersection with the quarter-circle of radius z0). The
number of bound states is floor(z0 / (pi/2)) + 1 -- even the shallowest well holds one.
The energies then are E_n = z_n^2 hbar^2 / (2 m a^2).

Ties to dgs.stability_of_matter / dgs.quantum_oscillator (bound states of a potential) and,
as V0 -> infinity, to the infinite square well. Verified: the state count, energies below
the infinite well, deeper penetration for weakly bound states, and the deep-well limit.
NumPy only; py-3.13.
"""

import numpy as np


def well_strength(V0, a, mass=1.0, hbar=1.0):
    """The dimensionless well parameter z0 = a sqrt(2 m V0)/hbar. Bigger z0 (deeper or
    wider well) holds more bound states."""
    if V0 <= 0 or a <= 0 or mass <= 0:
        raise ValueError("V0, a, mass must be positive")
    return a * np.sqrt(2 * mass * V0) / hbar


def num_bound_states(V0, a, mass=1.0, hbar=1.0):
    """Number of bound states = floor(z0/(pi/2)) + 1 -- a finite well always holds at
    least one, and gains another every time z0 crosses a multiple of pi/2."""
    z0 = well_strength(V0, a, mass, hbar)
    return int(np.floor(z0 / (np.pi / 2)) + 1)


def _bound_z(z0, n_scan=200000):
    """Solve the even/odd transcendental conditions for z in (0, z0): find where
    z tan(z) or -z cot(z) crosses sqrt(z0^2 - z^2). Returns sorted z-roots."""
    z = np.linspace(1e-6, z0 - 1e-9, n_scan)
    rhs = np.sqrt(np.clip(z0 ** 2 - z ** 2, 0, None))
    roots = []
    for lhs in (z * np.tan(z), -z / np.tan(z)):        # even, odd
        g = lhs - rhs
        for i in range(len(z) - 1):
            gi, gj = g[i], g[i + 1]
            # a real root: sign change with small, finite values (skip tan/cot poles)
            if np.isfinite(gi) and np.isfinite(gj) and gi * gj < 0 \
                    and abs(gi) < 5 and abs(gj) < 5:
                roots.append(z[i] - gi * (z[i + 1] - z[i]) / (gj - gi))
    return sorted(roots)


def bound_state_energies(V0, a, mass=1.0, hbar=1.0):
    """The bound-state energies E_n = z_n^2 hbar^2/(2 m a^2), 0 < E_n < V0, sorted
    ascending -- the quantized levels of the finite well."""
    z0 = well_strength(V0, a, mass, hbar)
    zs = _bound_z(z0)
    return np.array([z ** 2 * hbar ** 2 / (2 * mass * a ** 2) for z in zs])


def penetration_depth(E, V0, mass=1.0, hbar=1.0):
    """How far the wavefunction leaks past the wall: delta = hbar/sqrt(2 m (V0 - E)),
    the 1/e decay length of e^{-x/delta}. Diverges as E -> V0 (barely bound states
    leak far); tiny for deep states."""
    if not 0 <= E < V0:
        raise ValueError("need 0 <= E < V0 (a bound state)")
    if mass <= 0:
        raise ValueError("mass must be positive")
    return hbar / np.sqrt(2 * mass * (V0 - E))


def infinite_well_energies(a, mass=1.0, hbar=1.0, n_levels=4):
    """Levels of the INFINITE well of the same full width L = 2a:
    E_n = n^2 pi^2 hbar^2 / (2 m L^2). The finite-well levels sit just below these."""
    if a <= 0 or n_levels < 1:
        raise ValueError("a > 0 and n_levels >= 1")
    L = 2 * a
    n = np.arange(1, n_levels + 1)
    return n ** 2 * np.pi ** 2 * hbar ** 2 / (2 * mass * L ** 2)


if __name__ == "__main__":
    V0, a = 20.0, 1.0        # hbar = m = 1
    z0 = well_strength(V0, a)
    print(f"well: V0={V0}, half-width a={a} -> strength z0 = {z0:.3f}")
    print(f"predicted bound states: {num_bound_states(V0, a)}")

    E = bound_state_energies(V0, a)
    Einf = infinite_well_energies(a, n_levels=len(E))
    print(f"\n{len(E)} bound levels (finite < infinite, because psi leaks out):")
    for n, (e, ei) in enumerate(zip(E, Einf)):
        d = penetration_depth(e, V0)
        print(f"  E_{n} = {e:6.3f}  (infinite well {ei:6.3f}),  "
              f"penetration depth delta = {d:.3f} a")

    print(f"\nhighest state penetrates {penetration_depth(E[-1], V0):.3f} a, "
          f"ground state only {penetration_depth(E[0], V0):.3f} a "
          f"(weakly bound leaks farther)")

    print("\ndeep-well limit (V0=2000): finite levels approach the infinite well")
    Ed = bound_state_energies(2000.0, a)
    Ei = infinite_well_energies(a, n_levels=3)
    print(f"  E_0,1,2 = {np.round(Ed[:3],3)}  vs infinite {np.round(Ei,3)}")
