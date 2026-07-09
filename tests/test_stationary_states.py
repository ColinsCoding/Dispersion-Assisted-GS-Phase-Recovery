"""Test dgs.stationary_states: orthonormal infinite-well eigenstates, a single
stationary state frozen in |Psi|^2, a two-state superposition sloshing at the Bohr
frequency (E2-E1)/hbar, norm conservation, and expansion reconstructing Psi(0)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import stationary_states as ss

L = 1.0
x = np.linspace(0, L, 4000)
states = [ss.infinite_well_eigenstate(n, x, L) for n in range(1, 6)]
energies = [ss.infinite_well_energy(n, L) for n in range(1, 6)]

# 1. eigenstates are orthonormal and vanish at the walls
for i in range(4):
    for j in range(4):
        overlap = np.trapezoid(states[i] * states[j], x)
        assert np.isclose(overlap, 1.0 if i == j else 0.0, atol=1e-3)
assert np.isclose(states[0][0], 0.0) and np.isclose(states[0][-1], 0.0, atol=1e-9)

# 2. energies E_n = n^2 pi^2 hbar^2 / 2mL^2
assert np.isclose(ss.infinite_well_energy(1), np.pi**2 / 2)
assert np.isclose(ss.infinite_well_energy(2) / ss.infinite_well_energy(1), 4.0)   # ~ n^2

# 3. a single stationary state: |Psi(t)|^2 is FROZEN, <x> constant
psi1, E1 = states[0], energies[0]
for t in (0.0, 0.1, 0.5, 3.0):
    P = ss.probability_density(ss.evolve([1.0], [psi1], [E1], t))
    assert np.allclose(P, psi1 ** 2)                             # stationary
    assert np.isclose(ss.expectation_position(ss.evolve([1.0], [psi1], [E1], t), x), 0.5)

# 4. a 50/50 superposition of n=1,2 OSCILLATES at the Bohr frequency
psi2, E2 = states[1], energies[1]
c = [1 / np.sqrt(2), 1 / np.sqrt(2)]
w = ss.bohr_frequency(E2, E1)
assert np.isclose(w, E2 - E1)                                   # (E2-E1)/hbar
T = 2 * np.pi / w
xt = lambda t: ss.expectation_position(ss.evolve(c, [psi1, psi2], [E1, E2], t), x)
x0 = xt(0.0)
assert not np.isclose(x0, 0.5, atol=1e-2)                       # displaced from center
assert np.isclose(xt(T), x0, atol=1e-3)                        # periodic: returns after T
assert np.isclose(xt(T / 2), 1.0 - x0, atol=1e-3)             # mirror about 0.5 at half period
assert np.isclose(xt(T / 4), 0.5, atol=1e-3)                  # passes through center
assert abs(xt(T / 2) - x0) > 0.1                              # it genuinely moved

# 5. norm is conserved for all time (only phases rotate)
for t in (0.0, 0.13, 0.5, 2.7):
    Psi = ss.evolve(c, [psi1, psi2], [E1, E2], t)
    assert np.isclose(np.trapezoid(ss.probability_density(Psi), x), 1.0, atol=1e-3)

# 6. expansion reconstructs an arbitrary initial state
psi0 = psi1 + 0.5 * states[2]                                  # n=1 + 0.5 n=3
psi0 = psi0 / np.sqrt(np.trapezoid(np.abs(psi0) ** 2, x))
coeffs = ss.expansion_coefficients(psi0, states, x)
assert np.isclose(np.abs(coeffs[0]) ** 2, 0.8, atol=1e-2)     # |c1|^2 : |c3|^2 = 4 : 1
assert np.isclose(np.abs(coeffs[2]) ** 2, 0.2, atol=1e-2)
assert np.isclose(np.sum(np.abs(coeffs) ** 2), 1.0, atol=1e-3)   # complete basis
rebuilt = ss.evolve(coeffs, states, energies, 0.0)
assert np.allclose(rebuilt.real, psi0, atol=1e-3)            # evolve at t=0 rebuilds psi0

# 7. kwarg bounds
for bad in (lambda: ss.infinite_well_eigenstate(0, x),
            lambda: ss.infinite_well_energy(1, L=0),
            lambda: ss.evolve([1, 1], [psi1], [E1], 0.0)):     # length mismatch
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_stationary_states: all checks passed")
