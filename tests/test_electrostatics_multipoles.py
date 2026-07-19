"""Test electrostatics_multipoles: the classic charge-quadrupole has zero
monopole and dipole, a symmetric traceless quadrupole tensor, and the
multipole expansion converges to the exact potential far from the charges."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import electrostatics_multipoles as em

charges, positions = em.quadrupole_square_example(q=1.0, a=1.0)

# 1. zero monopole and dipole by symmetry -- only the quadrupole survives
assert abs(em.monopole_moment(charges)) < 1e-12
assert np.allclose(em.dipole_moment(charges, positions), 0.0, atol=1e-12)

# 2. quadrupole tensor is symmetric and traceless (required of any quadrupole)
Theta = em.quadrupole_moment(charges, positions)
assert np.allclose(Theta, Theta.T)
assert abs(np.trace(Theta)) < 1e-10
assert not np.allclose(Theta, 0.0)   # the quadrupole itself must be nonzero

# 3. the multipole expansion converges to the exact potential as r -> infinity
errors = []
for r in [5.0, 50.0, 500.0]:
    fp = [r / np.sqrt(2), r / np.sqrt(2), 0.0]
    approx = em.multipole_potential(charges, positions, fp)["V_total"]
    exact = em.exact_potential(charges, positions, fp)
    errors.append(abs(approx - exact) / abs(exact))
assert errors[0] > errors[1] > errors[2]   # monotonically improving
assert errors[-1] < 1e-4

# 4. a single point charge has only a monopole (no dipole/quadrupole)
q_single = np.array([3.0])
pos_single = np.array([[0.0, 0.0, 0.0]])
assert abs(em.monopole_moment(q_single) - 3.0) < 1e-12
assert np.allclose(em.dipole_moment(q_single, pos_single), 0.0)
assert np.allclose(em.quadrupole_moment(q_single, pos_single), 0.0)

print("test_electrostatics_multipoles: all checks passed")
