"""Test the generalized (Robertson) uncertainty principle: Pauli + position-momentum."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import robertson_uncertainty as ru

X, Y, Z = ru.SIGMA_X, ru.SIGMA_Y, ru.SIGMA_Z

# 1. the Pauli commutator algebra: [s_x,s_y]=2i s_z (cyclic)
assert np.allclose(ru.commutator(X, Y), 2j * Z)
assert np.allclose(ru.commutator(Y, Z), 2j * X)
assert np.allclose(ru.commutator(Z, X), 2j * Y)

# 2. expectation / variance: |up_z> is a sigma_z eigenstate (sharp), random on x,y
up = np.array([1, 0]); down = np.array([0, 1])
assert np.isclose(ru.expectation(Z, up).real, 1.0) and np.isclose(ru.variance(Z, up), 0.0)
assert np.isclose(ru.expectation(Z, down).real, -1.0)
assert np.isclose(ru.std(X, up), 1.0)                       # fully uncertain in x

# 3. spin-up saturates Robertson: sigma_x sigma_y = |<sigma_z>| = 1 (equality)
lhs, rhs, ok = ru.check_uncertainty(X, Y, up)
assert ok and np.isclose(lhs, 1.0) and np.isclose(rhs, 1.0)

# 4. Robertson holds for EVERY state (random spinors)
rng = np.random.default_rng(0)
for _ in range(200):
    psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    _, _, holds = ru.check_uncertainty(X, Y, psi)
    assert holds

# 5. commuting observables -> zero bound (no joint constraint). [sigma_z, sigma_z]=0
assert np.isclose(ru.uncertainty_bound(Z, Z, up), 0.0)
# a state aligned with +x has <sigma_z>=0, so the x-y bound vanishes there
plus_x = np.array([1, 1])
assert np.isclose(ru.uncertainty_bound(X, Y, plus_x), 0.0)

# 6. position-momentum: Gaussian saturates hbar/2; a chirp raises it; hbar scales the bound
t = np.linspace(-12, 12, 4096); g = np.exp(-t**2 / 2)
prod, bound = ru.position_momentum_uncertainty(t, g, hbar=1.0)
assert np.isclose(bound, 0.5) and abs(prod - 0.5) < 0.02     # Heisenberg minimum
chirped = g * np.exp(1j * 1.0 * t**2)
assert ru.position_momentum_uncertainty(t, chirped)[0] > 1.0  # chirp -> above the minimum
prod2, bound2 = ru.position_momentum_uncertainty(t, g, hbar=2.0)
assert np.isclose(bound2, 1.0) and abs(prod2 - 1.0) < 0.04    # scales with hbar

print(f"TEST PASS  (Pauli [s_x,s_y]=2i s_z cyclic; |up_z> sharp in z; spin saturates "
      f"sigma_x sigma_y=|<s_z>|=1; Robertson holds for 200 random states; commuting "
      f"-> bound 0; Gaussian sigma_x sigma_p=hbar/2)")
