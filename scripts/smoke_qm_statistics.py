"""Smoke-test the QM statistical interpretation: <x>, sigma, Heisenberg (hbar=1)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import quantum as q

# 1. Gaussian packet: <x>=x0, sigma_x=sigma, <p>=k0 (hbar=1)
x = np.linspace(-30, 30, 8000)
x0, k0, sig = 3.0, 1.5, 1.2
psi = q.gaussian_packet(x, x0, k0, sig)
assert abs(q.expectation_x(psi, x) - x0) < 1e-3
assert abs(q.sigma_x(psi, x) - sig) < 1e-2
assert abs(q.expectation_p(psi, x) - k0) < 1e-3

# 2. the Gaussian SATURATES the uncertainty principle: sigma_x sigma_p = 1/2
prod = q.uncertainty_product(psi, x)
assert abs(prod - 0.5) < 1e-2, prod
# and sigma_p = 1/(2 sigma) for a Gaussian
assert abs(q.sigma_p(psi, x) - 1 / (2 * sig)) < 1e-2

# 3. tighter position (smaller sigma) -> wider momentum (product stays ~1/2)
narrow = q.gaussian_packet(x, 0.0, 0.0, 0.5)
wide = q.gaussian_packet(x, 0.0, 0.0, 2.0)
assert q.sigma_x(narrow, x) < q.sigma_x(wide, x)
assert q.sigma_p(narrow, x) > q.sigma_p(wide, x)            # the trade-off
assert abs(q.uncertainty_product(narrow, x) - 0.5) < 2e-2

# 4. a NON-Gaussian (infinite-well ground state) exceeds the bound: product > 1/2
L = 10.0
xb = np.linspace(0, L, 4000)
box = np.sqrt(2 / L) * np.sin(np.pi * xb / L)              # n=1 stationary state
assert q.uncertainty_product(box, xb) > 0.5
assert abs(q.expectation_x(box, xb) - L / 2) < 1e-2        # symmetric -> centered
assert abs(q.expectation_p(box, xb)) < 1e-6                # real state -> <p>=0

# 5. <p>=0 for any real wavefunction (no net momentum)
assert abs(q.expectation_p(q.gaussian_packet(x, 1.0, 0.0, 1.0), x)) < 1e-6

print(f"SMOKE PASS  (Gaussian saturates Heisenberg: sigma_x*sigma_p={prod:.4f} ~ 0.5; "
      f"box ground state = {q.uncertainty_product(box, xb):.3f} > 0.5)")
