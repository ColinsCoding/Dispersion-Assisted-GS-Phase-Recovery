"""Test dgs.quantum_operators: <x>, <p>, Delta_x, Delta_p, <T> on a Gaussian packet
(the minimum-uncertainty state, Delta_x Delta_p = hbar/2), the reality of the
expectation values (Hermitian observables), the sharp/fuzzy x-p tradeoff between narrow
and wide packets, and the Heisenberg bound Delta_x Delta_p >= hbar/2."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import quantum_operators as qo

hbar = 1.0
x = np.linspace(-40, 40, 8192)
x0, sigma, k0, m = 3.0, 1.5, 2.0, 1.0
psi = qo.gaussian_packet(x, x0, sigma, k0)

# 1. Gaussian packet: <x>=x0, Delta_x=sigma
assert np.isclose(qo.expectation_position(psi, x), x0, atol=1e-4)
assert np.isclose(qo.uncertainty_position(psi, x), sigma, rtol=1e-3)

# 2. <p> = hbar k0, Delta_p = hbar/(2 sigma)
assert np.isclose(qo.expectation_momentum(psi, x, hbar), hbar * k0, rtol=1e-3)
assert np.isclose(qo.uncertainty_momentum(psi, x, hbar), hbar / (2 * sigma), rtol=1e-3)

# 3. the Gaussian saturates Heisenberg: Delta_x Delta_p = hbar/2 exactly
assert np.isclose(qo.heisenberg_product(psi, x, hbar), hbar / 2, rtol=1e-3)

# 4. kinetic energy <T> = <p^2>/2m = (hbar^2 k0^2 + hbar^2/4sigma^2)/2m
T_exact = (hbar**2 * k0**2 + hbar**2 / (4 * sigma**2)) / (2 * m)
assert np.isclose(qo.kinetic_energy(psi, x, m, hbar), T_exact, rtol=1e-3)
assert qo.kinetic_energy(psi, x, m, hbar) > 0                    # KE is non-negative

# 5. observables give REAL expectation values (Hermitian operators)
dx = x[1] - x[0]
p_psi = -1j * hbar * qo._d_dx(qo.normalize(psi, x), dx)
val = np.trapezoid(np.conj(qo.normalize(psi, x)) * p_psi, x)
assert abs(val.imag) < 1e-6 * (abs(val.real) + 1)              # <p> is real
# a stationary (real) packet has <p> = 0
assert np.isclose(qo.expectation_momentum(qo.gaussian_packet(x, 0, 1.0, 0.0), x), 0.0, atol=1e-6)

# 6. SHARP vs FUZZY: narrow in x is fuzzy in p, wide in x is sharp in p
narrow = qo.gaussian_packet(x, 0, 0.3, 0)
wide = qo.gaussian_packet(x, 0, 5.0, 0)
dxn, dpn = qo.uncertainty_position(narrow, x), qo.uncertainty_momentum(narrow, x)
dxw, dpw = qo.uncertainty_position(wide, x), qo.uncertainty_momentum(wide, x)
assert dxn < dxw and dpn > dpw                                # traded off
assert np.isclose(dpw, hbar / (2 * 5.0), rtol=1e-2)          # wide -> sharp p ~0.1
assert np.isclose(dxn * dpn, hbar / 2, rtol=1e-2)           # both stay at the minimum
assert np.isclose(dxw * dpw, hbar / 2, rtol=1e-2)

# 7. a non-Gaussian state exceeds the Heisenberg minimum
two_bump = qo.gaussian_packet(x, -4, 1, 0) + qo.gaussian_packet(x, 4, 1, 0)
assert qo.heisenberg_product(two_bump, x, hbar) > hbar / 2 * 1.5   # well above the minimum

# 8. kwarg bounds
for bad in (lambda: qo.normalize(np.zeros(len(x)), x),
            lambda: qo.kinetic_energy(psi, x, mass=0),
            lambda: qo.gaussian_packet(x, 0, -1, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_quantum_operators: all checks passed")
