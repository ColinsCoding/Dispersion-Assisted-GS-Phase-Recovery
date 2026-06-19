"""Smoke-test EM waves in conductors: complex k~ and skin depth (Griffiths 9.4.1)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import electrodynamics as ed

eps0, mu0 = ed._EPS0, ed._MU0
sigma_cu = 5.96e7                       # copper conductivity [S/m]

# 1. the defining relation k~^2 = mu eps omega^2 + i mu sigma omega (eq 9.124)
for f in (60.0, 1e6, 1e9):
    w = 2 * np.pi * f
    kt = ed.conductor_wavenumber(w, sigma_cu)
    assert np.allclose(kt**2, mu0 * eps0 * w**2 + 1j * mu0 * sigma_cu * w, rtol=1e-9)

# 2. good-conductor limit: k ~ kappa ~ sqrt(omega mu sigma / 2)
w = 2 * np.pi * 1e6
kt = ed.conductor_wavenumber(w, sigma_cu)
approx = np.sqrt(w * mu0 * sigma_cu / 2)
assert abs(kt.real - approx) / approx < 1e-3
assert abs(kt.imag - approx) / approx < 1e-3
assert abs(kt.real - kt.imag) / kt.real < 1e-3       # k ~ kappa in a good conductor

# 3. skin depth matches sqrt(2/(omega mu sigma)) and the known copper numbers
d_1mhz = ed.skin_depth(2 * np.pi * 1e6, sigma_cu)
assert abs(d_1mhz - np.sqrt(2 / (2 * np.pi * 1e6 * mu0 * sigma_cu))) < 1e-12
assert 6e-5 < d_1mhz < 7e-5                          # ~66 um at 1 MHz
d_60hz = ed.skin_depth(2 * np.pi * 60, sigma_cu)
assert 8e-3 < d_60hz < 9e-3                          # ~8.5 mm at 60 Hz

# 4. skin depth shrinks as 1/sqrt(frequency)
d_100mhz = ed.skin_depth(2 * np.pi * 100e6, sigma_cu)
assert abs(d_1mhz / d_100mhz - 10.0) < 0.05          # 100x freq -> 10x thinner

# 5. validation
for bad in (lambda: ed.conductor_wavenumber(-1, sigma_cu), lambda: ed.skin_depth(2*np.pi*1e6, 0.0)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad input")

print(f"SMOKE PASS  (copper skin depth: {d_60hz*1e3:.1f} mm @60Hz, "
      f"{d_1mhz*1e6:.0f} um @1MHz; k~kappa in good conductor)")
