"""Test non-plane-wave solutions of Helmholtz: cylindrical (Bessel) and spherical."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import mpmath as mp
from griffiths import bessel as bz

# 1. cylindrical wave J_m(k rho) e^{i m phi} solves the 2-D Helmholtz equation
for m in (0, 1, 2, 3):
    assert bz.cylindrical_wave_residual(m) == 0, m

# 2. outgoing spherical wave e^{ikr}/r solves the 3-D radial Helmholtz equation
assert bz.spherical_wave_residual() == 0

# 3. far-field amplitude decay exponents (energy conservation)
assert bz.wave_amplitude_decay("plane") == 0.0
assert bz.wave_amplitude_decay("cylindrical") == 0.5
assert bz.wave_amplitude_decay("spherical") == 1.0
try:
    bz.wave_amplitude_decay("conical")
except ValueError:
    pass
else:
    raise AssertionError("unknown geometry should raise")

# 4. the cylindrical decay is real: J_0(x)*sqrt(x) is bounded (J_0 ~ sqrt(2/pi x) cos),
#    confirming amplitude ~ 1/sqrt(rho) far from a line source
xs = np.linspace(50, 200, 50)
env = np.array([abs(float(mp.besselj(0, x))) * np.sqrt(x) for x in xs])
assert env.max() < 1.0 and env.max() > 0.3            # bounded near sqrt(2/pi)=0.798

# 5. spherical decay: |e^{ikr}/r| * r = 1 exactly (amplitude ~ 1/r)
r = np.linspace(1, 100, 50)
assert np.allclose(np.abs(np.exp(1j * 2.0 * r) / r) * r, 1.0)

print("TEST PASS  (cylindrical J_m(k*rho)e^{i m phi} & spherical e^{ikr}/r solve "
      "Helmholtz; decay plane:0, cyl:1/2, sph:1)")
