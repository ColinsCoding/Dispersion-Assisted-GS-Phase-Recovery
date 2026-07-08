"""Test dgs.average_power: the trig identities <cos^2>=1/2 and <cos cos(+phi)>=
1/2 cos(phi) (symbolic + numeric), RMS = peak/sqrt2, AC real power 1/2 V0 I0 cos(phi)
= V_rms I_rms cos(phi) with power factor, and optical intensity 1/2 c eps0 E0^2 =
c eps0 E_rms^2 -- the SAME 1/2 in a circuit and in a light wave."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import average_power as ap

# 1. the trig identities, symbolic
sym = ap.average_identities_symbolic()
phi = sp.symbols("phi", real=True)
assert sym["cos_squared"] == sp.Rational(1, 2)
assert sp.simplify(sym["cos_product"] - sp.cos(phi) / 2) == 0

# 2. and numeric: <cos^2>=1/2, <cos cos(+phi)> = 1/2 cos(phi)
assert np.isclose(ap.average_cos_squared_numeric(), 0.5, atol=1e-4)
for p in (0.0, np.pi/3, np.pi/2, np.pi):
    assert np.isclose(ap.average_product_numeric(p), 0.5*np.cos(p), atol=1e-4)
assert np.isclose(ap.average_product_numeric(np.pi/2), 0.0, atol=1e-4)   # orthogonal

# 3. RMS = peak/sqrt(2), and matches the RMS of an actual sampled sinusoid
assert np.isclose(ap.rms_sinusoid(170), 170/np.sqrt(2))
t = np.linspace(0, 2*np.pi, 100000, endpoint=False)
assert np.isclose(ap.rms(5.0*np.cos(t)), ap.rms_sinusoid(5.0), rtol=1e-3)
assert np.isclose(ap.rms_sinusoid(170), 120.2, atol=0.1)                 # 120 V mains

# 4. AC real power = 1/2 V0 I0 cos(phi) = V_rms I_rms cos(phi), verified numerically
V0, I0 = 170.0, 10.0
assert np.isclose(ap.average_power_ac(V0, I0, 0.0), 0.5*V0*I0)           # resistive: max
assert np.isclose(ap.average_power_ac(V0, I0, np.pi/2), 0.0, atol=1e-9)  # reactive: zero
assert np.isclose(ap.average_power_ac(V0, I0, np.pi/3), 0.5*V0*I0*0.5)   # pf 0.5
# equals the RMS-product form
assert np.isclose(ap.average_power_ac(V0, I0, np.pi/4),
                  ap.rms_sinusoid(V0)*ap.rms_sinusoid(I0)*np.cos(np.pi/4))
# independent numeric time-average agrees
for p in (0.0, np.pi/3, np.pi/2, 1.0):
    assert np.isclose(ap.average_power_numeric(V0, I0, p),
                      ap.average_power_ac(V0, I0, p), atol=1e-2)

# 5. power factor
assert np.isclose(ap.power_factor(0.0), 1.0)
assert np.isclose(ap.power_factor(np.pi/2), 0.0, atol=1e-12)
assert np.isclose(ap.power_factor(np.pi/3), 0.5)

# 6. optical intensity = 1/2 c eps0 E0^2 = c eps0 E_rms^2 (the SAME 1/2)
E0 = 1000.0
I = ap.optical_intensity(E0)
assert np.isclose(I, 0.5 * ap.C_LIGHT * ap.EPS0 * E0**2)
assert np.isclose(I, ap.C_LIGHT * ap.EPS0 * ap.rms_sinusoid(E0)**2)      # = c eps0 E_rms^2
assert np.isclose(I, ap.intensity_numeric(E0), rtol=1e-4)               # numeric Poynting avg
assert np.isclose(ap.optical_intensity(2*E0), 4*I)                     # ~ amplitude^2
assert np.isclose(ap.optical_intensity(E0, 1.5), 1.5*I)               # ~ n

# 7. the unification: the AC-power 1/2 and the intensity 1/2 are the SAME <cos^2>
assert np.isclose(ap.average_power_ac(1, 1, 0), ap.average_cos_squared_numeric(), atol=1e-4)

# 8. kwarg bounds
for bad in (lambda: ap.rms_sinusoid(-1),
            lambda: ap.rms([]),
            lambda: ap.average_power_ac(-1, 1, 0),
            lambda: ap.optical_intensity(1, n_index=0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_average_power: all checks passed")
