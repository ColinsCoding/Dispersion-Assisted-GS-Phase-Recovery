"""Test dgs.maxwell_boltzmann: the speed distribution (normalized, peak at v_mp), the
three characteristic speeds and their ordering, the moments <v>=v_avg and
sqrt<v^2>=v_rms by numerical integration, equipartition <KE>=3/2 kT, and the T/m
scalings."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import maxwell_boltzmann as mb

k = mb.K_BOLTZ
m = 28 * mb.AMU        # N2
T = 300.0

# 1. the three characteristic speeds, closed forms
assert np.isclose(mb.rms_speed(m, T), np.sqrt(3 * k * T / m))
assert np.isclose(mb.mean_speed(m, T), np.sqrt(8 * k * T / (np.pi * m)))
assert np.isclose(mb.most_probable_speed(m, T), np.sqrt(2 * k * T / m))
assert np.isclose(mb.rms_speed(m, 300), 517, atol=1)          # N2 ~517 m/s

# 2. ordering v_mp < v_avg < v_rms with the fixed ratios
vmp, vavg, vrms = (mb.most_probable_speed(m, T), mb.mean_speed(m, T), mb.rms_speed(m, T))
assert vmp < vavg < vrms
assert np.isclose(vrms / vmp, np.sqrt(1.5))                   # sqrt(3/2)
assert np.isclose(vavg / vmp, np.sqrt(4 / np.pi))            # sqrt(4/pi)

# 3. the pdf: non-negative, peaks at v_mp, integrates to 1
v = np.linspace(1, 6 * vrms, 400000)
f = mb.maxwell_boltzmann_pdf(v, m, T)
assert np.all(f >= 0)
assert np.isclose(v[np.argmax(f)], vmp, rtol=1e-3)           # peak at most-probable speed
assert np.isclose(np.trapezoid(f, v), 1.0, atol=1e-3)        # normalized

# 4. moments match the closed-form speeds (independent numerical check)
norm, v1, v2 = mb._moments(m, T)
assert np.isclose(norm, 1.0, atol=1e-3)
assert np.isclose(v1, vavg, rtol=1e-3)                       # <v> = mean speed
assert np.isclose(np.sqrt(v2), vrms, rtol=1e-3)             # sqrt<v^2> = rms speed

# 5. equipartition: <KE> = 3/2 kT = 1/2 m v_rms^2
assert np.isclose(mb.average_kinetic_energy(T), 1.5 * k * T)
assert np.isclose(mb.average_kinetic_energy(T), 0.5 * m * vrms ** 2)
assert np.isclose(mb.average_kinetic_energy(T, dof=5), 2.5 * k * T)   # diatomic, more DOF

# 6. scalings: v_rms ~ sqrt(T) and ~ 1/sqrt(m)
assert np.isclose(mb.rms_speed(m, 4 * T) / mb.rms_speed(m, T), 2.0)   # 4x T -> 2x v
assert np.isclose(mb.rms_speed(4 * m, T) / mb.rms_speed(m, T), 0.5)   # 4x m -> half v

# 7. kwarg bounds
for bad in (lambda: mb.rms_speed(0, T),
            lambda: mb.maxwell_boltzmann_pdf(100, m, 0),
            lambda: mb.most_probable_speed(m, -1),
            lambda: mb.average_kinetic_energy(300, dof=0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_maxwell_boltzmann: all checks passed")
