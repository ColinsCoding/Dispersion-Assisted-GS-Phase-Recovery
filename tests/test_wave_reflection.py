"""Test dgs.wave_reflection: reflection/transmission coefficients, power conservation, VSWR,
standing-wave envelope, string ends, Fresnel/Brewster/critical angles, and impedance matching."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import wave_reflection as wr

# 1. reflection coefficient limits: matched, open, short
assert wr.reflection_coefficient(50, 50) == 0.0          # matched -> no reflection
assert wr.reflection_coefficient(50, np.inf) == 1.0      # open -> +1
assert wr.reflection_coefficient(50, 0) == -1.0          # short/fixed -> -1
# a partial mismatch
assert math.isclose(wr.reflection_coefficient(50, 100), (100-50)/(100+50))

# 2. transmission amplitude tau = 1 + Gamma
for Z2 in (50, 75, 100, 200):
    g = wr.reflection_coefficient(50, Z2)
    assert math.isclose(wr.transmission_coefficient(50, Z2), 1 + g)

# 3. power conservation R + T = 1 for any real mismatch
for Z2 in (10, 50, 75, 300):
    g = wr.reflection_coefficient(50, Z2)
    assert math.isclose(wr.power_reflectance(g) + wr.power_transmittance(g), 1.0)
# matched carries all the power, mismatch reflects some
assert wr.power_transmittance(wr.reflection_coefficient(50, 50)) == 1.0
assert wr.power_reflectance(wr.reflection_coefficient(50, 200)) > 0

# 4. VSWR: 1 when matched, infinite at total reflection, and the textbook 2:1 case
assert wr.vswr(0.0) == 1.0
assert math.isinf(wr.vswr(1.0))
assert math.isclose(wr.vswr(wr.reflection_coefficient(50, 100)), 2.0)   # 100/50 -> VSWR 2

# 5. standing-wave envelope: max = 1+|G|, min = 1-|G|, ratio = VSWR
g = wr.reflection_coefficient(50, 150)                    # |G| = 0.5
bz = np.linspace(0, 2*np.pi, 2000)
env = wr.standing_wave_pattern(g, bz)
assert math.isclose(env.max(), 1 + abs(g), rel_tol=1e-3)
assert math.isclose(env.min(), 1 - abs(g), rel_tol=1e-3)
assert math.isclose(env.max()/env.min(), wr.vswr(g), rel_tol=1e-3)

# 6. string boundaries: fixed inverts, free does not
assert wr.string_end_reflection("fixed") == -1.0
assert wr.string_end_reflection("free") == 1.0

# 7. optics: Fresnel 4% air-glass, Brewster and critical angles
r, R = wr.fresnel_normal_incidence(1.0, 1.5)
assert math.isclose(R, 0.04, abs_tol=1e-3)               # ~4% per surface
assert r < 0                                             # phase flip going into denser medium
# going the other way (glass->air) same reflectance, opposite sign
r2, R2 = wr.fresnel_normal_incidence(1.5, 1.0)
assert math.isclose(R, R2) and r2 > 0
assert math.isclose(np.degrees(wr.brewster_angle(1.0, 1.5)), 56.31, abs_tol=0.02)
assert math.isclose(np.degrees(wr.critical_angle(1.5, 1.0)), 41.81, abs_tol=0.02)
# defining identities: tan(theta_B)=n2/n1, sin(theta_c)=n2/n1
assert math.isclose(math.tan(wr.brewster_angle(1.0, 1.5)), 1.5 / 1.0, rel_tol=1e-9)
assert math.isclose(math.sin(wr.critical_angle(1.5, 1.0)), 1.0 / 1.5, rel_tol=1e-9)

# 8. impedance transforms: half-wave repeats the load, quarter-wave inverts about Z0
ZL, Z0 = 100.0, 50.0
Zhalf = wr.input_impedance(ZL, Z0, np.pi)                 # half wave: Z_in = Z_L
assert math.isclose(Zhalf.real, ZL, rel_tol=1e-6) and abs(Zhalf.imag) < 1e-6
# quarter-wave transformer matches 100 ohm load to 50 ohm source
Z0t = wr.quarter_wave_transformer(50, 100)
assert math.isclose(Z0t, math.sqrt(50*100))
Zin = wr.input_impedance(100, Z0t, np.pi/2)              # Z_in = Z0^2/Z_L = 50
assert math.isclose(Zin.real, 50.0, rel_tol=1e-6)
assert abs(wr.reflection_coefficient(50, Zin.real)) < 1e-9   # now matched

# 9. kwarg bounds
for bad in (lambda: wr.string_end_reflection("clamped"),
            lambda: wr.fresnel_normal_incidence(0, 1.5),
            lambda: wr.critical_angle(1.0, 1.5),            # n1 < n2: no TIR
            lambda: wr.quarter_wave_transformer(-1, 50)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_wave_reflection: all checks passed")
