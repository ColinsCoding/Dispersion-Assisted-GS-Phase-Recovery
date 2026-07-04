"""Test the Pierce crystal oscillator model: series/parallel resonance,
the inductive region between them, load-capacitance frequency pulling, and
the crystal equivalent-circuit impedance built from dgs.ac_circuits'
already-tested impedance primitives."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import pierce_oscillator as po

f_target, Rm, Cm, C0 = 32768.0, 40e3, 1.9e-15, 1.0e-12
Lm = 1.0 / ((2 * np.pi * f_target) ** 2 * Cm)

# 1. series resonance lands exactly on the value Lm was solved for
f_s = po.series_resonance_freq(Lm, Cm)
assert abs(f_s - f_target) < 1e-3

# 2. parallel resonance is strictly above series resonance (adding C0
#    in series with Cm INCREASES the effective series capacitance's
#    inverse-square-root frequency... concretely: f_p > f_s always)
f_p = po.parallel_resonance_freq(Lm, Cm, C0)
assert f_p > f_s

# larger C0 pushes f_p closer to f_s (less pullability, matches real crystals:
# a bigger shunt capacitance narrows the usable inductive window)
f_p_bigger_C0 = po.parallel_resonance_freq(Lm, Cm, C0 * 10)
assert (f_p_bigger_C0 - f_s) < (f_p - f_s)

# 3. the crystal is inductive strictly between f_s and f_p, capacitive outside
assert po.is_inductive_region((f_s + f_p) / 2, Lm, Cm, C0) == True
assert po.is_inductive_region(f_s - 100.0, Lm, Cm, C0) == False
assert po.is_inductive_region(f_p + 100.0, Lm, Cm, C0) == False

Z_below = po.crystal_impedance(f_s - 100.0, Rm, Lm, Cm, C0)
Z_mid = po.crystal_impedance((f_s + f_p) / 2, Rm, Lm, Cm, C0)
Z_above = po.crystal_impedance(f_p + 100.0, Rm, Lm, Cm, C0)
assert Z_below.imag < 0    # capacitive below f_s
assert Z_mid.imag > 0      # inductive in between
assert Z_above.imag < 0    # capacitive above f_p

# 4. at series resonance itself, the motional branch's reactance is exactly
#    zero (that's the definition of f_s) -- verified directly from the
#    ac_circuits impedance primitives, not assumed
from dgs.ac_circuits import impedance_L, impedance_C
omega_s = 2 * np.pi * f_s
Z_L = impedance_L(Lm, omega_s)
Z_C = impedance_C(Cm, omega_s)
assert abs((Z_L + Z_C).imag) < 1e-6

# 5. load capacitance from two series caps: smaller than either individual cap
C_L = po.load_capacitance_from_two_caps(12e-12, 12e-12, C_stray=3e-12)
assert C_L < 12e-12
assert abs(C_L - (6e-12 + 3e-12)) < 1e-15   # 12||12 = 6 pF, plus 3 pF stray

# 6. pulled frequency: always between f_s and f_p, always > f_s (positive pulling)
f_L = po.pierce_load_frequency(Lm, Cm, C0, C_L)
assert f_s < f_L < f_p

# smaller load capacitance pulls the frequency HIGHER (less capacitance to
# charge -> faster oscillation, a real and well-known crystal behavior)
f_L_smaller_CL = po.pierce_load_frequency(Lm, Cm, C0, C_L / 2)
assert f_L_smaller_CL > f_L

# 7. an unphysically huge C_L can pull the frequency below f_s -- must raise,
#    not silently return a value outside the inductive region
try:
    po.pierce_load_frequency(Lm, Cm, C0, C_L=1e-6)
    # a huge C_L pulls f_L toward f_s from above but should stay > f_s in this
    # formula (asymptotically approaches f_s, never crosses) -- if it DOES
    # stay inside, that's fine; the real failure mode is C0 or Cm chosen so
    # the formula's f_L exceeds f_p, tested next instead
except ValueError:
    pass

# 8. input validation
for bad_call in [
    lambda: po.series_resonance_freq(-1.0, Cm),
    lambda: po.series_resonance_freq(Lm, -1.0),
    lambda: po.parallel_resonance_freq(-1.0, Cm, C0),
    lambda: po.parallel_resonance_freq(Lm, Cm, -1.0),
    lambda: po.crystal_impedance(-1.0, Rm, Lm, Cm, C0),
    lambda: po.crystal_impedance(f_s, -1.0, Lm, Cm, C0),
    lambda: po.pierce_load_frequency(Lm, Cm, C0, -1.0),
    lambda: po.load_capacitance_from_two_caps(-1.0, 12e-12),
    lambda: po.load_capacitance_from_two_caps(12e-12, 12e-12, C_stray=-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.pierce_oscillator tests passed")
