"""Test dgs.waveguide_phase_group_velocity: v_p*v_g=c^2 must hold to high
precision at several f/f_c (checked against an INDEPENDENT numerical
domega/dbeta, not just the closed form re-evaluated), v_p must exceed c
while v_g stays below it, both must diverge/vanish correctly near cutoff,
and both must converge to c far above cutoff."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.waveguide_phase_group_velocity import (
    phase_velocity, group_velocity_closed_form, group_velocity_numerical,
    verify_vp_vg_equals_c2, velocities_near_cutoff,
)
from dgs.cylindrical_waveguide_resonance import waveguide_cutoff_frequency, C_LIGHT

a = 0.01   # 1 cm radius circular waveguide
fc = waveguide_cutoff_frequency(1, 1, a, "TE")

# 1. v_p > c and v_g < c for every propagating frequency tested
for ratio in (1.01, 1.1, 1.5, 3.0, 10.0):
    f = ratio * fc
    vp = phase_velocity(f, 1, 1, a, "TE")
    vg = group_velocity_closed_form(f, 1, 1, a, "TE")
    assert vp > C_LIGHT, f"v_p={vp:.3e} should exceed c at f/f_c={ratio}"
    assert vg < C_LIGHT, f"v_g={vg:.3e} should be below c at f/f_c={ratio}"

# 2. the closed-form and independently-computed numerical group velocity agree
for ratio in (1.05, 1.3, 2.0, 8.0):
    f = ratio * fc
    vg_closed = group_velocity_closed_form(f, 1, 1, a, "TE")
    vg_num = group_velocity_numerical(f, 1, 1, a, "TE")
    assert abs(vg_closed - vg_num) / vg_closed < 1e-4, \
        f"closed-form vg={vg_closed:.6e} vs numerical vg={vg_num:.6e} disagree at f/f_c={ratio}"

# 3. v_p * v_g = c^2 exactly (checked, both a boolean-safe call and the returned numbers)
for ratio in (1.02, 1.5, 4.0, 15.0):
    result = verify_vp_vg_equals_c2(ratio * fc, 1, 1, a, "TE")
    assert result["vp_vg_product_rel_err"] < 1e-4
    assert abs(result["v_p_times_v_g"] - C_LIGHT ** 2) / C_LIGHT ** 2 < 1e-4

# 4. near cutoff: v_p grows without bound, v_g shrinks toward 0, and v_p is
# monotonically DEcreasing while v_g is monotonically INcreasing as f/f_c grows
sweep = velocities_near_cutoff(1, 1, a, "TE", f_over_fc=np.array([1.0001, 1.001, 1.01, 1.1, 1.5]))
assert np.all(np.diff(sweep["v_p"]) < 0), "v_p should strictly decrease as f/f_c increases"
assert np.all(np.diff(sweep["v_g"]) > 0), "v_g should strictly increase as f/f_c increases"
assert sweep["v_p"][0] > 50 * C_LIGHT, "v_p should be huge extremely close to cutoff"
assert sweep["v_g"][0] < 0.05 * C_LIGHT, "v_g should be tiny extremely close to cutoff"

# 5. far above cutoff, both velocities converge to c (a waveguide mode
# becomes indistinguishable from a plane wave grazing down the guide)
far = velocities_near_cutoff(1, 1, a, "TE", f_over_fc=np.array([100.0, 1000.0]))
for vp, vg in zip(far["v_p"], far["v_g"]):
    assert abs(vp - C_LIGHT) / C_LIGHT < 1e-3
    assert abs(vg - C_LIGHT) / C_LIGHT < 1e-3

# 6. below cutoff, phase velocity is undefined (evanescent) -- must raise, not
# silently return a bogus number
try:
    phase_velocity(0.5 * fc, 1, 1, a, "TE")
    assert False, "should reject a below-cutoff frequency"
except ValueError:
    pass

print("all dgs.waveguide_phase_group_velocity tests passed")
