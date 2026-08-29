"""Test dgs.ptz_mount_bracket_statics: the cantilever-beam formulas match
known closed-form scaling behavior, the resonance-margin check correctly
flags a natural frequency inside the blade-pass keep-out band (and clears
one outside it), input validation rejects bad parameters, and the reused
BLADE_PASS_HZ/default_gimbal_params actually come from
dgs.ptz_helicopter_stabilization (not silently redefined/duplicated)."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.ptz_mount_bracket_statics import (
    rectangular_tube_second_moment, root_bending_stress, tip_deflection,
    tip_mass_natural_frequency, check_resonance_margin, evaluate_bracket,
    RESONANCE_MARGIN_FRACTION,
)
from dgs.ptz_helicopter_stabilization import BLADE_PASS_HZ, default_gimbal_params

# 1. This module reuses the helicopter-stabilization module's constants,
# it does not redefine its own copy (single source of truth for the
# excitation frequency and gimbal mass)
import dgs.ptz_mount_bracket_statics as bracket_mod
assert bracket_mod.BLADE_PASS_HZ is BLADE_PASS_HZ
assert bracket_mod.default_gimbal_params is default_gimbal_params

# 2. Second moment of area: a solid section (wall_thickness = half the
# smaller outer dimension) should be less stiff removed than a thin-walled
# tube of the same outer size (more material removed from the middle -> less I)
I_thick_wall = rectangular_tube_second_moment(0.02, 0.02, 0.0099)  # nearly solid
I_thin_wall = rectangular_tube_second_moment(0.02, 0.02, 0.001)
assert I_thick_wall > I_thin_wall > 0

for bad_call in [
    lambda: rectangular_tube_second_moment(-0.02, 0.02, 0.001),
    lambda: rectangular_tube_second_moment(0.02, 0.02, 0.011),  # wall too thick
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

# 3. Bending stress scales linearly with tip load and with length (M = F*L)
I_test = rectangular_tube_second_moment(0.025, 0.015, 0.002)
s1 = root_bending_stress(weight_N=5.0, length_m=0.3, second_moment_I=I_test, outer_height_m=0.015)
s2 = root_bending_stress(weight_N=10.0, length_m=0.3, second_moment_I=I_test, outer_height_m=0.015)
assert abs(s2 / s1 - 2.0) < 1e-9
s3 = root_bending_stress(weight_N=5.0, length_m=0.6, second_moment_I=I_test, outer_height_m=0.015)
assert abs(s3 / s1 - 2.0) < 1e-9

# g_load multiplies the effective force the same way weight_N does
s_g4 = root_bending_stress(weight_N=5.0, length_m=0.3, second_moment_I=I_test,
                            outer_height_m=0.015, g_load=4.0)
assert abs(s_g4 / s1 - 4.0) < 1e-9

# 4. Tip deflection scales as L^3 (classic cantilever result)
d1 = tip_deflection(weight_N=5.0, length_m=0.3, second_moment_I=I_test, E_modulus=69e9)
d2 = tip_deflection(weight_N=5.0, length_m=0.6, second_moment_I=I_test, E_modulus=69e9)
assert abs(d2 / d1 - 8.0) < 1e-6  # (0.6/0.3)^3 = 8

# 5. Natural frequency scales as 1/sqrt(mass) and as 1/L^1.5
f_light = tip_mass_natural_frequency(E_modulus=69e9, second_moment_I=I_test,
                                      length_m=0.3, tip_mass_kg=0.5)
f_heavy = tip_mass_natural_frequency(E_modulus=69e9, second_moment_I=I_test,
                                      length_m=0.3, tip_mass_kg=2.0)
assert abs(f_heavy / f_light - 0.5) < 1e-6  # sqrt(0.5/2.0) = 0.5

# 6. Resonance margin: inside the band is correctly flagged, well outside is clear
inside = check_resonance_margin(natural_freq_hz=BLADE_PASS_HZ, excitation_freq_hz=BLADE_PASS_HZ)
assert inside["clears_margin"] is False
far_below = check_resonance_margin(natural_freq_hz=BLADE_PASS_HZ * 0.1,
                                    excitation_freq_hz=BLADE_PASS_HZ)
assert far_below["clears_margin"] is True
far_above = check_resonance_margin(natural_freq_hz=BLADE_PASS_HZ * 4.0,
                                    excitation_freq_hz=BLADE_PASS_HZ)
assert far_above["clears_margin"] is True

# right at the edge of the keep-out band (just past the margin) should clear
just_outside = BLADE_PASS_HZ * (1 + RESONANCE_MARGIN_FRACTION) * 1.001
edge = check_resonance_margin(natural_freq_hz=just_outside, excitation_freq_hz=BLADE_PASS_HZ)
assert edge["clears_margin"] is True

try:
    check_resonance_margin(natural_freq_hz=10.0, excitation_freq_hz=BLADE_PASS_HZ, margin_fraction=1.5)
    assert False, "should have raised ValueError"
except ValueError:
    pass

# 7. evaluate_bracket: the module's own worked examples actually demonstrate
# what their docstrings/print statements claim
good = evaluate_bracket(length_m=0.30, outer_width_m=0.025, outer_height_m=0.015,
                         wall_thickness_m=0.002)
assert good["acceptable"] is True
assert good["factor_of_safety"] > 1.0
assert good["resonance_check"]["clears_margin"] is True

thin = evaluate_bracket(length_m=0.30, outer_width_m=0.025, outer_height_m=0.015,
                         wall_thickness_m=0.0001)
assert thin["factor_of_safety"] > 1.0        # stress alone looks acceptable...
assert thin["resonance_check"]["clears_margin"] is False   # ...but resonance check catches it
assert thin["acceptable"] is False           # so overall verdict is correctly NOT acceptable

# 8. evaluate_bracket defaults to dgs.ptz_helicopter_stabilization's own
# gimbal mass, not an independently chosen number
default_mass = default_gimbal_params()["mass"]
explicit = evaluate_bracket(length_m=0.30, outer_width_m=0.025, outer_height_m=0.015,
                             wall_thickness_m=0.002,
                             gimbal_params={"mass": default_mass})
assert abs(explicit["natural_freq_hz"] - good["natural_freq_hz"]) < 1e-9

print("all dgs.ptz_mount_bracket_statics tests passed")
