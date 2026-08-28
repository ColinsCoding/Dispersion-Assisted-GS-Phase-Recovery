"""Test dgs/ptz_helicopter_stabilization.py: the disturbance model behaves
as a real sinusoid, the relative-reference (unstabilized) controller
actually holds its own angle near zero, inertial-reference stabilization
gives a meaningfully lower RMS pointing error than relative-reference (the
real claim of this module -- same controller, same gains, same plant,
only the measured signal differs), and kwarg bounds reject non-physical
inputs."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.ptz_helicopter_stabilization import (
    base_disturbance_tilt, simulate_stabilization, verify_stabilization_reduces_error,
    default_gimbal_params, MAIN_ROTOR_HZ, BLADE_PASS_HZ, N_BLADES,
)

# 1. Disturbance model sanity: zero at t=0, correct amplitude/frequency
assert abs(base_disturbance_tilt(0.0, amplitude_deg=1.5)) < 1e-12
peak = base_disturbance_tilt(0.25 / BLADE_PASS_HZ, amplitude_deg=1.5)  # quarter period -> peak
assert abs(peak - np.radians(1.5)) < 1e-9

# 2. Disturbance-model bounds
for bad_call in [
    lambda: base_disturbance_tilt(0.0, amplitude_deg=-1.0),
    lambda: base_disturbance_tilt(0.0, freq_hz=0.0),
    lambda: base_disturbance_tilt(0.0, freq_hz=-5.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

# 3. Blade-pass frequency is derived correctly (2-bladed rotor -> 2x main rotor)
assert abs(BLADE_PASS_HZ - MAIN_ROTOR_HZ * N_BLADES) < 1e-12

# 4. simulate_stabilization validates inputs
try:
    simulate_stabilization(t_end=-1.0)
    assert False, "should have raised ValueError"
except ValueError:
    pass

try:
    simulate_stabilization(reference="sideways")
    assert False, "should have raised ValueError"
except ValueError:
    pass

# 5. Relative-reference control actually holds its own angle near zero
#    (confirms the "unstabilized" baseline isn't secretly broken/free-falling)
rel = simulate_stabilization(t_end=1.0, reference="relative")
assert np.sqrt(np.mean(rel["gimbal_relative_tilt"] ** 2)) < np.radians(0.5)

# 6. Inertial-reference tracks the base disturbance far better than
#    relative-reference does -- the actual physics claim, checked
#    end-to-end rather than assumed from the controller design alone
check = verify_stabilization_reduces_error()
assert check["meaningfully_improved"]
assert check["improvement_ratio"] > 5.0
assert check["inertial_reference_rms_rad"] < check["relative_reference_rms_rad"]

# 7. Zero disturbance amplitude -> both references settle near zero error
#    (no vibration to reject means there's nothing to distinguish them on)
zero_dist_relative = simulate_stabilization(t_end=1.0, disturbance_amplitude_deg=0.0, reference="relative")
zero_dist_inertial = simulate_stabilization(t_end=1.0, disturbance_amplitude_deg=0.0, reference="inertial")
assert zero_dist_relative["rms_inertial_error_rad"] < np.radians(0.5)
assert zero_dist_inertial["rms_inertial_error_rad"] < np.radians(0.5)

# 8. default_gimbal_params returns physically sensible (positive) values
params = default_gimbal_params()
for key in ("I_p0", "I_tilt", "mass", "cg_distance"):
    assert params[key] > 0

print("all dgs.ptz_helicopter_stabilization tests passed")
