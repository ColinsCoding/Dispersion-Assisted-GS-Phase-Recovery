"""Test dgs/shark_electroreception.py: the v-cross-B motional-EMF
geomagnetic-sensing hypothesis (checked against Kalmijn's cited
detection threshold), the exact dipole-field formula (on-axis/equatorial
ratio), and the distributed-array gradient estimate (checked to actually
point toward the true source, not assumed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.shark_electroreception import (
    motional_field, field_to_nV_per_cm, is_detectable, verify_geomagnetic_sensing_plausible,
    dipole_field_magnitude, verify_dipole_axis_ratio, detection_range_on_axis,
    point_charge_field_vector, point_charge_potential, ampullae_array_positions,
    estimate_gradient_from_array, verify_gradient_points_to_source, KALMIJN_THRESHOLD_NV_PER_CM,
)

# 1. motional_field: E = v x B, known cross-product cases
E = motional_field([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
assert np.allclose(E, [0.0, 0.0, 1.0])   # x-hat cross y-hat = z-hat

E_parallel = motional_field([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
assert np.allclose(E_parallel, [0.0, 0.0, 0.0])   # parallel v and B -> zero induced field

# 2. field_to_nV_per_cm: exact unit conversion
assert abs(field_to_nV_per_cm(1.0) - 1e7) < 1e-3   # 1 V/m = 1e7 nV/cm

# 3. is_detectable: correct threshold comparison
below = is_detectable(1e-9)   # 10 nV/cm, above the 5 nV/cm threshold... check carefully
# 1e-9 V/m = 1e-2 nV/cm (field_to_nV_per_cm(1e-9)=1e-9*1e7=1e-2), well BELOW threshold
assert below["detectable"] is False
above = is_detectable(1e-6)   # 1e-6 V/m = 1e1 nV/cm = 10 nV/cm, above 5 nV/cm threshold
assert above["detectable"] is True

print("dgs.shark_electroreception: motional-field / detectability checks passed")

# 4. verify_geomagnetic_sensing_plausible: realistic swim speeds through
#    Earth's field must be detectable -- the actual claim, checked with
#    real numbers, across several (speed, field) combinations
for v in (0.3, 1.0, 2.0, 3.0):
    for B in (25e-6, 50e-6, 65e-6):
        result = verify_geomagnetic_sensing_plausible(v, B)
        assert result["detectable"] is True, f"v={v}, B={B}: {result}"
        assert result["margin_factor"] > 1.0

# a much slower speed / weaker field should still be detectable, but with
# a smaller margin -- confirms the margin actually tracks the physics,
# not a constant regardless of input
slow = verify_geomagnetic_sensing_plausible(0.3, 25e-6)
fast = verify_geomagnetic_sensing_plausible(3.0, 65e-6)
assert fast["margin_factor"] > slow["margin_factor"]

for bad in [dict(swim_speed_m_s=-1.0, B_earth_T=50e-6), dict(swim_speed_m_s=1.0, B_earth_T=-1.0)]:
    try:
        verify_geomagnetic_sensing_plausible(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.shark_electroreception: geomagnetic-sensing plausibility checks passed")

# 5. dipole_field_magnitude / verify_dipole_axis_ratio: exact Griffiths
#    identity, on-axis = 2x equatorial, for several (p, r)
for p, r in [(1e-19, 0.1), (1e-18, 0.05), (1e-20, 0.3)]:
    check = verify_dipole_axis_ratio(p, r)
    assert check["matches_theory"] is True
    assert abs(check["ratio"] - 2.0) < 1e-9

try:
    dipole_field_magnitude(p=-1.0, r=0.1, theta_rad=0.0)
    raise AssertionError("expected ValueError for p <= 0")
except ValueError:
    pass

print("dgs.shark_electroreception: dipole field checks passed")

# 6. detection_range_on_axis: a bigger dipole -> a bigger detection range
#    (monotonic, not just "returns a number")
r_small = detection_range_on_axis(1e-20)
r_large = detection_range_on_axis(1e-18)
assert r_large > r_small
# and the returned range, plugged back into dipole_field_magnitude at
# theta=0, must reproduce the threshold exactly (round-trip check)
p_test = 1e-19
r_test = detection_range_on_axis(p_test)
field_at_range = dipole_field_magnitude(p_test, r_test, 0.0)
threshold_V_per_m = KALMIJN_THRESHOLD_NV_PER_CM / 1e7
assert abs(field_at_range - threshold_V_per_m) / threshold_V_per_m < 1e-9

print("dgs.shark_electroreception: detection-range round-trip check passed")

# 7. point_charge_field_vector / point_charge_potential: E = -grad(V),
#    checked via finite difference at one point
q, source = 1e-9, np.array([1.0, 0.0, 0.0])
field_pos = np.array([0.0, 0.0, 0.0])
E_at_origin = point_charge_field_vector(q, source, field_pos)
h = 1e-6
dVdx = (point_charge_potential(q, source, field_pos + [h, 0, 0]) -
        point_charge_potential(q, source, field_pos - [h, 0, 0])) / (2 * h)
assert abs(E_at_origin[0] - (-dVdx)) / abs(E_at_origin[0]) < 1e-4

try:
    point_charge_field_vector(q, source, source)
    raise AssertionError("expected ValueError when field_pos == source_pos")
except ValueError:
    pass

print("dgs.shark_electroreception: point-charge field/potential checks passed")

# 8. ampullae_array_positions: correct shape, all at x=0 (facing +x),
#    correct spread
positions = ampullae_array_positions(half_width_m=0.1, n_per_axis=4)
assert positions.shape == (16, 3)
assert np.all(positions[:, 0] == 0.0)
assert positions[:, 1].max() == 0.1 and positions[:, 1].min() == -0.1

for bad in [dict(half_width_m=-1.0, n_per_axis=4), dict(half_width_m=0.1, n_per_axis=1)]:
    try:
        ampullae_array_positions(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

# 9. verify_gradient_points_to_source: the actual directional-sensing
#    claim -- estimated gradient must point toward the true source,
#    checked across several source positions, not one lucky case
for source_pos in [(1.0, 0.4, -0.2), (2.0, -0.3, 0.1), (0.8, 0.0, 0.5)]:
    check = verify_gradient_points_to_source(source_pos=source_pos)
    assert check["well_aligned"] is True, f"source={source_pos}: {check}"
    assert check["angle_error_deg"] < 1.0

# a denser array should give an even better (smaller) angle error --
# confirms the estimate actually improves with more sensors, not a fluke
coarse = verify_gradient_points_to_source(n_per_axis=3)
fine = verify_gradient_points_to_source(n_per_axis=8)
assert fine["angle_error_deg"] <= coarse["angle_error_deg"] + 1e-6

print("dgs.shark_electroreception: directional-sensing (array gradient) checks passed")
print("all dgs.shark_electroreception tests passed")
