"""Test the classical-spinning-sphere failure, gyromagnetic anomaly, and
Thomas precession calculations against real reference values (classical
electron radius ~2.818e-15 m, charge-to-mass ratio ~1.759e11 C/kg,
electron g-factor ~2.0023) -- and the error-propagation reuse."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import thomas_precession_gyromagnetic_anomaly as tp

# 1. classical electron radius matches the real known value
r_e = tp.classical_electron_radius_m()
assert abs(r_e - 2.818e-15) / 2.818e-15 < 0.01

# 2. charge-to-mass ratio matches the real Thomson value
e_over_m = tp.charge_to_mass_ratio_c_per_kg()
assert abs(e_over_m - 1.759e11) / 1.759e11 < 0.01

# 3. moment of inertia scales correctly: I ~ m*r^2
I1 = tp.moment_of_inertia_solid_sphere(1.0, 1.0)
I2 = tp.moment_of_inertia_solid_sphere(2.0, 1.0)
I3 = tp.moment_of_inertia_solid_sphere(1.0, 2.0)
assert abs(I2 / I1 - 2.0) < 1e-9
assert abs(I3 / I1 - 4.0) < 1e-9   # r^2 scaling

# 4. parallel axis theorem: I_axis >= I_cm always, equals I_cm at d=0
I_cm = 5.0
assert abs(tp.parallel_axis_theorem(I_cm, 2.0, 0.0) - I_cm) < 1e-9
assert tp.parallel_axis_theorem(I_cm, 2.0, 3.0) > I_cm

# 5. required surface speed for the electron's real spin (hbar/2) using its
#    classical radius is WAY above c -- the real, famous textbook result
v, v_over_c = tp.required_surface_speed_m_s(9.1093837015e-31, r_e)
assert v_over_c > 10.0   # order-of-magnitude check: known result is ~100x c

# 6. classical g-factor is exactly 1, and the anomaly dict is self-consistent
assert tp.classical_g_factor() == 1.0
anomaly = tp.gyromagnetic_anomaly(2.0023)
assert abs(anomaly["ratio"] - 2.0023) < 1e-9
assert abs(anomaly["qed_anomaly_a_e"] - 0.00115) < 1e-4   # real QED value ~0.00116

# 7. Thomas precession frequency scales as v^2 (quadratic, not linear)
omega_orb = 1e16
omega_T_v1 = tp.thomas_precession_frequency_rad_s(1e6, omega_orb)
omega_T_v2 = tp.thomas_precession_frequency_rad_s(2e6, omega_orb)
assert abs(omega_T_v2 / omega_T_v1 - 4.0) < 1e-6   # (2x)^2 = 4x

# 8. Thomas precession scales linearly with orbital frequency
omega_T_orb1 = tp.thomas_precession_frequency_rad_s(1e6, 1e16)
omega_T_orb2 = tp.thomas_precession_frequency_rad_s(1e6, 2e16)
assert abs(omega_T_orb2 / omega_T_orb1 - 2.0) < 1e-9

# 9. error propagation: relative uncertainty in frequency is preserved
#    exactly in the inferred field (B is linear in f)
f_hz, sigma_f_hz = 9.5e9, 0.02e9
B_value, B_sigma = tp.propagate_field_uncertainty_tesla(f_hz, sigma_f_hz)
assert abs(B_sigma / B_value - sigma_f_hz / f_hz) < 1e-9
# and this matches field_for_resonance_tesla's own value (no bias introduced)
from dgs.electron_spin_resonance import field_for_resonance_tesla
assert abs(B_value - field_for_resonance_tesla(f_hz)) < 1e-9

# 10. input validation
for bad_call in [
    lambda: tp.charge_to_mass_ratio_c_per_kg(m=-1.0),
    lambda: tp.moment_of_inertia_solid_sphere(-1.0, 1.0),
    lambda: tp.moment_of_inertia_solid_sphere(1.0, -1.0),
    lambda: tp.parallel_axis_theorem(-1.0, 1.0, 1.0),
    lambda: tp.parallel_axis_theorem(1.0, -1.0, 1.0),
    lambda: tp.parallel_axis_theorem(1.0, 1.0, -1.0),
    lambda: tp.required_surface_speed_m_s(-1.0, 1.0),
    lambda: tp.required_surface_speed_m_s(1.0, -1.0),
    lambda: tp.required_surface_speed_m_s(1.0, 1.0, target_angular_momentum=-1.0),
    lambda: tp.gyromagnetic_anomaly(-1.0),
    lambda: tp.thomas_precession_frequency_rad_s(-1.0, 1e16),
    lambda: tp.thomas_precession_frequency_rad_s(3e8, 1e16),  # >= c
    lambda: tp.thomas_precession_frequency_rad_s(1e6, -1.0),
    lambda: tp.propagate_field_uncertainty_tesla(-1.0, 0.1e9),
    lambda: tp.propagate_field_uncertainty_tesla(9.5e9, -1.0),
    lambda: tp.propagate_field_uncertainty_tesla(9.5e9, 0.1e9, g_factor=-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.thomas_precession_gyromagnetic_anomaly tests passed")
