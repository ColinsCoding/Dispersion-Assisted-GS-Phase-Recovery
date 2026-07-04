"""Test classical electrodynamics applied to computer engineering: skin-
effect resistance via logarithmic differentiation (cross-checked against
direct numerical differentiation), transmission-line theory, and a
simulated Time-Domain Reflectometry (TDR) measurement that must correctly
INFER an unknown load impedance and fault distance from reflection data
alone."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import transmission_line_tdr as tdr

rho, mu_r, a = 1.68e-8, 1.0, 0.5e-3

# 1. skin depth shrinks as frequency increases (1/sqrt(f))
d1 = tdr.skin_depth(1e6, rho, mu_r)
d2 = tdr.skin_depth(4e6, rho, mu_r)
assert abs(d1 / d2 - 2.0) < 1e-9   # quadrupling f halves delta (1/sqrt(4)=1/2)

# 2. AC resistance per length scales as sqrt(f): quadrupling f doubles R
R1 = tdr.ac_resistance_per_length(1e6, rho, mu_r, a)
R2 = tdr.ac_resistance_per_length(4e6, rho, mu_r, a)
assert abs(R2 / R1 - 2.0) < 1e-6

# 3. dR/df via logarithmic differentiation matches direct numerical
#    differentiation, at several frequencies
for f in [1e6, 1e7, 1e8, 1e9]:
    dR_log = tdr.dR_df_via_log_differentiation(f, rho, mu_r, a)
    dR_num = tdr.dR_df_direct(f, rho, mu_r, a)
    assert abs(dR_log - dR_num) / dR_num < 1e-6

# 4. characteristic impedance and propagation velocity: known PCB values
L_per_len, C_per_len = 250e-9, 100e-12
Z0 = tdr.characteristic_impedance(L_per_len, C_per_len)
v = tdr.propagation_velocity(L_per_len, C_per_len)
assert abs(Z0 - 50.0) < 1e-6          # designed to hit the standard 50 Ohm target
assert 0 < v < 3e8                     # must be slower than c (dielectric-filled line)

# 5. reflection coefficient: matched load gives zero reflection; open/short
#    give the textbook +1/-1 extremes
assert abs(tdr.reflection_coefficient(50.0, 50.0)) < 1e-12
assert abs(tdr.reflection_coefficient(50.0, 1e12) - 1.0) < 1e-6   # near-open
assert abs(tdr.reflection_coefficient(50.0, 1e-12) - (-1.0)) < 1e-6  # near-short

# 6. load_impedance_from_reflection is the EXACT inverse of
#    reflection_coefficient -- round-trip test across several loads
for Z_load in [10.0, 50.0, 75.0, 120.0, 300.0]:
    gamma = tdr.reflection_coefficient(50.0, Z_load)
    Z_recovered = tdr.load_impedance_from_reflection(gamma, 50.0)
    assert abs(Z_recovered - Z_load) < 1e-9

# 7. full TDR simulation: correctly infers both the load impedance AND
#    the fault/load distance from the simulated trace alone
Z_load_true, Z_source, line_length = 75.0, 50.0, 0.3
t = np.linspace(-1e-9, 6e-9, 500)
v_trace, gamma, t_rt = tdr.tdr_step_response(Z0, Z_load_true, Z_source, line_length, v, t)
Z_load_inferred = tdr.load_impedance_from_reflection(gamma, Z0)
distance_inferred = t_rt * v / 2
assert abs(Z_load_inferred - Z_load_true) < 1e-9
assert abs(distance_inferred - line_length) < 1e-9

# before the reflection returns, the trace sits at the incident level;
# after, it steps to incident*(1+gamma)
v_incident = 1.0 * Z0 / (Z_source + Z0)
assert np.allclose(v_trace[t < 0], 0.0)
assert np.allclose(v_trace[(t >= 0) & (t < t_rt)], v_incident)
assert np.allclose(v_trace[t >= t_rt], v_incident * (1 + gamma))

# 8. input validation
for bad_call in [
    lambda: tdr.skin_depth(-1.0, rho, mu_r),
    lambda: tdr.ac_resistance_per_length(1e6, rho, mu_r, -1.0),
    lambda: tdr.characteristic_impedance(-1.0, C_per_len),
    lambda: tdr.reflection_coefficient(-1.0, 50.0),
    lambda: tdr.reflection_coefficient(50.0, -50.0),   # Z_load=-Z0, singular
    lambda: tdr.load_impedance_from_reflection(1.0, 50.0),   # gamma=1, infinite Z
    lambda: tdr.tdr_step_response(-1.0, 75.0, 50.0, 0.3, v, t),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.transmission_line_tdr tests passed")
