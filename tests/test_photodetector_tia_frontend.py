"""Test dgs/photodetector_tia_frontend.py: the transimpedance amplifier's
KCL feedback loop, solved via sp.dsolve for the closed-form step response
and cross-checked against a real scipy ODE integration of the same
equation, plus the -3dB bandwidth definition and the Rf-independent
gain-bandwidth product."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.photodetector_tia_frontend import (
    photocurrent, tia_transimpedance_gain_dc, tia_bandwidth_hz,
    tia_step_response_analytic, tia_step_response_ode,
    verify_step_response_matches_ode, tia_transfer_function,
    verify_bandwidth_is_minus_3db, gain_bandwidth_tradeoff,
)

# 1. photocurrent: known value, and validation
I_ph = photocurrent(1e-6, 0.9)
assert abs(I_ph - 0.9e-6) < 1e-15
for bad in [(-1.0, 0.9), (1e-6, -0.9), (0.0, 0.9)]:
    try:
        photocurrent(*bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

# 2. tia_transimpedance_gain_dc: exactly -Rf, and validation
assert tia_transimpedance_gain_dc(1e4) == -1e4
try:
    tia_transimpedance_gain_dc(-1.0)
    raise AssertionError("expected ValueError for Rf <= 0")
except ValueError:
    pass

# 3. tia_bandwidth_hz: known formula, doubling Rf halves bandwidth exactly
Rf, Cf = 1e4, 0.5e-12
f_p = tia_bandwidth_hz(Rf, Cf)
assert abs(f_p - 1.0 / (2 * np.pi * Rf * Cf)) < 1e-6
f_p_2x = tia_bandwidth_hz(2 * Rf, Cf)
assert abs(f_p_2x - f_p / 2) / (f_p / 2) < 1e-9

# 4. tia_step_response_analytic: known limits -- V(0)=0, V(inf)->-I0*Rf,
#    and matches the sp.dsolve-derived closed form exactly at a spot value
I0 = 1e-6
tau = Rf * Cf
V0 = tia_step_response_analytic(np.array([0.0]), I0, Rf, Cf)[0]
V_inf = tia_step_response_analytic(np.array([50 * tau]), I0, Rf, Cf)[0]
assert abs(V0) < 1e-15
assert abs(V_inf - (-I0 * Rf)) / abs(I0 * Rf) < 1e-15   # fully settled after 50 tau
V_at_tau = tia_step_response_analytic(np.array([tau]), I0, Rf, Cf)[0]
expected_at_tau = -I0 * Rf * (1 - np.exp(-1))   # one time constant: 1-1/e of the way there
assert abs(V_at_tau - expected_at_tau) < 1e-15

# 5. tia_step_response_ode / verify_step_response_matches_ode: real ODE
#    integration must match the closed form, across several Rf/Cf scales
for Rf_test, Cf_test in [(1e3, 1e-12), (1e4, 0.5e-12), (1e6, 1e-15)]:
    assert verify_step_response_matches_ode(1e-6, Rf_test, Cf_test) is True

# 6. tia_transfer_function: DC value must equal -Rf exactly (matches
#    tia_transimpedance_gain_dc, an independent cross-check of two
#    different derivations of the same quantity)
H0 = tia_transfer_function(np.array([0.0]), Rf, Cf)[0]
assert abs(H0 - tia_transimpedance_gain_dc(Rf)) < 1e-9

# 7. verify_bandwidth_is_minus_3db: must pass, across several Rf/Cf
for Rf_test, Cf_test in [(1e3, 1e-12), (1e5, 1e-13), (1e4, 1e-9)]:
    assert verify_bandwidth_is_minus_3db(Rf_test, Cf_test) is True

# 8. gain_bandwidth_tradeoff: gain and bandwidth individually change with
#    Rf (gain UP, bandwidth DOWN), but their product is Rf-independent
tradeoff = gain_bandwidth_tradeoff(np.array([1e3, 1e4, 1e5, 1e6]), Cf)
assert np.all(np.diff(tradeoff["gains_ohm"]) > 0), "gain must increase with Rf"
assert np.all(np.diff(tradeoff["bandwidths_Hz"]) < 0), "bandwidth must decrease with Rf"
assert tradeoff["product_is_constant"] is True
expected_product = 1.0 / (2 * np.pi * Cf)
assert abs(tradeoff["gain_bandwidth_products"][0] - expected_product) / expected_product < 1e-9

# 9. gain_bandwidth_tradeoff: input validation
try:
    gain_bandwidth_tradeoff(np.array([1e3, -1e4]), Cf)
    raise AssertionError("expected ValueError for a negative Rf in the sweep")
except ValueError:
    pass
try:
    gain_bandwidth_tradeoff(np.array([1e3, 1e4]), -1e-12)
    raise AssertionError("expected ValueError for Cf <= 0")
except ValueError:
    pass

print("all dgs.photodetector_tia_frontend tests passed")
