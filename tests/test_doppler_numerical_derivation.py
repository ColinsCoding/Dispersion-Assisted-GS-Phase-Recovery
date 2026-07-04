"""Test the numerical derivation of relativistic Doppler shift: simulating
an emitter/receiver couple's actual pulse-emission and light-arrival events
must reproduce dgs.special_relativity.relativistic_doppler's closed-form
result, without that formula ever being used in the derivation itself."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.special_relativity import C_SI, relativistic_doppler
from dgs import doppler_numerical_derivation as dd

f0 = 1.0e14

# 1. approaching and receding both match the closed-form formula to
#    machine precision, across several speeds
for v_frac in (0.1, 0.3, 0.6, 0.9):
    v = v_frac * C_SI
    for approaching in (True, False):
        numeric, analytic = dd.compare_numerical_vs_analytic_doppler(f0, v, approaching, n_pulses=20)
        rel_err = abs(numeric["f_obs"] - analytic["f_obs"]) / analytic["f_obs"]
        assert rel_err < 1e-9, f"v={v_frac}c approaching={approaching}: rel_err={rel_err}"
        # the simulated pulse train, at constant v, must show a genuinely
        # constant observed period (this is a physical requirement, not
        # just an artifact of the derivation)
        assert numeric["period_std"] / numeric["period_obs"] < 1e-9

# 2. approaching blueshifts (higher f_obs), receding redshifts (lower f_obs)
v = 0.6 * C_SI
numeric_approach, _ = dd.compare_numerical_vs_analytic_doppler(f0, v, True, n_pulses=20)
numeric_recede, _ = dd.compare_numerical_vs_analytic_doppler(f0, v, False, n_pulses=20)
assert numeric_approach["f_obs"] > f0
assert numeric_recede["f_obs"] < f0

# 3. low-speed limit: at v << c, relativistic Doppler should reduce toward
#    the classical (non-relativistic) approximation f_obs/f0 ~ 1 +/- beta
v_slow = 0.001 * C_SI
numeric_slow, analytic_slow = dd.compare_numerical_vs_analytic_doppler(f0, v_slow, True, n_pulses=20)
classical_approx = f0 * (1 + v_slow / C_SI)
assert abs(numeric_slow["f_obs"] - classical_approx) / f0 < 1e-5

# 4. the emission schedule itself reflects time dilation: lab-frame spacing
#    between emission events must be exactly gamma times the proper period
t_emit, x_emit, t_arrival = dd.simulate_pulse_train_arrival_times(
    f0, v, n_pulses=10, x0=0.0, x_receiver=1.0)
from dgs.special_relativity import lorentz_factor
gamma = lorentz_factor(v, C_SI)["gamma"]
dt_emit = np.diff(t_emit)
assert np.allclose(dt_emit, gamma / f0)

# 5. input validation
for bad_call in [
    lambda: dd.simulate_pulse_train_arrival_times(f0, C_SI * 1.1, 10, 0.0, 1.0),
    lambda: dd.simulate_pulse_train_arrival_times(-1.0, v, 10, 0.0, 1.0),
    lambda: dd.simulate_pulse_train_arrival_times(f0, v, 1, 0.0, 1.0),
    lambda: dd.compare_numerical_vs_analytic_doppler(f0, -v, True),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.doppler_numerical_derivation tests passed")
