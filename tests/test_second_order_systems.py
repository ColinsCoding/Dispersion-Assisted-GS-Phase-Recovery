"""Test dgs.second_order_systems: zeta and omega_n from an RLC, the poles, the
damping regimes, overshoot depending only on zeta, the step response satisfying
its defining ODE (all three regimes), and the resonant peak existing only for
zeta < 1/sqrt(2). Cross-checked against spice.py / circuit_energy."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import second_order_systems as so

# 1. the two numbers from a series RLC (the R=20,L=1mH,C=1uF from circuit_energy)
R, L, C = 20.0, 1e-3, 1e-6
wn = so.natural_frequency_rlc(L, C)
zeta = so.damping_ratio_rlc(R, L, C)
assert np.isclose(wn, 1 / np.sqrt(L * C))
assert np.isclose(wn / (2 * np.pi), 5033, rtol=1e-3)          # matches spice f_n
assert np.isclose(zeta, (R / 2) * np.sqrt(C / L))
assert np.isclose(so.damping_ratio_rlc(0, L, C), 0.0)         # lossless -> zeta=0
assert np.isclose(so.quality_factor(0.5), 1.0)

# 2. poles match circuit_energy's rlc_damping roots: -10000 +/- 30000j
p1, p2 = so.poles(zeta, wn)
assert np.isclose(p1.real, -10000, rtol=1e-3) and np.isclose(abs(p1.imag), 30000, rtol=1e-3)
assert np.isclose(p1, np.conj(p2))                            # complex-conjugate pair
# overdamped -> two real poles
q1, q2 = so.poles(2.0, 1.0)
assert abs(q1.imag) < 1e-12 and abs(q2.imag) < 1e-12 and q1.real < 0 and q2.real < 0

# 3. regimes and damped frequency
assert so.damping_regime(0.3) == "under"
assert so.damping_regime(1.0) == "critical"
assert so.damping_regime(2.0) == "over"
assert np.isclose(so.damped_frequency(zeta, wn) / (2*np.pi), 4775, rtol=1e-3)  # matches
assert so.damped_frequency(1.5, wn) == 0.0                    # overdamped -> no ring

# 4. overshoot depends ONLY on zeta (16.3% at 0.5, ~4.3% at 0.707, 0 for zeta>=1)
assert np.isclose(so.percent_overshoot(0.5), 16.303, atol=0.05)
assert np.isclose(so.percent_overshoot(0.7071067811865476), 4.32, atol=0.05)
assert so.percent_overshoot(1.0) == 0.0 and so.percent_overshoot(2.0) == 0.0
# and it really is the step-response peak
t = np.linspace(0, 40, 40000)
y = so.step_response(0.5, 1.0, t)
assert np.isclose((np.max(y) - 1) * 100, so.percent_overshoot(0.5), atol=0.1)

# 5. the step response satisfies y'' + 2 zeta wn y' + wn^2 y = wn^2, all regimes
def ode_residual(zeta, wn):
    tt = np.linspace(0, 6/(zeta*wn) if zeta > 0 else 20, 40000); dt = tt[1]-tt[0]
    yy = so.step_response(zeta, wn, tt)
    r = np.gradient(np.gradient(yy, dt), dt) + 2*zeta*wn*np.gradient(yy, dt) + wn**2*yy - wn**2
    return np.max(np.abs(r[20:-20]))
for z in (0.4, 1.0, 2.0):                                     # under, critical, over
    assert ode_residual(z, 1.0) < 1e-4
# every step response settles to the DC gain of 1
for z in (0.4, 1.0, 2.0):
    assert np.isclose(so.step_response(z, 1.0, 50.0), 1.0, atol=1e-3)

# 6. settling time ~ 4/(zeta wn) for 2%
assert np.isclose(so.settling_time(0.5, 2.0, tol=0.02), -np.log(0.02)/(0.5*2.0))

# 7. resonant peak exists only for zeta < 1/sqrt(2); magnitude exceeds DC there
assert so.resonant_peak_frequency(0.3, 1.0) is not None
assert so.resonant_peak_frequency(1/np.sqrt(2), 1.0) is None  # Butterworth: no peak
assert so.resonant_peak_frequency(0.9, 1.0) is None
wr = so.resonant_peak_frequency(0.3, 1.0)
assert so.magnitude_response(0.3, 1.0, wr) > so.magnitude_response(0.3, 1.0, 0.0)  # peak > DC(=1)
assert so.magnitude_response(0.3, 1.0, wr) >= so.magnitude_response(0.3, 1.0, wr*1.3)
assert np.isclose(so.magnitude_response(0.5, 1.0, 0.0), 1.0)  # DC gain 1

# 8. kwarg bounds
for bad in (lambda: so.natural_frequency_rlc(0, C),
            lambda: so.quality_factor(0),
            lambda: so.settling_time(1.5, 1.0),               # not underdamped
            lambda: so.poles(0.5, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_second_order_systems: all checks passed")
