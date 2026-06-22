"""Test RLC resonance: the peak at 1/sqrt(LC), Q, bandwidth, and the torch optimizer."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import resonance as r

R, L, C = 5.0, 1e-3, 1e-6
w0 = r.resonant_frequency(L, C)

# 1. resonant frequency omega0 = 1/sqrt(LC)
assert np.isclose(w0, 1.0 / np.sqrt(L * C))

# 2. at resonance the reactances cancel: |H| = 1 and the phase is zero
H0 = r.rlc_response(w0, R, L, C)
assert np.isclose(abs(H0), 1.0, atol=1e-9)
assert abs(np.angle(H0)) < 1e-9
# off resonance |H| < 1 (it is a bandpass peak)
assert abs(r.rlc_response(w0 * 1.5, R, L, C)) < 1.0
assert abs(r.rlc_response(w0 / 1.5, R, L, C)) < 1.0

# 3. Q and bandwidth: Q = (1/R)sqrt(L/C), bandwidth = R/L = omega0/Q
Q = r.quality_factor(R, L, C)
assert np.isclose(Q, (1 / R) * np.sqrt(L / C))
assert np.isclose(r.bandwidth(R, L), w0 / Q)
# lower R -> higher Q -> narrower bandwidth
assert r.quality_factor(1.0, L, C) > Q and r.bandwidth(1.0, L) < r.bandwidth(R, L)

# 4. the "experiment": sweep and read the peak + Q off the curve
omega, H = r.frequency_sweep(R, L, C)
w_pk, Q_meas = r.find_resonance(omega, H)
assert abs(w_pk - w0) / w0 < 0.01                      # peak within 1% of omega0
assert abs(Q_meas - Q) / Q < 0.05                      # measured Q within 5%

# 5. -3 dB point: at omega0 + bandwidth/2 the magnitude is ~ 1/sqrt(2)
w_half = w0 + r.bandwidth(R, L) / 2
assert abs(abs(r.rlc_response(w_half, R, L, C)) - 1 / np.sqrt(2)) < 0.03

# 6. torch optimization finds the same peak (skipped if torch is unavailable)
try:
    w_found, traj = r.find_resonance_torch(R, L, C)
    assert abs(w_found - w0) / w0 < 0.01               # gradient descent lands on omega0
    torch_msg = f"torch found {w_found:.0f} = omega0"
except ImportError:
    torch_msg = "torch optimizer skipped (no torch here)"

print(f"TEST PASS  (omega0={w0:.0f} rad/s, |H|=1 phase=0 at resonance; Q={Q:.2f}, "
      f"bandwidth=omega0/Q; swept peak within 1%; {torch_msg})")
