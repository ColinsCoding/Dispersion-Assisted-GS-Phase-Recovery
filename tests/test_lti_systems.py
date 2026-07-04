"""Test LTI systems taken to engineering depth: impulse response, step
response as its integral, the convolution theorem, and the frequency
response as the Fourier transform of the impulse response -- all
cross-checked against each other and the analytic RC low-pass."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import lti_systems as lti

R, C = 1.0e3, 1.0e-6
tau = R * C
dt = tau / 300.0
t = np.arange(0, 15 * tau, dt)

# 1. impulse response: causal, positive, decaying, correct value at t=0
h = lti.impulse_response_rc(t, R, C)
assert np.all(h >= 0)
assert abs(h[0] - 1.0 / tau) < 1e-9
assert h[-1] < h[0] * 1e-3   # decayed to <0.1% by t=15*tau

# zero for t<0
t_neg = np.array([-1e-3, -1e-6, 0.0, 1e-6])
h_neg = lti.impulse_response_rc(t_neg, R, C)
assert h_neg[0] == 0.0 and h_neg[1] == 0.0

# 2. step = integral of impulse: numeric integral matches analytic 1-e^(-t/RC)
s_numeric = lti.step_response_from_impulse(t, h)
s_analytic = lti.step_response_rc_analytic(t, R, C)
assert np.max(np.abs(s_numeric - s_analytic)) < 1e-3
assert abs(s_analytic[-1] - 1.0) < 1e-4   # settles to 1 (full charge)

# 3. convolution theorem: convolving x with h in time domain must match
#    multiplying their Fourier transforms in frequency domain
rng = np.random.default_rng(0)
n = 256
x = rng.normal(size=n)
h_short = lti.impulse_response_rc(np.arange(n) * dt, R, C)

y_time = lti.convolve_continuous(x, h_short, dt)

n_fft = len(x) + len(h_short) - 1
y_direct = np.convolve(x, h_short)
Y_via_fft = np.fft.irfft(np.fft.rfft(x, n=n_fft) * np.fft.rfft(h_short, n=n_fft), n=n_fft)
assert np.max(np.abs(y_direct - Y_via_fft[:len(y_direct)])) < 1e-8
# and convolve_continuous is just np.convolve scaled by dt
assert np.allclose(y_time, y_direct * dt)

# 4. frequency response: FFT of h(t) matches the analytic RC transfer function
#    in the passband (finite-window truncation limits high-frequency accuracy)
freqs, H_numeric = lti.frequency_response_from_impulse(h, dt)
H_analytic = lti.transfer_function_rc(freqs, R, C)
f_c = lti.cutoff_frequency_rc(R, C)
band = freqs < 3 * f_c
mag_err = np.max(np.abs(np.abs(H_numeric[band]) - np.abs(H_analytic[band])))
assert mag_err < 0.05

# 5. cutoff frequency: |H(f_c)| = 1/sqrt(2) exactly (by construction of f_c)
H_at_fc = lti.transfer_function_rc(f_c, R, C)
assert abs(abs(H_at_fc) - 1 / np.sqrt(2)) < 1e-9

# DC gain is exactly 1 (H(0) = 1, no attenuation at zero frequency)
assert abs(lti.transfer_function_rc(0.0, R, C) - 1.0) < 1e-12

# 6. doubling RC halves the cutoff frequency (f_c = 1/(2*pi*RC), inversely proportional)
f_c_2 = lti.cutoff_frequency_rc(2 * R, C)
assert abs(f_c_2 - f_c / 2) < 1e-6

# 7. input validation
for bad_call in [
    lambda: lti.impulse_response_rc(t, -1.0, C),
    lambda: lti.impulse_response_rc(t, R, 0.0),
    lambda: lti.step_response_rc_analytic(t, -1.0, C),
    lambda: lti.step_response_from_impulse(t[:-1], h),
    lambda: lti.convolve_continuous(x, h_short, -1.0),
    lambda: lti.transfer_function_rc(freqs, -1.0, C),
    lambda: lti.frequency_response_from_impulse(h, 0.0),
    lambda: lti.cutoff_frequency_rc(0.0, C),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.lti_systems tests passed")
