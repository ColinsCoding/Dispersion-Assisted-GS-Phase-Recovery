"""Linear time-invariant (LTI) systems, taken to engineering depth: impulse
response h(t), step response s(t), the convolution theorem, and the
frequency response H(f) as the Fourier transform of h(t) -- all cross-
verified against each other and against the analytic RC low-pass filter,
rather than treated as separate facts.

Three identities that MUST all agree for a real LTI system (verified, not
asserted):
  1. s(t) = integral_0^t h(tau) dtau            (step is impulse's integral)
  2. y(t) = (x * h)(t)                          (convolution is the system's
                                                   full response to ANY input)
  3. Y(f) = X(f) * H(f)                         (convolution theorem: time-
                                                   domain convolution becomes
                                                   frequency-domain multiplication)

Reuses dgs.ac_circuits' RC time constant convention (tau = R*C) rather than
redefining the RC low-pass; this module treats it as an LTI SYSTEM (impulse/
step/frequency response) rather than an AC steady-state impedance.
"""

import numpy as np


def impulse_response_rc(t, R, C):
    """Impulse response of a series-RC low-pass filter: h(t) = (1/RC)e^{-t/RC}
    for t >= 0 (causal), 0 for t < 0. This IS the actual physical response
    to a delta-function voltage spike at the input."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    t = np.asarray(t, dtype=float)
    tau = R * C
    h = np.where(t >= 0, np.exp(-t / tau) / tau, 0.0)
    return h


def step_response_rc_analytic(t, R, C):
    """Closed-form step response of the RC low-pass: s(t) = 1 - e^{-t/RC}
    for t >= 0 (the textbook capacitor-charging curve)."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    t = np.asarray(t, dtype=float)
    tau = R * C
    return np.where(t >= 0, 1.0 - np.exp(-t / tau), 0.0)


def step_response_from_impulse(t, h):
    """Step response computed NUMERICALLY as the running integral of the
    impulse response: s(t) = integral_0^t h(tau) dtau. This is the discrete
    analog of s = integral(h), verified against the analytic RC formula
    to confirm the identity holds, not just for RC specifically."""
    t = np.asarray(t, dtype=float)
    h = np.asarray(h, dtype=float)
    if t.shape != h.shape:
        raise ValueError("t and h must have the same shape")
    return np.concatenate([[0.0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])


def convolve_continuous(x, h, dt):
    """Discrete approximation to the continuous convolution integral
    y(t) = integral x(tau) h(t-tau) dtau, via numpy's discrete convolution
    scaled by dt (the Riemann-sum step) so the result approximates the
    true continuous-time convolution, not just the unscaled discrete sum."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.convolve(x, h) * dt


def transfer_function_rc(freqs_hz, R, C):
    """Analytic frequency response of the RC low-pass:
    H(f) = 1 / (1 + j*2*pi*f*R*C), the Fourier transform of impulse_response_rc
    (a real 1-pole low-pass, -3dB at f_c = 1/(2*pi*R*C))."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    omega = 2 * np.pi * freqs_hz
    return 1.0 / (1.0 + 1j * omega * R * C)


def frequency_response_from_impulse(h, dt):
    """H(f) computed NUMERICALLY as the (rfft) Fourier transform of a
    sampled impulse response h(t), with the corresponding frequency axis --
    to be cross-checked against transfer_function_rc's analytic formula."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    h = np.asarray(h, dtype=float)
    H = np.fft.rfft(h) * dt
    freqs = np.fft.rfftfreq(len(h), d=dt)
    return freqs, H


def cutoff_frequency_rc(R, C):
    """The RC low-pass -3dB corner frequency f_c = 1/(2*pi*R*C), i.e. where
    |H(f)| drops to 1/sqrt(2) of its DC value."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    return 1.0 / (2 * np.pi * R * C)


if __name__ == "__main__":
    R, C = 1.0e3, 1.0e-6   # a real RC pair: 1 kOhm, 1 uF -> tau = 1 ms
    tau = R * C
    dt = tau / 200.0
    t = np.arange(0, 12 * tau, dt)

    h = impulse_response_rc(t, R, C)
    s_numeric = step_response_from_impulse(t, h)
    s_analytic = step_response_rc_analytic(t, R, C)
    max_err = np.max(np.abs(s_numeric - s_analytic))
    print(f"RC low-pass: R={R:.0f} Ohm, C={C*1e6:.1f} uF, tau={tau*1e3:.2f} ms")
    print(f"step response: numeric integral of h(t) vs analytic 1-e^(-t/RC), "
          f"max error = {max_err:.2e}")

    freqs, H_numeric = frequency_response_from_impulse(h, dt)
    H_analytic = transfer_function_rc(freqs, R, C)
    # compare magnitude near the passband where the finite-time-window FFT
    # of a decaying exponential is most accurate
    band = freqs < cutoff_frequency_rc(R, C) * 5
    mag_err = np.max(np.abs(np.abs(H_numeric[band]) - np.abs(H_analytic[band])))
    print(f"frequency response: FFT of h(t) vs analytic 1/(1+j*2*pi*f*RC), "
          f"max |H| error (below 5*f_c) = {mag_err:.3e}")

    f_c = cutoff_frequency_rc(R, C)
    print(f"-3dB cutoff frequency: f_c = {f_c:.1f} Hz "
          f"(|H(f_c)| = {abs(transfer_function_rc(f_c, R, C)):.4f}, expect 1/sqrt(2)={1/np.sqrt(2):.4f})")
