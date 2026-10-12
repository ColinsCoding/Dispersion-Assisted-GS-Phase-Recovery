"""The lock-in amplifier: how quantum sensors pull a tiny signal out of 1/f noise.

  * MIX AND FILTER. Multiply the signal A cos(w t + phi) by references cos(w t)
    and sin(w t); the products contain DC terms (A/2) cos phi and (A/2) sin phi
    plus junk at 2w. A low-pass filter keeps the DC: X, Y, R = sqrt(X^2+Y^2),
    theta = atan2(Y, X). Derived in SymPy by averaging over a period.
  * NOISE BANDWIDTH. An n-pole RC low-pass with time constant tau passes white
    noise as if it were a brick-wall filter of width ENBW_n:
        1/(4 tau), 1/(8 tau), 3/(32 tau), 5/(64 tau), ...
    Output noise = noise density * sqrt(ENBW). Longer tau -> less noise, slower.
  * WHY MODULATE. Real detectors have 1/f (flicker) noise and drift. Moving the
    measurement to a modulation frequency f_m where the noise floor is white
    dodges them. A digital lock-in simulation shows the gain.
  * QUANTUM SENSING. Frequency-modulate the microwave drive of an NV
    magnetometer: f(t) = f_c + delta sin(w_m t). The lock-in's first-harmonic
    output is delta * dS/df for small delta -- a dispersive curve whose zero
    crossing sits exactly on resonance. That error signal, fed to an integrator,
    makes the microwave frequency TRACK the magnetic field in real time.

Builds on dgs.nv_magnetometer (Lorentzian ODMR, gamma_e). Synthetic data. Education.
"""
import numpy as np
import sympy as sp

from dgs import nv_magnetometer as nv


# ── 1. demodulation, derived ─────────────────────────────────────────

def demod_symbolic():
    """Period-average of A cos(w t + phi) times 2 cos(w t) and 2 sin(w t).
    (The factor 2 is the usual lock-in convention so that X = A cos phi.)
    Returns (X, Y, R, theta)."""
    A, w, t, phi = sp.symbols("A omega t phi", positive=True)
    Tp = 2 * sp.pi / w
    s = A * sp.cos(w * t + phi)
    X = sp.simplify(sp.integrate(2 * s * sp.cos(w * t), (t, 0, Tp)) / Tp)
    Y = sp.simplify(sp.integrate(-2 * s * sp.sin(w * t), (t, 0, Tp)) / Tp)
    R = sp.simplify(sp.sqrt(X**2 + Y**2))
    return X, Y, R, sp.atan2(Y, X)


def mixer_products_symbolic():
    """A cos(w t + phi) cos(w t) = (A/2)[cos(phi) + cos(2 w t + phi)]: DC + 2w.
    Returns (sum form, residual) -- the residual simplifies to 0."""
    A, w, t, phi = sp.symbols("A omega t phi", positive=True)
    prod = A * sp.cos(w * t + phi) * sp.cos(w * t)
    target = A / 2 * (sp.cos(phi) + sp.cos(2 * w * t + phi))
    return target, sp.simplify(sp.expand_trig(prod - target))


def off_frequency_rejection_symbolic():
    """Mixing a signal at w + dw with the w reference leaves a beat cos(dw t)
    (plus a 2w term): it is attenuated by the low-pass |H(dw)|.
    Returns (sum form, residual = 0)."""
    A, w, dw, t = sp.symbols("A omega Delta_omega t", positive=True)
    prod = 2 * A * sp.cos((w + dw) * t) * sp.cos(w * t)
    target = A * (sp.cos(dw * t) + sp.cos((2 * w + dw) * t))
    return target, sp.simplify(sp.expand_trig(prod - target))


def square_wave_reference_symbolic(n_terms=4):
    """Fourier series of a +/-1 square-wave reference: (4/pi) sum sin((2k+1) w t)/(2k+1).
    A square-wave lock-in also responds at 3w, 5w, ... with weights 1/3, 1/5, ..."""
    w, t = sp.symbols("omega t", positive=True)
    k = sp.symbols("k", integer=True, nonnegative=True)
    T = 2 * sp.pi / w
    coeffs = []
    for n in range(1, 2 * n_terms, 2):
        bn = sp.simplify(2 / T * (sp.integrate(sp.sin(n * w * t), (t, 0, T / 2))
                                  - sp.integrate(sp.sin(n * w * t), (t, T / 2, T))))
        coeffs.append((n, bn))
    return coeffs


# ── 2. low-pass filters and noise bandwidth ─────────────────────────

def enbw_symbolic(order):
    """ENBW of an n-pole RC filter: integral_0^inf |H(f)|^2 df, H = 1/(1 + j 2 pi f tau)^n."""
    if order < 1:
        raise ValueError("order must be >= 1")
    f, tau = sp.symbols("f tau", positive=True)
    return sp.simplify(sp.integrate(1 / (1 + (2 * sp.pi * f * tau) ** 2) ** order, (f, 0, sp.oo)))


def enbw(order, tau):
    """Closed form: ENBW_n = Gamma(n - 1/2) / (4 sqrt(pi) tau Gamma(n))."""
    from math import gamma, pi, sqrt
    if order < 1 or tau <= 0:
        raise ValueError("order >= 1 and tau > 0 required")
    return gamma(order - 0.5) / (4 * sqrt(pi) * tau * gamma(order))


def settle_time(order, tau, fraction=0.99):
    """Time for an n-pole filter's step response to reach `fraction` (bisection
    on the regularized gamma CDF: step = P(n, t/tau))."""
    from math import exp, factorial
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    step = lambda x: 1 - exp(-x) * sum(x**k / factorial(k) for k in range(order))
    lo, hi = 0.0, 1.0
    while step(hi) < fraction:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if step(mid) < fraction else (lo, mid)
    return hi * tau


def lowpass(x, dt, tau, order=1):
    """Cascade of `order` exponential (discrete RC) filters."""
    a = dt / (tau + dt)
    y = np.asarray(x, float)
    for _ in range(order):
        out = np.empty_like(y)
        acc = y[0] * 0.0
        for i, v in enumerate(y):
            acc += a * (v - acc)
            out[i] = acc
        y = out
    return y


def lowpass_fast(x, dt, tau, order=1):
    """Same filter via scipy.signal.lfilter (vectorized)."""
    from scipy.signal import lfilter
    a = dt / (tau + dt)
    y = np.asarray(x, float)
    for _ in range(order):
        y = lfilter([a], [1, -(1 - a)], y)
    return y


# ── 3. digital lock-in and 1/f noise ─────────────────────────────────

def pink_noise(n, dt, density_at_1hz, white_density, rng=None):
    """Noise with PSD S(f) = density^2 / f + white^2 (one-sided, V^2/Hz),
    synthesized by shaping white Gaussian noise in the frequency domain."""
    rng = np.random.default_rng(rng)
    f = np.fft.rfftfreq(n, dt)
    f[0] = f[1]
    psd = density_at_1hz**2 / f + white_density**2
    spec = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) * np.sqrt(psd / (2 * dt) * n) / np.sqrt(2)
    spec[0] = 0
    return np.fft.irfft(spec, n)


def lockin(signal, t, f_ref, tau, order=4, phase_deg=0.0):
    """Digital dual-phase lock-in. Returns (X, Y, R, theta_deg) time series."""
    ph = np.deg2rad(phase_deg)
    dt = t[1] - t[0]
    X = lowpass_fast(2 * signal * np.cos(2 * np.pi * f_ref * t + ph), dt, tau, order)
    Y = lowpass_fast(-2 * signal * np.sin(2 * np.pi * f_ref * t + ph), dt, tau, order)
    return X, Y, np.hypot(X, Y), np.degrees(np.arctan2(Y, X))


def dc_vs_lockin_demo(A=1e-3, f_mod=1000.0, fs=20000.0, T=20.0, tau=0.1,
                      flicker=3e-3, white=1e-4, drift=5e-3, rng=0):
    """A small signal A buried in drift + 1/f + white noise.
    DC method: switch the source on and off slowly (0.5 Hz) and difference.
    Lock-in: chop the source at f_mod and demodulate.
    Returns dict with estimated amplitudes and their scatter."""
    rng = np.random.default_rng(rng)
    n = int(T * fs)
    t = np.arange(n) / fs
    noise = pink_noise(n, 1 / fs, flicker, white, rng) + drift * t / T
    slow = (np.sin(2 * np.pi * 0.5 * t) > 0).astype(float)          # on/off at 0.5 Hz
    fast = 0.5 * (1 + np.sign(np.sin(2 * np.pi * f_mod * t)))        # chopped at f_mod
    # DC method: mean(on) - mean(off) in each 2 s cycle
    v_dc = A * slow + noise
    cyc = (t // 2).astype(int)
    dc_est = np.array([v_dc[(cyc == c) & (slow == 1)].mean() - v_dc[(cyc == c) & (slow == 0)].mean()
                       for c in range(int(T // 2))])
    # lock-in: square-wave chop has fundamental amplitude 2A/pi
    v_li = A * fast + noise
    X, Y, R, th = lockin(v_li, t, f_mod, tau, order=4, phase_deg=-90.0)   # chop is a sine: reference sin
    settle = int(10 * tau * fs)
    blocks = np.array_split(X[settle:], int(T // 2) - 1)
    li_est = np.array([b.mean() for b in blocks]) * np.pi / 2
    return {"true": A, "dc_mean": float(dc_est.mean()), "dc_std": float(dc_est.std()),
            "lockin_mean": float(li_est.mean()), "lockin_std": float(li_est.std())}


# ── 4. quantum sensing: FM lock-in on NV ODMR, and field tracking ────

def fm_first_harmonic_symbolic(order=3):
    """S(f_c + delta sin(w t)), Taylor-expanded in delta, projected on 2 sin(w t):
    first harmonic = delta S' + delta^3 S''' / 8 + ..."""
    d, t, fc = sp.symbols("delta t f_c", real=True)
    w = sp.symbols("omega", positive=True)
    S = sp.Function("S")
    u = sp.symbols("u", real=True)
    taylor = sum(sp.diff(S(u), u, k).subs(u, fc) * (d * sp.sin(w * t)) ** k / sp.factorial(k)
                 for k in range(order + 1))
    T = 2 * sp.pi / w
    h1 = sp.integrate(sp.expand(2 * taylor * sp.sin(w * t)), (t, 0, T)) / T
    return sp.simplify(h1)


def fm_error_signal(detuning_hz, delta_hz, contrast, fwhm_hz, n_phase=256):
    """Exact first-harmonic lock-in output for a Lorentzian ODMR dip under
    sinusoidal FM of depth delta, versus detuning f_c - f0 (numerical
    Fourier projection)."""
    ph = 2 * np.pi * (np.arange(n_phase) + 0.5) / n_phase
    det = np.atleast_1d(np.asarray(detuning_hz, float))[:, None]
    S = nv.lorentzian_dip(det + delta_hz * np.sin(ph)[None, :], 0.0, contrast, fwhm_hz)
    return 2 * np.mean(S * np.sin(ph)[None, :], axis=1)


def zero_crossing_slope(delta_hz, contrast, fwhm_hz, h=None):
    """d(error)/d(detuning) at resonance [1/Hz] -- the discriminator gain."""
    h = fwhm_hz * 1e-4 if h is None else h
    e = fm_error_signal([-h, h], delta_hz, contrast, fwhm_hz)
    return float((e[1] - e[0]) / (2 * h))


def optimal_modulation_depth(contrast, fwhm_hz):
    """Modulation depth delta that maximizes the zero-crossing slope (golden search)."""
    lo, hi = 0.01 * fwhm_hz, 3.0 * fwhm_hz
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(80):
        if abs(zero_crossing_slope(a, contrast, fwhm_hz)) > abs(zero_crossing_slope(b, contrast, fwhm_hz)):
            hi, b = b, a
            a = hi - g * (hi - lo)
        else:
            lo, a = a, b
            b = lo + g * (hi - lo)
    return (lo + hi) / 2


def track_field(B_of_t, t, contrast=0.03, fwhm=8e6, delta=None,
                photon_rate=1e11, gain=0.3, rng=0):
    """Closed-loop magnetometer: at each step the lock-in error (with shot
    noise) drives an integrator that steers the microwave frequency onto the
    upper ODMR line, f = D + gamma_e B. Returns the field estimate (f - D)/gamma_e.
    The lock-in is modeled at the level of its output: the exact FM error
    signal plus shot noise 1/sqrt(R dt) per step. `gain` is the fraction of the
    measured frequency error corrected each step (0 < gain < 1)."""
    rng = np.random.default_rng(rng)
    delta = optimal_modulation_depth(contrast, fwhm) if delta is None else delta
    k = zero_crossing_slope(delta, contrast, fwhm)                  # 1/Hz
    dt = t[1] - t[0]
    f_c = nv.D_ZFS + nv.GAMMA_E * B_of_t[0]
    est = np.empty(len(t))
    sigma = 1.0 / np.sqrt(photon_rate * dt)                         # relative shot noise per step
    for i, B in enumerate(B_of_t):
        f_line = nv.D_ZFS + nv.GAMMA_E * B
        err = fm_error_signal([f_c - f_line], delta, contrast, fwhm)[0] + sigma * rng.normal()
        f_c -= gain * err / k
        est[i] = (f_c - nv.D_ZFS) / nv.GAMMA_E
    return est


def predicted_tracking_noise(contrast=0.03, fwhm=8e6, photon_rate=1e11, dt=1e-4, gain=0.3, delta=None):
    """RMS field noise of the tracking loop for a constant field.
    Per step the frequency error has noise sigma_f = 1/(|k| sqrt(R dt)); a
    discrete integrator x <- (1-g) x + g n passes white noise with variance
    factor g / (2 - g). Field noise = sigma_f sqrt(g/(2-g)) / gamma_e.
    Loop bandwidth ~ g / (2 pi dt): faster tracking costs more noise."""
    delta = optimal_modulation_depth(contrast, fwhm) if delta is None else delta
    k = abs(zero_crossing_slope(delta, contrast, fwhm))
    sigma_f = 1.0 / (k * np.sqrt(photon_rate * dt))
    return sigma_f * np.sqrt(gain / (2 - gain)) / nv.GAMMA_E


def loop_bandwidth_hz(gain, dt):
    """-3 dB bandwidth of the discrete integrator loop (small-gain approximation)."""
    return gain / (2 * np.pi * dt)
