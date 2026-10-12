"""Bessel functions, interference and the Doppler effect, all in one instrument:
the laser Doppler vibrometer (LDV), plus the Airy pattern.

    moving mirror      f' = f (1 + beta) / (1 - beta) ~ f (1 + 2 v / c)       (two Lorentz boosts)
    Michelson          I = I1 + I2 + 2 sqrt(I1 I2) cos(dphi),  dphi = 4 pi x / lambda
    fringe rate        (1/2pi) d(dphi)/dt = 2 v / lambda   == the Doppler beat, exactly to O(beta)
    vibrating mirror   x = a sin(W t)  ->  dphi = phi0 + b sin(W t),  b = 4 pi a / lambda
    Jacobi-Anger       cos(phi0 + b sin W t) = J0(b) cos phi0
                         + 2 sum_k J_2k(b) cos(2k W t) cos phi0
                         - 2 sum_k J_(2k-1)(b) sin((2k-1) W t) sin phi0
    Bessel null        the carrier term vanishes at J0(b) = 0 -> a = j_{0,m} lambda / (4 pi)
    heterodyne LDV     a frequency-shifted reference (Bragg cell, f_B) makes the detector
                       signal cos(2 pi f_B t + dphi(t)); IQ demodulation recovers dphi(t),
                       and v(t) = (lambda / 4 pi) d(dphi)/dt
    Airy pattern       circular aperture: I(theta) = I0 [2 J1(x) / x]^2, x = k R sin(theta);
                       first dark ring at x = 3.8317 -> theta = 1.22 lambda / D

Reuses dgs.pm_sideband_bessel (Jacobi-Anger) and dgs.special_relativity
(relativistic_doppler). All signals are synthetic.
"""
import numpy as np
import sympy as sp
from scipy.special import j0, j1, jv, jn_zeros

from dgs.pm_sideband_bessel import jacobi_anger_sidebands
from dgs.special_relativity import C_SI, relativistic_doppler

HENE_NM = 632.8


# ── 1. Doppler from a moving mirror ─────────────────────────────────

def mirror_doppler_symbolic():
    """Light at f hits a mirror approaching at beta. In the mirror frame the
    light is Doppler-shifted once (f D); the mirror re-emits that frequency,
    which the lab sees shifted again: f' = f D^2, D = sqrt((1+beta)/(1-beta)).
    Returns (exact f'/f, its series to O(beta^3))."""
    beta = sp.symbols("beta", real=True)
    D = sp.sqrt((1 + beta) / (1 - beta))
    ratio = sp.simplify(D**2)
    return ratio, sp.series(ratio, beta, 0, 3).removeO()


def mirror_doppler_shift_hz(speed_m_s, wavelength_nm=HENE_NM):
    """Exact shift f' - f for a mirror approaching at speed v (negative = receding).
    Uses dgs.special_relativity.relativistic_doppler twice (in, then back out)."""
    f = C_SI / (wavelength_nm * 1e-9)
    once = relativistic_doppler(f, speed_m_s, approaching=True)["f_obs"]
    return relativistic_doppler(once, speed_m_s, approaching=True)["f_obs"] - f


# ── 2. interference and the fringe rate ─────────────────────────────

def michelson_intensity(dphi, I1=1.0, I2=1.0):
    """Two-beam interference. Constructive at dphi = 2 pi m (I = (sqrt I1 + sqrt I2)^2),
    destructive at (2m+1) pi (I = (sqrt I1 - sqrt I2)^2)."""
    if I1 < 0 or I2 < 0:
        raise ValueError("intensities must be non-negative")
    return I1 + I2 + 2 * np.sqrt(I1 * I2) * np.cos(dphi)


def fringe_rate_equals_doppler():
    """Mirror at x(t): round-trip phase 4 pi x / lambda. The fringe (beat) rate is
    (1/2pi) d/dt of that = 2 v / lambda. The first-order Doppler shift is
    2 (v/c) f = 2 v / lambda. Returns their difference (must be 0)."""
    t, lam, c = sp.symbols("t lambda c", positive=True)
    x = sp.Function("x")(t)
    fringe_rate = sp.diff(4 * sp.pi * x / lam, t) / (2 * sp.pi)
    beta = sp.diff(x, t) / c
    doppler_first_order = 2 * beta * (c / lam)
    return sp.simplify(fringe_rate - doppler_first_order)


# ── 3. vibrating mirror: Bessel sidebands ───────────────────────────

def modulation_index(amplitude_m, wavelength_nm=HENE_NM):
    """b = 4 pi a / lambda (round trip doubles the path change)."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    return 4 * np.pi * np.asarray(amplitude_m, float) / (wavelength_nm * 1e-9)


def detector_harmonics(b, phi0, n_max=8):
    """Amplitudes of the detector signal cos(phi0 + b sin W t) at DC and each
    harmonic n W (one-sided, as a spectrum analyzer shows them). Even n scale
    with cos(phi0), odd n with sin(phi0)."""
    ns = np.arange(n_max + 1)
    amp = np.where(ns % 2 == 0, np.abs(np.cos(phi0)), np.abs(np.sin(phi0))) * np.abs(jv(ns, b))
    amp[1:] *= 2
    return ns, amp


def fft_harmonics(b, phi0, n_max=8, n_samples=4096):
    """Same amplitudes measured by FFT of one period of the synthetic signal."""
    th = 2 * np.pi * np.arange(n_samples) / n_samples
    s = np.cos(phi0 + b * np.sin(th))
    S = np.abs(np.fft.rfft(s)) / n_samples
    S[1:] *= 2
    return np.arange(n_max + 1), S[: n_max + 1]


def jacobi_anger_check(b, n_max=30):
    """max |exp(i b sin x) - sum_n J_n(b) exp(i n x)| over one period, using the
    sidebands from dgs.pm_sideband_bessel."""
    xs = np.linspace(0, 2 * np.pi, 400)
    ns, c = jacobi_anger_sidebands(b, n_max)
    series = (c[None, :] * np.exp(1j * np.outer(xs, ns))).sum(axis=1)
    return float(np.max(np.abs(np.exp(1j * b * np.sin(xs)) - series)))


def bessel_null_amplitudes(n_zeros=3, wavelength_nm=HENE_NM):
    """Vibration amplitudes at which the carrier vanishes (J0(b) = 0): the
    'Bessel null' calibration method. Returns (zeros of J0, amplitudes in m)."""
    z = jn_zeros(0, n_zeros)
    return z, z * wavelength_nm * 1e-9 / (4 * np.pi)


# ── 4. heterodyne LDV: recover v(t) from a noisy detector signal ────

def heterodyne_ldv(velocity_fn, duration_s=2e-3, fs_hz=200e6, f_bragg_hz=40e6,
                   wavelength_nm=HENE_NM, snr_db=30.0, lp_cut_hz=10e6, seed=0):
    """Synthesize the detector signal cos(2 pi f_B t + dphi(t)) + noise, where
    dphi = 4 pi x(t) / lambda and x = integral of v. Demodulate: multiply by
    exp(-i 2 pi f_B t), low-pass in the frequency domain, take the unwrapped angle,
    differentiate. Returns dict with t, true v, recovered v, RMS error.
    The Doppler shift 2v/lambda must stay well inside f_B, or the sign of v is lost."""
    if fs_hz <= 2 * f_bragg_hz:
        raise ValueError("sample rate must exceed 2 f_B")
    rng = np.random.default_rng(seed)
    lam = wavelength_nm * 1e-9
    t = np.arange(int(duration_s * fs_hz)) / fs_hz
    v = velocity_fn(t)
    x = np.cumsum(v) / fs_hz
    dphi = 4 * np.pi * x / lam
    s = np.cos(2 * np.pi * f_bragg_hz * t + dphi)
    s += np.sqrt(0.5) * 10 ** (-snr_db / 20) * rng.standard_normal(len(t))   # signal power 1/2 -> SNR = snr_db
    base = s * np.exp(-2j * np.pi * f_bragg_hz * t)
    F = np.fft.fft(base)
    F[np.abs(np.fft.fftfreq(len(t), 1 / fs_hz)) > lp_cut_hz] = 0
    ph = np.unwrap(np.angle(np.fft.ifft(F)))
    v_rec = lam / (4 * np.pi) * np.gradient(ph, 1 / fs_hz)
    edge = len(t) // 20                              # ignore FFT-filter edge ringing
    err = v_rec[edge:-edge] - v[edge:-edge]
    return {"t": t, "v": v, "v_rec": v_rec, "rms_error": float(np.sqrt(np.mean(err**2))),
            "max_doppler_hz": float(2 * np.max(np.abs(v)) / lam)}


# ── 5. Airy pattern: Bessel functions in diffraction ────────────────

def airy(x):
    """[2 J1(x) / x]^2 with the x -> 0 limit 1."""
    x = np.asarray(x, float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = (2 * j1(x[nz]) / x[nz]) ** 2
    return out


def airy_symbolic():
    """Fraunhofer amplitude of a disk of radius R: integral over the aperture of
    exp(-i k r sin(theta) cos(phi)) r dr dphi. The phi integral gives 2 pi J0(q r),
    and int_0^R J0(q r) r dr = R J1(q R) / q. Returns (that radial integral, its
    check against R J1(qR)/q, which must be 0)."""
    r, R, q = sp.symbols("r R q", positive=True)
    radial = sp.integrate(sp.besselj(0, q * r) * r, (r, 0, R))
    return radial, sp.simplify(radial - R * sp.besselj(1, q * R) / q)


def airy_numeric_check(n=1024, radius_px=24, n_samples=40):
    """2D FFT of a pixelated disk vs the Airy formula along one axis.
    Returns (x values, FFT intensity, Airy intensity)."""
    yy, xx = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
    disk = (xx**2 + yy**2 <= radius_px**2).astype(float)
    P = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(disk)))) ** 2
    P /= P[n // 2, n // 2]
    k = np.arange(n_samples)
    x = 2 * np.pi * radius_px * k / n          # k R sin(theta) on the FFT grid
    return x, P[n // 2, n // 2 + k], airy(x)


def rayleigh_factor():
    """First zero of J1 / pi: theta_min = (j_{1,1} / pi) lambda / D = 1.2197 lambda / D."""
    return float(jn_zeros(1, 1)[0] / np.pi)


# ── 6. hardware-style demodulator (CIC) and multi-channel test signal ──
#
# FPGA and GPU demodulators avoid FFTs: they mix to baseband and filter with
# cascaded moving averages (a CIC filter). A length-L boxcar at fs = 200 MS/s
# with L = 5 has nulls at every multiple of 40 MHz, so the 2 f_B = 80 MHz mixing
# product lands on a null. The Doppler shift moves it slightly off the null, so
# one boxcar leaks ~2.6 %; three in cascade leak ~2e-5. Group delay: order*(L-1)/2.

def vibration_signal(n_samples, v_peak, f_vib=5e3, fs_hz=200e6, f_bragg_hz=40e6,
                     wavelength_nm=HENE_NM):
    """Noise-free heterodyne signal for v(t) = v_peak sin(2 pi f_vib t), using the
    exact displacement x(t) = v_peak (1 - cos(2 pi f_vib t)) / (2 pi f_vib).
    Returns (t, s, v_true). The CUDA and MATLAB versions use the same formulas."""
    t = np.arange(n_samples) / fs_hz
    w = 2 * np.pi * f_vib
    x = v_peak * (1 - np.cos(w * t)) / w
    s = np.cos(2 * np.pi * f_bragg_hz * t + 4 * np.pi * x / (wavelength_nm * 1e-9))
    return t, s, v_peak * np.sin(w * t)


def cic_demodulate(s, fs_hz=200e6, f_bragg_hz=40e6, wavelength_nm=HENE_NM, length=5, order=3):
    """Mix to baseband, CIC low-pass (order cascaded boxcars of `length`), unwrap
    the phase, central-difference derivative. Returns (v_rec, delay_samples):
    v_rec[n] estimates v at sample n - delay_samples. Invalid first samples are NaN."""
    if length < 1 or order < 1:
        raise ValueError("length and order must be >= 1")
    t = np.arange(len(s)) / fs_hz
    z = s * np.exp(-2j * np.pi * f_bragg_hz * t)
    k = np.ones(length) / length
    for _ in range(order):
        z = np.convolve(z, k)[: len(s)]                 # causal boxcar
    ph = np.unwrap(np.angle(z))
    v = np.full(len(s), np.nan)
    v[1:-1] = wavelength_nm * 1e-9 / (4 * np.pi) * (ph[2:] - ph[:-2]) * fs_hz / 2
    v[: order * (length - 1) + 1] = np.nan             # filter not yet filled
    return v, order * (length - 1) / 2
