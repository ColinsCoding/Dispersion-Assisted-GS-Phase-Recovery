"""Optical frequency combs through Calculus 2: geometric series, Fourier series,
orthogonality integrals -- plus self-referencing, optical clockwork with exact
integers, and laser-safety arithmetic for ultrashort pulses.

  * PULSE TRAIN -> COMB. N pulses spaced T apart, each slipping its carrier
    phase by dphi, have spectrum A(f) * sum_k exp(i k (dphi - 2 pi f T)).
    The geometric series sums to a Dirichlet kernel whose peaks sit at
        f_n = n f_rep + f_ceo,   f_rep = 1/T,   f_ceo = (dphi / 2 pi) f_rep,
    each about f_rep / N wide. (Calc 2: sum of a geometric series.)
  * CARRIER-ENVELOPE OFFSET. The envelope travels at the group velocity, the
    carrier at the phase velocity; per round trip of length L the carrier slips
    dphi = omega_c L (1/v_g - 1/v_p). Dispersion sets f_ceo -- the same physics
    as this repo's group-delay notebooks.
  * f-2f. Double tooth n and beat it against tooth 2n:
        2 (n f_rep + f_ceo) - (2n f_rep + f_ceo) = f_ceo.
  * MEASURING A LASER. Beat it against the nearest tooth: nu = n f_rep + f_ceo +/- f_b.
    The integer n and the sign are ambiguous (like a 2 pi ambiguity); changing
    f_rep slightly resolves both (two measurements, like two dispersions in GS).
  * CLOCKWORK. Lock a tooth to an optical clock and f_rep becomes a microwave
    signal with the optical clock's fractional stability. Exact integer /
    rational arithmetic avoids the float precision trap from atomic_clock.
  * INNER PRODUCT. Comb teeth exp(i 2 pi f_n t) are orthogonal over one
    repetition period: the comb is an orthonormal basis, and the number of
    teeth in the optical bandwidth is the dimension of that signal space.
  * SAFETY. A comb's average power hides enormous peak power: E_pulse = P_avg / f_rep,
    P_peak ~ 0.88 E_pulse / tau (sech^2). MPE is an INPUT (see dgs.laser_safety).

Synthetic numbers; typical magnitudes. Education.
"""
from fractions import Fraction

import numpy as np
import sympy as sp

from dgs import dimensions_inner_product as dip
from dgs import laser_safety as ls

C = 299792458.0


# ── 1. pulse train -> comb: the geometric series ─────────────────────

def comb_geometric_sum_symbolic():
    """sum_{k=0}^{N-1} r^k with r = exp(i x), x = dphi - 2 pi f T, and |sum|^2."""
    x = sp.symbols("x", real=True)
    k, N = sp.symbols("k N", integer=True, positive=True)
    r = sp.exp(sp.I * x)
    S = sp.summation(r**k, (k, 0, N - 1))
    S_closed = (1 - r**N) / (1 - r)
    mag2 = sp.simplify(sp.expand_complex(S_closed * sp.conjugate(S_closed)))
    target = sp.sin(N * x / 2) ** 2 / sp.sin(x / 2) ** 2
    return S, sp.simplify(sp.trigsimp((mag2 - target).rewrite(sp.exp)).rewrite(sp.cos))


def comb_peak_frequencies_symbolic():
    """Peaks where x = 2 pi n:  dphi - 2 pi f T = -2 pi n  ->  f = n/T + dphi/(2 pi T)."""
    f, T, dphi = sp.symbols("f T Delta_phi", real=True)
    n = sp.symbols("n", integer=True)
    sol = sp.solve(sp.Eq(dphi - 2 * sp.pi * f * T, -2 * sp.pi * n), f)[0]
    return sp.expand(sol)


def dirichlet_kernel(x, N):
    """|sum_{k<N} e^{i k x}|^2 = sin^2(N x/2)/sin^2(x/2), with the limit N^2 at x = 2 pi n."""
    x = np.asarray(x, float)
    num, den = np.sin(N * x / 2) ** 2, np.sin(x / 2) ** 2
    out = np.full_like(x, float(N**2))
    ok = den > 1e-24
    out[ok] = num[ok] / den[ok]
    return out


def comb_spectrum(f, f_rep, f_ceo, n_pulses, pulse_fwhm_s=None):
    """|E(f)|^2 for a finite train: envelope |A(f)|^2 (Gaussian pulse) times the Dirichlet kernel."""
    f = np.asarray(f, float)
    x = 2 * np.pi * f_ceo / f_rep - 2 * np.pi * f / f_rep
    env = 1.0
    if pulse_fwhm_s is not None:
        sig_f = 0.4413 / pulse_fwhm_s / 2.3548        # Gaussian TBP 0.441
        env = np.exp(-0.5 * (f / sig_f) ** 2)
    return env * dirichlet_kernel(x, n_pulses)


def simulate_pulse_train(n_pulses, f_rep, dphi, fs, f_carrier, pulse_fwhm_s):
    """Time-domain train of Gaussian pulses with carrier phase slipping by dphi per pulse."""
    T = 1 / f_rep
    t = np.arange(int(n_pulses * T * fs)) / fs
    sig = 0.0 * t
    s = pulse_fwhm_s / 2.3548
    for k in range(n_pulses):
        tk = (k + 0.5) * T
        sig = sig + np.exp(-0.5 * ((t - tk) / s) ** 2) * np.cos(2 * np.pi * f_carrier * (t - tk) + k * dphi)
    return t, sig


# ── 2. carrier-envelope offset and f-2f ──────────────────────────────

def ceo_slip_symbolic():
    """dphi = omega_c L (1/v_g - 1/v_p) per round trip; f_ceo = dphi f_rep / (2 pi)."""
    w, L, vg, vp, frep = sp.symbols("omega_c L v_g v_p f_rep", positive=True)
    dphi = w * L * (1 / vg - 1 / vp)
    return dphi, sp.simplify(dphi * frep / (2 * sp.pi))


def ceo_from_slip(dphi_rad, f_rep):
    """f_ceo in [0, f_rep) from the per-pulse phase slip."""
    return (dphi_rad / (2 * np.pi) % 1.0) * f_rep


def f2f_symbolic():
    n, fr, f0 = sp.symbols("n f_rep f_ceo", positive=True)
    tooth = lambda m: m * fr + f0
    return sp.simplify(2 * tooth(n) - tooth(2 * n))


# ── 3. measuring an unknown laser: integer and sign ambiguity ────────

def tooth(n, f_rep, f_ceo):
    return n * f_rep + f_ceo


def beat_note(nu, f_rep, f_ceo):
    """|nu - nearest tooth| (what an RF counter sees) -- sign and n are lost."""
    n = round((nu - f_ceo) / f_rep)
    return abs(nu - tooth(n, f_rep, f_ceo))


def candidates(f_b, f_rep, f_ceo, nu_guess, span):
    """All frequencies consistent with one beat measurement inside a coarse window."""
    lo, hi = nu_guess - span, nu_guess + span
    n0, n1 = int(np.floor((lo - f_ceo) / f_rep)), int(np.ceil((hi - f_ceo) / f_rep))
    out = []
    for n in range(n0, n1 + 1):
        for sgn in (+1, -1):
            v = tooth(n, f_rep, f_ceo) + sgn * f_b
            if lo <= v <= hi:
                out.append((n, sgn, v))
    return out


def resolve_with_two_reps(nu_true, f_rep1, f_rep2, f_ceo, nu_guess, span, tol=1.0):
    """Measure the beat at two repetition rates; keep the candidates (n, sign)
    consistent with both. Returns the list of surviving frequencies."""
    b1, b2 = beat_note(nu_true, f_rep1, f_ceo), beat_note(nu_true, f_rep2, f_ceo)
    c1 = candidates(b1, f_rep1, f_ceo, nu_guess, span)
    c2 = candidates(b2, f_rep2, f_ceo, nu_guess, span)
    return sorted({round(v1, 3) for _, _, v1 in c1 for _, _, v2 in c2 if abs(v1 - v2) < tol}), len(c1)


# ── 4. optical clockwork with exact arithmetic ───────────────────────

def frep_from_optical(nu_opt, n, f_ceo, f_beat):
    """f_rep = (nu - f_ceo - f_beat) / n, evaluated EXACTLY with Fractions.
    Inputs may be ints, Fractions, or decimal strings ('429228004229873.0')."""
    F = lambda v: Fraction(str(v)) if not isinstance(v, Fraction) else v
    return (F(nu_opt) - F(f_ceo) - F(f_beat)) / n


def stability_transfer(frac_stability_optical):
    """Locked comb: delta f_rep / f_rep = delta nu / nu (division is noiseless in principle)."""
    return frac_stability_optical


# ── 5. comb teeth as an orthonormal basis ────────────────────────────

def tooth_orthogonality_symbolic(n, m):
    """<e^{i 2 pi n t/T}, e^{i 2 pi m t/T}> over one period T (via dgs.dimensions_inner_product)."""
    T = sp.symbols("T", positive=True)
    return sp.simplify(dip.fourier_inner_symbolic(n, m) * T / (2 * sp.pi))


def tooth_gram_matrix(n_list, f_rep, samples=4096):
    """Numeric Gram matrix of sampled teeth over one period, using the explicit-loop inner product."""
    T = 1 / f_rep
    t = np.arange(samples) * T / samples
    vecs = [np.exp(2j * np.pi * n * f_rep * t) / np.sqrt(samples) for n in n_list]
    return np.array([[dip.inner(u, v) for v in vecs] for u in vecs])


def comb_dimension(bandwidth_hz, f_rep):
    """Number of teeth inside the optical bandwidth = dimension of the signal space."""
    if bandwidth_hz <= 0 or f_rep <= 0:
        raise ValueError("bandwidth and f_rep must be positive")
    return int(bandwidth_hz // f_rep)


def minkowski_interval(dt_s, dx_m):
    """s^2 = c^2 dt^2 - dx^2 and its class (light-like / time-like / space-like).
    The Minkowski 'inner product' is indefinite, unlike the complex Hermitian
    inner product above, which is positive-definite."""
    s2 = (C * dt_s) ** 2 - dx_m**2
    tol = 1e-9 * max((C * dt_s) ** 2, dx_m**2, 1e-300)
    kind = "light-like" if abs(s2) <= tol else ("time-like" if s2 > 0 else "space-like")
    return s2, kind


# ── 6. ultrashort-pulse safety arithmetic ────────────────────────────

def pulse_energy(P_avg, f_rep):
    if P_avg <= 0 or f_rep <= 0:
        raise ValueError("P_avg and f_rep must be positive")
    return P_avg / f_rep


def peak_power(P_avg, f_rep, tau_fwhm, shape="sech2"):
    """P_peak = k E / tau with k = 0.88 (sech^2) or 0.94 (Gaussian)."""
    k = {"sech2": 0.8814, "gaussian": 0.9394}[shape]
    return k * pulse_energy(P_avg, f_rep) / tau_fwhm


def safety_summary(P_avg, f_rep, tau_fwhm, beam_diam_m, mpe_W_per_m2):
    """Average and peak irradiance and the eyewear OD for a caller-supplied MPE
    (look the MPE up in the current ANSI Z136.1 for your wavelength, pulse
    duration, and exposure; ultrashort-pulse rules differ from CW)."""
    E_avg = ls.beam_irradiance(P_avg, beam_diam_m)
    return {"pulse_energy_J": pulse_energy(P_avg, f_rep), "peak_power_W": peak_power(P_avg, f_rep, tau_fwhm),
            "avg_irradiance_W_m2": E_avg, "od_for_given_mpe": ls.optical_density_required(E_avg, mpe_W_per_m2)}
