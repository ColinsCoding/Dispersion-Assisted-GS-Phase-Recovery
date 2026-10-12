"""Why one intensity measurement cannot give the phase: zero flipping.

A length-L signal x[n] has spectrum X(w) = sum x[n] w^n on the unit circle
w = e^{-j omega}: a polynomial in w, fixed (up to a constant) by its L-1 zeros w_k.

ZERO FLIP: on |w| = 1,  |w - a| = |1 - conj(a) w|.  So replacing the factor
(w - a) by (1 - conj(a) w) -- which moves the zero from a to 1/conj(a), its
reflection through the unit circle -- leaves |X(w)| UNCHANGED everywhere on the
circle while changing the phase. With L-1 zeros there are up to 2^(L-1)
different signals sharing one Fourier magnitude: the 1-D phase-retrieval
ambiguity (flipping ALL zeros = time-reversed conjugate, a "trivial" one).

THE TIME-STRETCH LINK: in dgs.gs_core.disperse, H = exp(j pi D nu^2). For very
large D the dispersed intensity becomes the spectrum mapped onto time (the
dispersive Fourier transform behind photonic time-stretch), so a far-field
measurement sees ONLY |X| -- and cannot separate zero-flipped signals. At
moderate D the quadratic phase mixes phase into intensity, the flipped
signals give DIFFERENT traces, and two diversities (D1, D2) constrain the zeros
-- which is why the repo's retrieve_phase needs two sufficiently different D.
"""
import itertools

import numpy as np
from numpy.polynomial import polynomial as P

from dgs.gs_core import disperse


def zeros_of(x):
    """Zeros w_k of X(w) = sum x[n] w^n (ascending coefficients)."""
    x = np.asarray(x, complex)
    if x.size < 2 or x[-1] == 0:
        raise ValueError("need >= 2 samples and a nonzero last sample")
    return P.polyroots(x)


def signal_from_zeros(zeros, lead):
    """x[n] = ascending coefficients of lead * prod (w - w_k)."""
    return lead * P.polyfromroots(zeros)


def flip_zero(x, k):
    """Replace factor (w - a) by (1 - conj(a) w): zero a -> 1/conj(a), |X| unchanged.

    (1 - conj(a) w) = -conj(a) (w - 1/conj(a)), so the new leading constant is
    lead * (-conj(a))."""
    x = np.asarray(x, complex)
    z = zeros_of(x)
    a = z[k]
    z_new = z.copy()
    z_new[k] = 1 / np.conj(a)
    return signal_from_zeros(z_new, x[-1] * (-np.conj(a)))


def flip_set(x, mask):
    """Flip every zero whose index is in mask (iterable of bools)."""
    x = np.asarray(x, complex)
    z = zeros_of(x)
    lead = x[-1]
    z_new = z.copy()
    for k, m in enumerate(mask):
        if m:
            lead = lead * (-np.conj(z[k]))
            z_new[k] = 1 / np.conj(z[k])
    return signal_from_zeros(z_new, lead)


def all_flips(x):
    """All 2^(L-1) zero-flip variants (including x itself)."""
    L1 = len(zeros_of(x))
    return [flip_set(x, mask) for mask in itertools.product((False, True), repeat=L1)]


def fourier_magnitude(x, nfft=256):
    return np.abs(np.fft.fft(np.asarray(x, complex), nfft))


def upsample_smooth(x, sps=8, width=1.5):
    """Place samples on a fine grid with a Gaussian pulse per sample (band-limits the signal)."""
    n = len(x) * sps
    t = np.arange(n)
    out = np.zeros(n, complex)
    for k, xk in enumerate(x):
        out += xk * np.exp(-0.5 * ((t - (k + 0.5) * sps) / (width * sps / 2)) ** 2)
    return out


def dispersed_intensity(x, D, N=4096, sps=8):
    """Time intensity |disperse(E, D)|^2 of the upsampled signal centred in an N-sample window."""
    e = upsample_smooth(x, sps)
    E = np.zeros(N, complex)
    s = N // 2 - e.size // 2
    E[s:s + e.size] = e
    return np.abs(disperse(E, D)) ** 2


def distinguishability(signals, D, **kw):
    """Smallest relative L2 difference between the dispersed intensities of any two
    signals in the set (0 = some pair is indistinguishable at this D)."""
    I = [dispersed_intensity(s, D, **kw) for s in signals]
    best = np.inf
    for i, j in itertools.combinations(range(len(I)), 2):
        best = min(best, np.linalg.norm(I[i] - I[j]) / np.linalg.norm(I[i]))
    return float(best)


def symbolic_flip_identity():
    """SymPy: |w - a|^2 - |1 - conj(a) w|^2 = (|w|^2 - 1)(1 - |a|^2) -> 0 on |w| = 1."""
    import sympy as sp
    wr, wi, ar, ai = sp.symbols("w_r w_i a_r a_i", real=True)
    w, a = wr + sp.I * wi, ar + sp.I * ai
    lhs = sp.expand(sp.Abs(w - a) ** 2 - sp.Abs(1 - sp.conjugate(a) * w) ** 2)
    lhs = sp.expand(sp.expand_complex(lhs))
    factored = sp.factor(lhs)
    return factored, (wr, wi, ar, ai)


def max_dispersion_without_wrap(N, nu_max=0.15):
    """Group delay is -D nu samples; keep |D| nu_max < N/2 so the dispersed pulse
    does not wrap around the periodic FFT window (a numerical artefact, not physics)."""
    if N <= 0 or nu_max <= 0:
        raise ValueError("N and nu_max must be positive")
    return N / (2 * nu_max)
