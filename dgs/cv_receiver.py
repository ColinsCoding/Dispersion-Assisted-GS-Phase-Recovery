"""Continuous-variable (CV) photonic receiver: where continuous fields become discrete bits.

In plain words: a CV photonic processor stores quantum information in the continuous
quadratures x, p of light modes (like the position and momentum of an oscillator). Reading it
out needs a HOMODYNE receiver: interfere the signal with a strong local oscillator (LO) and
subtract two photodiode currents; the difference is proportional to one quadrature. Three
receiver facts decide whether quantum information survives:

  1. Loss and detector inefficiency eta mix in vacuum noise: V -> eta V + (1 - eta)/2.
     Squeezing (variance below the vacuum 1/2) leaks away quickly.
  2. Amplifiers. A phase-INsensitive amplifier of gain G (an EDFA, the kind time-stretch
     uses) adds noise: V -> G V + (G - 1)/2, i.e. at least half a vacuum unit referred to the
     input -- squeezing is destroyed. A phase-SENSITIVE amplifier (optical parametric
     amplifier) can amplify one quadrature noiselessly. So classical time-stretch cannot carry
     squeezed light; a quantum "time-stretch" would need phase-sensitive gain.
  3. Mode matching. The homodyne only sees the part of the signal that overlaps the LO's
     temporal mode: eta_total = eta_det x |<LO|signal>|^2. Dispersion chirps the signal mode;
     an unchirped LO then wastes squeezing. Measuring the chirp is a CLASSICAL phase-retrieval
     job: send a bright pulse through the same mode, record intensities at zero and two
     dispersions, and run Gerchberg-Saxton (dgs.gs_core operators). Two planes are not enough
     (two chirps fit the same pair of widths); three planes plus a few quadratic-phase seeds
     recover the mode, and the LO can be shaped to match.

The discrete/continuous boundary: a GKP qubit (Gottesman-Kitaev-Preskill) stores a bit as a
comb of narrow peaks in x, spaced sqrt(pi); the receiver decodes bit = round(x / sqrt(pi)) mod 2.
Finite squeezing, loss, and ADC quantization all act like Gaussian/uniform displacement noise;
an error happens when the total shift rounds to the wrong cell.

Conventions: x = (a + a^dagger)/sqrt(2), vacuum variance 1/2, hbar = 1. NumPy + SciPy.
"""

import numpy as np
from scipy.special import erfc
from scipy.stats import norm

from dgs import gs_core

VACUUM = 0.5
SQRT_PI = np.sqrt(np.pi)


def _frac(name, v):
    if not np.all((np.asarray(v) > 0) & (np.asarray(v) <= 1)):
        raise ValueError(f"{name} must be in (0, 1], got {v}")


# ---------------------------------------------------------------- 1-2. loss and amplifiers

def db(variance):
    """Noise relative to vacuum in dB: negative = squeezed."""
    return 10 * np.log10(np.asarray(variance, float) / VACUUM)


def variance_from_db(s_db):
    return VACUUM * 10 ** (np.asarray(s_db, float) / 10)


def after_loss(V, eta):
    """Beam-splitter loss / detector efficiency: eta V + (1 - eta) / 2."""
    _frac("eta", eta)
    eta = np.asarray(eta, float)
    return eta * np.asarray(V, float) + (1 - eta) * VACUUM


def after_pia(V, G):
    """Phase-insensitive amplifier (Caves limit): G V + (G - 1)/2."""
    if np.any(np.asarray(G) < 1):
        raise ValueError("gain G must be >= 1")
    G = np.asarray(G, float)
    return G * np.asarray(V, float) + (G - 1) * VACUUM


def after_psa(V_x, V_p, G):
    """Ideal phase-sensitive amplifier: x amplified by G, p de-amplified by 1/G, no added noise."""
    if np.any(np.asarray(G) < 1):
        raise ValueError("gain G must be >= 1")
    return G * np.asarray(V_x, float), np.asarray(V_p, float) / G


def measured_squeezing_db(s_db, eta_total):
    """Observed squeezing after total efficiency eta_total for an initial s_db (< 0) squeezed state."""
    return db(after_loss(variance_from_db(s_db), eta_total))


def homodyne_samples(V, n, seed=0):
    """n homodyne outcomes of a Gaussian state with quadrature variance V (zero mean)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.sqrt(V) * np.random.default_rng(seed).standard_normal(n)


# ---------------------------------------------------------------- 3. mode matching with GS calibration

def chirped_gaussian(t, T0, C):
    """exp(-(1 + iC) t^2 / (2 T0^2)): a chirped temporal mode."""
    if T0 <= 0:
        raise ValueError("T0 must be positive")
    return np.exp(-(1 + 1j * C) * np.asarray(t, float) ** 2 / (2 * T0 ** 2))


def mode_overlap(a, b):
    """|<a|b>|^2 / (<a|a><b|b>): the fraction of mode b seen by an LO in mode a."""
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    return float(abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def gaussian_chirp_overlap(C):
    """Closed form overlap of an unchirped and a chirped Gaussian of equal T0: 2 / sqrt(4 + C^2)."""
    return 2 / np.sqrt(4 + np.asarray(C, float) ** 2)


def calibration_intensities(field, dispersions, snr_db=None, seed=0):
    """Intensities |field|^2 at D = 0 and at each D (gs_core.disperse), with optional noise."""
    rng = np.random.default_rng(seed)
    out = [np.abs(field) ** 2] + [np.abs(gs_core.disperse(field, D)) ** 2 for D in dispersions]
    if snr_db is not None:
        out = [np.maximum(I + I.max() * 10 ** (-snr_db / 20) * rng.standard_normal(I.size), 0) for I in out]
    return out


def denoise_intensity(I, cutoff=0.03, snr_db=None, floor_sigmas=3.0):
    """Low-pass the measured intensity (keep |f| <= cutoff cycles/sample) and zero what is left
    below floor_sigmas x the filtered noise level. Noise in the dark tails otherwise feeds
    the GS amplitude projections and pulls the retrieval to a wrong mode."""
    if not 0 < cutoff <= 0.5:
        raise ValueError("cutoff must be in (0, 0.5]")
    I = np.asarray(I, float)
    F = np.fft.fft(I)
    F[np.abs(np.fft.fftfreq(I.size)) > cutoff] = 0
    J = np.fft.ifft(F).real
    if snr_db is None:
        return np.maximum(J, 0)
    sigma_f = I.max() * 10 ** (-snr_db / 20) * np.sqrt(2 * cutoff)   # white noise after the filter
    return np.where(J > floor_sigmas * sigma_f, J, 0.0)


def gs_multiplane(I0, Is, Ds, seed_field, n_iter):
    """Cycle amplitude projections: sqrt(I0) at D = 0 and sqrt(I_k) at each D_k. Returns (field, mismatch)."""
    g = np.asarray(seed_field, complex)
    a0 = np.sqrt(np.maximum(I0, 0))
    amps = [np.sqrt(np.maximum(I, 0)) for I in Is]
    for _ in range(n_iter):
        for A, D in zip(amps, Ds):
            gd = gs_core.disperse(g, D)
            g = gs_core.undisperse(A * np.exp(1j * np.angle(gd)), D)
            g = a0 * np.exp(1j * np.angle(g))
    mismatch = float(sum(np.mean((np.abs(gs_core.disperse(g, D)) ** 2 - I) ** 2) for I, D in zip(Is, Ds)))
    return g, mismatch


def retrieve_mode(I0, Is, Ds, width_scale, seeds=np.linspace(-12, 12, 13), n_seed_iter=60, n_final_iter=400):
    """Multistart GS: try quadratic-phase seeds exp(-i a t^2 / 2 w^2), keep the best, refine."""
    if len(Is) != len(Ds) or len(Ds) < 1:
        raise ValueError("need one intensity per dispersion")
    n = len(I0)
    t = np.arange(n) - n / 2
    best = None
    for a in seeds:
        g, e = gs_multiplane(I0, Is, Ds, np.sqrt(I0) * np.exp(-1j * a * t ** 2 / (2 * width_scale ** 2)), n_seed_iter)
        if best is None or e < best[1]:
            best = (g, e, a)
    g, e = gs_multiplane(I0, Is, Ds, best[0], n_final_iter)
    return g, e, float(best[2])


# ---------------------------------------------------------------- 4. GKP decoding at the boundary

def gkp_noise_variance(s_db, eta=1.0, adc_bits=None, adc_range=4 * SQRT_PI):
    """Total effective displacement variance: finite squeezing + loss (rescaled, (1-eta)/(2 eta))
    + ADC quantization LSB^2/12 over +-adc_range."""
    _frac("eta", eta)
    var = variance_from_db(s_db) + (1 - eta) / (2 * eta)
    if adc_bits is not None:
        lsb = 2 * adc_range / (2 ** int(adc_bits))
        var = var + lsb ** 2 / 12
    return float(var)


def gkp_error_exact(sigma):
    """P(round(noise / sqrt(pi)) is odd) for Gaussian noise of std sigma (sums the wrapped cells)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    k = np.arange(1, 400, 2)
    return float(np.sum(2 * (norm.cdf((k + 0.5) * SQRT_PI / sigma) - norm.cdf((k - 0.5) * SQRT_PI / sigma))))


def gkp_error_approx(sigma):
    """Leading term: erfc(sqrt(pi) / (2 sqrt(2) sigma))."""
    return float(erfc(SQRT_PI / (2 * np.sqrt(2) * sigma)))


def gkp_monte_carlo(s_db, eta=1.0, adc_bits=None, adc_range=4 * SQRT_PI, n=200000, seed=0):
    """Simulate the receiver: GKP peaks at sqrt(pi)(2m + bit) + Gaussian noise (squeezing + loss),
    then a b-bit ADC over +-adc_range, then decode bit = round(x / sqrt(pi)) mod 2. Returns the bit error rate."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n)
    m = rng.integers(-1, 2, n)                                     # a few peaks of the comb
    sigma = np.sqrt(variance_from_db(s_db) + (1 - eta) / (2 * eta))
    x = SQRT_PI * (2 * m + bits) + sigma * rng.standard_normal(n)
    if adc_bits is not None:
        lsb = 2 * adc_range / (2 ** int(adc_bits))
        x = np.clip(np.floor(x / lsb) * lsb + lsb / 2, -adc_range, adc_range)
    decoded = np.mod(np.round(x / SQRT_PI).astype(int), 2)
    return float(np.mean(decoded != bits))
