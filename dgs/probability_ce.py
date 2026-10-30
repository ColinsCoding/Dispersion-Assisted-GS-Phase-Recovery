"""Calculus-based probability for computer engineering, derived from the physics of the hardware.

Each result is derived symbolically (SymPy) and then checked by Monte Carlo (NumPy); the general
statements are proved in lean/ProbabilityCE.lean.

    binomial_to_poisson()     bit errors / photon arrivals: Binomial(n, lambda/n) -> Poisson(lambda)
    photons_per_bit(...)      optical power -> mean photon count per bit (E = h c / lambda)
    shot_noise_snr(N)         Poisson variance = mean, so SNR = sqrt(N)
    johnson_sigma(T, R, B)    thermal (Johnson-Nyquist) noise voltage sqrt(4 k T R B)
    ber_derivation()          integrate the Gaussian tail: BER = (1/2) erfc(A / (sigma sqrt 2))
    packet_error(p, n)        1 - (1 - p)^n, bounded by n p (union bound / Bernoulli)
    expected_transmissions()  geometric retries: E[N] = 1 / (1 - q) by summing the series
    memoryless()              exponential inter-arrival times forget the past
    fermi(E, mu, T)           semiconductor occupancy; f(mu + x) + f(mu - x) = 1
    monte_carlo(seed)         simulate each one and compare with the formula
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

H_PLANCK = 6.62607015e-34     # J s
C_LIGHT = 2.99792458e8        # m / s
K_B = 1.380649e-23            # J / K
EV = 1.602176634e-19          # J


def _positive(**kw):
    for k, v in kw.items():
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"{k} must be a positive finite number, got {v!r}")


def _probability(**kw):
    for k, v in kw.items():
        if not (isinstance(v, (int, float)) and 0 <= v <= 1):
            raise ValueError(f"{k} must be a probability in [0, 1], got {v!r}")


# ---- counting: binomial -> Poisson, photons, shot noise --------------------------------------
def binomial_to_poisson():
    """Limit of C(n,k) (lam/n)^k (1 - lam/n)^(n-k) as n -> oo. Returns (pmf_n, limit)."""
    n, k, lam = sp.symbols("n k lambda", positive=True)
    pmf = sp.binomial(n, k) * (lam / n)**k * (1 - lam / n)**(n - k)
    # C(n,k)/n^k -> 1/k!  and  (1 - lam/n)^n -> e^-lam ;  (1 - lam/n)^-k -> 1
    limit = lam**k / sp.factorial(k) * sp.limit((1 - lam / n)**n, n, sp.oo)
    return pmf, limit


def poisson_moments():
    """Mean and variance of Poisson(lam), by summing the series symbolically."""
    k, lam = sp.Symbol("k", integer=True, nonnegative=True), sp.Symbol("lambda", positive=True)
    p = lam**k * sp.exp(-lam) / sp.factorial(k)
    mean = sp.simplify(sp.summation(k * p, (k, 0, sp.oo)))
    second = sp.simplify(sp.summation(k**2 * p, (k, 0, sp.oo)))
    return mean, sp.simplify(second - mean**2)


def photons_per_bit(power_w, bit_rate, wavelength_m):
    """Mean photons in one bit slot: (P / bit_rate) / (h c / lambda)."""
    _positive(power_w=power_w, bit_rate=bit_rate, wavelength_m=wavelength_m)
    return power_w / bit_rate / (H_PLANCK * C_LIGHT / wavelength_m)


def shot_noise_snr(mean_count):
    """Signal / noise for a Poisson count: N / sqrt(N) = sqrt(N)."""
    _positive(mean_count=mean_count)
    return math.sqrt(mean_count)


def johnson_sigma(T, R, B):
    """RMS thermal noise voltage of a resistor R at temperature T over bandwidth B."""
    _positive(T=T, R=R, B=B)
    return math.sqrt(4 * K_B * T * R * B)


# ---- bit error rate from Gaussian noise ------------------------------------------------------
def ber_derivation():
    """Symbols +A / -A plus Gaussian noise sigma, threshold 0. Returns (integral, closed form).

    An error on a +A bit means the noise is below -A: integrate the Gaussian pdf from -oo to -A.
    """
    x, A, s = sp.Symbol("x", real=True), sp.Symbol("A", positive=True), sp.Symbol("sigma", positive=True)
    pdf = sp.exp(-x**2 / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))
    integral = sp.Integral(pdf, (x, -sp.oo, -A))
    return integral, sp.simplify(integral.doit())


def ber(A, sigma):
    """Numeric BER for antipodal +-A signaling in Gaussian noise: (1/2) erfc(A / (sigma sqrt 2))."""
    _positive(A=A, sigma=sigma)
    return 0.5 * math.erfc(A / (sigma * math.sqrt(2)))


# ---- packets and retries ---------------------------------------------------------------------
def packet_error(p, n):
    """Probability a packet of n bits has at least one error, and the union bound n p."""
    _probability(p=p)
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    return 1 - (1 - p)**n, min(1.0, n * p)


def expected_transmissions():
    """Each try fails with probability q. E[N] = sum_{k>=1} k q^(k-1) (1 - q) = 1 / (1 - q)."""
    k, q = sp.Symbol("k", integer=True, positive=True), sp.Symbol("q", positive=True)
    s = sp.simplify(sp.summation(k * q**(k - 1) * (1 - q), (k, 1, sp.oo)))
    return sp.simplify(s.args[0][0] if isinstance(s, sp.Piecewise) else s)   # convergent branch, q < 1


def memoryless():
    """For T ~ Exp(lam): P(T > s + t | T > s) - P(T > t), simplified (it is 0)."""
    lam, s, t, x = sp.symbols("lambda s t x", positive=True)
    tail = lambda a: sp.integrate(lam * sp.exp(-lam * x), (x, a, sp.oo))
    return sp.simplify(tail(s + t) / tail(s) - tail(t))


# ---- semiconductor occupancy -----------------------------------------------------------------
def fermi(E_ev, mu_ev, T):
    """Fermi-Dirac occupancy 1 / (1 + exp((E - mu) / kT)), energies in eV."""
    _positive(T=T)
    return 1.0 / (1.0 + np.exp((np.asarray(E_ev, float) - mu_ev) * EV / (K_B * T)))


def fermi_symmetry():
    """f(mu + x) + f(mu - x) - 1, simplified (it is 0): a state above mu is as likely full as one
    the same distance below is empty."""
    x = sp.Symbol("x", real=True)
    f = lambda u: 1 / (1 + sp.exp(u))
    return sp.simplify(f(x) + f(-x) - 1)


# ---- simulation ------------------------------------------------------------------------------
def monte_carlo(seed=0, n=400_000):
    """Simulate each model and return {name: (formula, simulated)}."""
    rng = np.random.default_rng(seed)
    out = {}
    A, sigma = 1.0, 0.4
    bits = rng.choice([-A, A], n)
    out["BER, A/sigma = 2.5"] = (ber(A, sigma), np.mean(np.sign(bits + rng.normal(0, sigma, n)) != np.sign(bits)))
    lam = 7.0
    counts = rng.binomial(10_000, lam / 10_000, n)
    out["photon count mean (binomial n=1e4)"] = (lam, counts.mean())
    out["photon count variance"] = (lam, counts.var())
    p, nbits = 1e-4, 1500
    out["packet error, 1500 bits, p = 1e-4"] = (packet_error(p, nbits)[0], np.mean(rng.binomial(nbits, p, n) > 0))
    q = 0.3
    out["E[transmissions], q = 0.3"] = (1 / (1 - q), rng.geometric(1 - q, n).mean())
    T = rng.exponential(2.0, n)
    out["P(T > 3 | T > 1) vs P(T > 2)"] = (np.mean(T > 2), np.mean(T[T > 1] > 3))
    X = rng.exponential(1.0, n)
    out["Markov: P(X >= 3) <= E[X]/3"] = (1 / 3, np.mean(X >= 3))
    sums = rng.uniform(0, 1, (n // 10, 12)).sum(1) - 6           # CLT: sum of 12 uniforms ~ N(0, 1)
    out["CLT: std of sum of 12 uniforms - 6"] = (1.0, sums.std())
    return out
