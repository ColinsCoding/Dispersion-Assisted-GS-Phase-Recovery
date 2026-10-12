"""Why the plum-pudding atom failed: Thomson vs Rutherford, quantitatively
(Knight Ch. 37.6). Order-of-magnitude estimates with every assumption labelled.

THOMSON ATOM: positive charge Q = Ze spread uniformly over a sphere of radius R,
electrons embedded; outside the atom the field is screened (neutral atom).
Inside, Gauss's law gives E(r) = k Q r / R^3 (linear in r). An alpha (charge q,
kinetic energy K) on a straight line at impact parameter b < R gets a sideways
impulse (calculus: integrate F_perp dt along the chord)
    dp = k q Q b / R^3 * 2 sqrt(R^2 - b^2) / v     ->   theta(b) = (k q Q / K) b sqrt(R^2 - b^2) / R^3
maximised at b = R/sqrt2:   theta_max = k q Q / (2 K R)   ~ 1e-4 rad.
Electrons can't do better: an alpha hitting an electron at rest turns by at most
m_e / m_alpha ~ 1.4e-4 rad.

MANY SMALL KICKS: crossing a foil of thickness t means N ~ t / a collisions
(a = n^(-1/3)); random directions add like a 2-D random walk,
    <theta^2> = N theta_1^2,   P(theta > T) = exp(-T^2 / <theta^2>).
For T = 90 deg this is ~10^(-thousands): plum pudding forbids back-scattering.

RUTHERFORD ATOM: one close encounter with a point nucleus,
    P(theta > T) = pi b(T)^2 n t,   b(T) = (d/2) cot(T/2),   d = k q Q / K,
which is ~1e-5 -- small but observable, as Geiger and Marsden saw.
"""
import numpy as np
import sympy as sp

KE2_MEV_FM = 1.439964     # e^2 / (4 pi eps0) in MeV fm
FM = 1e-15

# Assumptions (labelled, change freely)
GOLD = dict(Z=79, R_atom=1.44e-10, n=5.90e28)   # metallic radius (m), number density (m^-3)
ALPHA = dict(q=2, K_MeV=7.7, mass_ratio_e=7294.3)  # K ~ RaC' alphas; m_alpha / m_e
FOIL_T = 4.0e-7                                    # 0.4 um thin gold foil (assumed)


def thomson_field_symbolic():
    """Gauss's law for a uniform sphere: enclosed charge Q r^3/R^3 -> E(r) = kQr/R^3."""
    k, Q, R, r = sp.symbols("k Q R r", positive=True)
    q_enc = Q * r ** 3 / R ** 3
    return sp.simplify(k * q_enc / r ** 2)


def thomson_deflection_symbolic():
    """theta(b) from the impulse integral, and its maximum over b."""
    k, q, Q, R, b, x, v, m, K = sp.symbols("k q Q R b x v m K", positive=True)
    Fperp = k * q * Q * b / R ** 3                   # perpendicular force inside the sphere
    L = sp.sqrt(R ** 2 - b ** 2)
    dp = sp.integrate(Fperp / v, (x, -L, L))         # dt = dx / v along the straight chord
    theta = sp.simplify((dp / (m * v)).subs(m * v ** 2, 2 * K))
    theta = sp.simplify(theta.subs(v, sp.sqrt(2 * K / m)))
    b_star = [s for s in sp.solve(sp.diff(theta, b), b) if s.is_positive][0]
    theta_max = sp.simplify(theta.subs(b, b_star))
    return theta, b_star, theta_max


def theta_max_thomson(Z=GOLD["Z"], R=GOLD["R_atom"], q=ALPHA["q"], K_MeV=ALPHA["K_MeV"]):
    """k q Q / (2 K R), radians."""
    if min(Z, R, q, K_MeV) <= 0:
        raise ValueError("all inputs must be positive")
    return q * Z * KE2_MEV_FM / (2 * K_MeV * R / FM)


def theta_max_electron(mass_ratio=ALPHA["mass_ratio_e"]):
    """Largest turn of a heavy alpha off a light electron at rest: arcsin(m_e/m_alpha)."""
    return float(np.arcsin(1 / mass_ratio))


def collisions_in_foil(t=FOIL_T, n=GOLD["n"]):
    return t * n ** (1 / 3)


def theta_rms_multiple(theta_1, N):
    """2-D random walk of N kicks of size theta_1 with random directions."""
    return np.sqrt(N) * theta_1


def log10_prob_beyond_multiple(T, theta_rms):
    """log10 P(theta > T) = -(T / theta_rms)^2 / ln 10 (too small for a float)."""
    return -(T / theta_rms) ** 2 / np.log(10)


def rutherford_prob_beyond(T, Z=GOLD["Z"], q=ALPHA["q"], K_MeV=ALPHA["K_MeV"], n=GOLD["n"], t=FOIL_T):
    d = q * Z * KE2_MEV_FM / K_MeV * FM
    b = 0.5 * d / np.tan(T / 2)
    return np.pi * b ** 2 * n * t


def mc_random_walk_rms(theta_1, N, trials=20000, seed=0):
    """Monte Carlo check of sqrt(N) theta_1 for random-direction kicks."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, size=(trials, N))
    tx = theta_1 * np.cos(phi).sum(axis=1)
    ty = theta_1 * np.sin(phi).sum(axis=1)
    return float(np.sqrt(np.mean(tx ** 2 + ty ** 2)))


TIMELINE = [
    (1803, "Dalton", "atomic theory of chemistry: matter is made of atoms"),
    (1897, "J. J. Thomson", "discovers the electron (cathode rays, e/m)"),
    (1904, "J. J. Thomson", "plum-pudding model: electrons in a sphere of positive charge"),
    (1909, "Geiger & Marsden", "a few alphas bounce back from gold foil"),
    (1911, "Rutherford", "nuclear model: tiny dense positive nucleus"),
    (1913, "Bohr", "quantized orbits around Rutherford's nucleus"),
]
