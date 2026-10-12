"""Label-free particle classification from SEALS angular spectra (synthetic).

"Label-free" = no fluorescent tags: the particle is identified from how it
scatters light alone. (The classifier itself is still trained with known
classes.) This module builds a synthetic dataset with the tested Mie model and
the published SEALS wavelength-to-angle map, and the two feature sets the SEALS
paper contrasts:

    conventional cytometer : 2 numbers, forward scatter + side scatter
    SEALS                  : the whole angular spectrum (one value per resolvable angle)

Gini impurity (the CART split criterion) G = 1 - sum_k p_k^2 is derived here
symbolically: its maximum over the probability simplex is at p_k = 1/K, G = 1 - 1/K.

Everything here is SYNTHETIC: Mie theory + stated noise model. No measured data.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from scarlett import physics as ph
from scarlett import seals_hardware as sh

DEFAULT_CLASSES = {"7.32 um bead": 7.32e-6, "8.5 um bead": 8.5e-6, "9.94 um bead": 9.94e-6}


def seals_angles(n_angles=42, offset_deg=20.0):
    """Scattering angles (rad) seen through the SEALS map, detector offset_deg off axis."""
    if n_angles < 2:
        raise ValueError("need at least 2 angles")
    lam, _, th = sh.mapping(np.linspace(sh.LAM_MIN, sh.LAM_MAX, n_angles))
    return np.abs(np.radians(np.degrees(th) + offset_deg))


def make_dataset(classes=None, n_per_class=150, size_cv=0.02, n_mean=1.39, n_sigma=0.01,
                 pixel_noise=0.05, gain_jitter=0.3, n_angles=42, offset_deg=20.0, seed=0):
    """X = log10 intensity at each SEALS angle, y = class index, plus metadata.

    Noise model (all multiplicative, log-normal):
      size_cv      particle-to-particle diameter spread within a class
      n_sigma      refractive-index spread
      pixel_noise  independent per-angle noise (detector / shot noise proxy)
      gain_jitter  one random brightness factor per particle (laser power,
                   position in the focus) -- hurts absolute-intensity features
    """
    classes = DEFAULT_CLASSES if classes is None else classes
    for v in (n_per_class,):
        if v < 1:
            raise ValueError("n_per_class must be >= 1")
    if min(size_cv, n_sigma, pixel_noise, gain_jitter) < 0:
        raise ValueError("noise levels must be non-negative")
    rng = np.random.default_rng(seed)
    ang = seals_angles(n_angles, offset_deg)
    X, y = [], []
    for c, d0 in enumerate(classes.values()):
        for _ in range(n_per_class):
            d = d0 * (1 + size_cv * rng.standard_normal())
            n = n_mean + n_sigma * rng.standard_normal()
            I = ph.mie_scattering(n, 1.0, d, sh.LAM0, ang, 0.1).i_unpolarized
            I = I * np.exp(pixel_noise * rng.standard_normal(I.size)) * np.exp(gain_jitter * rng.standard_normal())
            X.append(np.log10(I))
            y.append(c)
    return np.array(X), np.array(y), ang, list(classes)


def conventional_features(X, angles, fsc_deg=None, ssc_deg=40.0):
    """Two-angle 'flow cytometer' features: the most forward angle and a side angle."""
    deg = np.degrees(angles)
    i_f = int(np.argmin(deg)) if fsc_deg is None else int(np.argmin(abs(deg - fsc_deg)))
    i_s = int(np.argmin(abs(deg - ssc_deg)))
    return X[:, [i_f, i_s]], (float(deg[i_f]), float(deg[i_s]))


def split(n, train_frac=0.6, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(train_frac * n)
    return idx[:k], idx[k:]


# ── Gini, symbolically ───────────────────────────────────────────────

def gini_symbolic(K=3):
    """Maximise G = 1 - sum p_i^2 subject to sum p_i = 1 (Lagrange multipliers)."""
    p = sp.symbols(f"p1:{K + 1}", nonnegative=True)
    mu = sp.Symbol("mu")
    G = 1 - sum(pi ** 2 for pi in p)
    L = G - mu * (sum(p) - 1)
    sol = sp.solve([sp.diff(L, pi) for pi in p] + [sum(p) - 1], list(p) + [mu], dict=True)[0]
    return G, sol, sp.simplify(G.subs(sol))


def gini_binary_vs_entropy():
    """G(p) = 2p(1-p) and H(p) = -p log2 p - (1-p) log2(1-p): both peak at p = 1/2."""
    p = sp.Symbol("p", positive=True)
    G = 1 - p ** 2 - (1 - p) ** 2
    H = -p * sp.log(p, 2) - (1 - p) * sp.log(1 - p, 2)
    return sp.factor(G), H, p


def export_csv(path, X, y):
    """Write X | y as CSV for MATLAB (fitctree)."""
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", fmt="%.8g")
    return path
