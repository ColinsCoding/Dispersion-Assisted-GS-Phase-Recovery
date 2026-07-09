"""Spectral unmixing: what is each pixel made of, from its colors across many bands.

A multispectral / hyperspectral camera measures, at every pixel, a whole SPECTRUM -- the
brightness in many wavelength bands, not just R/G/B. Most pixels are a MIXTURE of a few pure
materials (say healthy tissue, tumor, background), each with its own known spectral signature
(an ENDMEMBER). To leading order the mixing is LINEAR: a pixel's spectrum is a weighted sum
        s = E a,     s in R^bands,  E = [e_1 | e_2 | ... ] the endmember matrix,
where the ABUNDANCES a_i are how much of each material is present. Recovering a from s is
spectral unmixing -- and the physical constraints make it a CONSTRAINED least-squares problem:
abundances are non-negative (a_i >= 0) and, for a full mixture, sum to one. Solve it and the
DOMINANT abundance classifies the pixel (tumor vs healthy), turning a stack of band images
into a labeled map.

This is the same over-determined A x = b least squares as dgs.gnss_positioning and
dgs.statics_linalg, now with a >= 0: non-negative least squares (scipy.optimize.nnls), plus a
softly-enforced sum-to-one for fractional abundances. Verified: a pure endmember unmixes to a
one-hot vector, a known mixture recovers its fractions (summing to 1), the reconstruction
E a matches the pixel, classification survives noise, and it runs over a whole image cube.
NumPy + SciPy; py-3.13.
"""

import numpy as np
from scipy.optimize import nnls


def unmix(spectrum, endmembers):
    """Non-negative least squares: the abundances a >= 0 minimizing ||E a - s||.
    endmembers E is (n_bands, n_endmembers), each column a pure spectrum."""
    E = np.asarray(endmembers, float)
    s = np.asarray(spectrum, float)
    if E.ndim != 2 or E.shape[0] != len(s):
        raise ValueError("endmembers must be (n_bands, k) matching the spectrum length")
    a, _ = nnls(E, s)
    return a


def unmix_fractional(spectrum, endmembers, sum_weight=50.0):
    """Fully-constrained unmixing: abundances non-negative AND summing to 1, by
    appending a heavily weighted sum-to-one row to the least-squares system (the
    standard FCLS trick). Returns fractional abundances."""
    E = np.asarray(endmembers, float)
    s = np.asarray(spectrum, float)
    if E.ndim != 2 or E.shape[0] != len(s):
        raise ValueError("endmembers must be (n_bands, k) matching the spectrum length")
    E_aug = np.vstack([E, sum_weight * np.ones(E.shape[1])])
    s_aug = np.append(s, sum_weight)
    a, _ = nnls(E_aug, s_aug)
    return a


def reconstruct(abundances, endmembers):
    """The spectrum implied by a set of abundances: s_hat = E a."""
    return np.asarray(endmembers, float) @ np.asarray(abundances, float)


def reconstruction_error(spectrum, abundances, endmembers):
    """RMS residual between the measured spectrum and its unmixed reconstruction."""
    r = np.asarray(spectrum, float) - reconstruct(abundances, endmembers)
    return float(np.sqrt(np.mean(r ** 2)))


def classify(spectrum, endmembers, labels=None, fractional=True):
    """Label a pixel by its DOMINANT material: the endmember with the largest
    abundance. Returns the label (or the index if labels is None)."""
    a = unmix_fractional(spectrum, endmembers) if fractional else unmix(spectrum, endmembers)
    idx = int(np.argmax(a))
    return labels[idx] if labels is not None else idx


def unmix_cube(cube, endmembers, fractional=True):
    """Unmix an image cube of shape (H, W, n_bands): returns an abundance map of
    shape (H, W, n_endmembers). Per-pixel constrained least squares -- the map that
    turns a spectral stack into 'how much of each material is here'."""
    cube = np.asarray(cube, float)
    if cube.ndim != 3:
        raise ValueError("cube must be (H, W, n_bands)")
    H, W, B = cube.shape
    k = np.asarray(endmembers, float).shape[1]
    out = np.zeros((H, W, k))
    fn = unmix_fractional if fractional else unmix
    for i in range(H):
        for j in range(W):
            out[i, j] = fn(cube[i, j], endmembers)
    return out


if __name__ == "__main__":
    # three tissue endmembers over 6 spectral bands
    healthy = np.array([0.20, 0.45, 0.65, 0.55, 0.35, 0.25])
    tumor = np.array([0.60, 0.50, 0.30, 0.25, 0.40, 0.55])
    background = np.array([0.10, 0.12, 0.10, 0.08, 0.09, 0.11])
    E = np.column_stack([healthy, tumor, background])
    labels = ["healthy", "tumor", "background"]

    print("unmix a known 60% tumor / 40% healthy pixel:")
    s = 0.4 * healthy + 0.6 * tumor
    a = unmix_fractional(s, E)
    print(f"  abundances = {np.round(a, 3)} (sum {a.sum():.3f}), "
          f"classified as '{classify(s, E, labels)}'")
    print(f"  reconstruction RMS error = {reconstruction_error(s, a, E):.2e}")

    print("\nwith 3% measurement noise:")
    rng = np.random.default_rng(0)
    sn = s + rng.normal(0, 0.03, len(s))
    an = unmix_fractional(sn, E)
    print(f"  abundances = {np.round(an, 3)}, still '{classify(sn, E, labels)}'")

    print("\nclassify a tiny 2x2 image cube:")
    cube = np.array([[0.9*healthy+0.1*tumor, 0.2*healthy+0.8*tumor],
                     [background, 0.5*healthy+0.5*tumor]])
    amap = unmix_cube(cube, E)
    for i in range(2):
        for j in range(2):
            print(f"  pixel ({i},{j}): {labels[int(np.argmax(amap[i,j]))]:10s} "
                  f"{np.round(amap[i,j],2)}")
