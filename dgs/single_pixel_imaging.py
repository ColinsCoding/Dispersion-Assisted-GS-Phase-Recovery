"""Single-pixel imaging: a camera with ONE detector and many light patterns.

A single-pixel camera has no image sensor. It shines a sequence of structured light
PATTERNS onto the scene (with a DMD micromirror array) and, for each pattern, reads a
single number off ONE photodiode -- the total light that gets through:
        y_k = < pattern_k , scene >.
Collect enough patterns and you invert those scalar readings back into a full image.
Why bother? At wavelengths where megapixel sensors are expensive or impossible
(terahertz, mid-IR, single-photon), one good detector plus a pattern generator is far
cheaper -- and it is the same spirit as Jalali time-stretch imaging, which reads a
scene through one fast photodiode.

Two regimes, both here:

  FULL BASIS (N patterns for N pixels). Use the Hadamard matrix (all +/-1 patterns,
  mutually orthogonal). N measurements exactly determine N pixels: because H H = N I,
  the reconstruction is just x = (1/N) H y -- no approximation, no noise amplification.

  COMPRESSED SENSING (M < N patterns). Real images are compressible -- sparse in some
  basis -- so you can recover them from FEWER measurements than pixels with random
  patterns, by finding the sparsest x that fits. Orthogonal Matching Pursuit (OMP), a
  greedy sparse solver implemented here, recovers a k-sparse scene exactly from about
  O(k log N) random measurements -- fewer detectors reads than pixels.

Verified: Hadamard reconstruction is exact to machine precision, and OMP recovers a
sparse scene's support and values from a fraction of the measurements. NumPy only;
py-3.13.
"""

import numpy as np


def hadamard_matrix(n):
    """The n x n Sylvester-Hadamard matrix of +/-1 patterns (n a power of 2).
    Symmetric and orthogonal: H H = n I, so its inverse is (1/n) H."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2")
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def single_pixel_measure(scene, patterns, noise_std=0.0, rng_seed=0):
    """Simulate the single detector: for each pattern (a row), return the one
    number y_k = <pattern_k, scene> -- the total transmitted light. Optional
    additive read noise. scene is the flattened image (length N)."""
    scene = np.asarray(scene, float).ravel()
    patterns = np.asarray(patterns, float)
    if patterns.shape[1] != len(scene):
        raise ValueError("each pattern must have one entry per scene pixel")
    y = patterns @ scene
    if noise_std > 0:
        y = y + np.random.default_rng(rng_seed).normal(0, noise_std, len(y))
    return y


def reconstruct_hadamard(y, n):
    """Exact reconstruction from a FULL set of n Hadamard measurements:
    x = (1/n) H y (since H H = n I). No approximation."""
    y = np.asarray(y, float)
    if len(y) != n:
        raise ValueError("need exactly n measurements for the n-pixel Hadamard basis")
    return (hadamard_matrix(n) @ y) / n


def random_patterns(m, n, rng_seed=0):
    """m random +/-1 patterns over n pixels -- the compressed-sensing sensing
    matrix. Fewer patterns (m < n) than pixels is the whole point."""
    if m < 1 or n < 1:
        raise ValueError("m and n must be positive")
    return np.random.default_rng(rng_seed).choice([-1.0, 1.0], size=(m, n))


def reconstruct_least_squares(y, patterns):
    """Minimum-norm least-squares reconstruction from any patterns. Exact when
    the patterns span the pixel space (m >= n, full rank); an approximation when
    m < n (underdetermined -- use OMP if the scene is sparse)."""
    x, *_ = np.linalg.lstsq(np.asarray(patterns, float), np.asarray(y, float),
                            rcond=None)
    return x


def omp(patterns, y, sparsity):
    """Orthogonal Matching Pursuit: recover a `sparsity`-sparse scene from m < n
    measurements y = patterns @ scene. Greedily picks the pixel most correlated
    with the residual, re-fits the chosen support by least squares, and repeats.
    Recovers a k-sparse image exactly from ~O(k log n) random patterns."""
    A = np.asarray(patterns, float)
    y = np.asarray(y, float)
    m, n = A.shape
    if sparsity < 1 or sparsity > n:
        raise ValueError("sparsity must be in 1..n")
    residual = y.copy()
    support = []
    x = np.zeros(n)
    for _ in range(sparsity):
        corr = A.T @ residual
        corr[support] = 0.0                       # do not re-pick a chosen column
        j = int(np.argmax(np.abs(corr)))
        support.append(j)
        coefs, *_ = np.linalg.lstsq(A[:, support], y, rcond=None)
        residual = y - A[:, support] @ coefs
        if np.linalg.norm(residual) < 1e-12:
            break
    x[support] = coefs
    return x


def reconstruction_rmse(true_scene, recovered):
    """Root-mean-square reconstruction error, the imaging quality metric."""
    a = np.asarray(true_scene, float).ravel()
    b = np.asarray(recovered, float).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 64                                          # an 8x8 image, flattened

    # FULL BASIS: Hadamard patterns give exact reconstruction
    scene = rng.random(N)
    H = hadamard_matrix(N)
    y = single_pixel_measure(scene, H)
    recon = reconstruct_hadamard(y, N)
    print("FULL BASIS (Hadamard):")
    print(f"  {N} patterns for {N} pixels -> RMSE = {reconstruction_rmse(scene, recon):.2e} "
          f"(exact)")

    # COMPRESSED SENSING: a sparse scene from FEWER measurements via OMP
    k = 5
    sparse_scene = np.zeros(N)
    idx = rng.choice(N, k, replace=False)
    sparse_scene[idx] = rng.uniform(1, 5, k)
    print(f"\nCOMPRESSED SENSING (a {k}-sparse {N}-pixel scene):")
    for M in (8, 20, 30):
        P = random_patterns(M, N, rng_seed=1)
        ycs = single_pixel_measure(sparse_scene, P)
        xhat = omp(P, ycs, sparsity=k)
        print(f"  M={M:2d} patterns ({M/N:.0%} of pixels): "
              f"RMSE = {reconstruction_rmse(sparse_scene, xhat):.2e}")
    print("  -> a handful of measurements, not 64, recover the sparse image.")
