"""Test dgs.spectral_unmixing: a pure endmember unmixes to a one-hot vector, a known
mixture recovers its fractions (non-negative, summing to 1), the reconstruction E a
matches the pixel, classification picks the dominant material and survives noise, and
it runs over an image cube."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import spectral_unmixing as su

healthy = np.array([0.20, 0.45, 0.65, 0.55, 0.35, 0.25])
tumor = np.array([0.60, 0.50, 0.30, 0.25, 0.40, 0.55])
background = np.array([0.10, 0.12, 0.10, 0.08, 0.09, 0.11])
E = np.column_stack([healthy, tumor, background])
labels = ["healthy", "tumor", "background"]

# 1. a PURE endmember unmixes to a one-hot abundance vector
assert np.allclose(su.unmix(healthy, E), [1, 0, 0], atol=1e-6)
assert np.allclose(su.unmix(tumor, E), [0, 1, 0], atol=1e-6)

# 2. a KNOWN mixture recovers its fractions (non-negative, summing to 1)
s = 0.4 * healthy + 0.6 * tumor
a = su.unmix_fractional(s, E)
assert np.allclose(a, [0.4, 0.6, 0.0], atol=1e-3)
assert np.all(a >= 0) and np.isclose(a.sum(), 1.0, atol=1e-3)

# 3. the reconstruction E a matches the measured pixel
assert np.allclose(su.reconstruct(a, E), s, atol=1e-3)
assert su.reconstruction_error(s, a, E) < 1e-6

# 4. classification picks the dominant material
assert su.classify(healthy, E, labels) == "healthy"
assert su.classify(s, E, labels) == "tumor"                 # 60% tumor dominates
assert su.classify(background, E, labels) == "background"
assert su.classify(s, E) == 1                               # index form (tumor = col 1)

# 5. robust to measurement noise
rng = np.random.default_rng(0)
noisy = s + rng.normal(0, 0.03, len(s))
an = su.unmix_fractional(noisy, E)
assert np.allclose(an, [0.4, 0.6, 0.0], atol=0.05)          # fractions ~ recovered
assert su.classify(noisy, E, labels) == "tumor"            # class survives

# 6. unmix a whole image cube (H, W, bands) -> (H, W, endmembers)
cube = np.array([[0.9*healthy + 0.1*tumor, 0.2*healthy + 0.8*tumor],
                 [background,               0.7*tumor + 0.3*healthy]])
amap = su.unmix_cube(cube, E)
assert amap.shape == (2, 2, 3)
assert np.all(amap >= 0)
expected = [["healthy", "tumor"], ["background", "tumor"]]
for i in range(2):
    for j in range(2):
        assert labels[int(np.argmax(amap[i, j]))] == expected[i][j]

# 7. kwarg bounds
for bad in (lambda: su.unmix(healthy[:3], E),               # length mismatch
            lambda: su.unmix_fractional(np.ones(6), np.ones(3)),   # E not 2-D matching
            lambda: su.unmix_cube(np.ones((6,)), E)):        # not a cube
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_spectral_unmixing: all checks passed")
