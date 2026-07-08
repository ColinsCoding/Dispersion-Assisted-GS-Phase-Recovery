"""Test dgs.single_pixel_imaging: the Hadamard patterns (orthogonal, H H = nI),
exact full-basis reconstruction from single-detector readings, and compressed
sensing -- OMP recovering a k-sparse scene (support and values) from far fewer
measurements than pixels, while too few measurements fail."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import single_pixel_imaging as spi

rng = np.random.default_rng(0)
N = 64

# 1. Hadamard matrix: +/-1, symmetric, and H H = N I (orthogonal patterns)
H = spi.hadamard_matrix(N)
assert H.shape == (N, N)
assert set(np.unique(H)).issubset({-1.0, 1.0})
assert np.allclose(H, H.T)
assert np.allclose(H @ H, N * np.eye(N))
for bad in (3, 0, 6):
    try:
        spi.hadamard_matrix(bad); assert False
    except ValueError:
        pass

# 2. the single detector returns one scalar per pattern
scene = rng.random(N)
y = spi.single_pixel_measure(scene, H)
assert y.shape == (N,)
assert np.isclose(y[0], np.dot(H[0], scene))          # each y_k = <pattern_k, scene>

# 3. FULL BASIS: Hadamard reconstruction is exact
recon = spi.reconstruct_hadamard(y, N)
assert spi.reconstruction_rmse(scene, recon) < 1e-12
try:
    spi.reconstruct_hadamard(y[:10], N); assert False   # wrong number of measurements
except ValueError:
    pass

# 4. random patterns and full-rank least squares (m >= n) also recover exactly
P_full = spi.random_patterns(N, N, rng_seed=2)
assert P_full.shape == (N, N) and set(np.unique(P_full)).issubset({-1.0, 1.0})
recon_ls = spi.reconstruct_least_squares(spi.single_pixel_measure(scene, P_full), P_full)
assert spi.reconstruction_rmse(scene, recon_ls) < 1e-9

# 5. COMPRESSED SENSING: a k-sparse scene recovered from M < N measurements by OMP
k = 5
sparse = np.zeros(N)
support_true = np.sort(rng.choice(N, k, replace=False))
sparse[support_true] = rng.uniform(1, 5, k)

# enough measurements (~O(k log N)) -> exact recovery of support AND values
P20 = spi.random_patterns(20, N, rng_seed=1)
xhat = spi.omp(P20, spi.single_pixel_measure(sparse, P20), sparsity=k)
assert spi.reconstruction_rmse(sparse, xhat) < 1e-9
assert np.array_equal(np.sort(np.nonzero(xhat)[0]), support_true)   # right pixels

# more measurements still work
P30 = spi.random_patterns(30, N, rng_seed=1)
xhat30 = spi.omp(P30, spi.single_pixel_measure(sparse, P30), sparsity=k)
assert spi.reconstruction_rmse(sparse, xhat30) < 1e-9

# too few measurements -> recovery fails (much larger error)
P8 = spi.random_patterns(8, N, rng_seed=1)
xhat8 = spi.omp(P8, spi.single_pixel_measure(sparse, P8), sparsity=k)
assert spi.reconstruction_rmse(sparse, xhat8) > 1e-3
assert spi.reconstruction_rmse(sparse, xhat8) > 100 * spi.reconstruction_rmse(sparse, xhat)

# 6. rmse metric sanity
assert spi.reconstruction_rmse([1, 2, 3], [1, 2, 3]) == 0.0
assert np.isclose(spi.reconstruction_rmse([0, 0], [3, 4]), np.sqrt((9 + 16) / 2))

# 7. kwarg bounds
for bad in (lambda: spi.single_pixel_measure(scene, np.ones((3, 5))),   # wrong width
            lambda: spi.random_patterns(0, N),
            lambda: spi.omp(P20, y[:20], sparsity=0),
            lambda: spi.omp(P20, y[:20], sparsity=N + 1)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_single_pixel_imaging: all checks passed")
