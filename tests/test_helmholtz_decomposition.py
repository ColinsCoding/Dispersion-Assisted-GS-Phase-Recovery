"""Test dgs/helmholtz_decomposition.py: the Fourier-space
longitudinal/transverse split of a generic (genuinely non-irrotational,
non-solenoidal) synthetic field, exact to near machine precision for a
band-limited periodic field, cross-checked by an independent torch.fft
implementation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.helmholtz_decomposition import (
    synthetic_test_field, spectral_divergence, spectral_curl,
    helmholtz_decompose, verify_decomposition,
)

# 1. synthetic_test_field: well-formed, and genuinely has both nonzero
#    curl and nonzero divergence (not an accidental special case)
F = synthetic_test_field(N=16, n_modes=4, seed=1)
assert F.shape == (16, 16, 16, 3)
div_F = spectral_divergence(F)
curl_F = spectral_curl(F)
assert np.max(np.abs(div_F)) > 1.0, "test field should have real, non-trivial divergence"
assert np.max(np.abs(curl_F)) > 1.0, "test field should have real, non-trivial curl"

for bad_N, bad_modes in [(2, 3), (16, 0)]:
    try:
        synthetic_test_field(N=bad_N, n_modes=bad_modes)
        raise AssertionError(f"expected ValueError for N={bad_N}, n_modes={bad_modes}")
    except ValueError:
        pass

# 2. A pure constant (DC-only) field must have exactly zero divergence
#    and curl everywhere -- the simplest possible sanity check on the
#    spectral derivative operators themselves
const_field = np.ones((8, 8, 8, 3)) * np.array([1.0, 2.0, 3.0])
assert np.max(np.abs(spectral_divergence(const_field))) < 1e-10
assert np.max(np.abs(spectral_curl(const_field))) < 1e-10

# 3. A single-mode field with a KNOWN closed-form divergence/curl:
#    F = (cos(kx), 0, 0) on a periodic grid -> div(F) = -k*sin(kx),
#    curl(F) = (0, 0, 0) (F depends only on x, has no y/z component that
#    varies with x in a way that produces curl... check: curl_z = dFy/dx
#    - dFx/dy = 0 - 0 = 0, curl_y = dFx/dz - dFz/dx = 0, all zero)
N, L = 32, 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
k = 2
F_known = np.zeros((N, N, N, 3))
F_known[..., 0] = np.cos(k * X)
div_known = spectral_divergence(F_known, L)
expected_div = -k * np.sin(k * X)
assert np.max(np.abs(div_known - expected_div)) < 1e-9
assert np.max(np.abs(spectral_curl(F_known, L))) < 1e-9

# 4. helmholtz_decompose: reconstruction, and each part's defining property
parts = helmholtz_decompose(F)
F_irrot, F_sol = parts["F_irrotational"], parts["F_solenoidal"]
assert np.max(np.abs((F_irrot + F_sol) - F)) < 1e-9, "parts must sum back to the original field"
assert np.max(np.abs(spectral_curl(F_irrot))) < 1e-9, "irrotational part must have ~0 curl"
assert np.max(np.abs(spectral_divergence(F_sol))) < 1e-9, "solenoidal part must have ~0 divergence"

# each part individually should be NONTRIVIAL (not the whole answer being
# dumped into one side) -- both parts carry real signal
assert np.max(np.abs(F_irrot)) > 0.01
assert np.max(np.abs(F_sol)) > 0.01

# 5. verify_decomposition: the full pipeline, across a couple of seeds/sizes
for seed in (0, 1, 2):
    check = verify_decomposition(N=20, n_modes=5, seed=seed)
    assert check["reconstruction_error"] < 1e-9, check
    assert check["max_abs_curl_of_irrotational_part"] < 1e-9, check
    assert check["max_abs_div_of_solenoidal_part"] < 1e-9, check
    assert check["original_max_abs_divergence"] > 0.1, "expected a genuinely non-trivial test field"
    assert check["original_max_abs_curl"] > 0.1, "expected a genuinely non-trivial test field"

print("dgs.helmholtz_decomposition: numpy checks passed")

# 6. torch (py 3.12 only): independent cross-check
try:
    import torch  # noqa: F401
    from dgs.helmholtz_decomposition import torch_verify_decomposition
    tcheck = torch_verify_decomposition(N=20, n_modes=5, seed=0)
    assert tcheck["max_abs_diff_numpy_vs_torch_irrotational"] < 1e-9
    assert tcheck["max_abs_diff_numpy_vs_torch_solenoidal"] < 1e-9
    assert tcheck["torch_max_abs_curl_of_irrotational_part"] < 1e-9
    assert tcheck["torch_max_abs_div_of_solenoidal_part"] < 1e-9
    print("dgs.helmholtz_decomposition: torch cross-check passed")
except ImportError:
    print("dgs.helmholtz_decomposition: torch not available, skipped cross-check")

print("all dgs.helmholtz_decomposition tests passed")
