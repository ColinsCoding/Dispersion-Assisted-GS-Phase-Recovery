"""Test dgs.imaging_inverse: the forward blur model must be physically
sane (energy-preserving, reproduces the kernel from an impulse), and the
Tikhonov inverse must show the actual bias/variance tradeoff its docstring
claims -- naive inversion (lambda~0) unstable under noise, oversmoothing at
large lambda, and a genuine interior minimum of reconstruction error in
between. That interior minimum is the substantive claim; a test that only
checks "runs without crashing" would miss it entirely."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.imaging_inverse import (
    gaussian_blur_kernel, apply_blur, add_gaussian_noise,
    tikhonov_deconvolve, reconstruction_error, synthetic_object,
)

# 1. kernel is normalized, symmetric, peaked at center
kernel = gaussian_blur_kernel(9, sigma=1.5)
assert abs(kernel.sum() - 1.0) < 1e-12
assert np.allclose(kernel, kernel.T)
assert kernel[4, 4] == kernel.max()

# 2. rejects a kernel that has no center pixel
try:
    gaussian_blur_kernel(8, sigma=1.5)
    assert False, "should reject an even kernel size"
except ValueError:
    pass

# 3. blur preserves total energy (kernel sums to 1 -> DC component unchanged
# by a circular convolution)
x_true = synthetic_object(64)
y_clean = apply_blur(x_true, kernel)
assert abs(y_clean.sum() - x_true.sum()) < 1e-8

# 4. blurring is smoothing: max pixel value can only go down, never up
assert y_clean.max() <= x_true.max() + 1e-9

# 5. impulse response: blurring a single bright pixel reproduces the kernel
# itself (mod circular wraparound) -- the actual correctness check on the
# PSF->OTF machinery, not just "output has the right shape"
delta = np.zeros((31, 31))
delta[15, 15] = 1.0
response = apply_blur(delta, kernel)
k = kernel.shape[0]
c = 15   # response is centered on the impulse's own position, no shift needed
patch = response[c - k // 2:c + k // 2 + 1, c - k // 2:c + k // 2 + 1]
assert np.allclose(patch, kernel, atol=1e-10)

# 6. noiseless case: Tikhonov with a tiny lambda recovers the true object
# almost exactly (no noise to amplify, so the near-naive inverse is fine).
# lambda must be well below min|H|^2 (~1.4e-14 for this kernel/size) or it
# starts suppressing the highest spatial frequencies itself -- checked here
# rather than assumed, since that's the whole point of the tradeoff below.
x_hat_clean = tikhonov_deconvolve(y_clean, kernel, lam=1e-16)
assert reconstruction_error(x_true, x_hat_clean) < 1e-8

# 7. noisy case: the classic bias/variance tradeoff must actually appear --
# naive inversion (lambda~0) amplifies noise, heavy regularization
# oversmooths, and there is a strictly better lambda in between
y_noisy = add_gaussian_noise(y_clean, sigma=0.05, seed=1)
lambdas = np.logspace(-6, 1, 25)
errors = np.array([reconstruction_error(x_true, tikhonov_deconvolve(y_noisy, kernel, lam))
                    for lam in lambdas])
best_idx = int(np.argmin(errors))

assert 0 < best_idx < len(lambdas) - 1, "expected an INTERIOR minimum, not an endpoint"
assert errors[best_idx] < errors[0], "regularized reconstruction should beat the naive inverse"
assert errors[best_idx] < errors[-1], "regularized reconstruction should beat heavy oversmoothing"

# 8. mismatched shapes and negative lambda are rejected, not silently broadcast
try:
    reconstruction_error(x_true, x_true[:-1])
    assert False, "should reject mismatched shapes"
except ValueError:
    pass

try:
    tikhonov_deconvolve(y_noisy, kernel, lam=-1.0)
    assert False, "should reject negative lambda"
except ValueError:
    pass

print(f"all dgs.imaging_inverse tests passed  "
      f"(best lambda={lambdas[best_idx]:.2e}, MSE={errors[best_idx]:.5f}, "
      f"naive={errors[0]:.5f}, oversmoothed={errors[-1]:.5f})")
