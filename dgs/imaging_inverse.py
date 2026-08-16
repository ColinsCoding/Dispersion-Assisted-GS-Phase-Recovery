"""The 2D computational-imaging forward model and its Tikhonov inverse.

    y = H x + n

x is the true object, H a (here: Gaussian) blur operator, n additive noise,
y the measurement -- the same ill-posed-inverse-problem shape as this repo's
own H(f)=exp(i*pi*D*f^2) dispersion operator (dgs.dispersion_gs_prototype)
and the 1D Wiener-filter demo in dgs.inverse_calculus.deconvolution_demo,
just in two spatial dimensions with a REAL (not complex) point-spread
function. Recovering x from a noisy y by naive inversion (divide by H in
Fourier space) blows up wherever H is small, because noise gets divided by
the same tiny number as signal -- classic ill-posedness. Tikhonov
regularization

    x_hat = argmin_x  ||H x - y||^2 + lambda ||x||^2

has the closed-form Fourier-domain solution

    X_hat(k) = H*(k) Y(k) / (|H(k)|^2 + lambda),

which is exactly a Wiener filter with lambda playing the role of the
noise-to-signal power ratio: lambda -> 0 recovers the (unstable) naive
inverse, lambda -> infinity oversmooths toward zero, and some interior
lambda minimizes the true reconstruction error -- the bias/variance
tradeoff, verified numerically in tests/test_imaging_inverse.py rather than
assumed. FFT-based (circular) convolution throughout; NumPy only, py-3.13.
"""

import numpy as np


def gaussian_blur_kernel(size, sigma):
    """A normalized (sums to 1) 2D Gaussian point-spread function, size x size,
    peak at the center -- the blur operator H as a small spatial kernel."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer (so the kernel has a center pixel)")
    ax = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _psf_to_otf(kernel, shape):
    """Embed a small spatial kernel into a zero array of the image's shape and
    roll it so the kernel's center lands at index (0,0), THEN FFT -- the
    standard PSF->OTF construction that makes np.fft.fft2(image)*OTF a
    correct (wraparound/circular) convolution instead of a shifted one."""
    kh, kw = kernel.shape
    psf = np.zeros(shape)
    psf[:kh, :kw] = kernel
    psf = np.roll(psf, -(kh // 2), axis=0)
    psf = np.roll(psf, -(kw // 2), axis=1)
    return np.fft.fft2(psf)


def apply_blur(image, kernel):
    """The forward model H x: circular convolution of image with kernel via
    the FFT (fast, and exact up to wraparound at the edges)."""
    H = _psf_to_otf(kernel, image.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(image) * H))


def add_gaussian_noise(image, sigma, seed=0):
    """The +n in y = Hx + n: i.i.d. Gaussian sensor noise of the given
    standard deviation, reproducibly seeded."""
    rng = np.random.default_rng(seed)
    return image + sigma * rng.standard_normal(image.shape)


def tikhonov_deconvolve(blurred, kernel, lam):
    """Closed-form Tikhonov/Wiener inverse: X_hat = H* Y / (|H|^2 + lambda).
    lam=0 is the naive (unregularized) inverse -- unstable wherever H is
    near zero; lam>0 trades bias for noise suppression."""
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    H = _psf_to_otf(kernel, blurred.shape)
    Y = np.fft.fft2(blurred)
    X_hat = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    return np.real(np.fft.ifft2(X_hat))


def reconstruction_error(x_true, x_hat):
    """Mean-squared error between the true object and a reconstruction --
    the metric a lambda sweep is trying to minimize."""
    x_true = np.asarray(x_true, float)
    x_hat = np.asarray(x_hat, float)
    if x_true.shape != x_hat.shape:
        raise ValueError("x_true and x_hat must have the same shape")
    return float(np.mean((x_true - x_hat) ** 2))


def synthetic_object(n=64):
    """A simple synthetic 'true object': a bright square and a bright disk on
    a dark background, sharp edges -- easy to see blurring and deconvolution
    act on visually, and with enough high-spatial-frequency content that
    blur is not a no-op."""
    obj = np.zeros((n, n))
    a, b = n // 4, 3 * n // 4
    obj[a:b, a // 2:a // 2 + (b - a) // 2] = 1.0     # square, left half
    yy, xx = np.mgrid[0:n, 0:n]
    cy, cx, r = n // 2, 3 * n // 4, n // 8
    obj[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1.0   # disk, right side
    return obj


if __name__ == "__main__":
    x_true = synthetic_object(64)
    kernel = gaussian_blur_kernel(9, sigma=1.5)

    y_clean = apply_blur(x_true, kernel)
    y_noisy = add_gaussian_noise(y_clean, sigma=0.02, seed=0)

    print("=== y = Hx + n forward model ===")
    print(f"  true object: sum={x_true.sum():.1f}, blurred: sum={y_clean.sum():.1f} (energy-preserving)")
    print(f"  noisy measurement std added: 0.02")

    print("\n=== Tikhonov lambda sweep ===")
    lambdas = np.logspace(-6, 0, 13)
    errors = [reconstruction_error(x_true, tikhonov_deconvolve(y_noisy, kernel, lam)) for lam in lambdas]
    best = lambdas[int(np.argmin(errors))]
    for lam, err in zip(lambdas, errors):
        marker = "  <-- best" if lam == best else ""
        print(f"  lambda={lam:9.2e}  MSE={err:.5f}{marker}")
    print(f"\n  naive inverse (lambda~0) MSE = {errors[0]:.5f}  (unstable, noise-amplified)")
    print(f"  over-smoothed (lambda=1)  MSE = {errors[-1]:.5f}  (blurry, biased)")
    print(f"  best lambda = {best:.2e}, MSE = {min(errors):.5f}  (bias/variance tradeoff optimum)")
