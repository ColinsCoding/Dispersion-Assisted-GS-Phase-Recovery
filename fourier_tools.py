"""Fourier transform at engineering depth -- and its bridge to vector calculus.

The DFT is a *linear operator*: a unitary change of basis into the frequency
domain. So you can write it as a matrix (W), compute it fast with the radix-2
FFT (Cooley-Tukey, O(N log N) instead of O(N^2)), and use its two killer
theorems:

  * derivative theorem:  F{ d f/dt } = i*omega * F{f}      (calculus -> algebra)
  * convolution theorem: F{ a (*) b } = F{a} . F{b}        (convolution -> product)

The derivative theorem is the whole reason this repo's dispersion is a spectral
multiplier H(f)=exp(i pi D f^2): d/dt in time is i*omega in frequency, so a
differential operator becomes pointwise multiplication. NumPy only (everything
verified against numpy.fft). Education.
"""

import numpy as np


# ── the DFT as a matrix (a linear operator / change of basis) ───────
def dft_matrix(N):
    """The N x N DFT matrix W, W[j,k] = exp(-2 pi i j k / N).

    X = W @ x is the DFT. W/sqrt(N) is unitary -- the transform is just a
    rotation into the Fourier basis.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.exp(-2j * np.pi * j * k / N)


def dft(x):
    """DFT via the matrix (matches numpy.fft.fft)."""
    x = np.asarray(x, dtype=complex)
    return dft_matrix(len(x)) @ x


def idft(X):
    """Inverse DFT: (1/N) W^* @ X."""
    X = np.asarray(X, dtype=complex)
    N = len(X)
    return np.conj(dft_matrix(N)) @ X / N


# ── the fast algorithm: radix-2 Cooley-Tukey ────────────────────────
def fft_radix2(x):
    """Recursive radix-2 FFT (N a power of 2). Splits even/odd, O(N log N).

    Identical result to dft(x)/numpy.fft.fft, but the divide-and-conquer is why
    the FFT changed the world -- N=2^20 goes from ~10^12 to ~2x10^7 operations.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N & (N - 1):
        raise ValueError("length must be a power of 2")
    if N <= 1:
        return x
    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    tw = np.exp(-2j * np.pi * np.arange(N // 2) / N)      # twiddle factors
    return np.concatenate([even + tw * odd, even - tw * odd])


# ── the two theorems ────────────────────────────────────────────────
def spectral_derivative(x, dt, order=1):
    """d^order/dt^order of a periodic, band-limited signal via the FFT.

    Uses F{d/dt} = i*omega: multiply the spectrum by (i*omega)^order and invert.
    For a band-limited periodic signal this is *exact* (to machine precision),
    unlike finite differences. This is the derivative theorem -- the link from
    Fourier to calculus.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)
    return np.fft.ifft((1j * omega) ** order * np.fft.fft(x)).real


def circular_convolution(a, b):
    """Circular convolution via the convolution theorem: ifft(fft(a) * fft(b))."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must be the same length")
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real


if __name__ == "__main__":
    N = 8
    x = np.random.default_rng(0).standard_normal(N)
    print("matrix DFT vs numpy:", np.allclose(dft(x), np.fft.fft(x)))
    print("radix-2 FFT vs numpy:", np.allclose(fft_radix2(x), np.fft.fft(x)))
    t = np.linspace(0, 1, 256, endpoint=False)
    f = np.sin(2 * np.pi * 5 * t)
    d = spectral_derivative(f, t[1] - t[0])
    print("spectral d/dt error vs analytic:",
          np.max(np.abs(d - 2 * np.pi * 5 * np.cos(2 * np.pi * 5 * t))))
