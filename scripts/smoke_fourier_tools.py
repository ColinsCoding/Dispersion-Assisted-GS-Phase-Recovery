"""Smoke-test the engineering-FT toolkit: DFT matrix, radix-2 FFT, the two theorems."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import fourier_tools as ft

rng = np.random.default_rng(0)

# 1. DFT matrix is sqrt(N)-unitary: W^H W = N I (a rotation into the Fourier basis)
W = ft.dft_matrix(16)
assert np.allclose(W.conj().T @ W, 16 * np.eye(16), atol=1e-10)

# 2. matrix DFT and inverse match numpy and round-trip
x = rng.standard_normal(13) + 1j * rng.standard_normal(13)
assert np.allclose(ft.dft(x), np.fft.fft(x), atol=1e-10)
assert np.allclose(ft.idft(ft.dft(x)), x, atol=1e-10)

# 3. radix-2 FFT equals numpy for power-of-2 lengths; rejects non-powers
for N in (2, 4, 8, 256):
    xx = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    assert np.allclose(ft.fft_radix2(xx), np.fft.fft(xx), atol=1e-9), N
try:
    ft.fft_radix2(np.zeros(6))
except ValueError:
    pass
else:
    raise AssertionError("non-power-of-2 should raise")

# 4. derivative theorem: spectral d/dt of a band-limited periodic signal is EXACT
t = np.linspace(0, 1, 512, endpoint=False)
dt = t[1] - t[0]
f = np.sin(2 * np.pi * 7 * t) + 0.5 * np.cos(2 * np.pi * 3 * t)
d1 = ft.spectral_derivative(f, dt, order=1)
exact1 = 2 * np.pi * 7 * np.cos(2 * np.pi * 7 * t) - 0.5 * 2 * np.pi * 3 * np.sin(2 * np.pi * 3 * t)
assert np.max(np.abs(d1 - exact1)) < 1e-8, np.max(np.abs(d1 - exact1))
# second derivative too
d2 = ft.spectral_derivative(f, dt, order=2)
exact2 = -(2 * np.pi * 7)**2 * np.sin(2 * np.pi * 7 * t) - 0.5 * (2 * np.pi * 3)**2 * np.cos(2 * np.pi * 3 * t)
assert np.max(np.abs(d2 - exact2)) < 1e-7

# 5. convolution theorem == direct circular convolution
a = rng.standard_normal(32)
b = rng.standard_normal(32)
direct = np.array([np.sum(a * np.roll(b[::-1], i + 1)) for i in range(32)])
assert np.allclose(ft.circular_convolution(a, b), direct, atol=1e-9)

# 6. tie to the repo: multiplying the spectrum by a quadratic phase IS dispersion
#    (the derivative theorem is why H(f)=exp(i pi D f^2) acts as a differential op)
assert np.allclose(ft.spectral_derivative(f, dt, 2),
                   ft.spectral_derivative(ft.spectral_derivative(f, dt, 1), dt, 1), atol=1e-6)

print(f"SMOKE PASS  (DFT=FFT=numpy; spectral d/dt exact to "
      f"{np.max(np.abs(d1-exact1)):.1e}; convolution theorem holds)")
