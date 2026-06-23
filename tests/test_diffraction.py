"""Test single-slit diffraction: sinc^2 pattern, minima, FFT identity, resolving power."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import diffraction as df

a, lam = 50e-6, 600e-9

# 1. central maximum: I(0) = I0
assert np.isclose(df.single_slit_intensity(0.0, a, lam, I0=2.5), 2.5)

# 2. minima at sin(theta) = m lambda/a -> intensity ~ 0 there
mins = df.single_slit_minima(a, lam, m_max=3)
assert len(mins) == 3
for m, th in zip((1, 2, 3), mins):
    assert np.isclose(np.sin(th), m * lam / a)
    assert df.single_slit_intensity(th, a, lam) < 1e-12        # dark fringe

# 3. central-lobe half-width = arcsin(lambda/a) = the first minimum
assert np.isclose(df.central_lobe_halfwidth(a, lam), np.arcsin(lam / a))
assert np.isclose(df.central_lobe_halfwidth(a, lam), mins[0])

# 4. narrower slit -> WIDER pattern (the diffraction-uncertainty reciprocity)
assert df.central_lobe_halfwidth(a / 2, lam) > df.central_lobe_halfwidth(a, lam)

# 5. intensity is symmetric and the side lobes decay (1st side lobe ~ 4.7% of peak)
th = np.linspace(-np.radians(3), np.radians(3), 4001)
I = df.single_slit_intensity(th, a, lam)
assert np.allclose(I, I[::-1], atol=1e-9)                      # symmetric
# the first side lobe (between 1st and 2nd minima) peaks at ~0.045 of the center
side = I[(th > mins[0]) & (th < mins[1])]
assert 0.04 < side.max() < 0.05

# 6. Rayleigh resolving power: lambda/D (slit), 1.22 lambda/D (circular)
assert np.isclose(df.rayleigh_resolution(lam, 1e-2), lam / 1e-2)
assert np.isclose(df.rayleigh_resolution(lam, 1e-2, circular=True), 1.22 * lam / 1e-2)
assert df.rayleigh_resolution(lam, 2e-2) < df.rayleigh_resolution(lam, 1e-2)   # bigger D -> finer

# 7. THE deep identity: |FFT(aperture)|^2 == the sinc^2 formula (diffraction = Fourier)
x = np.linspace(-5e-3, 5e-3, 200000); dx = x[1] - x[0]
ap = (np.abs(x) < a / 2).astype(float)                         # a rectangular slit
theta, I_fft = df.aperture_diffraction(ap, dx, lam)
I_formula = df.single_slit_intensity(theta, a, lam)
core = np.abs(theta) < np.radians(3)
assert np.max(np.abs(I_fft - I_formula)[core]) < 1e-4          # FFT pattern == sinc^2

print(f"TEST PASS  (I(0)=I0; minima at m lambda/a (dark); lobe half-width=arcsin(lambda/a); "
      f"narrower slit -> wider; side lobe {side.max():.3f}; Rayleigh lambda/D & 1.22 lambda/D; "
      f"|FFT(slit)|^2 == sinc^2)")
