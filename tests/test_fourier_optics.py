"""Test dgs.fourier_optics: the far-field of a slit matches the sinc^2 formula, the
reciprocal scaling (width*sin(theta_null)=lambda), the EXACT shift-invariance of the
diffraction intensity (magnitude of the FT is unchanged by a shift though the phase
is not -- the lost-phase fact behind phase retrieval), and the double-slit fringes /
grating orders + sharpening."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import fourier_optics as fo
from dgs.diffraction import single_slit_intensity

lam = 600e-9
x = fo.make_grid(200000, 1e-7)      # 20 mm window
dx = x[1] - x[0]

# 1. aperture constructors
w = 50e-6
ap = fo.rect_aperture(x, w)
assert set(np.unique(ap)) <= {0.0, 1.0}
assert np.isclose(ap.sum() * dx, w, rtol=2e-2)                 # total open width (~sample-quantized)
ds = fo.double_slit(x, 20e-6, 200e-6)
assert np.isclose(ds.sum() * dx, 2 * 20e-6, rtol=2e-2)         # two slits' width
gr = fo.grating(x, 10e-6, 40e-6, 5)
assert np.isclose(gr.sum() * dx, 5 * 10e-6, rtol=2e-2)         # five slits

# 2. the far field of a slit IS the sinc^2 pattern
theta, A, I = fo.far_field(ap, dx, lam)
assert len(theta) == len(A) == len(I) and np.isclose(I.max(), 1.0)
core = np.abs(theta) < np.radians(2)
assert np.max(np.abs(I[core] - single_slit_intensity(theta, w, lam)[core])) < 1e-2

# 3. reciprocal scaling: width * sin(theta_null) = lambda for ANY width
assert np.isclose(fo.first_null_angle(w, lam), np.arcsin(lam / w))
for width in (20e-6, 50e-6, 100e-6):
    assert np.isclose(fo.scaling_product(width, lam), lam)     # constant = lambda
# narrower slit diffracts wider
assert fo.first_null_angle(20e-6, lam) > fo.first_null_angle(100e-6, lam)

# 4. SHIFT-INVARIANT INTENSITY: |FT| unchanged by a shift, though the FT itself moves
assert fo.intensity_is_shift_invariant(ap, 5000)              # intensity identical
assert fo.intensity_is_shift_invariant(ds, -1234)
# the complex amplitude DOES change (a linear phase) -- that's the lost phase
assert not np.allclose(np.fft.fft(ap), np.fft.fft(np.roll(ap, 5000)))
assert np.allclose(np.abs(np.fft.fft(ap)), np.abs(np.fft.fft(np.roll(ap, 5000))))

# 5. double slit: bright fringes at m lambda/d, dark between -> strong contrast
d = 200e-6
assert np.isclose(fo.double_slit_fringe_angle(d, lam, 1), np.arcsin(lam / d))
th2, _, I2 = fo.far_field(fo.double_slit(x, 20e-6, d), dx, lam)
a_bright = fo.double_slit_fringe_angle(d, lam, 1)
a_dark = np.arcsin(0.5 * lam / d)
assert np.interp(a_bright, th2, I2) > 10 * np.interp(a_dark, th2, I2)

# 6. grating: principal maxima at m lambda/period, and more slits -> sharper central peak
period = 10e-6
maxima = fo.grating_maxima(period, lam, max_order=3)
assert np.isclose(maxima[0], 0.0)
assert np.isclose(maxima[1], np.arcsin(lam / period))
# central peak narrows ~1/N: at a fixed small off-axis angle, more slits -> less light
th_a, _, Ia = fo.far_field(fo.grating(x, 2e-6, period, 2), dx, lam)
th_b, _, Ib = fo.far_field(fo.grating(x, 2e-6, period, 8), dx, lam)
ang = np.radians(1.0)                                          # inside the 2-slit lobe, past the 8-slit's
assert np.interp(ang, th_b, Ib) < np.interp(ang, th_a, Ia)

# 7. kwarg bounds
for bad in (lambda: fo.rect_aperture(x, 0),
            lambda: fo.first_null_angle(100e-9, lam),          # lambda/width >= 1
            lambda: fo.double_slit_fringe_angle(100e-9, lam),  # order lam/d >= 1
            lambda: fo.make_grid(4, 1e-7),
            lambda: fo.grating(x, 1e-6, 1e-6, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_fourier_optics: all checks passed")
