"""Test N-slit Fraunhofer diffraction grating: central max, grating-equation orders,
N=1 reduction to a single slit, missing orders, and energy bounds."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import diffraction_grating as dgt

d, a, wavelength = 2.0e-6, 0.4e-6, 589e-9
theta = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 20001)

# 1. central maximum (theta=0) is exactly 1 for any N, d, a
for N in (1, 4, 12):
    I0 = dgt.grating_intensity(np.array([0.0]), d, a, N, wavelength)[0]
    assert abs(I0 - 1.0) < 1e-9, (N, I0)

# 2. N=1 grating IS a single slit: the array factor is identically 1
array_N1 = dgt.grating_interference_factor(theta, d, 1, wavelength)
assert np.allclose(array_N1, 1.0, atol=1e-9)
I_N1 = dgt.grating_intensity(theta, d, a, 1, wavelength)
I_envelope = dgt.single_slit_envelope(theta, a, wavelength)
assert np.allclose(I_N1, I_envelope, atol=1e-9)

# 3. principal maxima land exactly where the grating equation d*sin(theta)=m*lambda says,
#    and the intensity there equals the single-slit envelope (array factor = 1)
N = 8
m, theta_max = dgt.principal_maxima_angles(d, wavelength, m_max=3)
assert np.allclose(d * np.sin(theta_max), m * wavelength, atol=1e-15)
I_at_max = dgt.grating_intensity(theta_max, d, a, N, wavelength)
env_at_max = dgt.single_slit_envelope(theta_max, a, wavelength)
assert np.allclose(I_at_max, env_at_max, atol=1e-6), (I_at_max, env_at_max)

# 4. intensity is bounded in [0, 1] everywhere (normalized central max is the global max)
I_full = dgt.grating_intensity(theta, d, a, N, wavelength)
assert np.all(I_full >= -1e-12) and np.all(I_full <= 1.0 + 1e-9), I_full.max()

# 5. missing order: with d = 3a exactly, the envelope's first zero (a*sin(theta)=lambda)
#    coincides with the m=3 principal maximum (d*sin(theta)=3*lambda) -> that order vanishes
a_missing = d / 3.0
m2, theta_max2 = dgt.principal_maxima_angles(d, wavelength, m_max=4)
idx3 = np.where(m2 == 3)[0][0]
I_missing = dgt.grating_intensity(np.array([theta_max2[idx3]]), d, a_missing, N, wavelength)[0]
assert I_missing < 1e-6, I_missing
# the neighboring, non-missing order (m=2) is NOT suppressed
idx2 = np.where(m2 == 2)[0][0]
I_present = dgt.grating_intensity(np.array([theta_max2[idx2]]), d, a_missing, N, wavelength)[0]
assert I_present > 1e-3, I_present

# 6. sharper fringes with more slits: peak-to-first-zero width shrinks as N grows
def fwhm_like_width(N):
    th = np.linspace(-0.05, 0.05, 4001)
    I = dgt.grating_intensity(th, d, a, N, wavelength)
    above_half = th[I >= 0.5]
    return above_half.max() - above_half.min()

assert fwhm_like_width(4) > fwhm_like_width(16)

# 7. wavelength_to_rgb: red end is red-dominant, blue end is blue-dominant, out-of-band is black
r_red = dgt.wavelength_to_rgb(700)
r_blue = dgt.wavelength_to_rgb(450)
assert r_red[0] > r_red[2] and r_blue[2] > r_blue[0]
assert dgt.wavelength_to_rgb(200) == (0.0, 0.0, 0.0)

# 8. input validation
for bad_call in [
    lambda: dgt.single_slit_envelope(theta, -1.0, wavelength),
    lambda: dgt.grating_interference_factor(theta, d, 0, wavelength),
    lambda: dgt.principal_maxima_angles(-1.0, wavelength, 3),
    lambda: dgt.angle_to_screen_position(theta, -1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.diffraction_grating tests passed")
