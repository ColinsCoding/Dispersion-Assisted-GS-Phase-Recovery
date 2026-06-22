"""Test the linear algebra of Bessel functions: eigenvalue zeros, recurrence, series."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import bessel_linalg as bl

# 1. the Bessel zeros are the eigenvalues of the radial operator matrix
known = {0: [2.4048, 5.5201, 8.6537], 1: [3.8317, 7.0156], 2: [5.1356, 8.4172]}
for n, ref in known.items():
    z = bl.bessel_zeros(n, k=len(ref))
    assert np.allclose(z, ref, atol=2e-3), (n, z, ref)

# 2. _besselj0 matches known values: J0(0)=1, J0(1)=0.7652, J0(2.4048)~0
assert abs(bl._besselj0(np.array([0.0]))[0] - 1.0) < 1e-6
assert abs(bl._besselj0(np.array([1.0]))[0] - 0.76519) < 1e-4
assert abs(bl._besselj0(np.array([2.4048]))[0]) < 1e-3        # first zero

# 3. the first J0 eigenmode matches J0(alpha_1 r/R) in shape
r, zeros, modes = bl.bessel_modes(0, k=3)
j0_exact = bl._besselj0(zeros[0] * r)
m0 = modes[0] / modes[0][np.argmax(np.abs(modes[0]))]
assert np.corrcoef(m0, j0_exact / np.max(np.abs(j0_exact)))[0, 1] > 0.9999

# 4. spherical Bessel: j0 = sin x/x, j1 = sin x/x^2 - cos x/x, and the recurrence
x = 2.5
j = bl.spherical_jn(5, x)
assert abs(j[0] - np.sin(x)/x) < 1e-12
assert abs(j[1] - (np.sin(x)/x**2 - np.cos(x)/x)) < 1e-12
for n in range(1, 5):                                          # j_{n-1}+j_{n+1} = (2n+1)/x j_n
    assert abs(j[n-1] + j[n+1] - (2*n+1)/x * j[n]) < 1e-12

# 5. Fourier-Bessel: expand a smooth f(r)=1-r^2 (zero at R=1) and reconstruct
r = np.linspace(1e-4, 1.0, 1500)
f = 1 - r**2
zeros = bl.bessel_zeros(0, k=18)
c = bl.fourier_bessel_coeffs(f, r, zeros)
recon = bl.fourier_bessel_reconstruct(c, r, zeros)
assert np.max(np.abs(recon - f)[50:-50]) < 0.03                # interior reconstruction

# 6. orthogonality: distinct J0(alpha_m r) are orthogonal with weight r (~0 overlap)
b1 = bl._besselj0(zeros[0]*r); b2 = bl._besselj0(zeros[1]*r)
overlap = np.trapezoid(b1*b2*r, r) / np.sqrt(np.trapezoid(b1**2*r, r)*np.trapezoid(b2**2*r, r))
assert abs(overlap) < 1e-2

print(f"TEST PASS  (J0/J1/J2 zeros = matrix eigenvalues incl. 2.4048; J0 mode matches; "
      f"spherical j_n recurrence exact; Fourier-Bessel reconstructs 1-r^2; "
      f"orthogonality {abs(overlap):.1e})")
