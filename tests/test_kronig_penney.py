"""Test the Kronig-Penney band model: free-electron limit, gaps, band edges."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import kronig_penney as kp

z = np.linspace(1e-6, 30, 5000)

# 1. P=0 is the FREE electron: f(z)=cos(z), allowed everywhere, no gaps
assert np.allclose(kp.kp_rhs(z, 0.0), np.cos(z))
assert np.all(kp.allowed(z, 0.0))                         # |cos z| <= 1 always
assert abs(kp.allowed_fraction(0.0) - 1.0) < 1e-6
assert len(kp.band_edges(0.0)) == 0                       # no band edges

# 2. band edges sit at z = n*pi for ANY P: f(n*pi) = (-1)^n
for n in range(1, 5):
    assert abs(kp.kp_rhs(np.array([n * np.pi]), 3.7)[0] - (-1)**n) < 1e-9

# 3. P>0 opens gaps: allowed fraction < 1, and shrinks as the lattice strengthens
assert kp.allowed_fraction(1.0) < 1.0
assert kp.allowed_fraction(5.0) < kp.allowed_fraction(1.0) < kp.allowed_fraction(0.2)

# 4. the band edges at z=pi and z=2pi are present for P>0
edges = kp.band_edges(2.0)
assert np.any(np.abs(edges - np.pi) < 1e-2)
assert np.any(np.abs(edges - 2 * np.pi) < 1e-2)

# 5. there IS a forbidden gap just above z=pi (|f|>1)
zg = np.linspace(np.pi + 0.05, np.pi + 0.7, 200)
assert np.any(~kp.allowed(zg, 2.0))                       # some z here are forbidden

# 6. dispersion: P=0 recovers the free-electron parabola E = (K a)^2 in the 1st band
Ka, E = kp.dispersion(0.0, z_max=np.pi - 0.05, n=2000)
good = ~np.isnan(Ka)
assert np.allclose(E[good], Ka[good]**2, atol=1e-6)       # E ~ K^2, free electron
# and P>0 introduces NaN (gap) regions in the dispersion
Ka2, _ = kp.dispersion(3.0)
assert np.any(np.isnan(Ka2)) and np.any(~np.isnan(Ka2))

print(f"TEST PASS  (P=0 free electron E=(Ka)^2, no gaps; band edges at n*pi; gaps open "
      f"for P>0 (allowed frac {kp.allowed_fraction(1.0):.2f}@P=1, "
      f"{kp.allowed_fraction(5.0):.2f}@P=5); first gap above z=pi)")
