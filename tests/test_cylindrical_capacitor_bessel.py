"""Test the cylindrical capacitor + Bessel-function boundary-value problem:
the simple coaxial formula, the Bessel-Fourier coefficients, and (the real
claim) that the truncated series reconstructs the V0 boundary condition in
the can's interior."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import j0
from dgs import cylindrical_capacitor_bessel as ccb

# 1. simple coaxial-cylinder capacitance matches the textbook formula exactly
a, b, L = 1.0, 2.0, 3.0
C = ccb.simple_cylindrical_capacitance(a, b, L)
expected = 2 * np.pi * ccb.EPS0 * L / np.log(b / a)
assert abs(C - expected) < 1e-20

for bad in [(2.0, 1.0, 1.0), (1.0, 1.0, 1.0)]:
    try:
        ccb.simple_cylindrical_capacitance(*bad, 1.0)
        assert False, "should reject a >= b"
    except ValueError:
        pass
try:
    ccb.simple_cylindrical_capacitance(1.0, 2.0, -1.0)
    assert False
except ValueError:
    pass

# 2. k_n are genuinely the zeros of J0 -- verify J0(k_n) ~ 0 for each one
a_can, L_can, V0 = 1.0, 2.0, 100.0
k_n, A_n = ccb.bessel_coefficients(V0, a_can, L_can, n_terms=10)
assert np.allclose(j0(k_n), 0.0, atol=1e-9)

# 3. THE central claim: the truncated series reconstructs V(r,L) ~ V0 in the
#    interior (away from the r=a discontinuity, where Gibbs phenomenon is
#    real physics, not a bug)
r_interior = np.linspace(0, 0.9 * a_can, 100)
V_top = ccb.potential(r_interior, L_can, V0, a_can, L_can, n_terms=50)
assert np.mean(np.abs(V_top - V0)) < 2.0   # within ~2% of V0 on average
assert np.all(np.abs(V_top - V0) < 15.0)   # no wild divergence anywhere in the interior

# 4. the bottom boundary condition V(r,0)=0 holds EXACTLY (sinh(0)=0 for
#    every term, not approximately -- this one is exact by construction)
V_bottom = ccb.potential(r_interior, 0.0, V0, a_can, L_can, n_terms=50)
assert np.allclose(V_bottom, 0.0, atol=1e-9)

# 5. the side-wall boundary condition V(a,z)=0 holds for all z (J0(k_n)=0
#    at r=a for every term, exactly, since k_n are defined as J0's zeros)
z_test = np.linspace(0, L_can, 20)
V_side = ccb.potential(np.full_like(z_test, a_can), z_test, V0, a_can, L_can, n_terms=50)
assert np.allclose(V_side, 0.0, atol=1e-6)

# 6. more terms should not make the interior reconstruction WORSE (should
#    converge, or at least not diverge, as n_terms grows)
err_10 = np.mean(np.abs(ccb.potential(r_interior, L_can, V0, a_can, L_can, 10) - V0))
err_50 = np.mean(np.abs(ccb.potential(r_interior, L_can, V0, a_can, L_can, 50) - V0))
assert err_50 <= err_10 + 1e-6

# 7. torch fallback path (torch not installed under py-3.13 here) agrees
#    exactly with the NumPy/SciPy coefficients
k_n_t, A_n_t = ccb.bessel_coefficients_torch(V0, a_can, L_can, n_terms=20)
k_n_n, A_n_n = ccb.bessel_coefficients(V0, a_can, L_can, n_terms=20)
assert np.allclose(A_n_t, A_n_n)
assert np.allclose(k_n_t, k_n_n)

# 8. input validation
for bad_call in [
    lambda: ccb.bessel_coefficients(V0, -1.0, L_can),
    lambda: ccb.bessel_coefficients(V0, a_can, -1.0),
    lambda: ccb.bessel_coefficients(V0, a_can, L_can, n_terms=0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.cylindrical_capacitor_bessel tests passed")
