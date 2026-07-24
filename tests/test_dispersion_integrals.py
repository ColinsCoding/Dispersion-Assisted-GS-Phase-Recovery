"""Test dgs/dispersion_integrals.py's two new closed-form integrals of
H(f)=exp(i*pi*D*f^2): the analytic impulse response (Fresnel/Gaussian
integral) and the Gaussian-pulse GVD broadening law. Each is checked against
an independent numeric evaluation (Riemann-sum Fourier integral / FFT
second-moment fit), not just against itself."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.dispersion_integrals import (
    derive_completed_square,
    impulse_response,
    verify_impulse_response_numeric,
    gaussian_broadening_T1,
    verify_gaussian_broadening_numeric,
)


# 1. The shared completed-square identity must hold exactly (symbolic residual 0)
derive_completed_square()  # raises AssertionError internally if it doesn't hold

# 2. Impulse response: closed form vs. Riemann-sum Fourier integral,
#    across positive and negative D
for D in [5.0, -5.0, 12.3, -0.8]:
    v = verify_impulse_response_numeric(D)
    assert v["max_rel_err"] < 0.01, (
        f"D={D}: impulse_response disagrees with direct Fourier-sum "
        f"integral by {v['max_rel_err']:.4f} (expected <1% truncation error)")

# 3. D=0 must raise (delta function, not the Gaussian/Fresnel kernel derived here)
try:
    impulse_response(0.0, 1.0)
    raise AssertionError("impulse_response(D=0, ...) should have raised ValueError")
except ValueError:
    pass

# 4. Gaussian broadening: closed form vs. FFT second-moment fit, exact to
#    numerical precision (unlike the impulse response, this isn't a
#    truncated oscillatory integral)
for D, T0 in [(0.0, 1.0), (5.0, 1.0), (20.0, 1.0), (-8.0, 0.7), (50.0, 2.0)]:
    v = verify_gaussian_broadening_numeric(D, T0)
    assert v["rel_err"] < 1e-6, (
        f"D={D}, T0={T0}: gaussian_broadening_T1={v['T1_analytic']} disagrees "
        f"with FFT second-moment fit={v['T1_numeric']} (rel_err={v['rel_err']:.2e})")

# 5. Sanity checks on the broadening formula itself
assert abs(gaussian_broadening_T1(0.0, 2.0) - 2.0) < 1e-12, \
    "D=0 (no dispersion) must leave the pulse width unchanged"
assert gaussian_broadening_T1(100.0, 1.0) > gaussian_broadening_T1(10.0, 1.0), \
    "more dispersion must broaden the pulse more"
assert abs(gaussian_broadening_T1(-30.0, 1.0) - gaussian_broadening_T1(30.0, 1.0)) < 1e-12, \
    "broadening depends on D^2 -- sign of D must not matter"

# 6. T0<=0 must raise (unphysical pulse width)
try:
    gaussian_broadening_T1(5.0, T0=0.0)
    raise AssertionError("gaussian_broadening_T1(T0=0) should have raised ValueError")
except ValueError:
    pass

print("all dgs.dispersion_integrals tests passed")
