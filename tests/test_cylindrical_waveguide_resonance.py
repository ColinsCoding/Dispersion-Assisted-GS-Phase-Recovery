"""Test dgs/cylindrical_waveguide_resonance.py: Bessel-function radial
modes of a cylindrical waveguide/cavity, boundary conditions checked
against scipy (not assumed), TE11 confirmed as the dominant mode, cavity
resonant frequencies, and the driven-resonance lineshape including the
Q-dependent peak-shift subtlety (the peak is NOT exactly at f0 for finite
Q -- a real bug this session caught and fixed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import numpy as np
from dgs.cylindrical_waveguide_resonance import (
    radial_wavenumber, verify_boundary_condition, radial_mode_profile,
    waveguide_cutoff_frequency, waveguide_propagation_constant,
    cavity_resonant_frequency, dominant_mode_cutoff,
    driven_resonance_response, resonance_peak_frequency, verify_resonance_peak,
    C_LIGHT,
)

a = 0.01   # 1 cm radius

# 1. radial_wavenumber: known values (scipy's own jn_zeros/jnp_zeros, spot-checked)
assert abs(radial_wavenumber(0, 1, a, "TM") - 240.482556) < 1e-4
assert abs(radial_wavenumber(1, 1, a, "TE") - 184.118378) < 1e-4

# 2. radial_wavenumber: input validation
for bad in [dict(m=-1, n=1, a=a), dict(m=0, n=0, a=a), dict(m=0, n=1, a=-1.0)]:
    try:
        radial_wavenumber(**bad, boundary="TM")
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass
try:
    radial_wavenumber(0, 1, a, boundary="bogus")
    raise AssertionError("expected ValueError for bad boundary")
except ValueError:
    pass

# 3. verify_boundary_condition: must pass for real modes, and must FAIL if
#    the boundary condition genuinely isn't satisfied (tests the checker
#    actually discriminates, not just returns True)
assert verify_boundary_condition(0, 1, a, "TM") is True
assert verify_boundary_condition(1, 1, a, "TE") is True

import dgs.cylindrical_waveguide_resonance as cwr
_orig = cwr.radial_wavenumber
cwr.radial_wavenumber = lambda m, n, a, boundary: _orig(m, n, a, boundary) * 1.1   # wrong on purpose
try:
    cwr.verify_boundary_condition(0, 1, a, "TM")
    raise AssertionError("expected AssertionError for a k_c that doesn't satisfy the boundary condition")
except AssertionError as e:
    assert "boundary condition not satisfied" in str(e)
finally:
    cwr.radial_wavenumber = _orig

# 4. radial_mode_profile: J_0(0) = 1 (finite, nonzero field on axis for m=0);
#    J_m(0) = 0 for m>0 (field vanishes on axis for m>0, a real physical fact)
prof_m0 = radial_mode_profile(0, 1, a, np.array([0.0]))
prof_m1 = radial_mode_profile(1, 1, a, np.array([0.0]))
assert abs(prof_m0[0] - 1.0) < 1e-9
assert abs(prof_m1[0] - 0.0) < 1e-9
# and the profile must vanish at r=a for a TM (Dirichlet) mode
prof_at_a = radial_mode_profile(0, 1, a, np.array([a]), boundary="TM")
assert abs(prof_at_a[0]) < 1e-9

# 5. radial_mode_profile: r outside [0, a] must raise
try:
    radial_mode_profile(0, 1, a, np.array([a * 1.5]))
    raise AssertionError("expected ValueError for r > a")
except ValueError:
    pass

# 6. waveguide_cutoff_frequency / dominant_mode_cutoff: TE11 dominant, checked
dom = dominant_mode_cutoff(a)
assert dom["dominant"] == "TE11"
assert dom["cutoffs_Hz"]["TE11"] < dom["cutoffs_Hz"]["TM01"] < dom["cutoffs_Hz"]["TE21"]

# 7. waveguide_propagation_constant: real beta above cutoff, imaginary below
f_c = waveguide_cutoff_frequency(1, 1, a, "TE")
beta_above = waveguide_propagation_constant(f_c * 1.5, 1, 1, a, "TE")
beta_below = waveguide_propagation_constant(f_c * 0.5, 1, 1, a, "TE")
assert beta_above.imag == 0 and beta_above.real > 0, f"expected real propagating beta, got {beta_above}"
assert beta_below.real == 0 and beta_below.imag > 0, f"expected evanescent (imaginary) beta, got {beta_below}"

# 8. cavity_resonant_frequency: p=0 (no axial variation) must equal the
#    plain waveguide cutoff frequency exactly
f_cavity_p0 = cavity_resonant_frequency(1, 1, 0, a, L=0.03, boundary="TE")
assert abs(f_cavity_p0 - f_c) / f_c < 1e-9   # relative tol: these are ~1e9 Hz values
# increasing p must strictly increase the resonant frequency
f_p1 = cavity_resonant_frequency(1, 1, 1, a, L=0.03, boundary="TE")
f_p2 = cavity_resonant_frequency(1, 1, 2, a, L=0.03, boundary="TE")
assert f_cavity_p0 < f_p1 < f_p2

# 9. driven_resonance_response: exact identity at f=f0 -> response = Q^2,
#    for several Q -- independent of the peak-location subtlety
f0 = 8.78e9
for Q in (2.0, 10.0, 100.0):
    r = driven_resonance_response(np.array([f0]), f0, Q)[0]
    assert abs(r - Q**2) / Q**2 < 1e-9, f"Q={Q}: response(f0)={r}, expected {Q**2}"

# 10. resonance_peak_frequency: matches the closed-form
#     f_peak = f0*sqrt(1-1/(2Q^2)), and is STRICTLY LESS than f0 for finite Q
#     (the actual bug this session caught: peak != f0 in general)
for Q in (1.0, 5.0, 50.0):
    f_peak = resonance_peak_frequency(f0, Q)
    assert f_peak < f0, f"Q={Q}: peak {f_peak} should be strictly < f0={f0}"
    expected = f0 * math.sqrt(1 - 1 / (2 * Q**2))
    assert abs(f_peak - expected) / expected < 1e-12

# 11. resonance_peak_frequency: converges to f0 as Q -> infinity (large Q)
f_peak_highQ = resonance_peak_frequency(f0, 10_000.0)
assert abs(f_peak_highQ - f0) / f0 < 1e-6

# 12. resonance_peak_frequency: must raise for Q <= 1/sqrt(2) (no interior peak)
try:
    resonance_peak_frequency(f0, 0.5)
    raise AssertionError("expected ValueError for Q <= 1/sqrt(2)")
except ValueError:
    pass

# 13. verify_resonance_peak: the module's own self-check passes for a
#     range of Q, including the low-Q case where the shift is largest
for Q in (1.0, 5.0, 50.0, 500.0):
    assert verify_resonance_peak(f0, Q) is True

# 14. driven_resonance_response / resonance_peak_frequency: input validation
for bad_f0, bad_Q in [(-1.0, 5.0), (1.0, -5.0), (1.0, 0.0)]:
    try:
        driven_resonance_response(np.array([1.0]), bad_f0, bad_Q)
        raise AssertionError(f"expected ValueError for f0={bad_f0}, Q={bad_Q}")
    except ValueError:
        pass

print("all dgs.cylindrical_waveguide_resonance tests passed")
