"""Test dgs/contour_integration_residues.py: the residue theorem (with the
pole-enclosure trap deliberately exercised, not just the happy path),
Jordan's lemma real-integral evaluation, the Lorentz susceptibility's pole
location (lower half plane, the causality condition), and Kramers-Kronig
derived from contour integration, cross-checked against dgs.causality's
independent FFT-based method."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs.contour_integration_residues import (
    residues_symbolic, contour_integral_numeric, verify_residue_theorem,
    real_integral_via_residues, lorentz_susceptibility_poles,
    verify_poles_in_lower_half_plane, kramers_kronig_via_contour_integration,
    cross_check_against_causality_module,
)

z = sp.symbols('z')

# 1. residues_symbolic: known exact residues for 1/(z^2+1) at +i and -i
res = residues_symbolic(1 / (z**2 + 1), z, [sp.I, -sp.I])
assert sp.simplify(res[sp.I] - (-sp.I / 2)) == 0
assert sp.simplify(res[-sp.I] - (sp.I / 2)) == 0

# 2. verify_residue_theorem: a contour enclosing ONLY +i must match 2*pi*i*Res(+i)
single_pole = verify_residue_theorem(lambda zz: 1 / (zz**2 + 1), 1 / (z**2 + 1), z, [sp.I],
                                      center=1j, radius=0.3)
assert single_pole["abs_diff"] < 1e-6
assert abs(single_pole["numeric_contour_integral"] - np.pi) < 1e-6   # 2*pi*i*(-i/2) = pi

# 3. THE ENCLOSURE TRAP: a contour enclosing BOTH +i and -i must give a
#    DIFFERENT (here, exactly zero) answer, since the two residues cancel
#    -- this is not a bug, it's the correct answer for that contour, and
#    the test asserts the residue SUM must be passed to match
both_poles = verify_residue_theorem(lambda zz: 1 / (zz**2 + 1), 1 / (z**2 + 1), z, [sp.I, -sp.I],
                                     center=0.0, radius=5.0)
assert both_poles["abs_diff"] < 1e-6
assert abs(both_poles["numeric_contour_integral"]) < 1e-6   # residues cancel exactly

# passing the WRONG pole list for a given contour must NOT spuriously agree
mismatched = verify_residue_theorem(lambda zz: 1 / (zz**2 + 1), 1 / (z**2 + 1), z, [sp.I],
                                     center=0.0, radius=5.0)   # contour has BOTH poles, list has only one
assert mismatched["abs_diff"] > 1.0, "mismatched pole list vs. contour should clearly disagree"

print("dgs.contour_integration_residues: residue theorem checks passed")

# 4. real_integral_via_residues: three independent routes to pi/a, for
#    several a values
for a in (0.5, 2.0, 5.0):
    r = real_integral_via_residues(a=a)
    assert r["max_abs_diff"] < 1e-6, f"a={a}: {r}"
    assert abs(r["expected_pi_over_a"] - np.pi / a) < 1e-12

try:
    real_integral_via_residues(a=-1.0)
    raise AssertionError("expected ValueError for a <= 0")
except ValueError:
    pass

print("dgs.contour_integration_residues: Jordan's lemma checks passed")

# 5. lorentz_susceptibility_poles / verify_poles_in_lower_half_plane: for
#    ANY physical (gamma>0) damping, both poles sit in Im<0 -- checked
#    across several (omega0, gamma) pairs, not just the default
for omega0, gamma in [(1.0, 0.2), (2.5, 0.05), (0.5, 1.0)]:
    check = verify_poles_in_lower_half_plane(omega0, gamma)
    assert check["all_in_lower_half_plane"] is True, f"omega0={omega0}, gamma={gamma}: {check}"
    assert len(check["poles"]) == 2

for bad_omega0, bad_gamma in [(-1.0, 0.2), (1.0, -0.2), (0.0, 0.2)]:
    try:
        lorentz_susceptibility_poles(bad_omega0, bad_gamma)
        raise AssertionError(f"expected ValueError for omega0={bad_omega0}, gamma={bad_gamma}")
    except ValueError:
        pass

print("dgs.contour_integration_residues: pole-location checks passed")

# 6. kramers_kronig_via_contour_integration: must match the Lorentz
#    susceptibility's own closed-form real part, at several query points
for w0_query in (0.0, 0.5, 1.5, 2.0, -1.0):
    kk = kramers_kronig_via_contour_integration(w0_query)
    assert kk["abs_diff"] < 1e-5, f"omega0={w0_query}: {kk}"

# 7. cross_check_against_causality_module: BOTH independent methods (FFT
#    Hilbert transform, contour-integration KK) must agree with the true
#    closed-form values to a small tolerance
omega_grid = np.linspace(-50, 50, 4000)
cross = cross_check_against_causality_module(omega_grid)
assert cross["max_abs_diff_fft_hilbert_vs_true"] < 1e-2
assert cross["max_abs_diff_contour_integration_vs_true"] < 1e-4
assert len(cross["contour_results"]) == 5

print("dgs.contour_integration_residues: Kramers-Kronig cross-check passed")
print("all dgs.contour_integration_residues tests passed")
