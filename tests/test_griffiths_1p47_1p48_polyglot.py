"""Test dgs/griffiths_1p47_1p48_polyglot.py: Griffiths Problems 1.47 and
1.48's delta-sifting integrals, proven exactly with SymPy (including fully
symbolic vector arguments where the problem's domain allows it), and
cross-checked against an independently-coded deterministic grid-quadrature
method in Python and (if installed) MATLAB."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import sympy as sp
from dgs.griffiths_1p47_1p48_polyglot import (
    point_charge_density_proof, dipole_density_proof, shell_density_proof,
    integral_1p48a_symbolic, integral_1p48b_symbolic, integral_1p48c, integral_1p48d,
    cross_validate_1p48, MATLAB_DEFAULT,
)

# 1. point_charge_density_proof: total charge is EXACTLY q, symbolically
q = sp.Symbol('q', real=True)
result = point_charge_density_proof()
assert sp.simplify(result - q) == 0

# 2. dipole_density_proof: net charge exactly 0, dipole moment exactly -q*a
#    (this module's -q-at-origin/+q-at-a sign convention)
dip = dipole_density_proof()
assert sp.simplify(dip["total_charge"]) == 0
ax, ay, az, q_ = sp.symbols('a_x a_y a_z q', real=True)
expected_p = sp.Matrix([-q_ * ax, -q_ * ay, -q_ * az])
assert sp.simplify(dip["dipole_moment"] - expected_p) == sp.zeros(3, 1)

# 3. shell_density_proof: total charge is EXACTLY Q, for symbolic R too
#    (not just as R -> infinity)
Q = sp.Symbol('Q', positive=True)
assert sp.simplify(shell_density_proof() - Q) == 0

# 4. integral_1p48a_symbolic: fully general 3*a.a
ax_, ay_, az_ = sp.symbols('a_x a_y a_z', real=True)
result_a = integral_1p48a_symbolic()
expected_a = 3 * (ax_**2 + ay_**2 + az_**2)
assert sp.simplify(result_a - expected_a) == 0
# numeric spot check at a concrete vector
assert float(result_a.subs({ax_: 1, ay_: 2, az_: 2})) == 27.0   # 3*(1+4+4)=27

# 5. integral_1p48b_symbolic: general formula AND Griffiths' own numbers
b_res = integral_1p48b_symbolic()
bx_, by_, bz_ = sp.symbols('b_x b_y b_z', real=True)
expected_b_general = (bx_**2 + by_**2 + bz_**2) / 125
assert sp.simplify(b_res["general"] - expected_b_general) == 0
assert b_res["numeric"] == sp.Rational(1, 5)

# a different b should give a different numeric result (sanity: not a
# hardcoded constant)
b_res2 = integral_1p48b_symbolic(b_vec=(1, 1, 1))
assert b_res2["numeric"] == sp.Rational(3, 125)

# 6. integral_1p48c: c is OUTSIDE V (|c|^2=38 > 36=R^2), so answer is 0
c_res = integral_1p48c()
assert c_res["c_mag_sq"] == 38
assert c_res["R_sq"] == 36
assert c_res["outside_V"] is True
assert c_res["final_answer"] == 0

# a point INSIDE the sphere should NOT automatically give zero -- confirms
# the containment check actually gates the answer, not always returning 0
c_res_inside = integral_1p48c(c_vec=(1, 1, 1), R=6)
assert c_res_inside["outside_V"] is False
assert c_res_inside["final_answer"] != 0

# 7. integral_1p48d: e is INSIDE V (|e-center|^2=2 < 2.25=R^2), answer -4
d_res = integral_1p48d()
assert d_res["dist_sq"] == 2
assert d_res["R_sq"] == sp.Rational(9, 4)
assert d_res["inside_V"] is True
assert d_res["final_answer"] == -4

# a center placed far away should push e outside V, giving 0 instead of -4
d_res_outside = integral_1p48d(center=(100, 100, 100), R=1.5)
assert d_res_outside["inside_V"] is False
assert d_res_outside["final_answer"] == 0

print("dgs.griffiths_1p47_1p48_polyglot: SymPy exact-proof checks passed")

# 8. cross_validate_1p48: Python grid quadrature must match SymPy exact
#    answers to within the Gaussian-regularization's expected small bias
check = cross_validate_1p48()
tolerances = {"a": 1e-2, "b": 1e-3, "c": 1e-9, "d": 1e-2}
for part, tol in tolerances.items():
    assert check["abs_diff"][part] < tol, (
        f"part {part}: exact={check['exact'][part]}, "
        f"numeric={check['numeric_python'][part]}, diff={check['abs_diff'][part]}")
print("dgs.griffiths_1p47_1p48_polyglot: Python grid-quadrature cross-check passed")

# 9. MATLAB (if installed): independently-coded grid quadrature must match too
if os.path.exists(MATLAB_DEFAULT):
    import tempfile
    from dgs.griffiths_1p47_1p48_polyglot import run_matlab_1p48
    with tempfile.TemporaryDirectory() as tmp:
        matlab_result = run_matlab_1p48(tmp)
    for part, tol in tolerances.items():
        diff = abs(check["exact"][part] - matlab_result[part])
        assert diff < tol, f"part {part}: exact={check['exact'][part]}, matlab={matlab_result[part]}, diff={diff}"
    print("dgs.griffiths_1p47_1p48_polyglot: MATLAB grid-quadrature cross-check passed")
else:
    print("dgs.griffiths_1p47_1p48_polyglot: MATLAB not found, skipped MATLAB checks")

print("all dgs.griffiths_1p47_1p48_polyglot tests passed")
