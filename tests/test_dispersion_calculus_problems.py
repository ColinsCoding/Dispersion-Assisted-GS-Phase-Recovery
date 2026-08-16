"""Test dgs/dispersion_calculus_problems.py's three photonics calculus
problems built on H(f)=exp(i*pi*D*f^2): the impulse response (Fresnel
integral, cross-checked against dgs.dispersion_integrals), the linear
group-delay law (checked against a numeric phase-gradient), and the
all-pass energy-conservation identity |H(f)|=1."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from dgs.dispersion_calculus_problems import (
    problem1_solve_symbolic, problem1_verify,
    problem2_solve_symbolic, problem2_verify,
    problem3_solve_symbolic, problem3_verify,
)

# 1. Impulse response: cross-check this module's own from-scratch
#    derivation against dgs.dispersion_integrals' independently verified
#    result, across positive and negative D
for D in [5.0, -5.0, 12.3, -0.8]:
    v = problem1_verify(D)
    assert v["max_abs_diff_vs_dispersion_integrals"] < 1e-10, (
        f"D={D}: this module's h(t) disagrees with dgs.dispersion_integrals "
        f"by {v['max_abs_diff_vs_dispersion_integrals']:.2e} (expected exact agreement)")

# D=0 must raise (delta function, not the Gaussian/Fresnel kernel solved here)
try:
    problem1_verify(0.0)
    raise AssertionError("problem1_verify(D=0) should have raised ValueError")
except ValueError:
    pass

# 2. Group delay: symbolic result must be exactly -D*f
D_sym, f_sym = sp.symbols("D f", real=True)
assert sp.simplify(problem2_solve_symbolic() - (-D_sym * f_sym)) == 0, \
    "Problem 2 symbolic solution must equal -D*f exactly"

# Numeric check across signs of D
for D in [5.0, -5.0, 20.0, -0.3]:
    v = problem2_verify(D)
    assert v["max_abs_err"] < 1e-6, (
        f"D={D}: numeric group-delay gradient disagrees with -D*f by "
        f"{v['max_abs_err']:.2e}")

# F<=0 and n<3 must raise
for bad_kwargs in [{"F": 0.0}, {"F": -1.0}, {"n": 2}]:
    try:
        problem2_verify(5.0, **bad_kwargs)
        raise AssertionError(f"problem2_verify({bad_kwargs}) should have raised ValueError")
    except ValueError:
        pass

# 3. All-pass identity: symbolic result must be exactly 1
assert problem3_solve_symbolic() == 1, "Problem 3 symbolic |H(f)|^2 must simplify to exactly 1"

# Numeric check, including D=0 (no dispersion is still all-pass, trivially)
for D in [0.0, 5.0, -600.0, 1e-3]:
    v = problem3_verify(D)
    assert v["max_abs_dev_from_1"] < 1e-9, (
        f"D={D}: |H(f)| deviates from 1 by {v['max_abs_dev_from_1']:.2e}")

print("all dgs.dispersion_calculus_problems tests passed")
