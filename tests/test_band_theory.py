"""Test the Kronig-Penney band theory model: the SymPy-derived dispersion
relation matches the known textbook closed form, P=0 has zero gaps (free
particle), P>0 gaps open exactly at alpha*a=n*pi, and the torch/numpy
sweeps agree."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import band_theory as bt

# 1. the SymPy derivation reproduces the known textbook closed form exactly:
#    cos(ka) = cos(alpha*a) + (P/(alpha*a))*sin(alpha*a)
import sympy as sp
rhs_expr, alpha, a, P = bt.derive_dispersion_relation()
expected = sp.cos(alpha * a) + (P / (alpha * a)) * sp.sin(alpha * a)
assert sp.simplify(rhs_expr - expected) == 0, (rhs_expr, expected)

# 2. numeric spot check of kronig_penney_rhs against the same closed form,
#    at a value where alpha*a is NOT a multiple of pi (avoids 0/0 edge case)
alpha_a_test, P_test = 1.3, 2.5
rhs_numeric = bt.kronig_penney_rhs(alpha_a_test, P_test)
rhs_manual = np.cos(alpha_a_test) + (P_test / alpha_a_test) * np.sin(alpha_a_test)
assert abs(rhs_numeric - rhs_manual) < 1e-12

# 3. the alpha*a -> 0 limit is finite (sin(x)/x -> 1), not a division blow-up
rhs_at_zero = bt.kronig_penney_rhs(0.0, P_test)
assert abs(rhs_at_zero - (1.0 + P_test)) < 1e-9   # cos(0) + P*1 = 1+P

# 4. P=0 (no potential at all): free particle, rhs=cos(alpha*a) always in
#    [-1,1] -- NO gaps anywhere, one continuous allowed band
gaps_free = bt.gap_edges(0.0, alpha_a_max=20.0, n_points=20000)
assert len(gaps_free) == 0, gaps_free

# 5. P>0: gaps open, and their edges land at (or very near) integer multiples
#    of pi -- the actual physical signature of a periodic potential, not an
#    arbitrary numerical artifact
P_val = 5.0
alpha_a, rhs, allowed = bt.find_bands(P_val, alpha_a_max=10.0, n_points=40000)
flips = alpha_a[np.where(np.diff(allowed.astype(int)) != 0)[0]]
assert len(flips) >= 4   # at least a couple of full gap/band cycles in this range

# every OTHER flip point (starting from the 2nd, 0-indexed) should sit at
# very close to an integer multiple of pi (gaps start exactly at n*pi for
# this sign convention -- verified in the actual sweep above, not assumed)
pi_multiples_found = 0
for edge in flips:
    n = round(edge / np.pi)
    if n > 0 and abs(edge - n * np.pi) < 0.01:
        pi_multiples_found += 1
assert pi_multiples_found >= 2, (flips, flips / np.pi)

# 6. more potential (larger P) means WIDER gaps -- a stronger periodic
#    scattering potential should suppress a wider range of energies
def total_gap_width(P_val, alpha_a_max=10.0, n_points=40000):
    _, _, allowed = bt.find_bands(P_val, alpha_a_max, n_points)
    return float(np.mean(~allowed) * alpha_a_max)

width_small_P = total_gap_width(1.0)
width_large_P = total_gap_width(10.0)
assert width_large_P > width_small_P

# 7. torch fallback path (torch not installed under py-3.13 here) agrees
#    exactly with the pure-NumPy sweep
alpha_a_t, rhs_t, allowed_t = bt.find_bands_torch(P_val, alpha_a_max=10.0, n_points=5000)
alpha_a_n, rhs_n, allowed_n = bt.find_bands(P_val, alpha_a_max=10.0, n_points=5000)
assert np.allclose(rhs_t, rhs_n)
assert np.array_equal(allowed_t, allowed_n)

# 8. input validation
for bad_call in [
    lambda: bt.find_bands(1.0, alpha_a_max=-1.0),
    lambda: bt.find_bands(1.0, n_points=1),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.band_theory tests passed")
