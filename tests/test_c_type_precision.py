"""Test C's char/int/float/double, compiled and run for real via gcc (same
pattern as dgs.circuits_polyglot), cross-checked against NumPy's IEEE 754
type-info tables. NOTE: requires mingw64 on PATH (gcc's driver looks up
cc1/as/ld via PATH even when invoked by full path) -- run via PowerShell,
not the Bash tool, which lacks mingw64 on PATH in this environment."""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import c_type_precision as ctp

with tempfile.TemporaryDirectory() as tmp:
    exe = ctp.compile_c_type_info(tmp)
    c_info = ctp.run_c_type_info(exe)

# 1. real, platform-actual sizes (not just textbook-assumed values)
assert c_info["sizeof_char"] == 1
assert c_info["sizeof_int"] == 4
assert c_info["sizeof_float"] == 4
assert c_info["sizeof_double"] == 8

# 2. char and int ranges match this compiler's actual signed limits
assert c_info["char_min"] == -128 and c_info["char_max"] == 127
assert c_info["int_min"] == -2147483648 and c_info["int_max"] == 2147483647

# 3. float has ~6-7 significant decimal digits, double has ~15-17
#    (IEEE 754 single vs double precision, not arbitrary numbers)
assert 6 <= c_info["flt_dig"] <= 7
assert 15 <= c_info["dbl_dig"] <= 17

# 4. C's float/double really are the same IEEE 754 formats NumPy uses --
#    verified numerically, not assumed because the names match
checks = ctp.cross_check_against_numpy(c_info)
assert all(checks.values()), checks

# 5. epsilon values match np.finfo exactly (both are the IEEE 754 constant,
#    not independently-computed approximations)
np_f32, np_f64 = np.finfo(np.float32), np.finfo(np.float64)
assert abs(c_info["flt_epsilon"] - np_f32.eps) / np_f32.eps < 1e-6
assert abs(c_info["dbl_epsilon"] - np_f64.eps) / np_f64.eps < 1e-6

# 6. the (epsilon)^(1/3) central-difference sweet-spot rule of thumb lands
#    within an order of magnitude of dgs.dual_autodiff's empirically-found
#    optimum (~1.17e-6) -- confirming the two independent results
#    (a textbook formula and an empirical sweep) actually agree
h_predicted = ctp.sqrt_epsilon_predicts_fd_sweet_spot(c_info["dbl_epsilon"])
h_empirical = 1.17e-6   # from dgs.dual_autodiff's finite_difference_error_sweep
assert 0.1 < h_predicted / h_empirical < 10

# 7. input validation
try:
    ctp.sqrt_epsilon_predicts_fd_sweet_spot(-1.0)
    assert False, "should have raised ValueError"
except ValueError:
    pass

# 8. empirical epsilon (halve-until-1+eps==1 loop, computed at runtime)
#    matches float.h's looked-up DBL_EPSILON exactly -- same IEEE 754
#    ULP-at-1.0, derived two independent ways
with tempfile.TemporaryDirectory() as tmp:
    eps_empirical = ctp.compute_epsilon_empirically(tmp)
assert eps_empirical == c_info["dbl_epsilon"]

# 9. catastrophic cancellation: (1.0+b)-b survives essentially intact well
#    above DBL_EPSILON, and is destroyed completely well below it
with tempfile.TemporaryDirectory() as tmp:
    pairs = ctp.catastrophic_cancellation_demo(tmp)
assert len(pairs) == 5
(b0, c0), (b1, c1), (b2, c2), (b3, c3), (b4, c4) = pairs
assert abs(c0 - b0) / b0 < 1e-3           # b=1e-8: well above epsilon, well preserved
assert c3 == 0.0 and c4 == 0.0            # b=1e-16, 1e-17: below DBL_EPSILON~2.22e-16, lost completely

# 10. Kahan compensated summation is dramatically more accurate than naive
#     summation for the same input (summing 0.1 x 1e6 in float)
with tempfile.TemporaryDirectory() as tmp:
    naive_sum, kahan_sum, true_sum = ctp.kahan_summation_demo(tmp)
assert abs(kahan_sum - true_sum) < 1e-3
assert abs(naive_sum - true_sum) > 100 * abs(kahan_sum - true_sum)

# 11. kahan_summation_demo input validation
with tempfile.TemporaryDirectory() as tmp:
    try:
        ctp.kahan_summation_demo(tmp, n=0)
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.c_type_precision tests passed")
