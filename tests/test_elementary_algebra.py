"""Test dgs/elementary_algebra.py: Feynman Ch. 22's direct/inverse operation
table, root-vs-logarithm distinction, log-table multiplication, and hardware
LUT addressing -- cross-checked, not just re-run against itself."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from dgs.elementary_algebra import (
    verify_direct_operation_rules,
    verify_power_law_rules,
    solve_for_missing_addend,
    solve_for_missing_factor,
    solve_for_root,
    solve_for_logarithm,
    verify_root_and_log_are_distinct_inverses,
    build_log_lookup_table,
    multiply_via_log_table,
    verify_log_table_multiplication,
    lut_address_bits_for_depth,
    lut_quantization_error_bound,
    ln1pr_taylor_coefficients,
    doubling_time_taylor_correction,
    doubling_time_correction_check,
)

# 1. all Eq 22.1 (a)-(e),(i)-(k) identities must hold
direct_results = verify_direct_operation_rules()
assert all(direct_results.values()), direct_results

# 2. power-law rules (f)-(h) must hold at the default positive sample
power_results = verify_power_law_rules()
assert all(power_results.values()), power_results

# 3. power laws must raise on non-positive inputs (continuity/ordering caveat)
try:
    verify_power_law_rules(a_val=-2.0, b_val=3.0, c_val=0.5)
    raise AssertionError("expected ValueError for negative base")
except ValueError:
    pass

# 4. the four inverse operations: known values
assert solve_for_missing_addend(3, 12) == 9
assert solve_for_missing_factor(3, 12) == 4
root_val = float(solve_for_root(3, 8))
assert abs(root_val - 2.0) < 1e-9   # cube root of 8 is 2
log_val = float(solve_for_logarithm(2, 8))
assert abs(log_val - 3.0) < 1e-9    # log base 2 of 8 is 3

# 5. division by zero and invalid log bases must raise clear errors
try:
    solve_for_missing_factor(0, 12)
    raise AssertionError("expected ValueError for a=0")
except ValueError:
    pass
try:
    solve_for_logarithm(1, 8)
    raise AssertionError("expected ValueError for base=1")
except ValueError:
    pass
try:
    solve_for_logarithm(2, -8)
    raise AssertionError("expected ValueError for c<=0")
except ValueError:
    pass

# 6. root and logarithm must be genuinely distinct inverses
distinct_result = verify_root_and_log_are_distinct_inverses()
assert distinct_result["distinct"] is True
assert abs(distinct_result["root (b^2=8)"] - 2.0 * (2 ** 0.5)) < 1e-6
assert abs(distinct_result["log (2^b=8)"] - 3.0) < 1e-9

# 7. log-table multiplication must match direct multiplication closely
lut_result = verify_log_table_multiplication(x1=2.5, x2=3.5, n_entries=100_000)
assert lut_result["relative_error"] < 1e-6

# 8. build_log_lookup_table / multiply_via_log_table used directly
xs, logs = build_log_lookup_table(1.0, 10.0, 50_000)
product = multiply_via_log_table(4.0, 2.0, xs, logs)
assert abs(product - 8.0) < 1e-4

# 9. build_log_lookup_table input validation
try:
    build_log_lookup_table(5.0, 1.0)
    raise AssertionError("expected ValueError for x_max <= x_min")
except ValueError:
    pass

# 10. LUT address bits: known powers of two, and the non-power-of-two case
assert lut_address_bits_for_depth(1) == 0
assert lut_address_bits_for_depth(16) == 4
assert lut_address_bits_for_depth(256) == 8
assert lut_address_bits_for_depth(1024) == 10
assert lut_address_bits_for_depth(1000) == 10   # not a power of 2 -> rounds up

# 11. quantization error bound: doubling address bits should ~quarter the error
err_8bit = lut_quantization_error_bound(input_bits=8, function_range=2.0)
err_9bit = lut_quantization_error_bound(input_bits=9, function_range=2.0)
assert abs(err_8bit / err_9bit - 2.0) < 1e-9   # one more bit halves the step

# 12. ln(1+r) Taylor coefficients: known values 0,1,-1/2,1/3,-1/4
coeffs = ln1pr_taylor_coefficients(order=4)
assert coeffs == [0, 1, sp.Rational(-1, 2), sp.Rational(1, 3), sp.Rational(-1, 4)]

# 13. Taylor-corrected doubling time must be closer to exact than first-order alone
for rate in (1, 6, 12):
    r = doubling_time_correction_check(rate)
    assert r["corrected_error"] < r["first_order_error"]
    assert r["corrected_error"] < 1e-2   # order-2 correction should be tight

# 14. invalid rate must raise
try:
    doubling_time_correction_check(0)
    raise AssertionError("expected ValueError for rate_pct<=0")
except ValueError:
    pass

print("all elementary_algebra tests passed")
