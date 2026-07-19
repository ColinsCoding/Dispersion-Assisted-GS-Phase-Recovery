"""Test the Quake III fast inverse square root (Q_rsqrt), actually
compiled with gcc and run, cross-checked against an independent Python
reimplementation of the same bit-hack, and both checked against the
exact 1/sqrt(x). Requires mingw64 on PATH -- run via PowerShell."""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import famous_c_code as fcc

test_values = [1.0, 2.0, 4.0, 10.0, 100.0, 0.5, 3.14159, 1000.0]

with tempfile.TemporaryDirectory() as tmp:
    exe = fcc.compile_rsqrt(tmp)
    c_results = fcc.run_rsqrt_c(exe, test_values)

# 1. the real compiled C program's Q_rsqrt matches the independent Python
#    bit-hack reimplementation almost exactly (both do the SAME bit trick,
#    implemented two completely different ways)
for x, c_val in zip(test_values, c_results):
    py_val = fcc.python_bit_hack_rsqrt(x)
    assert abs(c_val - py_val) < 1e-5, f"x={x}: C={c_val}, python={py_val}"

# 2. both are within the well-documented ~0.2% accuracy of one Newton
#    iteration after the magic-number guess -- not exact, but bounded
for x, c_val in zip(test_values, c_results):
    exact = fcc.python_reference_rsqrt(x)
    rel_err = abs(c_val - exact) / exact
    assert rel_err < 0.002, f"x={x}: rel_err={rel_err} exceeds the known ~0.2% bound"

# 3. it's a genuine APPROXIMATION, not secretly exact -- error should be
#    bounded away from zero for most inputs (confirms the test isn't
#    trivially passing because the algorithm is actually perfect)
errors = [abs(fcc.python_bit_hack_rsqrt(x) - fcc.python_reference_rsqrt(x)) / fcc.python_reference_rsqrt(x)
          for x in test_values]
assert max(errors) > 1e-5   # genuinely nonzero error somewhere

# 4. scaling sanity: Q_rsqrt(4*x) should be close to Q_rsqrt(x)/2
#    (1/sqrt(4x) = 1/(2*sqrt(x)) exactly; the approximation should preserve
#    this relationship reasonably well since it's a property of the true function)
r1 = fcc.python_bit_hack_rsqrt(4.0)
r2 = fcc.python_bit_hack_rsqrt(16.0)
assert abs(r2 - r1/2) / (r1/2) < 0.01

# 5. input validation
for bad_call in [
    lambda: fcc.python_reference_rsqrt(-1.0),
    lambda: fcc.python_reference_rsqrt(0.0),
    lambda: fcc.python_bit_hack_rsqrt(-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.famous_c_code tests passed")
