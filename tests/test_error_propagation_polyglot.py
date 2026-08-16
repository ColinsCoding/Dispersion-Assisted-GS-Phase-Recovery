"""Test dgs/error_propagation_polyglot.py: the same emf = B*h*v uncertainty
propagation computed by Python (Measurement, propagate(), product_rule()),
C (explicit function calls -- no operator overloading available), and C++
(operator overloading), cross-checked to near machine precision. Requires
gcc/g++ on PATH (C:\\msys64\\mingw64\\bin)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tempfile
from dgs.error_propagation_polyglot import (
    cross_validate_languages, compile_c, compile_cpp, _run_exe,
)
from dgs.error_propagation import Measurement

with tempfile.TemporaryDirectory() as tmp:
    results = cross_validate_languages(tmp)

# 1. every implementation must be present
expected_keys = {"python_measurement", "python_propagate_linear",
                  "python_product_rule_closed_form", "c_explicit_function_calls",
                  "cpp_operator_overloading", "max_abs_diff_across_all_implementations"}
assert expected_keys <= set(results.keys()), f"missing keys: {expected_keys - set(results.keys())}"

# 2. all five implementations must agree to near machine precision -- the
#    actual claim, not an assumption that a faithful C/C++ port would agree
max_diff = results["max_abs_diff_across_all_implementations"]
assert max_diff < 1e-9, f"implementations disagree by {max_diff}, expected < 1e-9"

# 3. sanity: the known closed-form answer for B=0.5+/-0.01, h=2.0+/-0.05,
#    v=3.0+/-0.1 -- emf=3.0 exactly, sigma from relative errors in quadrature
#    (2%, 2.5%, 3.333...%) times |emf|
val, sig = results["python_measurement"]
assert abs(val - 3.0) < 1e-12
rel = ((0.01/0.5)**2 + (0.05/2.0)**2 + (0.1/3.0)**2) ** 0.5
expected_sigma = 3.0 * rel
assert abs(sig - expected_sigma) < 1e-9

# 4. run_c=False / run_cpp=False: must skip that language cleanly, and the
#    remaining implementations must still agree with each other
with tempfile.TemporaryDirectory() as tmp:
    py_only = cross_validate_languages(tmp, run_c=False, run_cpp=False)
assert "c_explicit_function_calls" not in py_only
assert "cpp_operator_overloading" not in py_only
assert py_only["max_abs_diff_across_all_implementations"] < 1e-9

# 5. changing the inputs must change the propagated sigma predictably (a
#    LARGER relative uncertainty on any single input must not DECREASE the
#    total propagated uncertainty, since sigmas add in quadrature)
with tempfile.TemporaryDirectory() as tmp:
    baseline = cross_validate_languages(tmp, B=(0.5, 0.01), h=(2.0, 0.05), v=(3.0, 0.1))
with tempfile.TemporaryDirectory() as tmp:
    larger_v_sigma = cross_validate_languages(tmp, B=(0.5, 0.01), h=(2.0, 0.05), v=(3.0, 0.5))
_, sig_baseline = baseline["python_measurement"]
_, sig_larger = larger_v_sigma["python_measurement"]
assert sig_larger > sig_baseline, "increasing an input's sigma must increase the propagated sigma"

# 6. compile_c / compile_cpp / _run_exe: exercised directly (not just via
#    cross_validate_languages) to confirm each stage works in isolation
with tempfile.TemporaryDirectory() as tmp:
    exe_c = compile_c(tmp)
    exe_cpp = compile_cpp(tmp)
    B_m, h_m, v_m = Measurement(2.0, 0.1), Measurement(3.0, 0.2), Measurement(4.0, 0.3)
    val_c, sig_c = _run_exe(exe_c, B_m, h_m, v_m)
    val_cpp, sig_cpp = _run_exe(exe_cpp, B_m, h_m, v_m)
    expected = B_m * h_m * v_m
    assert abs(val_c - expected.value) < 1e-9 and abs(sig_c - expected.sigma) < 1e-9
    assert abs(val_cpp - expected.value) < 1e-9 and abs(sig_cpp - expected.sigma) < 1e-9

print("all dgs.error_propagation_polyglot tests passed")
