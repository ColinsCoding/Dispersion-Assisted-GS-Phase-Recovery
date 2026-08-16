"""Test dgs/vector3_polyglot.py: parallel/perpendicular vector decomposition
computed in Python/numpy, C (explicit vec3_dot/add/sub/scale functions),
and C++ (operator+, operator-, operator*(scalar) overloaded; dot() stays a
named method deliberately), cross-checked to near machine precision, plus
the physical validity checks (reconstruction, orthogonality, trig
identities). Requires gcc/g++ on PATH (C:\\msys64\\mingw64\\bin)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import tempfile
import numpy as np
from dgs.vector3_polyglot import (
    parallel_perp_numpy, cross_validate_languages, verify_decomposition,
    compile_c, compile_cpp, _run_exe,
)

# 1. parallel_perp_numpy: known case, v=(3,4,0) relative to a=(1,0,0) --
#    v_par should be exactly the x-component, v_perp the y-component
v_par, v_perp = parallel_perp_numpy((3.0, 4.0, 0.0), (1.0, 0.0, 0.0))
assert np.allclose(v_par, [3.0, 0.0, 0.0])
assert np.allclose(v_perp, [0.0, 4.0, 0.0])

# 2. parallel_perp_numpy: zero direction vector must raise
try:
    parallel_perp_numpy((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    raise AssertionError("expected ValueError for zero-vector direction")
except ValueError:
    pass

# 3. parallel_perp_numpy: v parallel to a already -> v_perp should be ~0
v_par2, v_perp2 = parallel_perp_numpy((5.0, 0.0, 0.0), (2.0, 0.0, 0.0))
assert np.allclose(v_par2, [5.0, 0.0, 0.0], atol=1e-12)
assert np.allclose(v_perp2, [0.0, 0.0, 0.0], atol=1e-12)

# 4. parallel_perp_numpy: v perpendicular to a already -> v_par should be ~0
v_par3, v_perp3 = parallel_perp_numpy((0.0, 7.0, 0.0), (1.0, 0.0, 0.0))
assert np.allclose(v_par3, [0.0, 0.0, 0.0], atol=1e-12)
assert np.allclose(v_perp3, [0.0, 7.0, 0.0], atol=1e-12)

# 5. verify_decomposition: every check True for a valid decomposition, and
#    the reported error terms are all near zero
checks = verify_decomposition((3.0, 4.0, 0.0), (1.0, 0.0, 0.0), v_par, v_perp)
for name in ("reconstructs_v", "orthogonal", "matches_cos_theta_identity", "matches_sin_theta_identity"):
    assert checks[name] is True or checks[name] == np.True_, f"{name} failed: {checks}"

# 6. verify_decomposition: must actually CATCH a broken decomposition, not
#    just rubber-stamp -- feed it a v_par/v_perp pair that doesn't reconstruct v
bad_par, bad_perp = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])  # doesn't sum to (3,4,0)
bad_checks = verify_decomposition((3.0, 4.0, 0.0), (1.0, 0.0, 0.0), bad_par, bad_perp)
assert bad_checks["reconstructs_v"] is False

# 7. cross_validate_languages: full three-language cross-check on a
#    non-axis-aligned, non-trivial case
v, a = (1.7, -3.2, 5.9), (2.1, 0.4, -1.3)
with tempfile.TemporaryDirectory() as tmp:
    results = cross_validate_languages(tmp, v=v, a=a)
max_diff = results.pop("max_abs_diff_across_all_implementations")
assert max_diff < 1e-9, f"implementations disagree by {max_diff}, expected < 1e-9"
assert set(results.keys()) == {"python_numpy", "c_explicit_functions", "cpp_operator_overloading"}

# 8. the C and C++ results, independently, must ALSO pass the physical checks
#    (not just "match Python" -- match Python AND be a valid decomposition)
for name in ("c_explicit_functions", "cpp_operator_overloading"):
    par, perp = results[name]
    c = verify_decomposition(v, a, par, perp)
    assert c["reconstructs_v"] and c["orthogonal"], f"{name} failed physical checks: {c}"

# 9. run_c=False / run_cpp=False: must skip cleanly
with tempfile.TemporaryDirectory() as tmp:
    py_only = cross_validate_languages(tmp, v=v, a=a, run_c=False, run_cpp=False)
assert "c_explicit_functions" not in py_only
assert "cpp_operator_overloading" not in py_only
assert py_only["max_abs_diff_across_all_implementations"] < 1e-12

# 10. compile_c / compile_cpp / _run_exe exercised directly
with tempfile.TemporaryDirectory() as tmp:
    exe_c = compile_c(tmp)
    exe_cpp = compile_cpp(tmp)
    par_c, perp_c = _run_exe(exe_c, (3.0, 4.0, 0.0), (1.0, 0.0, 0.0))
    par_cpp, perp_cpp = _run_exe(exe_cpp, (3.0, 4.0, 0.0), (1.0, 0.0, 0.0))
    assert np.allclose(par_c, [3.0, 0.0, 0.0]) and np.allclose(perp_c, [0.0, 4.0, 0.0])
    assert np.allclose(par_cpp, [3.0, 0.0, 0.0]) and np.allclose(perp_cpp, [0.0, 4.0, 0.0])

print("all dgs.vector3_polyglot tests passed")
