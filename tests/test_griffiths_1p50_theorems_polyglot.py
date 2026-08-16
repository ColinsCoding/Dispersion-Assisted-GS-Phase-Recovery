"""Test dgs/griffiths_1p50_theorems_polyglot.py: Griffiths Problem 1.50's
three fields (F1 purely solenoidal, F2 purely irrotational, F3 both),
the two generic curl-of-gradient/div-of-curl identities, direct
numerical instantiations of Theorem 1's and Theorem 2's (b)/(c) links,
and (if MATLAB is installed) the finite-difference cross-check + timing
benchmark."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import numpy as np
from dgs.griffiths_1p50_theorems_polyglot import (
    field_F1_properties, field_F2_properties, field_F3_properties,
    curl_of_gradient_is_zero_generic, divergence_of_curl_is_zero_generic,
    closed_loop_line_integral_F2, two_path_independence_F2,
    closed_cube_surface_flux_F1, two_surface_independence_F1,
    MATLAB_DEFAULT,
)

# 1. field_F1_properties: purely solenoidal (div=0), NOT irrotational
#    (curl != 0), vector potential verified
p1 = field_F1_properties()
assert p1["div_F1"] == 0
assert p1["curl_F1"] != 0   # F1 is NOT irrotational -- must be nonzero, not just "truthy"
assert p1["curl_A1_matches_F1"] is True

# 2. field_F2_properties: purely irrotational (curl=0), NOT solenoidal
#    (div != 0), scalar potential verified
p2 = field_F2_properties()
assert p2["div_F2"] == 3
assert p2["curl_F2_is_zero"] is True
assert p2["minus_grad_V2_matches_F2"] is True

# 3. field_F3_properties: BOTH (div=0 AND curl=0), both potentials verified
p3 = field_F3_properties()
assert p3["div_F3"] == 0
assert p3["curl_F3_is_zero"] is True
assert p3["grad_U3_matches_F3"] is True
assert p3["curl_A3_matches_F3"] is True

print("dgs.griffiths_1p50_theorems_polyglot: Problem 1.50 field checks passed")

# 4. The two generic identities: must hold for an UNDEFINED (sp.Function)
#    scalar/vector field, not just a specific example -- proven exactly
assert curl_of_gradient_is_zero_generic() is True
assert divergence_of_curl_is_zero_generic() is True

print("dgs.griffiths_1p50_theorems_polyglot: generic identity checks passed")

# 5. Theorem 1(c): closed-loop integral of F2 is ~0, checked on a
#    genuinely non-planar loop (not one that's trivially zero by planar
#    symmetry)
loop = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0), (0, 0, 0)]
circ = closed_loop_line_integral_F2(loop)
assert abs(circ) < 1e-9

# a loop that ISN'T closed should be rejected
try:
    closed_loop_line_integral_F2([(0, 0, 0), (1, 0, 0), (1, 1, 1)])
    raise AssertionError("expected ValueError for a non-closed loop")
except ValueError:
    pass

# 6. Theorem 1(b): two genuinely different paths between the same two
#    points must give the SAME line integral
paths = two_path_independence_F2((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0))
assert paths["abs_diff"] < 1e-9
assert abs(paths["integral_path_I"] - 1.5) < 1e-6   # known closed-form value (r^2/2 at (1,1,1) minus 0)

print("dgs.griffiths_1p50_theorems_polyglot: Theorem 1 (b)/(c) checks passed")

# 7. Theorem 2(c): closed-surface flux of F1 over the unit cube is ~0
flux = closed_cube_surface_flux_F1()
assert abs(flux) < 1e-9

try:
    closed_cube_surface_flux_F1(n_per_axis=1)
    raise AssertionError("expected ValueError for n_per_axis < 2")
except ValueError:
    pass

# 8. Theorem 2(b): two genuinely different open surfaces sharing the SAME
#    boundary loop must give the SAME flux (consistent "upward" orientation)
surfaces = two_surface_independence_F1()
assert surfaces["abs_diff"] < 1e-4   # grid-quadrature tolerance, not machine-epsilon
assert abs(surfaces["flat_surface_flux"] - 1 / 3) < 1e-3   # known closed form: int x^2 dx dy over unit square = 1/3

# a bigger bump should still agree (not a coincidence at one specific height)
surfaces_tall = two_surface_independence_F1(bump_height=1.5)
assert surfaces_tall["abs_diff"] < 1e-3

print("dgs.griffiths_1p50_theorems_polyglot: Theorem 2 (b)/(c) checks passed")

# 9. torch (py 3.12 only): cross-check div/curl for all three fields
try:
    import torch  # noqa: F401
    from dgs.griffiths_1p50_theorems_polyglot import torch_div_curl
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, size=(50, 3))

    t1 = torch_div_curl("F1", pts)
    assert np.max(np.abs(t1["divergence"])) < 1e-8

    t2 = torch_div_curl("F2", pts)
    assert np.max(np.abs(t2["curl"])) < 1e-8
    assert np.allclose(t2["divergence"], 3.0, atol=1e-8)   # div(F2)=3 exactly, everywhere

    t3 = torch_div_curl("F3", pts)
    assert np.max(np.abs(t3["divergence"])) < 1e-8
    assert np.max(np.abs(t3["curl"])) < 1e-8

    print("dgs.griffiths_1p50_theorems_polyglot: torch checks passed")
except ImportError:
    print("dgs.griffiths_1p50_theorems_polyglot: torch not available, skipped torch checks")

# 10. MATLAB (if installed): finite-difference div/curl must roughly match
#     the exact SymPy results at the same test point (1.3, -0.7, 2.1)
if os.path.exists(MATLAB_DEFAULT):
    import tempfile
    from dgs.griffiths_1p50_theorems_polyglot import run_matlab_1p50_check
    with tempfile.TemporaryDirectory() as tmp:
        matlab_result = run_matlab_1p50_check(tmp)

    # F1 = x^2*zhat at x=1.3: div=0, curl=(0,-2x,0)=(0,-2.6,0)
    assert abs(matlab_result["F1"]["divergence"]) < 1e-4
    assert np.allclose(matlab_result["F1"]["curl"], [0.0, -2.6, 0.0], atol=1e-4)

    # F2 = r: div=3, curl=0
    assert abs(matlab_result["F2"]["divergence"] - 3.0) < 1e-6
    assert np.max(np.abs(matlab_result["F2"]["curl"])) < 1e-6

    # F3: div=0, curl=0
    assert abs(matlab_result["F3"]["divergence"]) < 1e-4
    assert np.max(np.abs(matlab_result["F3"]["curl"])) < 1e-4

    print("dgs.griffiths_1p50_theorems_polyglot: MATLAB finite-difference checks passed")

    # 11. benchmark_solve_times (needs torch): must return positive times
    #     for all three tools, and MATLAB's process-launch overhead should
    #     make it the slowest by a wide margin (the actual, documented claim)
    try:
        import torch  # noqa: F401
        from dgs.griffiths_1p50_theorems_polyglot import benchmark_solve_times
        times = benchmark_solve_times(run_matlab=True)
        assert set(times.keys()) == {"sympy_seconds", "torch_seconds", "matlab_seconds"}
        assert all(v > 0 for v in times.values())
        assert times["matlab_seconds"] > times["sympy_seconds"]
        assert times["matlab_seconds"] > times["torch_seconds"]
        print("dgs.griffiths_1p50_theorems_polyglot: timing benchmark checks passed")
    except ImportError:
        print("dgs.griffiths_1p50_theorems_polyglot: torch not available, skipped timing benchmark")
else:
    print(f"dgs.griffiths_1p50_theorems_polyglot: MATLAB not found at {MATLAB_DEFAULT}, "
          f"skipped MATLAB and benchmark checks")

print("all dgs.griffiths_1p50_theorems_polyglot tests passed")
