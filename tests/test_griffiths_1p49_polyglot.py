"""Test dgs/griffiths_1p49_polyglot.py: Griffiths Problem 1.49's
by-parts decomposition (symbolic, and cross-checked numerically against
independent MATLAB quadrature), and -- where torch (py 3.12) is available
-- the autograd-exact off-origin divergence check and radial-quadrature
volume term.

torch is py-3.12-only in this repo, and MATLAB requires the installed
binary at C:\\Program Files\\MATLAB\\R2025b; both are skipped gracefully
here if unavailable so this test still runs on py-3.13 / CI without them,
while running the analytic/SymPy checks unconditionally."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import numpy as np
from dgs.griffiths_1p49_polyglot import (
    by_parts_terms_symbolic, by_parts_volume_term_analytic,
    by_parts_surface_term_analytic, J_analytic,
)

# 1. by_parts_terms_symbolic: symbolic total must be EXACTLY 4*pi
import sympy as sp
sym = by_parts_terms_symbolic()
assert sp.simplify(sym["total"] - 4 * sp.pi) == 0

# 2. by_parts_volume_term_analytic / surface_term_analytic: known limits
#    R -> large: volume_term -> 4*pi, surface_term -> 0
assert abs(by_parts_volume_term_analytic(50.0) - 4 * math.pi) < 1e-15
assert by_parts_surface_term_analytic(50.0) < 1e-15
#    R -> small: volume_term -> 0, surface_term -> 4*pi
assert by_parts_volume_term_analytic(1e-6) < 2e-5
assert abs(by_parts_surface_term_analytic(1e-6) - 4 * math.pi) < 2e-5

for bad in (0.0, -1.0):
    for fn in (by_parts_volume_term_analytic, by_parts_surface_term_analytic):
        try:
            fn(bad)
            raise AssertionError(f"expected ValueError for R={bad}")
        except ValueError:
            pass

# 3. J_analytic: EXACTLY 4*pi for every R tested -- the core textbook result
for R in (0.01, 0.5, 1.0, 2.0, 5.0, 100.0):
    J = J_analytic(R)
    assert abs(J - 4 * math.pi) < 1e-12, f"J({R})={J}, expected 4*pi"

print("dgs.griffiths_1p49_polyglot: analytic/SymPy checks passed")

# 4. torch (py 3.12 only): off-origin divergence ~0, radial quadrature matches
try:
    import torch  # noqa: F401
    from dgs.griffiths_1p49_polyglot import (
        torch_divergence_of_A_off_origin, torch_radial_quadrature_volume_term,
        torch_gradient_of_exp_minus_r,
    )

    rng = np.random.default_rng(1)
    pts = rng.uniform(-10, 10, size=(500, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 1e-2]
    div = torch_divergence_of_A_off_origin(pts)
    assert np.max(np.abs(div)) < 1e-8, f"divergence off-origin should be ~0, got max {np.max(np.abs(div))}"

    for R in (0.5, 2.0, 5.0):
        vol_torch = torch_radial_quadrature_volume_term(R, n_points=200_000)
        vol_analytic = by_parts_volume_term_analytic(R)
        assert abs(vol_torch - vol_analytic) < 1e-6, f"R={R}: torch={vol_torch}, analytic={vol_analytic}"

    grad_f = torch_gradient_of_exp_minus_r(pts)
    r = np.linalg.norm(pts, axis=1)
    r_hat = pts / r[:, None]
    grad_f_closed = -np.exp(-r)[:, None] * r_hat
    assert np.max(np.abs(grad_f - grad_f_closed)) < 1e-10

    print("dgs.griffiths_1p49_polyglot: torch checks passed")
except ImportError:
    print("dgs.griffiths_1p49_polyglot: torch not available, skipped torch checks")

# 5. MATLAB (if installed): run_matlab_by_parts must match analytic to ~1e-8
import os
matlab_path = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
if os.path.exists(matlab_path):
    import tempfile
    from dgs.griffiths_1p49_polyglot import run_matlab_by_parts
    R_values = (0.5, 2.0, 5.0)
    with tempfile.TemporaryDirectory() as tmp:
        rows = run_matlab_by_parts(tmp, R_values, matlab_path=matlab_path)
    assert len(rows) == len(R_values)
    for row, R in zip(rows, R_values):
        assert abs(row["total"] - 4 * math.pi) < 1e-6, f"R={R}: matlab total={row['total']}"
    print("dgs.griffiths_1p49_polyglot: MATLAB checks passed")
else:
    print("dgs.griffiths_1p49_polyglot: MATLAB not found, skipped MATLAB checks")

print("all dgs.griffiths_1p49_polyglot tests passed")
