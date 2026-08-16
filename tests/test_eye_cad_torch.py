"""Test dgs/eye_cad_torch.py: torch-differentiable eye geometry checked
against dgs.retinal_scan_imaging's numpy originals, the 6-rule
differentiable-CAD constraint solver, and the matplotlib CAD drawing.

torch is py-3.12-only in this repo; this test file is itself py-3.12-only
(no ImportError guard) since the whole module's point is torch-based
geometry -- run it under py -3.12."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs.eye_cad_torch import (
    torch_eye_focal_length_mm, torch_eye_power_diopters, torch_diffraction_spot_radius_um,
    verify_torch_matches_numpy, design_rule_residuals, solve_eye_design, draw_eye_cad,
    DEFAULT_R_BOUNDS_MM, DEFAULT_AXIAL_BOUNDS_MM, DEFAULT_N_BOUNDS,
)

# 1. torch geometry functions must match dgs.retinal_scan_imaging's numpy
#    originals exactly (same formulas, reimplemented for differentiability)
check = verify_torch_matches_numpy()
assert check["matches"] is True
assert check["focal_length_diff"] < 1e-9
assert check["power_diff"] < 1e-9
assert check["spot_radius_diff"] < 1e-9

# 2. torch functions are actually differentiable (grad flows through them --
#    the entire point of using torch instead of numpy here)
R = torch.tensor(5.55, dtype=torch.float64, requires_grad=True)
n = torch.tensor(1.336, dtype=torch.float64, requires_grad=True)
f = torch_eye_focal_length_mm(R, n)
f.backward()
assert R.grad is not None and abs(float(R.grad)) > 0
assert n.grad is not None and abs(float(n.grad)) > 0

# 3. design_rule_residuals: the DEFAULT textbook numbers must show a
#    nonzero emmetropia residual (the module's own documented claim that
#    R=5.55/n=1.336/axial=22.3 are not self-consistent under this rule)
R0 = torch.tensor(5.55, dtype=torch.float64)
axial0 = torch.tensor(22.3, dtype=torch.float64)
n0 = torch.tensor(1.336, dtype=torch.float64)
pupil0 = torch.tensor(4.0, dtype=torch.float64)
residuals0 = design_rule_residuals(R0, axial0, n0, pupil0)
assert float(residuals0["1_emmetropia"]) > 0.01, "expected a real emmetropia mismatch at textbook defaults"

# a self-consistent design (focal length forced to equal axial length)
# should give an exactly-zero emmetropia residual
f_matched = float(torch_eye_focal_length_mm(R0, n0))
axial_matched = torch.tensor(f_matched, dtype=torch.float64)
residuals_matched = design_rule_residuals(R0, axial_matched, n0, pupil0)
assert float(residuals_matched["1_emmetropia"]) < 1e-12

# 4. solve_eye_design: runs, returns a design, and improves emmetropia
#    relative to the (deliberately mismatched) starting point
design = solve_eye_design(n_steps=1500)
assert design["R_mm"] > 0 and design["axial_length_mm"] > 0
assert 1.0 < design["n_vitreous"] < 1.6
assert design["pupil_mm"] > 0
initial_mismatch = abs(22.0679 - 22.3)   # from the module's own documented starting mismatch
final_mismatch = abs(design["focal_length_mm"] - design["axial_length_mm"])
assert final_mismatch < initial_mismatch, f"solver did not improve emmetropia: {final_mismatch}"

# history must be non-empty and the recorded total_loss should generally
# decrease (checked via first vs. last, not monotonicity at every step --
# Adam is not strictly monotone)
history = design["history"]
assert len(history["total_loss"]) == 1500
assert history["total_loss"][-1] < history["total_loss"][0]

# satisfied dict has all 6 rules and each is genuinely a bool
assert set(design["satisfied"].keys()) == {
    "1_emmetropia", "2_target_power", "3_diffraction_vs_cone",
    "4_corneal_radius_bounds", "5_axial_length_bounds", "6_refractive_index_bounds"}
for v in design["satisfied"].values():
    assert isinstance(v, bool)

# 5. n_steps validation
try:
    solve_eye_design(n_steps=0)
    raise AssertionError("expected ValueError for n_steps=0")
except ValueError:
    pass

# 6. the "satisfied" check is honest about bound rules: a design forced
#    outside its bound must NOT be reported as satisfied even if the
#    solver were to (hypothetically) settle there -- checked directly by
#    calling solve_eye_design with a tiny step count so it barely moves
#    from a deliberately out-of-bounds start
design_oob = solve_eye_design(R0=3.0, n_steps=1)   # R0=3.0 is well outside DEFAULT_R_BOUNDS_MM
assert design_oob["satisfied"]["4_corneal_radius_bounds"] is False
assert not (DEFAULT_R_BOUNDS_MM[0] <= design_oob["R_mm"] <= DEFAULT_R_BOUNDS_MM[1])

print("dgs.eye_cad_torch: solver checks passed")

# 7. draw_eye_cad: must not raise, must return an Axes with content on it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
result_ax = draw_eye_cad(design, ax=ax)
assert result_ax is ax
assert len(ax.lines) > 0 or len(ax.patches) > 0
plt.close(fig)

print("dgs.eye_cad_torch: CAD drawing checks passed")
print("all dgs.eye_cad_torch tests passed")
