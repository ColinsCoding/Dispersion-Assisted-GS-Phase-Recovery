"""Test the torch differentiable truss FEM solver: cross-validated against
dgs.statics.three_bar_truss() (same geometry/loads, different method), a
closed-form single-bar sanity check, EA-independence of member forces for a
determinate truss, and autograd sensitivity against the analytic derivative.

torch is py-3.12 ONLY in this repo -- run with `py -3.12 -m pytest tests/test_truss_fem.py`.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from dgs.torch.truss_fem import solve_truss_fem, bridge_load_sweep, export_trajectory_json
from dgs.statics import three_bar_truss

dtype = torch.float64

# Same geometry/supports/load as dgs.statics.three_bar_truss(): A=(0,0), B=(2,0),
# C=(1,1.5); pin at A, roller at B; 10 kN down at apex C.
COORDS = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]], dtype=dtype)
MEMBERS = [(0, 2), (1, 2), (0, 1)]   # A-C, B-C, A-B
SUPPORTS = {0: "pin", 1: "roller"}
LOADS = {2: (0.0, -10000.0)}

# 1. member forces must match dgs.statics.three_bar_truss()'s independent
#    method-of-joints solution to near machine precision
result = solve_truss_fem(COORDS, MEMBERS, torch.tensor([2e5, 2e5, 2e5], dtype=dtype),
                          SUPPORTS, LOADS)
reference = three_bar_truss()["member_forces_N"]
computed = {"A-C": result["member_forces"][0].item(),
            "B-C": result["member_forces"][1].item(),
            "A-B": result["member_forces"][2].item()}
for key in reference:
    assert abs(computed[key] - reference[key]) < 1e-6, (key, computed[key], reference[key])

# 2. member forces of a statically DETERMINATE truss must be independent of EA
#    (only displacements depend on stiffness) -- try a completely different EA
result_diff_EA = solve_truss_fem(COORDS, MEMBERS, torch.tensor([5e4, 3e5, 1e6], dtype=dtype),
                                  SUPPORTS, LOADS)
assert torch.allclose(result["member_forces"], result_diff_EA["member_forces"], atol=1e-6)
# but displacements MUST differ (different stiffness -> different deflection)
assert not torch.allclose(result["displacements"], result_diff_EA["displacements"], atol=1e-9)

# 3. closed-form check: single horizontal bar, one end pinned, axial load F at
#    the free end -> displacement u = F*L/(EA) exactly (Hooke's law)
coords_bar = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=dtype)
members_bar = [(0, 1)]
EA_bar = torch.tensor([1.5e5], dtype=dtype, requires_grad=True)
F_applied = 1000.0
bar_result = solve_truss_fem(coords_bar, members_bar, EA_bar, {0: "pin", 1: "roller"},
                              {1: (F_applied, 0.0)})
u_expected = F_applied * 3.0 / EA_bar.item()
assert abs(bar_result["displacements"][1, 0].item() - u_expected) < 1e-9
assert abs(bar_result["member_forces"][0].item() - F_applied) < 1e-6

# 4. autograd sensitivity matches the analytic derivative of u=FL/(EA):
#    du/d(EA) = -F*L/(EA^2)
u_x = bar_result["displacements"][1, 0]
u_x.backward()
du_dEA_analytic = -F_applied * 3.0 / EA_bar.item() ** 2
assert abs(EA_bar.grad.item() - du_dEA_analytic) < 1e-6, (EA_bar.grad.item(), du_dEA_analytic)

# 5. bridge_load_sweep: zero load at step 0 -> zero displacement everywhere;
#    displacement grows monotonically (in magnitude) as load ramps up
EA3 = torch.tensor([2e5, 2e5, 2e5], dtype=dtype)
sweep = bridge_load_sweep(COORDS, MEMBERS, EA3, SUPPORTS, load_joint=2,
                           load_target=-10000.0, n_steps=5)
assert len(sweep) == 6
assert torch.allclose(sweep[0]["displacements"], torch.zeros_like(sweep[0]["displacements"]), atol=1e-12)
apex_y = [step["displacements"][2, 1].item() for step in sweep]
assert all(apex_y[i] < apex_y[i - 1] for i in range(1, len(apex_y)))  # increasingly negative (sagging)

# 6. n_steps validation
try:
    bridge_load_sweep(COORDS, MEMBERS, EA3, SUPPORTS, load_joint=2, load_target=-1.0, n_steps=0)
    assert False, "should reject n_steps < 1"
except ValueError:
    pass

# 7. export_trajectory_json writes valid JSON with the right structure
import json
import tempfile
import os

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "traj.json")
    export_trajectory_json(path, COORDS, MEMBERS, sweep, deflection_scale=50.0)
    with open(path) as f:
        data = json.load(f)
    assert len(data["steps"]) == 6
    assert data["deflection_scale"] == 50.0
    assert len(data["steps"][0]["joint_positions"]) == 3
    assert len(data["steps"][0]["member_forces"]) == 3
    # step 0 (no load) should be exactly the undeformed geometry
    assert data["steps"][0]["joint_positions"] == data["undeformed_joint_positions"]

print("all dgs.torch.truss_fem tests passed")
