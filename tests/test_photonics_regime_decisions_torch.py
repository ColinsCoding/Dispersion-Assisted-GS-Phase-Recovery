"""Test dgs/photonics_regime_decisions_torch.py: Fresnel/TIR as a hard
torch.where branch (checked continuous at the critical angle) and the
far-field dispersive-Fourier validity threshold as a discontinuous
Piecewise with a smooth torch.sigmoid surrogate for gradient-based design.
Requires py -3.12 (torch is py-3.12 only in this repo, not 3.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import sympy as sp
import torch
from dgs.photonics_regime_decisions_torch import (
    critical_angle, fresnel_reflectance_TE_symbolic,
    verify_fresnel_continuous_at_critical_angle, fresnel_reflectance_TE_torch,
    verify_torch_fresnel, far_field_hard_piecewise_symbolic,
    verify_far_field_penalty_discontinuous, far_field_soft_gate_torch,
    verify_far_field_soft_gate,
)

# 1. critical_angle: known value, and validation
n1, n2 = 1.5, 1.0
theta_c = critical_angle(n1, n2)
assert abs(theta_c - math.asin(n2 / n1)) < 1e-12
for bad in [(-1.0, 1.0), (1.0, -1.0), (1.0, 1.5), (1.0, 1.0)]:
    try:
        critical_angle(*bad)
        raise AssertionError(f"expected ValueError for critical_angle{bad}")
    except ValueError:
        pass

# 2. fresnel_reflectance_TE_symbolic: normal-incidence value must match the
#    textbook R0=((n1-n2)/(n1+n2))^2 exactly. NOTE: the module's symbols
#    are declared positive=True -- a freshly-created sp.symbols("n_1") with
#    NO assumptions is a structurally DIFFERENT SymPy object with the same
#    name, and .subs() silently no-ops instead of substituting, so the
#    replacement symbols here must carry matching assumptions.
expr, R_ordinary = fresnel_reflectance_TE_symbolic()
theta_i_mod, n1_mod, n2_mod = sp.symbols("theta_i n_1 n_2", positive=True)
n1_s, n2_s = sp.Rational(3, 2), sp.Rational(1, 1)
R0 = R_ordinary.subs({n1_mod: n1_s, n2_mod: n2_s, theta_i_mod: 0})
R0_expected = ((n1_s - n2_s) / (n1_s + n2_s))**2
assert sp.simplify(R0 - R0_expected) == 0

# 3. verify_fresnel_continuous_at_critical_angle: must succeed for n1>n2
assert verify_fresnel_continuous_at_critical_angle(n1, n2) is True

# 4. fresnel_reflectance_TE_torch: input validation
for bad in [(-1.0, 1.0), (1.0, -1.0), (1.0, 1.5), (1.0, 1.0)]:
    try:
        fresnel_reflectance_TE_torch(torch.tensor([0.5]), *bad)
        raise AssertionError(f"expected ValueError for fresnel_reflectance_TE_torch{bad}")
    except ValueError:
        pass

# 5. fresnel_reflectance_TE_torch: R must be antisymmetric-free (i.e. in [0,1])
#    across a full angle sweep, and monotonically increasing beyond ~30deg
#    toward the critical angle (physically: reflectance rises as you approach
#    grazing/TIR)
thetas_sweep = torch.linspace(0.0, theta_c, 50, dtype=torch.float64)
R_sweep = fresnel_reflectance_TE_torch(thetas_sweep, n1, n2)
assert torch.all(R_sweep >= -1e-10) and torch.all(R_sweep <= 1.0 + 1e-10), \
    "reflectance must stay in [0, 1]"
assert torch.all(R_sweep[1:] >= R_sweep[:-1] - 1e-9), \
    "TE reflectance should be monotonically non-decreasing from normal incidence to theta_c"

# 6. verify_torch_fresnel: every check must pass
result = verify_torch_fresnel(n1, n2)
for name, ok in result["checks"].items():
    assert ok, f"verify_torch_fresnel check failed: {name}"

# 7. far_field_hard_piecewise_symbolic / verify_far_field_penalty_discontinuous
#    (same positive=True assumption-matching requirement as check 2 above)
penalty = far_field_hard_piecewise_symbolic()
L_mod, L_D_mod = sp.symbols("L L_D", positive=True)
assert penalty.subs({L_mod: 20, L_D_mod: 1}) == 0   # far field: no penalty
assert penalty.subs({L_mod: 5, L_D_mod: 1}) == 1    # near field: full penalty
assert verify_far_field_penalty_discontinuous() is True

# 8. far_field_soft_gate_torch: validation
try:
    far_field_soft_gate_torch(torch.tensor(1.0), torch.tensor(1.0), steepness=0.0)
    raise AssertionError("expected ValueError for steepness <= 0")
except ValueError:
    pass

# 9. far_field_soft_gate_torch: must approximate the hard threshold's
#    ENDPOINTS (far from the boundary) while differing from it exactly AT
#    the boundary (0.5 vs. the hard function's 0-or-1) -- that gap IS the
#    point of using a soft surrogate
L_D = torch.tensor(1.0, dtype=torch.float64)
gate_far = far_field_soft_gate_torch(torch.tensor(1000.0, dtype=torch.float64), L_D)
gate_near = far_field_soft_gate_torch(torch.tensor(0.001, dtype=torch.float64), L_D)
gate_at_boundary = far_field_soft_gate_torch(torch.tensor(10.0, dtype=torch.float64), L_D)
assert gate_far.item() > 0.999
assert gate_near.item() < 0.001
assert abs(gate_at_boundary.item() - 0.5) < 1e-9

# 10. verify_far_field_soft_gate: every check must pass
result2 = verify_far_field_soft_gate()
for name, ok in result2["checks"].items():
    assert ok, f"verify_far_field_soft_gate check failed: {name}"

print("all dgs.photonics_regime_decisions_torch tests passed")
