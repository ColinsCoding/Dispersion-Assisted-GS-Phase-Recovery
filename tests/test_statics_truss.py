"""Test dgs/statics_truss.py: static determinacy, the method-of-joints
linear solver (checked by recomputing equilibrium residuals, not trusted
from the solve alone), the zero-force-member truss's classic result, and
an independent method-of-sections cross-check that caught a real sign
convention bug during development."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.statics_truss import (
    static_determinacy_check, solve_method_of_joints, verify_equilibrium_residuals,
    classify_forces, triangle_truss_example, zero_force_member_truss_example,
    method_of_sections_DC,
)

# 1. static_determinacy_check: determinate, understatic, overstatic cases
det = static_determinacy_check(3, 3, 3)
assert det["determinate"] is True and det["status"] == "determinate"

under = static_determinacy_check(3, 2, 3)
assert under["determinate"] is False and "understatic" in under["status"]

over = static_determinacy_check(4, 3, 3)
assert over["determinate"] is False and "overstatic" in over["status"]

for bad in [(-1, 3, 3), (3, -1, 3), (3, 3, 0)]:
    try:
        static_determinacy_check(*bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.statics_truss: determinacy checks passed")

# 2. solve_method_of_joints: rejects a non-determinate structure up front
joints = {"A": (0.0, 0.0), "B": (4.0, 0.0), "C": (2.0, 3.0)}
members_understatic = [("A", "B"), ("A", "C")]   # only 2 members -- a mechanism
try:
    solve_method_of_joints(joints, members_understatic, {"A": "pin", "B": "roller_y"}, {"C": (0.0, -1000.0)})
    raise AssertionError("expected ValueError for an understatic truss")
except ValueError:
    pass

# 3. triangle_truss_example: matches the module's own hand-verified closed form
tri = triangle_truss_example(P=1000.0)
assert abs(tri["member_forces"]["A-B"] - 1000.0 / 3) < 1e-6
# closed form: at apex C, symmetry gives F_AC=F_BC; y-equilibrium gives
# F_AC = -P / (2*(3/sqrt(13))) = -P*sqrt(13)/6
expected_diag = -1000.0 * np.sqrt(13) / 6
assert abs(tri["member_forces"]["A-C"] - expected_diag) < 1e-4
assert abs(tri["member_forces"]["B-C"] - expected_diag) < 1e-4
assert abs(tri["reactions"]["Ry@A"] - 500.0) < 1e-6
assert abs(tri["reactions"]["Ry@B"] - 500.0) < 1e-6
assert abs(tri["reactions"]["Rx@A"]) < 1e-9

eq = verify_equilibrium_residuals(tri["joints"], tri["members"], tri["member_forces"],
                                  tri["reactions"], tri["loads"])
assert eq["equilibrium_holds"] is True
assert eq["max_residual"] < 1e-6

cls = classify_forces(tri["member_forces"])
assert cls["A-B"].startswith("tension")
assert cls["A-C"].startswith("compression")
assert cls["B-C"].startswith("compression")

print("dgs.statics_truss: triangle truss checks passed")

# 4. zero_force_member_truss_example: F_BD must be EXACTLY zero (to
#    numerical precision) -- the module's headline structural result
zfm = zero_force_member_truss_example(P=1000.0)
assert abs(zfm["member_forces"]["B-D"]) < 1e-9, "expected a true zero-force member"
assert abs(zfm["member_forces"]["A-B"] - 2000.0 / 3) < 1e-6
assert abs(zfm["member_forces"]["B-C"] - 2000.0 / 3) < 1e-6
assert abs(zfm["member_forces"]["A-D"] - (-2500.0 / 3)) < 1e-6
assert abs(zfm["member_forces"]["D-C"] - (-2500.0 / 3)) < 1e-6

eq2 = verify_equilibrium_residuals(zfm["joints"], zfm["members"], zfm["member_forces"],
                                   zfm["reactions"], zfm["loads"])
assert eq2["equilibrium_holds"] is True

cls2 = classify_forces(zfm["member_forces"])
assert cls2["B-D"] == "zero-force member"

# a DIFFERENT load magnitude must still give F_BD=0 -- confirms it's not
# a coincidence at P=1000 specifically
for P_test in (250.0, 5000.0, 1.0):
    zfm_scaled = zero_force_member_truss_example(P=P_test)
    assert abs(zfm_scaled["member_forces"]["B-D"]) < 1e-9, f"P={P_test}: F_BD should still be 0"

print("dgs.statics_truss: zero-force-member truss checks passed")

# 5. method_of_sections_DC: must agree with the method-of-joints result
#    for F_DC (this cross-check is what caught a real sign-convention bug
#    during development -- keeping the test strict on both magnitude AND sign)
F_DC_sections = method_of_sections_DC(zfm["joints"], zfm["reactions"], zfm["loads"], P_ref=1000.0)
F_DC_joints = zfm["member_forces"]["D-C"]
assert abs(F_DC_sections - F_DC_joints) < 1e-6, (
    f"method of sections ({F_DC_sections}) disagrees with method of joints ({F_DC_joints})")

print("dgs.statics_truss: method-of-sections cross-check passed")
print("all dgs.statics_truss tests passed")
