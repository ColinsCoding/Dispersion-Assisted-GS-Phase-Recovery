"""Test the driven fold-then-reverse pizza box: the hinge profile
actually reaches 90deg and returns to 0deg (not just monotonically rising
or stuck), the simulation stays numerically stable while the lid sweeps
through the deformable mesh, and the sweep genuinely lifts pizza material
well above resting height during the fold -- the checkable definition of
"the pizza folded" used here, distinct from dgs.pizza_in_box_90deg's
oversized-floor mechanism."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.pizza_box_fold_reverse import hinge_profile, simulate_fold_and_reverse

# 1. hinge_profile actually goes 0 -> 90 -> 0, not stuck or monotonic
assert hinge_profile(0.0) == 0.0
assert abs(hinge_profile(1.3) - 90.0) < 1.0     # near peak during the hold
assert hinge_profile(4.5) == 0.0                # back down by the end

# 2. the full driven simulation stays numerically stable -- a real risk
# for a mass-spring mesh being actively swept by a rotating rigid panel
result = simulate_fold_and_reverse(t_max=4.5)
assert result["any_nan"] is False

# 3. the actual point of this module: the lid's sweep lifts pizza material
# well above where it rested flat (panel top is ~0.016m; resting drop
# height before any motion was ~0.031m) -- a clear climb during the fold,
# not just noise
assert result["peak_fold_max_z"] > 0.04

# 4. by the end (hinge back at 0, "put on the ground"), the mesh has
# settled back down close to the flat panel height -- much lower than the
# peak reached mid-fold, confirming the reverse rotation actually released
# it rather than leaving it stuck up in the air
assert result["final_mean_z"] < 0.5 * result["peak_fold_max_z"]

print("all dgs.pizza_box_fold_reverse tests passed")
