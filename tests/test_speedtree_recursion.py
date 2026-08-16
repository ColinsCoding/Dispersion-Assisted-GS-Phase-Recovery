"""Test the recursively-generated (SpeedTree-style) MuJoCo tree: the
recursion actually produces the expected number of branches for a given
depth/branching factor, the model loads and is stable under gravity alone
(a real bug caught here: the original stiffness guess let gravity alone
droop the tree by 0.93m -- almost its whole height -- before tuning),
the deepest branch tip settles rather than exploding, and it genuinely
oscillates once wind is applied rather than just leaning to a new static
position."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.speedtree_recursion import build_tree_model, simulate_tree_sway

# 1. recursion produces the expected branch count: a full n_children-ary
# tree of depth d (root at depth 0) has (n_children^(d+1) - 1)/(n_children - 1) branches
for max_depth, n_children in [(3, 2), (4, 2), (3, 3)]:
    model, n_branches = build_tree_model(max_depth=max_depth, n_children=n_children)
    expected = (n_children**(max_depth + 1) - 1) // (n_children - 1)
    assert n_branches == expected, f"depth={max_depth} n_children={n_children}: expected {expected}, got {n_branches}"
    assert model.nbody == n_branches + 1   # +1 for worldbody

# 2. gravity alone (no wind) does not cause the tree to collapse -- the
# specific regression caught during tuning (stiffness=0.015 let it droop
# by 0.93m; the fixed default keeps this small)
result_gravity_only = simulate_tree_sway(t_max=1.5, wind_amplitude=0.0, settle_time=999.0)
assert result_gravity_only["any_nan"] is False
assert result_gravity_only["tip_excursion"].max() < 0.15, (
    f"gravity-only sag should be small for a properly-stiff tree, got "
    f"{result_gravity_only['tip_excursion'].max():.4f} m")

# 3. the full sway simulation (settle, then oscillating wind) stays stable
# and genuinely oscillates -- not frozen (some excursion) and not exploding
# (bounded excursion, small final velocity)
result = simulate_tree_sway(t_max=3.0)
assert result["any_nan"] is False
assert result["final_max_qvel"] < 1.0
assert result["tip_excursion"].max() > 0.01     # genuinely moves under wind
assert result["tip_excursion"].max() < 0.5      # but doesn't fly apart

# 4. the tracked tip's excursion is NOT monotonic (rules out "it just leaned
# once and stopped" -- confirms actual back-and-forth oscillation)
post_settle = result["tip_excursion"][int(0.6 / 0.001):]
n_local_maxima = np.sum((post_settle[1:-1] > post_settle[:-2]) & (post_settle[1:-1] > post_settle[2:]))
assert n_local_maxima >= 1, "expected at least one oscillation peak after the wind starts"

print("all dgs.speedtree_recursion tests passed")
