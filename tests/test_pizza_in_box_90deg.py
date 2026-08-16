"""Test the deformable full-circle pizza settling into a static 90-degree
box corner: the mesh generates with no out-of-range face indices, the
simulation stays numerically stable, the pizza actually descends and
settles (not falls forever), and -- the real point of the sizing choice
(floor_depth < pizza_radius) -- the overhang half genuinely climbs the
vertical wall instead of just lying flat, which is what a badly-sized
scene (the first attempt, caught by exactly this kind of check) would
produce."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.pizza_in_box_90deg import (
    _full_circle_web_points, _full_circle_web_elements,
    build_pizza_in_box_model, simulate_pizza_settling,
)

# 1. mesh generation: point/element counts consistent, no out-of-range
# face index (the exact bug class caught earlier in pizza_to_calzone.py)
n_rings, n_spokes = 4, 16
points = _full_circle_web_points(radius=0.12, n_rings=n_rings, n_spokes=n_spokes)
elems = _full_circle_web_elements(n_rings, n_spokes)
assert len(points) == 1 + n_rings * n_spokes
assert len(elems) == n_spokes + 2 * n_spokes * (n_rings - 1)
assert max(max(e) for e in elems) < len(points)

# 2. the model loads with a single flex (one whole pizza, not two halves)
model = build_pizza_in_box_model()
assert model.nflex == 1

# 3. the full settling simulation stays numerically stable
result = simulate_pizza_settling(t_max=4.0)
assert result["any_nan"] is False

# 4. it actually falls and roughly settles: mean height drops sharply
# from the drop height, and by the second half of the run it's no longer
# still falling -- checked via the mean-height trend itself rather than
# instantaneous qvel, since a mass-spring mesh draped against a corner
# keeps small residual jitter indefinitely (verified directly: max|qvel|
# oscillates between ~0.02 and ~0.2 rad/s at t=1..5s, never cleanly
# settling below a fixed threshold, so a strict "settled" flag on qvel
# alone is too noisy a signal here)
assert result["mean_z"][0] > result["mean_z"][-1]
half = len(result["mean_z"]) // 2
late_drift = abs(result["mean_z"][-1] - result["mean_z"][half])
early_drop = result["mean_z"][0] - result["mean_z"][half]
assert late_drift < early_drop

# 5. no vertex ends up more than a small numerical margin below the
# ground -- a real failure mode if flex-vs-rigid contact margins are off
assert result["final_min_z"] > -0.02

# 6. the actual point of this module: the overhang half (rest y>0, no
# floor beneath it by construction since floor_depth < pizza_radius)
# ends up well above bare floor height (box_thickness=0.008), i.e. it
# genuinely drapes/climbs the wall rather than lying flat -- flat would
# mean the sizing failed to force any real bend (the bug in the first
# attempt at this scene, where the wall was too far away to ever be
# reached)
assert result["final_overhang_max_z"] > 0.03

print("all dgs.pizza_in_box_90deg tests passed")
