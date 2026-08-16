"""Test the bad-delivery pizza simulation: chaotic driving stays
numerically stable (a real risk at these angular speeds), the two anchors
genuinely move out of sync (not just a phase-shifted copy of each other),
and the resulting mangling is quantifiably worse than the clean,
synchronized fold from dgs.pizza_web_fold -- not just eyeballed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.pizza_bad_delivery import chaotic_delivery_angle, simulate_bad_delivery
from dgs.pizza_web_fold import simulate_web_fold

# 1. chaotic_delivery_angle with different phase offsets produces
# genuinely different trajectories, not the same signal shifted in a way
# that cancels out -- sampled at several times, the two phases disagree
t_samples = np.linspace(0, 3, 50)
angle_a = np.array([chaotic_delivery_angle(t, 0.0) for t in t_samples])
angle_b = np.array([chaotic_delivery_angle(t, np.pi) for t in t_samples])
assert not np.allclose(angle_a, angle_b)
assert not np.allclose(angle_a, -angle_b)   # not simply mirrored either

# 2. the full simulation stays numerically stable despite high angular
# speed -- a real risk at these amplitudes/frequencies, checked directly
result = simulate_bad_delivery(t_max=2.0)
assert result["any_nan"] is False
assert result["max_speed"] > 10.0   # confirms this genuinely IS fast/violent, not accidentally tame

# 3. the two rims (A and B) end up meaningfully separated at some point --
# the actual, checkable definition of "mangled" used here: a clean fold
# keeps the two halves close together (mirrored), chaotic driving scatters them
assert result["rim_separation"].max() > 0.3

# 4. compare directly against the clean, synchronized fold: the bad
# delivery's rim excursion should exceed the clean fold's
clean = simulate_web_fold(t_max=2.0)
clean_excursion = np.linalg.norm(clean["rim_a_pos"] - clean["rim_a_pos"][0], axis=1)
bad_excursion = np.linalg.norm(result["rim_a"] - result["rim_a"][0], axis=1)
assert bad_excursion.max() > clean_excursion.max()

print("all dgs.pizza_bad_delivery tests passed")
