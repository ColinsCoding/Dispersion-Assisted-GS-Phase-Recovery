"""Test differentiable_optics_tutorial: autograd basics, forward-model
gradients, the focusing-mask demo, and the GS-vs-gradient-descent comparison."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import differentiable_optics_tutorial as dot

# 1. scalar autograd matches the analytic derivative of x^2
r1 = dot.step1_scalar_autograd()
assert abs(r1["dy_dx_autograd"] - r1["dy_dx_analytic"]) < 1e-6

# 2. the forward optical model has a finite, nonzero gradient at a single pixel
r2 = dot.step2_forward_model_is_differentiable()
assert r2["grad_is_finite"]
assert r2["grad_norm"] > 1e-3

# 2b. gradient ascent on the phase mask concentrates energy at the target pixel
r2b = dot.step2b_design_a_focusing_phase_mask(n_iter=200)
assert r2b["final_share_at_target"] > 5 * r2b["initial_share_at_target"]
assert r2b["final_share_at_target"] > 0.5

# 4. GS and gradient-descent both reconstruct the measured intensity reasonably
cmp = dot.step4_compare_to_gs()
assert cmp["gs_reconstruction_mse"] < 1.0
assert cmp["gd_reconstruction_mse"] < 1.0

print("test_differentiable_optics_tutorial: all checks passed")
