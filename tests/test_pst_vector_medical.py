"""Test dgs/pst_vector_medical.py: PST's scalar phase-edge map turned into
a genuine 2-D vector field (wrap-safe spatial gradient), and checked
directly against a synthetic retinal-fundus-style test image -- the
vector field's magnitude must concentrate on the vessels, not just look
plausible."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.pst_vector_medical import (
    synthetic_fundus_image, pst_vector_field, verify_vector_field_highlights_vessels,
    _wrap_safe_gradient_1d,
)

# 1. synthetic_fundus_image: well-formed output, values in [0,1], vessel
#    mask actually marks SOME pixels (not empty, not everything)
data = synthetic_fundus_image(size=128, n_vessels=6, seed=0)
image, mask = data["image"], data["vessel_mask"]
assert image.shape == (128, 128)
assert image.min() >= 0.0 and image.max() <= 1.0
assert 0 < mask.sum() < mask.size * 0.5   # vessels are a minority of pixels

try:
    synthetic_fundus_image(size=16)
    raise AssertionError("expected ValueError for size too small")
except ValueError:
    pass

# 2. _wrap_safe_gradient_1d: a phase field with an artificial +pi/-pi wrap
#    boundary must NOT show a huge spurious gradient there (the whole
#    reason for the wrap-safe formulation instead of np.diff)
phase = np.zeros((10, 10))
phase[:, 5:] = np.pi - 0.01     # near +pi
phase[:, :5] = -np.pi + 0.01    # near -pi, adjacent column: TRUE difference is tiny
grad_x = _wrap_safe_gradient_1d(phase, axis=1)
# true angular difference across that boundary is ~0.02 rad, not ~2*pi
assert np.abs(grad_x[:, 4]).max() < 0.1, f"wrap-safe gradient failed: {grad_x[:, 4]}"

# a naive raw-radian diff WOULD have shown a huge jump here -- confirm that
# claim explicitly so the wrap-safe fix's necessity isn't just asserted
naive_diff = np.diff(phase, axis=1)[:, 4]
assert np.abs(naive_diff).max() > 3.0, "test setup didn't actually create a wrap discontinuity"

# 3. pst_vector_field: well-formed outputs, magnitude/direction consistent
#    with grad_x/grad_y by construction
field = pst_vector_field(image)
for key in ("phase", "grad_y", "grad_x", "magnitude", "direction"):
    assert field[key].shape == image.shape
assert np.all(field["magnitude"] >= 0)
recomputed_mag = np.sqrt(field["grad_x"]**2 + field["grad_y"]**2)
assert np.allclose(field["magnitude"], recomputed_mag)

# uniform image -> PST phase is ~0 everywhere -> vector field magnitude ~0
flat = np.full((64, 64), 0.5)
flat_field = pst_vector_field(flat)
assert flat_field["magnitude"].max() < 1e-8

# 4. verify_vector_field_highlights_vessels: the actual numeric claim --
#    vessel-region magnitude must substantially exceed background
check = verify_vector_field_highlights_vessels(size=128, n_vessels=6, seed=0)
assert check["highlights_vessels"] is True
assert check["ratio"] > 5.0, f"ratio only {check['ratio']}, expected strong vessel contrast"
assert check["vessel_mean_magnitude"] > check["background_mean_magnitude"]

# different seeds should also pass (not a lucky one-off image)
for seed in (1, 2, 3):
    c = verify_vector_field_highlights_vessels(size=96, n_vessels=4, seed=seed)
    assert c["highlights_vessels"] is True, f"seed={seed}: ratio={c['ratio']}"

print("all dgs.pst_vector_medical tests passed")
