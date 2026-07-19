"""Test the SymPy VR stereo window pipeline: focal-length derivation
from FOV (the real bug found and fixed this session -- an independently
chosen focal_px constant didn't match the projection's actual implied
focal length, giving a systematic ~92% depth-recovery error), stereo
disparity scaling with depth, and the round-trip consistency with
dgs.spatial_computing.stereo_depth."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import sympy_vr_windows as svr

# 1. focal_px_from_fov: wider FOV -> shorter effective focal length
#    (standard camera behavior: wide-angle lenses have short focal length)
f_narrow = svr.focal_px_from_fov(30, 640, 480)
f_wide = svr.focal_px_from_fov(90, 640, 480)
assert f_wide < f_narrow

# 2. make_vr_window builds 4 corners centered on position_3d
w = svr.make_vr_window(sp.Symbol('x'), (0.0, 0.0, 2.0), width_m=0.4, height_m=0.2)
assert w["corners_3d"].shape == (4, 3)
assert np.allclose(w["corners_3d"][:, 2], 2.0)   # all corners at the same depth
center_xy = w["corners_3d"][:, :2].mean(axis=0)
assert np.allclose(center_xy, [0.0, 0.0], atol=1e-9)

# 3. render_sympy_texture never crashes, even on expressions matplotlib's
#    mathtext can't parse (Integral emits \limits) -- falls back to plain text
tex_simple = svr.render_sympy_texture(sp.exp(-sp.Symbol('x')**2))
x = sp.Symbol('x')
tex_fallback = svr.render_sympy_texture(sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo)))
assert tex_simple.ndim == 3 and tex_simple.shape[2] == 4   # RGBA
assert tex_fallback.ndim == 3 and tex_fallback.shape[2] == 4

# 4. THE core fix: verify_disparity_matches_depth recovers the true depth
#    with near-zero error (this failed at ~92% error before deriving
#    focal_px consistently from fov/image size)
for true_z in [0.5, 1.0, 2.0, 5.0]:
    window = svr.make_vr_window(sp.Symbol('E'), (0.0, 0.0, true_z))
    result = svr.verify_disparity_matches_depth(window)
    assert result["relative_error"] < 1e-6
    assert abs(result["recovered_depth_m"] - true_z) < 1e-6

# 5. disparity DECREASES with depth (farther things shift less between
#    the two eyes -- the real stereo depth cue)
window_near = svr.make_vr_window(sp.Symbol('a'), (0.0, 0.0, 1.0))
window_far = svr.make_vr_window(sp.Symbol('a'), (0.0, 0.0, 5.0))
result_near = svr.verify_disparity_matches_depth(window_near)
result_far = svr.verify_disparity_matches_depth(window_far)
assert result_near["disparity_px"] > result_far["disparity_px"]

# 6. wider baseline -> larger disparity for the same depth (more eye
#    separation, more parallax)
window = svr.make_vr_window(sp.Symbol('b'), (0.0, 0.0, 2.0))
result_narrow_baseline = svr.verify_disparity_matches_depth(window, baseline_m=0.03)
result_wide_baseline = svr.verify_disparity_matches_depth(window, baseline_m=0.15)
assert result_wide_baseline["disparity_px"] > result_narrow_baseline["disparity_px"]

# 7. compose_stereo_image produces two correctly-shaped RGBA images
windows = [
    svr.make_vr_window(sp.Symbol('p'), (-0.2, 0.0, 1.5)),
    svr.make_vr_window(sp.Symbol('q'), (0.2, 0.0, 2.5)),
]
left_img, right_img = svr.compose_stereo_image(windows, image_w=320, image_h=240)
assert left_img.shape == (240, 320, 4)
assert right_img.shape == (240, 320, 4)

# 8. input validation
for bad_call in [
    lambda: svr.focal_px_from_fov(0, 640, 480),
    lambda: svr.focal_px_from_fov(180, 640, 480),
    lambda: svr.focal_px_from_fov(60, -1, 480),
    lambda: svr.make_vr_window(sp.Symbol('x'), (0, 0, 1.0), width_m=-1.0),
    lambda: svr.make_vr_window(sp.Symbol('x'), (0, 0, 1.0), height_m=-1.0),
    lambda: svr.make_vr_window(sp.Symbol('x'), (0, 0, -1.0)),   # depth must be positive
    lambda: svr.project_window_stereo(w, baseline_m=-1.0),
    lambda: svr.compose_stereo_image([]),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.sympy_vr_windows tests passed")
