"""Test dgs/laser_tag_raycaster.py's physics: DDA raycasting + fisheye
correction, the Airy-diffraction hit-probability model, Beer-Lambert fog
reuse, and mine-splatter kinematics. Pygame itself is not exercised here
(the game loop is lazy-imported and not unit-testable headlessly), matching
dgs/viewer_pygame.py's existing convention."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.laser_tag_raycaster import (
    cast_ray, cast_rays, correct_fisheye, wall_column_height,
    airy_intensity_fraction, airy_beam_x, airy_ring_profile,
    fog_attenuated_intensity,
    resolve_shot, mine_triggered, splatter_particles, billboard_scale,
    mouse_look_update, render_pitch_shift_px,
)

# 1. cast_ray must find a wall at the expected distance for a straight shot
grid = np.zeros((16, 16), dtype=int)
grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
d = cast_ray(2.5, 2.5, 0.0, grid)
assert abs(d - 12.52) < 0.05, f"expected ray to hit the right wall at ~12.5, got {d}"

# 2. cast_rays returns one distance per ray, all within [0, max_dist]
rays = cast_rays(2.5, 2.5, 0.0, np.pi / 3, 20, grid, max_dist=20.0)
assert rays.shape == (20,)
assert np.all((rays >= 0) & (rays <= 20.0))

# 3. Fisheye correction must never increase distance (cos <= 1)
perp = correct_fisheye(rays, 0.0, np.pi / 3)
assert np.all(perp <= rays + 1e-9), "perpendicular distance must be <= raw ray length"

# 4. Wall column height is inversely proportional to distance
h_near = wall_column_height(np.array([1.0]), 600)[0]
h_far = wall_column_height(np.array([10.0]), 600)[0]
assert abs(h_near - 600.0) < 1e-9
assert abs(h_far - 60.0) < 1e-9

# 5. cast_ray/cast_rays bounds
for bad_kwargs in [dict(px=1, py=1, angle=0, grid=grid, step=0), dict(px=1, py=1, angle=0, grid=grid, max_dist=0)]:
    try:
        cast_ray(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

# 6. Airy intensity fraction: exactly 1 at x=0, exactly 0 at the first dark
#    ring (x=3.8317, the first zero of J1), monotonically decreasing between
assert abs(airy_intensity_fraction(0.0) - 1.0) < 1e-12
assert airy_intensity_fraction(3.8317059702) < 1e-6
xs = np.linspace(0.01, 3.8, 20)
vals = airy_intensity_fraction(xs)
assert np.all(np.diff(vals) < 0), "Airy intensity should decrease monotonically out to the first dark ring"

# 7. airy_beam_x: at transverse_offset == spot_radius, x must equal the first Airy zero
x_at_edge = airy_beam_x(0.15, spot_radius_m=0.15)
assert abs(x_at_edge - 3.8317059702) < 1e-6
try:
    airy_beam_x(0.1, spot_radius_m=0.0)
    raise AssertionError("expected ValueError for spot_radius_m<=0")
except ValueError:
    pass

# 8. Fog attenuation: zero fog leaves intensity unchanged; more fog transmits less
assert abs(fog_attenuated_intensity(1.0, 0.0, 5.0) - 1.0) < 1e-12
low_fog = fog_attenuated_intensity(1.0, 0.1, 5.0)
high_fog = fog_attenuated_intensity(1.0, 1.0, 5.0)
assert 0.0 < high_fog < low_fog < 1.0, "more fog (higher mu) must transmit strictly less signal"

# 9. resolve_shot: a wall between shooter and target must fully block (P=0, hit=False)
grid2 = np.zeros((16, 16), dtype=int)
grid2[0, :] = grid2[-1, :] = grid2[:, 0] = grid2[:, -1] = 1
grid2[5, 3:9] = 1
res_blocked = resolve_shot((2.5, 2.5), np.pi / 4, (10.0, 10.0), grid2, rng=np.random.default_rng(0))
assert res_blocked["blocked"] is True
assert res_blocked["hit_probability"] == 0.0
assert res_blocked["hit"] is False

# 10. resolve_shot: a clear, perfectly-aimed shot at short range must have
#     hit probability very close to 1 (aim_fraction~1, fog~1 at mu_fog=0)
res_clear = resolve_shot((2.5, 2.5), np.arctan2(0.0, 1.0), (5.5, 2.5), grid2,
                          spot_radius_m=0.5, rng=np.random.default_rng(0))
assert res_clear["blocked"] is False
assert res_clear["hit_probability"] > 0.95, (
    f"perfectly-aimed clear-line-of-sight short shot should have high hit "
    f"probability, got {res_clear['hit_probability']:.3f}")

# 11. resolve_shot requires distinct shooter/target positions
try:
    resolve_shot((1.0, 1.0), 0.0, (1.0, 1.0), grid2)
    raise AssertionError("expected ValueError for identical shooter/target positions")
except ValueError:
    pass

# 12. mine_triggered: inside vs outside the trigger radius
assert mine_triggered((8.5, 8.5), (8.6, 8.5), trigger_radius_m=0.4) is True
assert mine_triggered((1.0, 1.0), (8.6, 8.5), trigger_radius_m=0.4) is False

# 13. splatter_particles: correct shape, and particles fall (y increases) over time
pts_t0 = splatter_particles((5.0, 5.0), n=10, t=0.0, rng=np.random.default_rng(1))
pts_t1 = splatter_particles((5.0, 5.0), n=10, t=1.0, rng=np.random.default_rng(1))
assert pts_t0.shape == (10, 2)
assert np.mean(pts_t1[:, 1]) > np.mean(pts_t0[:, 1]), "particles should fall under gravity over time"

# 14. billboard_scale: inverse-distance falloff
assert abs(billboard_scale(1.0, 600) - 600.0) < 1e-9
assert abs(billboard_scale(10.0, 600) - 60.0) < 1e-9

# 15. airy_ring_profile: correct shapes and monotonic falloff near x=0,
#     matching the same Airy pattern airy_intensity_fraction uses (the
#     fire-animation ring burst must be drawing REAL diffraction physics,
#     not an arbitrary gradient)
xs, ints = airy_ring_profile(n_rings=14, max_x=6.0)
assert xs.shape == (14,) and ints.shape == (14,)
assert ints[0] > ints[1] > ints[2], "Airy intensity should fall off monotonically near x=0"
for bad_kwargs in [dict(n_rings=1), dict(max_x=0.0)]:
    try:
        airy_ring_profile(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

# 16. Regression: the game's default spot_radius_m must be forgiving enough
#     for realistic key-based aiming, not require sub-3-degree precision --
#     a real usability bug found by actually running the game this session
for distance, delta_deg, min_expected_frac in [(5, 5, 0.5), (8, 3, 0.5)]:
    x = airy_beam_x(distance * np.radians(delta_deg), spot_radius_m=2.5)
    frac = float(airy_intensity_fraction(x))
    assert frac > min_expected_frac, (
        f"distance={distance}, {delta_deg} deg off: expected hit-probability "
        f"fraction > {min_expected_frac} at the game's spot_radius_m=2.5, got {frac:.3f}")

# 17. mouse_look_update: the look vector must always sit on the UNIT sphere
#     (genuinely norm 1), for any yaw/pitch, not just a 2D heading angle
yaw, pitch = 0.0, 0.0
for dx, dy in [(50, 0), (0, 20), (-30, -10), (100, 100)]:
    yaw, pitch, look = mouse_look_update(dx, dy, 0.003, yaw, pitch, np.radians(60))
    assert abs(np.linalg.norm(look) - 1.0) < 1e-9, f"look vector must be unit-norm, got {look}"

# 18. mouse_look_update: pitch must stay within +/-pitch_limit_rad even
#     under a huge mouse delta (the clamp must actually clamp)
_, pitch_extreme, _ = mouse_look_update(0, 1_000_000, 0.003, 0.0, 0.0, np.radians(50))
assert abs(pitch_extreme - np.radians(50)) < 1e-9, "pitch should clamp exactly at the limit"
_, pitch_extreme_neg, _ = mouse_look_update(0, -1_000_000, 0.003, 0.0, 0.0, np.radians(50))
assert abs(pitch_extreme_neg - (-np.radians(50))) < 1e-9, "pitch should clamp exactly at the negative limit"

# 19. mouse_look_update bounds: pitch_limit_rad outside (0, pi/2) must raise
for bad_limit in [0.0, np.pi / 2, -0.1, np.pi]:
    try:
        mouse_look_update(1, 1, 0.003, 0.0, 0.0, bad_limit)
        raise AssertionError(f"expected ValueError for pitch_limit_rad={bad_limit}")
    except ValueError:
        pass

# 20. render_pitch_shift_px: zero pitch gives zero shift; positive pitch
#     (looking up) gives a positive (downward, in screen-y) shift
assert render_pitch_shift_px(0.0, 600) == 0
assert render_pitch_shift_px(0.5, 600) > 0
assert render_pitch_shift_px(-0.5, 600) < 0

print("all dgs.laser_tag_raycaster tests passed")
