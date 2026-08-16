"""laser_tag_raycaster.py -- Wolfenstein-style raycasting pseudo-3D world,
with laser-tag hit detection built on real photonics (Airy diffraction +
Beer-Lambert fog attenuation), not just line-of-sight geometry.

PSEUDO-3D VIA RAYCASTING: cast_ray/cast_rays march (DDA-style) through a 2D
grid map, one ray per screen column; distance-to-wall becomes column height
-- the classic technique (Wolfenstein 3D, 1992) for real-time "3D" on
hardware with no 3D pipeline, still a clean way to do it in plain Pygame.
correct_fisheye removes the barrel-distortion artifact of using raw ray
length instead of perpendicular distance -- a real, well-known raycasting
bug, checked here rather than silently baked in.

THE PHOTONICS, NOT JUST "LASER = HITSCAN": a real laser-tag pulse is a
diffraction-limited spot, not a mathematical ray.
  - airy_intensity_fraction implements the Airy diffraction pattern
    I(x)/I0 = [2*J1(x)/x]^2 (scipy.special.j1) -- the SAME physics as
    dgs/retinal_scan_imaging.py's diffraction-limited spot formula
    (1.22*lambda*f/D), here determining how a near-miss still has a
    reduced but nonzero hit probability instead of a hard yes/no.
  - fog_attenuated_intensity reuses dgs/biophotonics.py's beer_lambert
    DIRECTLY (not reimplemented) for signal loss through fog/smoke in the
    arena -- the same Beer-Lambert law already in this repo.
  - resolve_shot combines both with the raycast line-of-sight check (a
    wall fully blocks a shot; fog only attenuates it) into a single hit
    probability, then draws the actual hit/miss outcome from an explicit
    rng (deterministic given a seed, for testability).

Mines/splatter (splatter_particles) are cosmetic ballistic kinematics
(x(t)=x0+v0*t+0.5*g*t^2), not photonics -- kept honestly separate.

Pygame is imported lazily inside rendering/game-loop functions only, so
this module's physics functions are importable and testable without
pygame installed, matching dgs/viewer_pygame.py's existing convention.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from scipy.special import j1

from dgs.biophotonics import beer_lambert


# ── 1. DDA raycasting (the pseudo-3D engine) ─────────────────────────────────

def cast_ray(px: float, py: float, angle: float, grid: np.ndarray,
             max_dist: float = 20.0, step: float = 0.02) -> float:
    """March along `angle` from (px,py) until hitting a nonzero cell of
    `grid` (a 2D array, 0=empty, nonzero=wall) or `max_dist`. Fixed-step
    marching (not true DDA-per-cell-boundary) -- simple and fast enough for
    a teaching-scale grid; `step` trades accuracy for speed.

    Bounds: step must be positive and much smaller than 1 grid cell.
    """
    if step <= 0:
        raise ValueError(f"step={step}: must be positive")
    if max_dist <= 0:
        raise ValueError(f"max_dist={max_dist}: must be positive")
    dx, dy = np.cos(angle), np.sin(angle)
    dist = 0.0
    ny, nx = grid.shape
    while dist < max_dist:
        x, y = px + dx * dist, py + dy * dist
        gx, gy = int(x), int(y)
        if gy < 0 or gy >= ny or gx < 0 or gx >= nx or grid[gy, gx] != 0:
            return dist
        dist += step
    return max_dist


def cast_rays(px: float, py: float, player_angle: float, fov: float,
              n_rays: int, grid: np.ndarray, max_dist: float = 20.0) -> np.ndarray:
    """One cast_ray per screen column across the field of view `fov`
    (radians), centered on player_angle. Returns raw (uncorrected)
    ray-length distances, length n_rays."""
    if n_rays < 1:
        raise ValueError(f"n_rays={n_rays}: must be >= 1")
    if fov <= 0:
        raise ValueError(f"fov={fov}: must be positive")
    angles = player_angle + np.linspace(-fov / 2, fov / 2, n_rays)
    return np.array([cast_ray(px, py, a, grid, max_dist=max_dist) for a in angles])


def correct_fisheye(raw_distances: np.ndarray, player_angle: float,
                     fov: float) -> np.ndarray:
    """Perpendicular distance = raw_distance * cos(ray_angle - player_angle)
    -- removes the fisheye/barrel-distortion artifact of using raw ray
    length for wall-column height (a real, well-documented raycasting bug:
    without this, straight walls render as curved)."""
    n = len(raw_distances)
    angles = player_angle + np.linspace(-fov / 2, fov / 2, n)
    return raw_distances * np.cos(angles - player_angle)


def wall_column_height(perp_distance: np.ndarray, screen_height: int,
                        eps: float = 1e-6) -> np.ndarray:
    """Projected wall-slice height on screen: h ~ screen_height/distance
    (perspective projection -- closer walls fill more of the screen,
    inverse-distance falloff)."""
    perp_distance = np.asarray(perp_distance, dtype=float)
    return screen_height / np.maximum(perp_distance, eps)


# ── 2. The laser-tag beam: Airy diffraction pattern ──────────────────────────

def airy_intensity_fraction(x) -> np.ndarray:
    """I(x)/I0 = [2*J1(x)/x]^2, the Airy diffraction pattern (x=0 -> 1,
    exactly; first dark ring at x=3.8317..., the first zero of J1) --
    the SAME physics as dgs/retinal_scan_imaging.py's diffraction-limited
    spot formula, used here to give a laser-tag shot a soft, physically
    real falloff instead of a hard hit/miss radius."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = (2.0 * j1(x[nz]) / x[nz]) ** 2
    return out if out.size > 1 else float(out[0])


def airy_beam_x(transverse_offset_m: float, spot_radius_m: float) -> float:
    """Map a physical transverse miss-offset to the Airy pattern's
    dimensionless argument x, scaled so x=3.8317 (first dark ring) occurs
    at transverse_offset_m=spot_radius_m -- i.e. spot_radius_m IS the
    Airy-disk radius (matching dgs/retinal_scan_imaging.py's convention)."""
    if spot_radius_m <= 0:
        raise ValueError(f"spot_radius_m={spot_radius_m}: must be positive")
    FIRST_AIRY_ZERO = 3.8317059702
    return FIRST_AIRY_ZERO * abs(transverse_offset_m) / spot_radius_m


# ── 3. Fog attenuation: reuse Beer-Lambert directly ──────────────────────────

def fog_attenuated_intensity(I0: float, mu_fog: float, path_length_m: float) -> float:
    """Signal remaining after `path_length_m` of fog -- a direct call into
    dgs/biophotonics.py's beer_lambert (not reimplemented)."""
    if mu_fog == 0.0:
        return I0
    result = beer_lambert(I0, mu_fog, path_length_m)
    return result["I"]


# ── 4. Resolving a shot: geometry + Airy + Beer-Lambert -> hit probability ──

def resolve_shot(shooter_pos: Tuple[float, float], shooter_angle: float,
                  target_pos: Tuple[float, float], grid: np.ndarray,
                  spot_radius_m: float = 0.15, mu_fog: float = 0.0,
                  rng: np.random.Generator | None = None) -> Dict:
    """Combine line-of-sight blocking (a wall fully blocks the shot),
    Airy-pattern aim tolerance (near-misses can still register, with
    reduced probability), and Beer-Lambert fog attenuation into one hit
    probability, then draw the actual outcome from `rng` (deterministic
    given a seed -- pass one for testability).

    Returns dict with keys: blocked, distance_m, transverse_miss_m,
    hit_probability, hit (bool).
    """
    if rng is None:
        rng = np.random.default_rng()
    sx, sy = shooter_pos
    tx, ty = target_pos
    dx, dy = tx - sx, ty - sy
    distance = float(np.hypot(dx, dy))
    if distance == 0:
        raise ValueError("shooter_pos and target_pos must differ")
    angle_to_target = float(np.arctan2(dy, dx))

    ray_dist = cast_ray(sx, sy, shooter_angle, grid, max_dist=distance + 1e-6)
    blocked = ray_dist < distance - 1e-3

    # small-angle transverse miss distance: r ~ distance * delta_angle
    delta_angle = float(np.angle(np.exp(1j * (angle_to_target - shooter_angle))))
    transverse_miss = distance * delta_angle

    if blocked:
        hit_probability = 0.0
    else:
        x = airy_beam_x(transverse_miss, spot_radius_m)
        aim_fraction = airy_intensity_fraction(x)
        signal = fog_attenuated_intensity(1.0, mu_fog, distance)
        hit_probability = float(np.clip(aim_fraction * signal, 0.0, 1.0))

    hit = bool(rng.random() < hit_probability)
    return {"blocked": blocked, "distance_m": distance,
            "transverse_miss_m": transverse_miss,
            "hit_probability": hit_probability, "hit": hit}


# ── 5. Mines: cosmetic ballistic splatter (no photonics) ────────────────────

def mine_triggered(player_pos: Tuple[float, float], mine_pos: Tuple[float, float],
                    trigger_radius_m: float = 0.4) -> bool:
    """True if player is within trigger_radius_m of the mine."""
    if trigger_radius_m <= 0:
        raise ValueError("trigger_radius_m must be positive")
    dx = player_pos[0] - mine_pos[0]
    dy = player_pos[1] - mine_pos[1]
    return bool(np.hypot(dx, dy) < trigger_radius_m)


def splatter_particles(center: Tuple[float, float], n: int = 24,
                        speed_range: Tuple[float, float] = (1.0, 4.0),
                        g: float = 9.8, t: float = 0.0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Positions of n ballistic splatter particles at time t after a mine
    triggers at `center`, isotropic launch directions, constant gravity g
    -- simple x(t)=x0+v0*t+0.5*g*t^2 kinematics, purely cosmetic (not
    modeling anything about the laser-tag photonics above).

    Returns array of shape (n, 2).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if t < 0:
        raise ValueError("t must be non-negative")
    rng = np.random.default_rng() if rng is None else rng
    angles = rng.uniform(0, 2 * np.pi, n)
    speeds = rng.uniform(*speed_range, n)
    vx, vy = speeds * np.cos(angles), speeds * np.sin(angles)
    cx, cy = center
    x = cx + vx * t
    y = cy + vy * t + 0.5 * g * t ** 2
    return np.stack([x, y], axis=1)


# ── 5b. Airy ring profile, for the on-fire visual (real Bessel physics) ────

def airy_ring_profile(n_rings: int = 12, max_x: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the SAME Airy pattern airy_intensity_fraction uses, at
    n_rings points from x=0 to max_x -- for drawing an animated ring burst
    on firing whose brightness genuinely follows the diffraction pattern
    (including the real dark ring near x=3.83 and the faint secondary
    maximum past it), not an arbitrary decorative gradient."""
    if n_rings < 2:
        raise ValueError(f"n_rings={n_rings}: must be >= 2")
    if max_x <= 0:
        raise ValueError(f"max_x={max_x}: must be positive")
    xs = np.linspace(0.01, max_x, n_rings)
    return xs, airy_intensity_fraction(xs)


# ── 5c. Mouse-look: 2D mouse delta -> a point on the unit sphere ────────────

def mouse_look_update(dx_px: float, dy_px: float, sensitivity: float,
                       yaw: float, pitch: float, pitch_limit_rad: float) -> Tuple[float, float, np.ndarray]:
    """Update (yaw, pitch) from a mouse-motion delta, and return the
    resulting look direction as a point on the UNIT SPHERE:
    look = (cos(pitch)cos(yaw), cos(pitch)sin(yaw), sin(pitch)) --
    genuinely norm-1 for any yaw/pitch (spherical coordinates), not just a
    2D angle. This raycaster only renders yaw (a flat 2.5D world has no
    true vertical geometry), so pitch is used purely as a cosmetic
    look-up/down screen shift (see render_pitch_shift_px) -- but the look
    vector itself is real spherical-coordinates 3D, matching a genuine
    360-degree-camera mouselook rather than a 2D-only heading.

    Bounds: pitch is clamped to +/-pitch_limit_rad (a flat raycaster has no
    way to render looking straight up/down, so an unclamped pitch would be
    physically meaningless here).
    """
    if pitch_limit_rad <= 0 or pitch_limit_rad >= np.pi / 2:
        raise ValueError(f"pitch_limit_rad={pitch_limit_rad}: must be in (0, pi/2)")
    new_yaw = yaw + dx_px * sensitivity
    new_pitch = float(np.clip(pitch + dy_px * sensitivity, -pitch_limit_rad, pitch_limit_rad))
    look = np.array([
        np.cos(new_pitch) * np.cos(new_yaw),
        np.cos(new_pitch) * np.sin(new_yaw),
        np.sin(new_pitch),
    ])
    return float(new_yaw), new_pitch, look


def render_pitch_shift_px(pitch: float, screen_height: int, vertical_gain: float = 1.0) -> int:
    """Cosmetic vertical screen shift from mouse-look pitch: shift the
    whole raycasting render up/down by pitch*vertical_gain*screen_height/pi
    -- the standard "fake pitch" trick classic raycasters use (shift
    columns rather than truly re-projecting geometry) since the underlying
    world has no vertical structure to actually re-render."""
    return int(pitch * vertical_gain * screen_height / np.pi)


# ── 6. Stick-figure billboard scale (pure geometry, testable) ────────────────

def billboard_scale(distance_m: float, screen_height: int,
                     focal_scale: float = 1.0, eps: float = 1e-6) -> float:
    """Projected screen height of a fixed-size sprite at `distance_m` --
    the same inverse-distance perspective projection as
    wall_column_height, applied to an NPC billboard sprite instead of a
    wall slice."""
    return focal_scale * screen_height / max(distance_m, eps)


# ── 7. Pygame game loop (lazy import; not covered by unit tests) ────────────

def main():
    try:
        import pygame
    except ImportError:
        raise SystemExit("Pygame not installed. Run:  pip install pygame")

    pygame.init()
    W, H = 960, 600
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Laser Tag Raycaster -- Airy-Pattern Hit Detection")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    grid = np.zeros((16, 16), dtype=int)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    grid[5, 3:9] = 1
    grid[10, 6:13] = 1

    px, py, p_angle = 2.5, 2.5, 0.0
    pitch = 0.0
    PITCH_LIMIT = np.radians(50.0)
    MOUSE_SENSITIVITY = 0.003
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    pygame.mouse.get_rel()  # discard the first (potentially large) delta from grabbing
    npc_pos = (12.5, 12.5)
    mine_pos = (8.5, 8.5)
    mine_active = True
    splatter_t0 = None
    fov = np.pi / 3
    n_rays = W // 4
    move_speed, turn_speed = 2.5, 2.5
    mu_fog = 0.0
    rng = np.random.default_rng()
    shot_effects: List[Dict] = []
    ring_xs, ring_intensities = airy_ring_profile(n_rings=14, max_x=6.0)
    EFFECT_DURATION_S = 0.4

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        now = pygame.time.get_ticks() / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # Mouse is grabbed below (set_grab(True)) for FPS-style
                # look -- without this, closing the window is the only way
                # out and the cursor stays trapped until then.
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # spot_radius_m=2.5 (not the physics-module default 0.15):
                # a 0.5-2.5 tolerance was found to require sub-3-degree aim
                # to register ANY nonzero hit probability at typical in-game
                # distances -- unplayable with discrete key-based turning.
                # 2.5 keeps the real Airy falloff shape but at a scale that
                # actually fits this game's movement/turn granularity.
                result = resolve_shot((px, py), p_angle, npc_pos, grid,
                                       spot_radius_m=2.5, mu_fog=mu_fog, rng=rng)
                print(f"shot: blocked={result['blocked']} "
                      f"P(hit)={result['hit_probability']:.3f} hit={result['hit']}")
                shot_effects.append({"t0": now, "hit": result["hit"], "blocked": result["blocked"]})

        # Mouse-look: 2D mouse delta -> unit-sphere (yaw, pitch) look direction.
        # pygame's mouse-y grows DOWNWARD, but "mouse up" should mean "look
        # up" (pitch increases) -- negate rel_dy so the convention matches.
        rel_dx, rel_dy = pygame.mouse.get_rel()
        p_angle, pitch, _look_dir = mouse_look_update(
            rel_dx, -rel_dy, MOUSE_SENSITIVITY, p_angle, pitch, PITCH_LIMIT)
        pitch_shift = render_pitch_shift_px(pitch, H)

        keys = pygame.key.get_pressed()
        new_px, new_py = px, py
        if keys[pygame.K_w]:
            new_px += np.cos(p_angle) * move_speed * dt
            new_py += np.sin(p_angle) * move_speed * dt
        if keys[pygame.K_s]:
            new_px -= np.cos(p_angle) * move_speed * dt
            new_py -= np.sin(p_angle) * move_speed * dt
        if keys[pygame.K_a]:
            p_angle -= turn_speed * dt
        if keys[pygame.K_d]:
            p_angle += turn_speed * dt
        if grid[int(new_py), int(new_px)] == 0:
            px, py = new_px, new_py
        if keys[pygame.K_f]:
            mu_fog = min(mu_fog + 0.5 * dt, 3.0)
        if keys[pygame.K_g]:
            mu_fog = max(mu_fog - 0.5 * dt, 0.0)

        if mine_active and mine_triggered((px, py), mine_pos):
            mine_active = False
            splatter_t0 = pygame.time.get_ticks() / 1000.0

        raw = cast_rays(px, py, p_angle, fov, n_rays, grid)
        perp = correct_fisheye(raw, p_angle, fov)
        heights = wall_column_height(perp, H)

        screen.fill((10, 10, 20))
        col_w = W / n_rays
        for i, (dist, h) in enumerate(zip(perp, heights)):
            shade = int(np.clip(255 - dist * 30, 40, 255))
            color = (shade // 3, shade // 2, shade)
            # +pitch_shift: looking up (pitch>0) moves the rendered world
            # DOWN the screen (view center tilts up and away from it).
            top = int((H - h) / 2) + pitch_shift
            pygame.draw.rect(screen, color, (int(i * col_w), max(top, 0), int(col_w) + 1,
                                              int(min(h, H))))

        # NPC stick figure billboard (only if not wall-occluded)
        dxn, dyn = npc_pos[0] - px, npc_pos[1] - py
        dist_npc = float(np.hypot(dxn, dyn))
        ray_to_npc = cast_ray(px, py, float(np.arctan2(dyn, dxn)), grid, max_dist=dist_npc + 1e-6)
        if ray_to_npc >= dist_npc - 1e-3:
            sc = billboard_scale(dist_npc, H, focal_scale=0.6)
            cx, cy = W // 2, int(H / 2) + pitch_shift
            r = max(int(sc * 0.08), 2)
            pygame.draw.circle(screen, (255, 220, 180), (cx, cy - int(sc * 0.35)), r)
            pygame.draw.line(screen, (255, 220, 180), (cx, cy - int(sc * 0.25)),
                              (cx, cy + int(sc * 0.15)), max(int(sc * 0.02), 1))
            pygame.draw.line(screen, (255, 220, 180), (cx, cy + int(sc * 0.15)),
                              (cx - int(sc * 0.1), cy + int(sc * 0.4)), max(int(sc * 0.02), 1))
            pygame.draw.line(screen, (255, 220, 180), (cx, cy + int(sc * 0.15)),
                              (cx + int(sc * 0.1), cy + int(sc * 0.4)), max(int(sc * 0.02), 1))

        if splatter_t0 is not None:
            t_elapsed = pygame.time.get_ticks() / 1000.0 - splatter_t0
            if t_elapsed < 1.2:
                pts = splatter_particles(mine_pos, n=20, t=t_elapsed, rng=rng)
                for wx, wy in pts:
                    dxp, dyp = wx - px, wy - py
                    dpr = float(np.hypot(dxp, dyp))
                    if dpr > 0.1:
                        sc = billboard_scale(dpr, H, focal_scale=0.3)
                        pygame.draw.circle(screen, (220, 60, 40),
                                            (W // 2 + int((np.arctan2(dyp, dxp) - p_angle) * W),
                                             H // 2), max(int(sc * 0.02), 1))
            else:
                splatter_t0 = None

        # Fire animation: an expanding, fading ring burst whose brightness
        # follows the REAL Airy diffraction pattern (ring_intensities), not
        # an arbitrary decorative gradient -- centered on the reticle, plus
        # a brief muzzle-to-reticle beam flash colored by the outcome.
        shot_effects[:] = [e for e in shot_effects if now - e["t0"] < EFFECT_DURATION_S]
        cx, cy = W // 2, H // 2 + pitch_shift
        for e in shot_effects:
            age_frac = (now - e["t0"]) / EFFECT_DURATION_S
            beam_color = (80, 220, 80) if e["hit"] else ((160, 160, 160) if e["blocked"] else (220, 60, 60))
            pygame.draw.line(screen, beam_color, (W // 2, H), (cx, cy), 3)
            growth = 1.0 + 2.0 * age_frac  # rings expand outward over the effect's lifetime
            fade = max(1.0 - age_frac, 0.0)
            for x_val, intensity in zip(ring_xs, ring_intensities):
                radius = int(x_val * 8 * growth)
                if radius < 1:
                    continue
                shade = int(np.clip(intensity * fade * 255, 0, 255))
                if shade > 3:
                    pygame.draw.circle(screen, (shade, shade, min(255, shade + 40)), (cx, cy), radius, width=1)

        hud = font.render(
            f"pos=({px:.1f},{py:.1f}) yaw={np.degrees(p_angle):.0f} deg pitch={np.degrees(pitch):.0f} deg  "
            f"mu_fog={mu_fog:.2f}  mouse=look SPACE=fire F/G=fog WASD=move ESC=quit", True, (255, 255, 255))
        screen.blit(hud, (10, 10))
        pygame.display.flip()

    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    pygame.quit()


if __name__ == "__main__":
    main()
