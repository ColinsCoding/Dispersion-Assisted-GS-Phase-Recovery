"""Chicken BBQ flip simulator: real projectile + rotational physics for a
tongs-flip, not a scripted animation.

The core mechanic is a single real-physics fact: an off-center strike
applies both a LINEAR impulse (throws the piece upward) and an ANGULAR
impulse (spins it) -- torque = force * offset, and angular impulse =
torque * contact_time = I * omega0 (impulse-momentum theorem for
rotation). Flip too close to center and it barely spins (lands the same
side down); flip too far off-center and it over-rotates and lands sideways
or upside down relative to the grill.

Once airborne, the piece is a genuine rigid-body projectile:
  x(t) = x0 + vx0*t
  y(t) = y0 + vy0*t - (1/2) g t^2
  angle(t) = angle0 + omega0*t
solved in closed form (the quadratic for when it returns to grill height),
not stepped with fixed-size numerical integration -- exact, not
approximate.

Doneness is tracked per side: whichever face is DOWN (in contact with the
grill) accumulates char over time; flipping switches which side chars.
Leave one side down too long and it burns -- the actual reason you flip
food on a real grill, not an arbitrary game rule.
"""

import numpy as np


# -- Flip impulse: an off-center strike gives BOTH linear and angular momentum

def apply_flip_impulse(strike_force, contact_time, offset_from_center, mass, moment_of_inertia):
    """A tongs strike of `strike_force` (N), applied for `contact_time` (s),
    at `offset_from_center` (m, can be negative) from the piece's center of
    mass, gives:
      linear impulse  J = F * dt          -> vy0 = J / m
      torque          tau = F * offset
      angular impulse = tau * dt          -> omega0 = (tau * dt) / I
    Returns (vy0, omega0)."""
    if mass <= 0:
        raise ValueError(f"mass={mass} must be positive")
    if moment_of_inertia <= 0:
        raise ValueError(f"moment_of_inertia={moment_of_inertia} must be positive")
    if contact_time <= 0:
        raise ValueError(f"contact_time={contact_time} must be positive")

    linear_impulse = strike_force * contact_time
    vy0 = linear_impulse / mass

    torque = strike_force * offset_from_center
    angular_impulse = torque * contact_time
    omega0 = angular_impulse / moment_of_inertia
    return vy0, omega0


# -- Projectile motion: closed form, not stepped -------------------------------

def time_to_return_to_height(y0, vy0, target_y, g=9.80665):
    """Solve y0 + vy0*t - (1/2)g*t^2 = target_y for the positive root --
    the exact (quadratic-formula) flight time, no numerical integration."""
    a = -0.5 * g
    b = vy0
    c = y0 - target_y
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None   # never reaches target_y (shouldn't happen for a<0, g>0, but guard anyway)
    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    positive_roots = [t for t in (t1, t2) if t > 1e-9]
    return float(max(positive_roots)) if positive_roots else None


def flip_trajectory(x0, y0, vx0, vy0, omega0, angle0, t, g=9.80665):
    """Position and orientation at time t along the flip arc -- exact
    closed-form kinematics, evaluable at any t (including the landing
    time from time_to_return_to_height)."""
    t = np.asarray(t, dtype=float)
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t ** 2
    angle = angle0 + omega0 * t
    return x, y, angle


# -- Landing classification -----------------------------------------------------

def normalize_angle(angle):
    """Wrap an angle (radians) into [0, 2*pi)."""
    return float(np.mod(angle, 2 * np.pi))


def classify_landing(angle_at_landing, tolerance=np.deg2rad(35)):
    """Which side faces down when the piece lands, given how far it
    rotated: near 0 or 2*pi (mod 2*pi) means the SAME side that started
    down is down again ('A'); near pi means the piece flipped ('B'); a
    landing that's neither close is a bad/sideways landing ('bad')."""
    a = normalize_angle(angle_at_landing)
    dist_to_0 = min(a, 2 * np.pi - a)
    dist_to_pi = abs(a - np.pi)
    if dist_to_0 <= tolerance:
        return "A"
    if dist_to_pi <= tolerance:
        return "B"
    return "bad"


def within_grill_bounds(x, grill_x_min, grill_x_max):
    """Did the piece land on the grill, or fall off the side?"""
    return grill_x_min <= x <= grill_x_max


# -- Doneness / char tracking ---------------------------------------------------

def update_char_level(char_levels, side_down, dt, char_rate=0.15, burn_threshold=1.0):
    """char_levels: {"A": float, "B": float} in [0, burn_threshold]. The
    side currently facing DOWN accumulates char at char_rate per second;
    the other side does not change (it's not in contact with the heat).
    Returns (new_char_levels, burnt: bool)."""
    if side_down not in ("A", "B"):
        raise ValueError(f"side_down must be 'A' or 'B', got {side_down!r}")
    new_levels = dict(char_levels)
    new_levels[side_down] = min(burn_threshold, new_levels[side_down] + char_rate * dt)
    burnt = new_levels[side_down] >= burn_threshold
    return new_levels, burnt


def char_color(char_level):
    """Map a char level in [0,1] to an RGB color: raw pink -> browned ->
    charred black, for rendering. Pure presentation, no physics."""
    char_level = float(np.clip(char_level, 0.0, 1.0))
    raw = np.array([235, 180, 170])
    browned = np.array([180, 110, 60])
    charred = np.array([25, 20, 18])
    if char_level < 0.5:
        t = char_level / 0.5
        color = raw * (1 - t) + browned * t
    else:
        t = (char_level - 0.5) / 0.5
        color = browned * (1 - t) + charred * t
    return tuple(int(c) for c in color)


def disk_moment_of_inertia(mass, radius):
    """A chicken piece modeled as a thin disk about its own center, for
    the flip's angular impulse calculation: I = (1/2) m R^2."""
    return 0.5 * mass * radius ** 2


# -- Teriyaki sauce: moisture that dries out under heat unless re-basted -------

def update_sauce_level(sauce_level, side_down, dt, dry_rate=0.08, dried_out_threshold=0.0):
    """The DOWN-facing side's sauce/moisture dries out over grill heat
    exposure (evaporation, same "only the side in contact with heat is
    affected" structure as update_char_level) -- the side NOT touching the
    grill doesn't dry out. Returns (new_sauce_level, dried_out: bool)."""
    if side_down not in ("A", "B"):
        raise ValueError(f"side_down must be 'A' or 'B', got {side_down!r}")
    new_levels = dict(sauce_level)
    new_levels[side_down] = max(dried_out_threshold, new_levels[side_down] - dry_rate * dt)
    dried_out = new_levels[side_down] <= dried_out_threshold
    return new_levels, dried_out


def baste(sauce_level, side, amount=0.4, max_level=1.0):
    """Apply teriyaki sauce to the given side (basting) -- only works on
    whichever side is currently reachable (in this game, both are always
    reachable between flips; a more elaborate version could restrict
    basting to the UP-facing side, since you can't easily baste the side
    touching the grill)."""
    if side not in ("A", "B"):
        raise ValueError(f"side must be 'A' or 'B', got {side!r}")
    new_levels = dict(sauce_level)
    new_levels[side] = min(max_level, new_levels[side] + amount)
    return new_levels


# -- Interactive pygame game (deferred import, same pattern as
# dgs.laser_tag_raycaster.main -- the physics above is fully testable
# headlessly, only this function needs a display) -----------------------------

def main():
    import pygame

    pygame.init()
    W, H = 640, 480
    GRILL_Y = 380           # pixel y of the grill surface
    GRILL_X_MIN, GRILL_X_MAX = 120, 520
    PPM = 300.0             # pixels per meter (for drawing world-space physics)
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Chicken BBQ Flip Simulator -- SPACE to flip, UP/DOWN to aim offset")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    mass, radius_m = 0.15, 0.06
    I = disk_moment_of_inertia(mass, radius_m)
    radius_px = int(radius_m * PPM)

    def world_to_screen(x_m, y_m):
        return int(x_m * PPM), int(GRILL_Y - y_m * PPM)

    state = {
        "x": 0.5, "y": 0.0, "vx": 0.0, "vy": 0.0, "angle": 0.0, "omega": 0.0,
        "airborne": False, "side_down": "A",
        "char": {"A": 0.0, "B": 0.0},
        "offset": 0.005, "score": 0, "game_over": False, "message": "",
        "tongs_t": 0.0, "tongs_striking": False,   # viewmodel animation state (purely cosmetic)
    }
    TONGS_STRIKE_DURATION = 0.22   # seconds -- matches the physics impulse's contact_time feel

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and not state["game_over"]:
                if event.key == pygame.K_SPACE and not state["airborne"]:
                    vy0, omega0 = apply_flip_impulse(
                        strike_force=12.0, contact_time=0.03,
                        offset_from_center=state["offset"], mass=mass, moment_of_inertia=I,
                    )
                    state["vy"] = vy0
                    state["omega"] = omega0
                    state["airborne"] = True
                    state["t_launch"] = 0.0
                    state["x0"], state["y0"], state["angle0"] = state["x"], 0.0, state["angle"]
                    state["tongs_striking"] = True
                    state["tongs_t"] = 0.0
                if event.key == pygame.K_UP:
                    state["offset"] = min(0.012, state["offset"] + 0.001)
                if event.key == pygame.K_DOWN:
                    state["offset"] = max(-0.012, state["offset"] - 0.001)

        if not state["game_over"]:
            if state["airborne"]:
                state["t_launch"] += dt
                x, y, angle = flip_trajectory(
                    state["x0"], state["y0"], 0.0, state["vy"], state["omega"],
                    state["angle0"], state["t_launch"],
                )
                state["x"], state["y"], state["angle"] = float(x), float(y), float(angle)
                if state["y"] <= 0.0 and state["t_launch"] > 0.05:
                    state["y"] = 0.0
                    state["airborne"] = False
                    if not within_grill_bounds(state["x"], GRILL_X_MIN / PPM, GRILL_X_MAX / PPM):
                        state["game_over"] = True
                        state["message"] = "Fell off the grill! Game over."
                    else:
                        verdict = classify_landing(state["angle"])
                        if verdict == "bad":
                            state["game_over"] = True
                            state["message"] = "Landed sideways! Game over."
                        else:
                            state["side_down"] = verdict
                            state["angle"] = 0.0 if verdict == "A" else np.pi
                            state["score"] += 1
                            state["message"] = f"Nice flip! Side {verdict} down. Score: {state['score']}"
            else:
                state["char"], burnt = update_char_level(state["char"], state["side_down"], dt, char_rate=0.12)
                if burnt:
                    state["game_over"] = True
                    state["message"] = f"Side {state['side_down']} burnt! Game over. Final score: {state['score']}"

        if state["tongs_striking"]:
            state["tongs_t"] += dt
            if state["tongs_t"] >= TONGS_STRIKE_DURATION:
                state["tongs_striking"] = False
                state["tongs_t"] = 0.0

        screen.fill((30, 30, 35))
        pygame.draw.rect(screen, (70, 70, 75), (GRILL_X_MIN, GRILL_Y, GRILL_X_MAX - GRILL_X_MIN, 12))
        for gx in range(GRILL_X_MIN, GRILL_X_MAX, 20):
            pygame.draw.line(screen, (50, 50, 55), (gx, GRILL_Y), (gx, GRILL_Y + 12))

        px, py = world_to_screen(state["x"], state["y"])
        color = char_color(state["char"][state["side_down"] if not state["airborne"] else "A"])
        surf = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(surf, color, (0, 0, radius_px * 2, radius_px * 2))
        pygame.draw.ellipse(surf, (0, 0, 0), (0, 0, radius_px * 2, radius_px * 2), 2)
        rotated = pygame.transform.rotate(surf, np.rad2deg(state["angle"]))
        rect = rotated.get_rect(center=(px, py - radius_px))
        screen.blit(rotated, rect)

        # -- CS:GO-style tongs viewmodel: a foreground pair of tongs anchored
        # at the bottom of the screen, drawn LAST (on top of everything else,
        # same layering as a first-person weapon viewmodel). Purely cosmetic
        # animation -- the actual flip physics already happened the instant
        # SPACE was pressed; this just gives it a visible "strike."
        frac = min(1.0, state["tongs_t"] / TONGS_STRIKE_DURATION) if state["tongs_striking"] else 0.0
        # extend-then-retract: rises for the first half, falls back for the second half
        extend = np.sin(frac * np.pi) if state["tongs_striking"] else 0.0
        pinch = np.sin(min(1.0, frac * 2.0) * np.pi / 2) if state["tongs_striking"] else 0.0

        anchor_x = px                        # tongs reach toward wherever the chicken actually is
        base_y = H + 10                      # tongs handle originates just below the screen
        tip_y = base_y - 90 - int(70 * extend)   # reaches upward toward the chicken when striking
        jaw_spread = int(26 * (1.0 - pinch))     # wide when idle/extending, pinched shut mid-strike

        left_arm = [(anchor_x - 18, base_y), (anchor_x - jaw_spread, tip_y)]
        right_arm = [(anchor_x + 18, base_y), (anchor_x + jaw_spread, tip_y)]
        tong_color = (200, 200, 210)
        pygame.draw.line(screen, tong_color, *left_arm, width=10)
        pygame.draw.line(screen, tong_color, *right_arm, width=10)
        pygame.draw.circle(screen, tong_color, (anchor_x - jaw_spread, tip_y), 6)
        pygame.draw.circle(screen, tong_color, (anchor_x + jaw_spread, tip_y), 6)

        hud_lines = [
            f"offset: {state['offset']*1000:+.1f} mm (UP/DOWN to aim)",
            f"char A: {state['char']['A']:.2f}   char B: {state['char']['B']:.2f}",
            f"score: {state['score']}",
            state["message"],
        ]
        for i, line in enumerate(hud_lines):
            screen.blit(font.render(line, True, (230, 230, 230)), (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()


def _print_physics_demo():
    mass, radius = 0.15, 0.06   # 150g piece, 6cm radius
    I = disk_moment_of_inertia(mass, radius)
    print(f"Chicken piece: mass={mass}kg, radius={radius}m, I={I:.6f} kg*m^2")

    print("\n=== A well-aimed flip (5mm offset -- tuned for a clean single flip) ===")
    vy0, omega0 = apply_flip_impulse(strike_force=12.0, contact_time=0.03,
                                      offset_from_center=0.005, mass=mass, moment_of_inertia=I)
    print(f"vy0={vy0:.3f} m/s, omega0={omega0:.3f} rad/s")

    t_land = time_to_return_to_height(y0=0.0, vy0=vy0, target_y=0.0)
    print(f"time in the air: {t_land:.4f} s")

    x_land, y_land, angle_land = flip_trajectory(x0=0.5, y0=0.0, vx0=0.0, vy0=vy0,
                                                  omega0=omega0, angle0=0.0, t=t_land)
    verdict = classify_landing(angle_land)
    print(f"landing angle: {np.rad2deg(normalize_angle(angle_land)):.1f} deg -> {verdict}")

    print("\n=== Char accumulation while side A cooks for 8 seconds ===")
    char = {"A": 0.0, "B": 0.0}
    burnt = False
    for _ in range(80):
        char, burnt = update_char_level(char, "A", dt=0.1)
    print(f"char levels after 8s: {char}, burnt={burnt}")
    print(f"char color for side A: {char_color(char['A'])}")


if __name__ == "__main__":
    _print_physics_demo()
    print("\nLaunching the interactive game (SPACE=flip, UP/DOWN=aim offset, ESC/close window=quit)...")
    main()
