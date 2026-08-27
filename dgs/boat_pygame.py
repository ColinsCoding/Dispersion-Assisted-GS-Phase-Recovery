"""A playable pygame boat, driven by the REAL heave/roll physics derived in
dgs.boat_lagrangian -- not an arbitrary bobbing animation. The natural
frequencies come straight out of that module's Euler-Lagrange derivation
(omega_heave = sqrt(rho g A_wp/(m+m_added)), omega_roll =
sqrt(rho g nabla GM/I_roll)); this module adds the two things a real boat
also has that a pure Lagrangian derivation deliberately leaves out for
clarity -- DAMPING (wave-radiation + viscous drag, phenomenological here:
critical-damping-ratio zeta, not derived from first principles) and an
external forcing (incoming waves, modeled as random impulses to velocity)
-- then integrates the resulting damped, driven oscillator with RK4.

THE GAME: waves hit at random, energizing heave and roll. Roll past
THETA_CAPSIZE_DEG capsizes the boat -- game over, survival time is the
score. Waves get more frequent over time (a difficulty ramp), so a stiffer
(larger GM) boat survives longer per hit but each hit snaps it harder --
the same comfort-vs-stability tradeoff the boat_lagrangian notebook's roll
plot shows numerically, now something you can feel by playing.

Controls: SPACE / click to restart after capsizing. Close the window to quit.
Run with `py -3.13 -m dgs.boat_pygame`.
"""
import math
import random

from dgs.boat_lagrangian import (
    G_STANDARD, RHO_SEAWATER, heave_natural_frequency, roll_natural_frequency,
)

THETA_CAPSIZE_DEG = 35.0


def _check_non_negative(value, name):
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def boat_accelerations(state, omega_heave, zeta_heave, omega_roll, zeta_roll):
    """state = (x, xdot, theta, thetadot). Returns (xdot, xddot, thetadot,
    thetaddot) for the damped oscillator
        x'' = -omega_heave^2 x - 2 zeta_heave omega_heave x'
        theta'' = -omega_roll^2 theta - 2 zeta_roll omega_roll theta'
    -- the same restoring terms dgs.boat_lagrangian derives from L = T-V,
    with an added phenomenological damping term (real boats lose energy to
    wave radiation and viscous drag; that dissipation has no potential-energy
    representation, so it cannot come out of a conservative Lagrangian --
    it is added here explicitly, not smuggled into L)."""
    x, xdot, theta, thetadot = state
    xddot = -omega_heave ** 2 * x - 2 * zeta_heave * omega_heave * xdot
    thetaddot = -omega_roll ** 2 * theta - 2 * zeta_roll * omega_roll * thetadot
    return xdot, xddot, thetadot, thetaddot


def rk4_step(state, dt, omega_heave, zeta_heave, omega_roll, zeta_roll):
    """One RK4 step of boat_accelerations. state = (x, xdot, theta, thetadot)."""
    def deriv(s):
        return boat_accelerations(s, omega_heave, zeta_heave, omega_roll, zeta_roll)

    s = state
    k1 = deriv(s)
    s2 = tuple(si + 0.5 * dt * ki for si, ki in zip(s, k1))
    k2 = deriv(s2)
    s3 = tuple(si + 0.5 * dt * ki for si, ki in zip(s, k2))
    k3 = deriv(s3)
    s4 = tuple(si + dt * ki for si, ki in zip(s, k3))
    k4 = deriv(s4)
    return tuple(
        si + (dt / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
        for si, k1i, k2i, k3i, k4i in zip(s, k1, k2, k3, k4)
    )


def analytic_underdamped_shm(x0, v0, omega, zeta, t):
    """Exact solution of x'' + 2 zeta omega x' + omega^2 x = 0 for the
    underdamped case (0 <= zeta < 1), used only to VERIFY rk4_step below --
    not used by the game itself (the game needs the coupled two-DOF state,
    this closed form is single-DOF)."""
    if not (0 <= zeta < 1):
        raise ValueError(f"analytic_underdamped_shm requires 0 <= zeta < 1, got {zeta}")
    omega_d = omega * math.sqrt(1 - zeta ** 2)
    A = x0
    B = (v0 + zeta * omega * x0) / omega_d
    envelope = math.exp(-zeta * omega * t)
    return envelope * (A * math.cos(omega_d * t) + B * math.sin(omega_d * t))


def verify_rk4_matches_analytic(omega=3.0, zeta=0.08, x0=1.0, v0=0.0, t_final=4.0, dt=0.001):
    """Step rk4_step with zero roll coupling (theta locked at 0) and compare
    the resulting heave trajectory against analytic_underdamped_shm at
    t_final -- the numerical integrator and the closed-form solution must
    agree to within O(dt^4) (RK4's own truncation order), confirming the
    integrator is actually solving the equation it claims to, not just
    producing plausible-looking numbers."""
    n_steps = int(round(t_final / dt))
    state = (x0, v0, 0.0, 0.0)
    for _ in range(n_steps):
        state = rk4_step(state, dt, omega, zeta, omega_roll=1.0, zeta_roll=0.0)
    x_numeric = state[0]
    x_analytic = analytic_underdamped_shm(x0, v0, omega, zeta, t_final)
    return {
        "x_numeric": x_numeric,
        "x_analytic": x_analytic,
        "abs_error": abs(x_numeric - x_analytic),
        "matches": abs(x_numeric - x_analytic) < 1e-6,
    }


def random_wave_impulse(rng, strength):
    """A wave hit: a random kick to heave velocity (always upward-biased,
    like a swell lifting the hull) and a correlated but independently
    randomized kick to roll velocity (waves rarely hit perfectly
    symmetrically)."""
    _check_non_negative(strength, "strength")
    d_xdot = strength * (0.6 + 0.4 * rng.random())
    d_thetadot = strength * rng.uniform(-1.0, 1.0) * 0.15
    return d_xdot, d_thetadot


def is_capsized(theta_rad, theta_capsize_deg=THETA_CAPSIZE_DEG):
    return abs(math.degrees(theta_rad)) >= theta_capsize_deg


def boat_hull_points(length=90, height=34):
    """Local (unrotated, un-translated) hull outline: a simple V-bottom
    boat silhouette, hull-fixed coordinate frame with (0,0) at the
    waterline center -- the point heave/roll are measured about."""
    hl, hh = length / 2, height
    return [
        (-hl, 0), (-hl * 0.7, -hh), (hl * 0.7, -hh), (hl, 0),
        (hl * 0.55, hh * 0.35), (-hl * 0.55, hh * 0.35),
    ]


def transform_hull(points, x_pixels, theta_rad, cx, cy):
    """Rotate the hull by theta (roll) and translate by (cx, cy - x_pixels)
    -- heave x is physically "up positive", pixels are "down positive",
    hence the subtraction."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    out = []
    for px, py in points:
        rx = px * c - py * s
        ry = px * s + py * c
        out.append((cx + rx, cy - x_pixels + ry))
    return out


def main():
    import pygame

    WIDTH, HEIGHT = 800, 500
    WATERLINE_Y = HEIGHT // 2 + 40
    PIXELS_PER_METER = 60.0

    # A real small boat's numbers, fed through dgs.boat_lagrangian
    omega_heave = heave_natural_frequency(m=1200.0, m_added=200.0, rho=RHO_SEAWATER,
                                           g=G_STANDARD, A_wp=6.0)
    omega_roll = roll_natural_frequency(I_roll=900.0, rho=RHO_SEAWATER, g=G_STANDARD,
                                         nabla=1.15, GM=0.9)
    zeta_heave, zeta_roll = 0.12, 0.10

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boat: real heave/roll physics from dgs.boat_lagrangian")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 48)

    rng = random.Random()

    def new_game():
        return {"state": (0.0, 0.0, 0.0, 0.0), "t": 0.0, "next_wave": rng.uniform(0.8, 1.8),
                "capsized": False}

    game = new_game()
    hull_local = boat_hull_points()
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if game["capsized"] and event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                game = new_game()

        if not game["capsized"]:
            game["t"] += dt
            game["state"] = rk4_step(game["state"], dt, omega_heave, zeta_heave,
                                      omega_roll, zeta_roll)
            game["next_wave"] -= dt
            if game["next_wave"] <= 0:
                # difficulty ramp: waves come faster the longer you survive
                base_gap = max(0.35, 1.6 - 0.01 * game["t"])
                game["next_wave"] = rng.uniform(0.5, 1.0) * base_gap
                strength = rng.uniform(1.0, 2.2)
                d_xdot, d_thetadot = random_wave_impulse(rng, strength)
                x, xdot, theta, thetadot = game["state"]
                game["state"] = (x, xdot + d_xdot, theta, thetadot + d_thetadot)
            if is_capsized(game["state"][2]):
                game["capsized"] = True

        x, xdot, theta, thetadot = game["state"]

        screen.fill((18, 26, 38))
        pygame.draw.rect(screen, (24, 60, 92), (0, WATERLINE_Y, WIDTH, HEIGHT - WATERLINE_Y))
        pygame.draw.line(screen, (120, 190, 220), (0, WATERLINE_Y), (WIDTH, WATERLINE_Y), 2)

        hull_screen = transform_hull(hull_local, x * PIXELS_PER_METER, theta,
                                      WIDTH // 2, WATERLINE_Y)
        hull_color = (200, 60, 60) if game["capsized"] else (230, 200, 90)
        pygame.draw.polygon(screen, hull_color, hull_screen)
        pygame.draw.polygon(screen, (30, 20, 10), hull_screen, 2)

        hud_lines = [
            f"survival time: {game['t']:.1f} s",
            f"roll: {math.degrees(theta):+.1f} deg   (capsize at +-{THETA_CAPSIZE_DEG:.0f} deg)",
            f"omega_heave={omega_heave:.2f} rad/s   omega_roll={omega_roll:.2f} rad/s",
        ]
        for i, line in enumerate(hud_lines):
            screen.blit(font.render(line, True, (230, 230, 230)), (12, 10 + 22 * i))

        if game["capsized"]:
            msg = big_font.render("CAPSIZED", True, (255, 120, 120))
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 60))
            sub = font.render(f"survived {game['t']:.1f} s -- click or press any key to retry",
                               True, (230, 230, 230))
            screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
