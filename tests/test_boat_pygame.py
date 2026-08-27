"""Test the pure physics/logic in dgs.boat_pygame (no pygame/display needed
for any of this): RK4 integration matches the exact damped-SHM solution,
damping actually dissipates energy, wave impulses only push the right
direction, and the capsize threshold triggers exactly at the boundary."""
import sys, pathlib, math, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.boat_pygame import (
    boat_accelerations, rk4_step, analytic_underdamped_shm,
    verify_rk4_matches_analytic, random_wave_impulse, is_capsized,
    boat_hull_points, transform_hull, THETA_CAPSIZE_DEG,
)

# 1. RK4 integration matches the closed-form damped-SHM solution
check = verify_rk4_matches_analytic()
assert check["matches"], check
assert check["abs_error"] < 1e-6

# 2. Undamped restoring acceleration points back toward equilibrium
xdot, xddot, thetadot, thetaddot = boat_accelerations((1.0, 0.0, 0.0, 0.0),
                                                        omega_heave=2.0, zeta_heave=0.0,
                                                        omega_roll=1.0, zeta_roll=0.0)
assert xddot < 0  # displaced positive x -> pulled back negative

# 3. Damping actually removes energy over time (zeta > 0 decays amplitude)
state_damped = (1.0, 0.0, 0.0, 0.0)
state_undamped = (1.0, 0.0, 0.0, 0.0)
dt = 0.01
for _ in range(200):
    state_damped = rk4_step(state_damped, dt, omega_heave=3.0, zeta_heave=0.15,
                             omega_roll=1.0, zeta_roll=0.0)
    state_undamped = rk4_step(state_undamped, dt, omega_heave=3.0, zeta_heave=0.0,
                               omega_roll=1.0, zeta_roll=0.0)


def energy(x, xdot, omega):
    return 0.5 * xdot ** 2 + 0.5 * omega ** 2 * x ** 2


e_damped = energy(state_damped[0], state_damped[1], 3.0)
e_undamped = energy(state_undamped[0], state_undamped[1], 3.0)
assert e_damped < e_undamped

# 4. Wave impulses: heave kick is always positive (upward), strength scales it
rng = random.Random(0)
for _ in range(50):
    d_xdot, d_thetadot = random_wave_impulse(rng, strength=1.0)
    assert d_xdot > 0
d_xdot_weak, _ = random_wave_impulse(random.Random(1), strength=1.0)
d_xdot_strong, _ = random_wave_impulse(random.Random(1), strength=3.0)
assert d_xdot_strong > d_xdot_weak

try:
    random_wave_impulse(rng, strength=-1.0)
    assert False, "should have raised ValueError"
except ValueError:
    pass

# 5. Capsize threshold: exactly at the boundary and beyond it, not before
just_under = math.radians(THETA_CAPSIZE_DEG) - 1e-4
just_over = math.radians(THETA_CAPSIZE_DEG) + 1e-4
assert not is_capsized(just_under)
assert is_capsized(just_over)
assert is_capsized(math.radians(THETA_CAPSIZE_DEG))  # exactly at threshold counts
assert not is_capsized(-just_under)
assert is_capsized(-just_over)

# 6. Hull transform: zero rotation/translation is the identity
hull = boat_hull_points()
identity = transform_hull(hull, x_pixels=0, theta_rad=0.0, cx=0, cy=0)
for (px, py), (qx, qy) in zip(hull, identity):
    assert abs(px - qx) < 1e-9 and abs(py - qy) < 1e-9

# 7. Hull transform: rotating by pi flips the hull through the origin
rotated = transform_hull(hull, x_pixels=0, theta_rad=math.pi, cx=0, cy=0)
for (px, py), (qx, qy) in zip(hull, rotated):
    assert abs(qx - (-px)) < 1e-9
    assert abs(qy - (-py)) < 1e-9

# 8. Hull transform: positive heave x_pixels moves the hull UP on screen
#    (screen y decreases), matching "heave up = pixels up" convention
lifted = transform_hull(hull, x_pixels=50, theta_rad=0.0, cx=0, cy=0)
for (px, py), (qx, qy) in zip(hull, lifted):
    assert abs(qy - (py - 50)) < 1e-9

print("all dgs.boat_pygame tests passed")
