"""Test the chicken BBQ flip physics: closed-form projectile motion matches
a direct quadratic-formula check, the flip impulse correctly couples
linear and angular momentum via the offset (torque) term, landing
classification is correct at the three physically meaningful angles, and
char accumulation only affects the side actually facing the grill."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.chicken_bbq_simulator import (
    apply_flip_impulse, time_to_return_to_height, flip_trajectory,
    normalize_angle, classify_landing, within_grill_bounds,
    update_char_level, char_color, disk_moment_of_inertia,
    update_sauce_level, baste,
)

mass, radius = 0.15, 0.06
I = disk_moment_of_inertia(mass, radius)

# 1. disk moment of inertia matches the textbook formula
assert I == 0.5 * mass * radius ** 2

# 2. zero offset gives ZERO spin -- a straight-up strike through the
# center of mass has no torque, physically
vy0, omega0 = apply_flip_impulse(strike_force=12.0, contact_time=0.03,
                                  offset_from_center=0.0, mass=mass, moment_of_inertia=I)
assert omega0 == 0.0
assert vy0 > 0

# 3. doubling the offset exactly doubles omega0 (torque is linear in offset)
_, omega_1x = apply_flip_impulse(12.0, 0.03, 0.005, mass, I)
_, omega_2x = apply_flip_impulse(12.0, 0.03, 0.010, mass, I)
assert abs(omega_2x - 2 * omega_1x) < 1e-9

# 4. time_to_return_to_height matches the textbook closed form 2*vy0/g
t = time_to_return_to_height(y0=0.0, vy0=4.0, target_y=0.0, g=9.80665)
assert abs(t - 2 * 4.0 / 9.80665) < 1e-9

# 5. a real, tuned flip (5mm offset) lands close to pi radians -- a
# genuine "flip" -- and classify_landing correctly reports side B
vy0, omega0 = apply_flip_impulse(strike_force=12.0, contact_time=0.03,
                                  offset_from_center=0.005, mass=mass, moment_of_inertia=I)
t_land = time_to_return_to_height(0.0, vy0, 0.0)
_, _, angle_land = flip_trajectory(0.0, 0.0, 0.0, vy0, omega0, 0.0, t_land)
assert abs(normalize_angle(angle_land) - np.pi) < np.deg2rad(10)
assert classify_landing(angle_land) == "B"

# 6. classify_landing at the three canonical angles
assert classify_landing(0.0) == "A"
assert classify_landing(2 * np.pi) == "A"
assert classify_landing(np.pi) == "B"
assert classify_landing(np.pi / 2) == "bad"          # a quarter-turn is neither flat side down

# 7. within_grill_bounds
assert within_grill_bounds(0.5, 0.0, 1.0) is True
assert within_grill_bounds(-0.1, 0.0, 1.0) is False
assert within_grill_bounds(1.5, 0.0, 1.0) is False

# 8. char only accumulates on the side actually facing down; the other
# side is untouched (it isn't in contact with the heat)
char = {"A": 0.0, "B": 0.0}
char, burnt = update_char_level(char, "A", dt=1.0, char_rate=0.1)
assert abs(char["A"] - 0.1) < 1e-9
assert char["B"] == 0.0
assert burnt is False

# 9. char is clamped at burn_threshold and reports burnt once reached
char = {"A": 0.95, "B": 0.0}
char, burnt = update_char_level(char, "A", dt=1.0, char_rate=0.5, burn_threshold=1.0)
assert char["A"] == 1.0
assert burnt is True

# 10. char_color interpolates raw -> browned -> charred monotonically
# darker as char level increases (sum of RGB channels decreases)
brightness = [sum(char_color(c)) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
assert all(brightness[i] > brightness[i + 1] for i in range(len(brightness) - 1))

# 11. bad input validation
try:
    apply_flip_impulse(12.0, 0.03, 0.005, mass=0.0, moment_of_inertia=I)
except ValueError:
    pass
else:
    raise AssertionError("should reject non-positive mass")

try:
    update_char_level({"A": 0.0, "B": 0.0}, "C", dt=1.0)
except ValueError:
    pass
else:
    raise AssertionError("should reject an invalid side")

# 12. sauce dries out only on the side facing the grill
sauce = {"A": 1.0, "B": 1.0}
sauce, dried = update_sauce_level(sauce, "A", dt=1.0, dry_rate=0.1)
assert abs(sauce["A"] - 0.9) < 1e-9
assert sauce["B"] == 1.0
assert dried is False

# 13. sauce is clamped at zero and reports dried_out once it hits zero
sauce = {"A": 0.05, "B": 1.0}
sauce, dried = update_sauce_level(sauce, "A", dt=1.0, dry_rate=0.1)
assert sauce["A"] == 0.0
assert dried is True

# 14. basting replenishes the given side, clamped at max_level, and
# doesn't touch the other side
sauce = {"A": 0.1, "B": 0.1}
sauce = baste(sauce, "A", amount=0.5, max_level=1.0)
assert abs(sauce["A"] - 0.6) < 1e-9
assert sauce["B"] == 0.1
sauce = baste(sauce, "A", amount=0.8, max_level=1.0)
assert sauce["A"] == 1.0   # clamped, not 1.4

# 15. invalid side is rejected for both sauce functions
try:
    update_sauce_level({"A": 1.0, "B": 1.0}, "C", dt=1.0)
except ValueError:
    pass
else:
    raise AssertionError("should reject an invalid side")

try:
    baste({"A": 1.0, "B": 1.0}, "C")
except ValueError:
    pass
else:
    raise AssertionError("should reject an invalid side")

print("all dgs.chicken_bbq_simulator tests passed")
