"""Test the F=ma digital-mechanics approximation: velocity is exact for
Euler under constant force (a real, provable special case, not a
coincidence), position error shrinks linearly with dt (genuine first-order
convergence, checked quantitatively, not just 'gets closer'), and MuJoCo's
RK4 integrator matches the exact solution far more tightly than Euler at
any practical step size -- a real demonstration that integrator CHOICE
matters, not just step size."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.spoon_physics import (
    exact_constant_force_solution, integrate_force_euler, simulate_force_mujoco, SpoonScoop,
)

F, m, v0, x0, duration = 5.0, 0.15, 0.0, 0.0, 0.5
v_exact, x_exact = exact_constant_force_solution(F, m, v0, x0, duration)

# 1. exact solution matches simple kinematics directly
a = F / m
assert abs(v_exact - a * duration) < 1e-12
assert abs(x_exact - 0.5 * a * duration ** 2) < 1e-12

# 2. Euler's velocity is essentially EXACT for constant force (dv/dt=a=const
# sums exactly regardless of step count) -- true at every dt tested, not
# just in the dt->0 limit
for dt in (0.1, 0.01, 0.001):
    _, v, _ = integrate_force_euler(F, m, v0, x0, duration, dt)
    assert abs(v[-1] - v_exact) < 1e-9

# 3. Euler's POSITION error is genuinely first-order: halving dt should
# roughly halve the error (checked as an order-of-magnitude ratio, not an
# exact factor, since duration/dt isn't always an integer multiple)
errors = []
for dt in (0.01, 0.001, 0.0001):
    _, _, x = integrate_force_euler(F, m, v0, x0, duration, dt)
    errors.append(abs(x[-1] - x_exact))
ratio_1 = errors[0] / errors[1]
ratio_2 = errors[1] / errors[2]
assert 8 < ratio_1 < 12   # dt shrinks 10x -> first-order error shrinks ~10x
assert 8 < ratio_2 < 12

# 4. MuJoCo's RK4 integrator matches the exact solution far more tightly
# than Euler does at ANY of the step sizes tested above
_, v_mj, x_mj = simulate_force_mujoco(F, m, v0, x0, duration, dt=0.001)
mujoco_v_err = abs(v_mj[-1] - v_exact)
mujoco_x_err = abs(x_mj[-1] - x_exact)
assert mujoco_x_err < 1e-9
assert mujoco_x_err < errors[-1]   # more accurate than Euler at dt=0.0001, using a coarser dt=0.001

# 5. SpoonScoop's force profile is smooth: zero at both ends, peaks at
# the midpoint, matches sin(pi*t/duration) exactly by construction
scoop = SpoonScoop(peak_force=0.1, mass=0.05, duration=0.6)
assert abs(scoop.force_at(0.0)) < 1e-12
assert abs(scoop.force_at(0.6)) < 1e-12
assert abs(scoop.force_at(0.3) - 0.1) < 1e-9   # midpoint = peak
assert scoop.force_at(0.7) == 0.0               # force is exactly zero after duration ends

# 6. SpoonScoop produces a realistic, bounded scoop (not a launch): peak
# velocity and net displacement stay in a sane kitchen-utensil range
t, v, x = scoop.integrate_euler(dt=1e-4)
assert 0.1 < v.max() < 5.0
assert 0.02 < x[-1] < 1.0
# the item decelerates back toward rest by the end (force goes to 0 at
# duration, but by symmetry of a half-sine, velocity should still be
# positive and roughly at its post-peak value, not exactly zero)
assert v[-1] > 0

print("all dgs.spoon_physics tests passed")
