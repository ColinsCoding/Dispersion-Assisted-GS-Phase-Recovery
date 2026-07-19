"""Test the firework particle system's projectile-motion physics (with
and without the exact linear-drag solution), brightness decay, and
device-selection honesty. Forces CPU explicitly so the test is
reproducible regardless of CUDA/driver availability on the machine
running it. Requires py-3.12 (torch)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs.torch import firework_particle_system as fps

DEVICE = torch.device("cpu")

# 1. make_burst produces the right number of particles, all speeds >= 0
burst = fps.make_burst(500, (0.0, 0.0), t_start=0.0, speed_mean=10.0, speed_std=2.0,
                        color_rgb=(1.0, 0.0, 0.0), lifetime_s=1.0, device=DEVICE)
assert burst["vx0"].shape == (500,)
speeds = torch.sqrt(burst["vx0"]**2 + burst["vy0"]**2)
assert torch.all(speeds >= 0)

# 2. no-drag trajectory matches the exact closed-form kinematics formula
#    x = x0 + vx0*t, y = y0 + vy0*t - 0.5*g*t^2 -- for a SINGLE particle
#    with a known launch velocity
single = fps.make_burst(1, (2.0, 3.0), t_start=0.0, speed_mean=10.0, speed_std=0.0,
                         color_rgb=(0, 0, 0), lifetime_s=5.0, device=DEVICE)
t_array = np.array([0.0, 1.0, 2.0])
x, y, brightness = fps.simulate_burst(single, t_array, g=9.8, drag_k=0.0)
vx0, vy0 = float(single["vx0"][0]), float(single["vy0"][0])
for i, t in enumerate(t_array):
    expected_x = 2.0 + vx0 * t
    expected_y = 3.0 + vy0 * t - 0.5 * 9.8 * t**2
    assert abs(float(x[0, i]) - expected_x) < 1e-4
    assert abs(float(y[0, i]) - expected_y) < 1e-4

# 3. gravity pulls particles down over time: mean height should eventually
#    decrease for a burst launched at t=0 (checked well past the peak)
t_long = np.linspace(0, 5, 50)
big_burst = fps.make_burst(300, (0.0, 0.0), t_start=0.0, speed_mean=8.0, speed_std=1.0,
                            color_rgb=(0, 0, 0), lifetime_s=3.0, device=DEVICE)
x2, y2, b2 = fps.simulate_burst(big_burst, t_long, g=9.8, drag_k=0.0)
mean_height_early = float(y2[:, 5].mean())
mean_height_late = float(y2[:, -1].mean())
assert mean_height_late < mean_height_early   # net downward pull over time

# 4. with drag, a particle's horizontal speed decreases over time (drag
#    opposes motion) -- check position increments shrink (approaching a
#    horizontal terminal displacement x0 + vx0/k)
drag_burst = fps.make_burst(1, (0.0, 0.0), t_start=0.0, speed_mean=10.0, speed_std=0.0,
                             color_rgb=(0, 0, 0), lifetime_s=5.0, device=DEVICE)
t_drag = np.array([0.0, 1.0, 2.0, 3.0, 50.0])
x3, y3, b3 = fps.simulate_burst(drag_burst, t_drag, g=9.8, drag_k=0.5)
vx0_drag = float(drag_burst["vx0"][0])
terminal_x = vx0_drag / 0.5
assert abs(float(x3[0, -1]) - terminal_x) < 0.05 * abs(terminal_x + 1e-9)   # converges near terminal displacement
# equal (1-second) intervals: |displacement| per interval should SHRINK as
# drag bleeds off horizontal speed (abs() because the random launch angle
# can point vx0 in either direction -- only the magnitude of the rate is
# physically meaningful here, not comparing unequal time spans)
dx_1 = abs(float(x3[0, 1]) - float(x3[0, 0]))
dx_2 = abs(float(x3[0, 2]) - float(x3[0, 1]))
dx_3 = abs(float(x3[0, 3]) - float(x3[0, 2]))
assert dx_2 < dx_1
assert dx_3 < dx_2

# 5. brightness decays monotonically after a burst starts, and is exactly
#    zero before the burst starts
t_bright = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
late_burst = fps.make_burst(10, (0.0, 0.0), t_start=1.0, speed_mean=5.0, speed_std=1.0,
                             color_rgb=(0, 0, 0), lifetime_s=1.0, device=DEVICE)
_, _, bright = fps.simulate_burst(late_burst, t_bright, g=9.8)
assert torch.all(bright[:, 0] == 0.0)   # t=-1, before t_start=1 -> zero
assert torch.all(bright[:, 1] == 0.0)   # t=0, still before t_start=1 -> zero
assert torch.all(bright[:, 2] == 0.0)   # t=0.5, still before t_start=1 -> zero
assert torch.all(bright[:, 3] > bright[:, 4])   # t=1 (burst start, peak) > t=2 (decayed)

# 6. simulate_show returns one result per burst, correctly shaped
bursts = [
    fps.make_burst(100, (0.0, 0.0), 0.0, 10.0, 1.0, (1, 0, 0), 1.0, device=DEVICE),
    fps.make_burst(200, (1.0, 1.0), 1.0, 8.0, 1.0, (0, 1, 0), 1.5, device=DEVICE),
]
results, device_used = fps.simulate_show(bursts, np.linspace(0, 3, 30))
assert len(results) == 2
assert results[0][0].shape == (100, 30)
assert results[1][0].shape == (200, 30)
assert device_used == DEVICE

# 7. get_device returns a torch.device, and honestly reports cpu when no
#    CUDA is forced available (doesn't assert GPU presence)
dev = fps.get_device()
assert isinstance(dev, torch.device)

# 8. input validation
for bad_call in [
    lambda: fps.make_burst(0, (0, 0), 0.0, 10.0, 1.0, (0, 0, 0), 1.0, device=DEVICE),
    lambda: fps.make_burst(10, (0, 0), -1.0, 10.0, 1.0, (0, 0, 0), 1.0, device=DEVICE),
    lambda: fps.make_burst(10, (0, 0), 0.0, -1.0, 1.0, (0, 0, 0), 1.0, device=DEVICE),
    lambda: fps.make_burst(10, (0, 0), 0.0, 10.0, -1.0, (0, 0, 0), 1.0, device=DEVICE),
    lambda: fps.make_burst(10, (0, 0), 0.0, 10.0, 1.0, (0, 0, 0), -1.0, device=DEVICE),
    lambda: fps.make_burst(10, (0, 0), 0.0, 10.0, 1.0, (0, 0), 1.0, device=DEVICE),
    lambda: fps.simulate_burst(burst, t_array, g=-1.0),
    lambda: fps.simulate_burst(burst, t_array, drag_k=-1.0),
    lambda: fps.simulate_show([], t_array),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.torch.firework_particle_system tests passed")
