"""Test speed_wobble: stable below critical speed, real exponential growth
above it, and a numerical-integrator artifact where coarse-step Euler fakes
instability on a physically stable system while RK4 does not."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import speed_wobble as sw

m, b0, b1, k = 1.0, 2.0, 0.5, 10.0
v_crit = sw.critical_speed(b0, b1)
assert abs(v_crit - 4.0) < 1e-9

t_fine = np.linspace(0, 30, 3000)

# 1. below critical speed: RK4 with a fine step shows real decay
traj_slow = sw.simulate_wobble(v_crit * 0.5, m, b0, b1, k, 0.1, 0.0, t_fine, method="rk4")
rate_slow = sw.envelope_growth_rate(traj_slow[:, 0], t_fine)
assert rate_slow < 0

# 2. above critical speed: RK4 with a fine step shows real growth
traj_fast = sw.simulate_wobble(v_crit * 1.5, m, b0, b1, k, 0.1, 0.0, t_fine, method="rk4")
rate_fast = sw.envelope_growth_rate(traj_fast[:, 0], t_fine)
assert rate_fast > 0

# 3. numerical artifact: at the SAME physically stable speed, a large-step
#    Euler integration fakes growth that a large-step RK4 does not
t_coarse = np.linspace(0, 30, 150)
traj_euler_coarse = sw.simulate_wobble(v_crit * 0.5, m, b0, b1, k, 0.1, 0.0, t_coarse, method="euler")
rate_euler_coarse = sw.envelope_growth_rate(traj_euler_coarse[:, 0], t_coarse)
traj_rk4_coarse = sw.simulate_wobble(v_crit * 0.5, m, b0, b1, k, 0.1, 0.0, t_coarse, method="rk4")
rate_rk4_coarse = sw.envelope_growth_rate(traj_rk4_coarse[:, 0], t_coarse)

assert rate_euler_coarse > 0       # Euler fakes instability
assert rate_rk4_coarse < 0         # RK4 correctly shows real decay

print("test_speed_wobble: all checks passed")
