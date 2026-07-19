"""Test memory_circuits: SRAM bistability/noise margin, DRAM RC decay and
refresh interval, and the DRAM-vs-EEPROM retention-time ratio."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import memory_circuits as mc

# 1. a symmetric SRAM cell has three fixed points: low-stable, unstable
#    threshold (near V_dd/2), high-stable
states = mc.sram_latch_stability(V_dd=1.0, V_th=0.5, gain=12.0)
assert len(states) == 3
low, mid, high = sorted(states)
assert low < 0.1 and high > 0.9
assert abs(mid - 0.5) < 1e-3

# 2. noise margin is the gap from each stable state to the unstable threshold
margin = mc.sram_noise_margin(states)
assert abs(margin - min(mid - low, high - mid)) < 1e-9
assert margin > 0.3   # this symmetric, high-gain cell should have a wide margin

# 3. DRAM cell decay follows exponential RC discharge
V0, R, C = 3.3, 3e12, 30e-15
V_at_tau = mc.dram_cell_decay(V0, t=R * C, R_leak=R, C_cell=C)
assert abs(V_at_tau - V0 / np.e) < 1e-6

# 4. refresh interval increases with leakage resistance (less leaky -> longer)
t_leaky = mc.dram_refresh_interval(V0, V_read_threshold=1.5, R_leak=1e9, C_cell=C)
t_tight = mc.dram_refresh_interval(V0, V_read_threshold=1.5, R_leak=1e12, C_cell=C)
assert t_tight > t_leaky

# 5. EEPROM retention is many orders of magnitude longer than DRAM's, driven
#    entirely by the oxide leakage resistance being many orders larger
cmp = mc.compare_retention_times()
assert cmp["eeprom_retention_s"] / cmp["dram_retention_s"] > 1e6
assert 0.01 < cmp["dram_retention_ms"] < 200       # realistic DRAM refresh ballpark

print("test_memory_circuits: all checks passed")
