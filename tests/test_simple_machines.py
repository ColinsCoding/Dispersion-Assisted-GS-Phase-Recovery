"""Test dgs.simple_machines: the six mechanical-advantage formulas, the conservation
of work (F_effort*d_effort = F_load*d_load, so MA = velocity ratio for an ideal
machine), and efficiency (real MA = ideal*eta < ideal; W_out < W_in with the loss to
friction, never a free lunch)."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import simple_machines as sm

# 1. mechanical advantage from geometry
assert sm.lever_ma(2, 0.5) == 4
assert sm.inclined_plane_ma(5, 1) == 5
assert sm.pulley_ma(4) == 4
assert math.isclose(sm.wheel_and_axle_ma(0.3, 0.05), 6)
assert math.isclose(sm.screw_ma(0.2, 0.005), 2 * math.pi * 0.2 / 0.005)
assert math.isclose(sm.wedge_ma(0.1, 0.02), 5)
# inclined plane MA = 1/sin(theta)
assert math.isclose(sm.inclined_plane_ma(1.0, math.sin(math.radians(30))),
                    1 / math.sin(math.radians(30)))

# 2. CONSERVATION OF WORK (ideal machine): W_in = W_out, and MA = velocity ratio
Fe, de, ma = 100.0, 4.0, 4
Fl = sm.output_force(Fe, ma)                 # ideal: efficiency 1
dl = de / ma                                 # load moves 1/MA as far
assert math.isclose(sm.work_in(Fe, de), sm.work_out(Fl, dl))     # 400 J = 400 J
assert math.isclose(ma, sm.velocity_ratio(de, dl))              # MA = VR (ideal)
assert Fl == 400.0                                              # force multiplied by MA

# 3. output force and ideal behavior
assert sm.output_force(100, 4) == 400                           # eff=1 default
assert sm.output_force(100, 4, efficiency=0.5) == 200

# 4. real machine: actual MA = ideal * efficiency, and it's smaller
assert sm.actual_mechanical_advantage(4, 0.8) == 3.2
assert sm.actual_mechanical_advantage(4, 0.8) < 4
assert sm.output_force(100, 4, 0.8) == 320                      # lifts 320 N, not 400

# 5. efficiency = W_out / W_in <= 1, and it accounts for the missing work
assert sm.efficiency(400, 400) == 1.0                           # ideal
assert math.isclose(sm.efficiency(350, 400), 0.875)            # 50 J lost to friction
assert sm.efficiency(350, 400) < 1.0
# real out is less than in (energy conservation with loss)
w_in = sm.work_in(100, 4)
w_out_real = sm.work_out(sm.output_force(100, 4, 0.8), 4 / 4)
assert w_out_real < w_in                                        # no free lunch

# 6. kwarg bounds
for bad in (lambda: sm.lever_ma(0, 1),
            lambda: sm.inclined_plane_ma(1, 2),                 # height > length
            lambda: sm.pulley_ma(2.5),                          # non-integer ropes
            lambda: sm.screw_ma(0.2, 0),
            lambda: sm.output_force(100, 4, efficiency=1.5),    # eff > 1
            lambda: sm.efficiency(1, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_simple_machines: all checks passed")
