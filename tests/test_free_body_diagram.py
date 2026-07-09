"""Test dgs.free_body_diagram: net force/torque, pivot-independence of torque when
sum F=0, the equilibrium check, solve_reactions on a simply supported beam (70/30) and
a ladder against a smooth wall (mu_min = 1/(2 tan theta)), and that the assembled
reactions put the body in equilibrium."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import free_body_diagram as fbd

# 1. net force and torque
forces = [((3.0, 4.0), (0.0, 0.0)), ((-3.0, -4.0), (2.0, 0.0))]
assert np.allclose(fbd.net_force(forces), [0.0, 0.0])
# a couple (net force 0): torque is the SAME about any pivot
couple = [((0.0, 10.0), (1.0, 0.0)), ((0.0, -10.0), (-1.0, 0.0))]
assert np.isclose(fbd.net_torque(couple, (0, 0)), 20.0)
assert np.isclose(fbd.net_torque(couple, (5, 5)), fbd.net_torque(couple, (0, 0)))

# 2. equilibrium check
assert fbd.is_in_equilibrium(couple) is False              # a couple spins it
bal = [((0, 10), (1, 0)), ((0, -10), (1, 0))]              # equal/opposite, same point
assert fbd.is_in_equilibrium(bal)

# 3. simply supported beam, L=10, 100 N at x=3 -> R_A=70, R_B=30
Ra, Rb = fbd.beam_reactions(10.0, [(100.0, 3.0)])
assert np.isclose(Ra, 70.0) and np.isclose(Rb, 30.0)
assert np.isclose(Ra + Rb, 100.0)                          # reactions carry the whole load
assert np.allclose(fbd.beam_reactions(10.0, [(100.0, 5.0)]), (50.0, 50.0))   # centered -> 50/50
# multiple loads superpose
Ra2, Rb2 = fbd.beam_reactions(10.0, [(100.0, 2.0), (100.0, 8.0)])
assert np.isclose(Ra2, 100.0) and np.isclose(Rb2, 100.0)   # symmetric pair

# 4. solve_reactions directly, and confirm the solution is in equilibrium
loads = [((0.0, -100.0), (3.0, 0.0))]
reactions = [((1.0, 0.0), (0.0, 0.0)), ((0.0, 1.0), (0.0, 0.0)), ((0.0, 1.0), (10.0, 0.0))]
mags, rforces = fbd.solve_reactions(loads, reactions)
assert np.allclose(mags, [0.0, 70.0, 30.0], atol=1e-9)     # R_x=0, R_A=70, R_B=30
assert fbd.is_in_equilibrium(loads + rforces)              # everything balances

# 5. ladder against a smooth wall: mu_min = 1/(2 tan theta)
W, th, L = 200.0, np.radians(60), 4.0
ld = [((0.0, -W), (0.5 * L * np.cos(th), 0.5 * L * np.sin(th)))]
rx = [((0.0, 1.0), (0.0, 0.0)),                            # floor normal
      ((1.0, 0.0), (0.0, 0.0)),                            # floor friction
      ((-1.0, 0.0), (L * np.cos(th), L * np.sin(th)))]     # wall normal
m, rf = fbd.solve_reactions(ld, rx)
N, f, Nw = m
assert np.isclose(N, W)                                    # floor holds the full weight
assert np.isclose(f, W / (2 * np.tan(th)))                # friction needed
assert np.isclose(f / N, 1 / (2 * np.tan(th)))            # mu_min = 1/(2 tan theta)
assert fbd.is_in_equilibrium(ld + rf)

# 6. determinacy / singular geometry
try:
    fbd.solve_reactions(loads, reactions[:2]); assert False    # only 2 reactions
except ValueError:
    pass
# three PARALLEL (all vertical) reactions cannot resist a horizontal load -> singular
parallel = [((0, 1), (0, 0)), ((0, 1), (5, 0)), ((0, 1), (10, 0))]
try:
    fbd.solve_reactions([((5.0, 0.0), (5, 0))], parallel); assert False
except ValueError:
    pass

# 7. kwarg bounds
for bad in (lambda: fbd.net_force([]),
            lambda: fbd.beam_reactions(0, [(1, 0)])):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_free_body_diagram: all checks passed")
