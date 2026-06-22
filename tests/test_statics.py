"""Test statics: equilibrium (sum F=0, sum tau=0), lever, beam reactions, bracket."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import statics as st

# 1. simply-supported beam: central load splits 50/50; reactions always sum to the load
Ra, Rb = st.beam_reactions([(5.0, 100.0)], 0.0, 10.0)
assert np.isclose(Ra, 50) and np.isclose(Rb, 50)
Ra, Rb = st.beam_reactions([(8.0, 100.0)], 0.0, 10.0)            # load near B -> B takes more
assert np.isclose(Ra, 20) and np.isclose(Rb, 80) and np.isclose(Ra + Rb, 100)
# two loads: reactions still sum to the total
Ra, Rb = st.beam_reactions([(2.0, 60.0), (7.0, 40.0)], 0.0, 10.0)
assert np.isclose(Ra + Rb, 100)

# 2. lever / torque balance: F1 d1 = F2 d2
F2 = st.lever(200.0, 1.0, 2.5)
assert np.isclose(200.0 * 1.0, F2 * 2.5) and np.isclose(F2, 80.0)

# 3. hinged bracket: T = weight/sin(theta), Hx = T cos(theta), Hy = 0
T, Hx, Hy = st.hinged_bracket(2.0, 500.0, 30.0)
assert np.isclose(T, 500.0 / np.sin(np.radians(30)))            # 1000 N
assert np.isclose(Hx, T * np.cos(np.radians(30))) and abs(Hy) < 1e-9

# 4. equilibrium check: the solved beam balances; perturb a reaction and it does not
Ra, Rb = st.beam_reactions([(5.0, 100.0)], 0.0, 10.0)
forces = [(0, Ra), (0, -100.0), (0, Rb)]; pts = [(0, 0), (5, 0), (10, 0)]
assert st.in_equilibrium(forces, pts)
bad = [(0, Ra + 5), (0, -100.0), (0, Rb)]                       # 5 N off -> not balanced
assert not st.in_equilibrium(bad, pts)

# 5. a balanced seesaw has zero net torque about the pivot
forces_ss = [(0, -200.0), (0, -80.0)]; pts_ss = [(-1.0, 0), (2.5, 0)]   # 200@-1, 80@+2.5
assert abs(st.net_torque(forces_ss, pts_ss, pivot=(0, 0))) < 1e-9       # 200*1 = 80*2.5

# 6. when forces balance, net torque is independent of the chosen pivot
assert np.allclose(np.abs(st.net_force(forces)), 0)
tau0 = st.net_torque(forces, pts, pivot=(0, 0))
tau1 = st.net_torque(forces, pts, pivot=(3.7, 1.2))
assert np.isclose(tau0, tau1)                                  # pivot-independent (and ~0)

print(f"TEST PASS  (beam 50/50 + 20/80 + sum=load; lever 200*1=80*2.5; bracket "
      f"T={T:.0f}N Hx={Hx:.0f}N Hy=0; equilibrium detects 5N imbalance; "
      f"torque pivot-independent when F balances)")
