"""Test dgs.finite_square_well: the well-strength z0, the bound-state count
floor(z0/(pi/2))+1, the transcendentally-solved energies (right count, all in
(0,V0), below the infinite well), the penetration depth (deeper for weakly bound
states, diverging as E->V0), and the deep-well -> infinite-well limit."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import finite_square_well as fw

# 1. well strength z0 = a sqrt(2 m V0)/hbar
assert np.isclose(fw.well_strength(20.0, 1.0), np.sqrt(2 * 20.0))
assert fw.well_strength(80.0, 1.0) == 2 * fw.well_strength(20.0, 1.0)   # ~sqrt(V0)

# 2. number of bound states = floor(z0/(pi/2)) + 1 (at least one, always)
assert fw.num_bound_states(1.0, 1.0) == 1                    # shallow well: one state
assert fw.num_bound_states(20.0, 1.0) == 5
z0 = fw.well_strength(20.0, 1.0)
assert fw.num_bound_states(20.0, 1.0) == int(np.floor(z0 / (np.pi / 2)) + 1)

# 3. the solved energies: right count, all bound, ordered, below the infinite well
for V0 in (5.0, 20.0, 50.0):
    E = fw.bound_state_energies(V0, 1.0)
    assert len(E) == fw.num_bound_states(V0, 1.0)            # solver finds them all
    assert np.all((E > 0) & (E < V0))                       # genuinely bound
    assert np.all(np.diff(E) > 0)                           # ascending, non-degenerate
    Einf = fw.infinite_well_energies(1.0, n_levels=len(E))
    assert np.all(E < Einf)                                 # finite well leaks -> lower

# 4. penetration depth = hbar/sqrt(2 m (V0-E)): deeper for weakly bound states
E = fw.bound_state_energies(20.0, 1.0)
assert np.isclose(fw.penetration_depth(E[0], 20.0), 1 / np.sqrt(2 * (20.0 - E[0])))
depths = [fw.penetration_depth(e, 20.0) for e in E]
assert np.all(np.diff(depths) > 0)                          # higher state penetrates more
assert depths[-1] > 10 * depths[0]                          # near-V0 state leaks far
# it diverges as E -> V0
assert fw.penetration_depth(19.9999, 20.0) > fw.penetration_depth(10.0, 20.0)

# 5. deep-well limit: finite levels approach the infinite-well levels
Ed = fw.bound_state_energies(2000.0, 1.0)
Ei = fw.infinite_well_energies(1.0, n_levels=3)
assert np.allclose(Ed[:3], Ei, rtol=0.05)                   # within 5% for a very deep well
assert np.all(Ed[:3] < Ei)                                  # still just below

# 6. kwarg bounds
for bad in (lambda: fw.well_strength(0, 1),
            lambda: fw.penetration_depth(25.0, 20.0),       # E > V0 (not bound)
            lambda: fw.penetration_depth(-1.0, 20.0),
            lambda: fw.infinite_well_energies(0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_finite_square_well: all checks passed")
