"""Test dgs.schrodinger_lennard_jones: the LJ well geometry, the harmonic frequency /
ladder near the bottom, and the numerically solved bound vibrational levels -- all E<0,
the ground state near (but below) the harmonic zero-point prediction, anharmonic crowding
toward dissociation, and more levels for a heavier molecule."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import schrodinger_lennard_jones as sj

# 1. well geometry: V(sigma)=0, minimum -eps at 2^(1/6) sigma
from dgs.lennard_jones import lj_potential
assert np.isclose(lj_potential(1.0, 1.0, 1.0), 0.0)               # V(sigma) = 0
r0, Vmin = sj.lj_minimum()
assert np.isclose(r0, 2 ** (1 / 6)) and Vmin == -1.0
assert np.isclose(lj_potential(r0, 1.0, 1.0), -1.0)              # bottom = -eps

# 2. harmonic frequency and ladder near the bottom
mu = 200.0
w = sj.harmonic_frequency(mu)
assert np.isclose(w, np.sqrt(36 * 2 ** (2 / 3) / mu))           # k = 36*2^(2/3), omega=sqrt(k/mu)
lv = sj.harmonic_levels(mu, 4)
assert np.allclose(lv, -1.0 + (np.arange(4) + 0.5) * w)         # E_n = -eps+(n+1/2)hbar w
assert np.allclose(np.diff(lv), w)                              # evenly spaced (harmonic)

# 3. the numerically solved bound states
E = sj.solve_bound_states(mu)
assert len(E) >= 4 and np.all(E < 0)                           # a few bound levels, all E<0
assert np.all(E > Vmin)                                        # every level above the floor (zero-point)
# ground state near the harmonic zero-point prediction, but BELOW it (anharmonic softening)
E0_harm = Vmin + 0.5 * w
assert np.isclose(E[0], E0_harm, rtol=0.05)
assert E[0] < E0_harm

# 4. anharmonic crowding: the gaps SHRINK toward dissociation (unlike the even oscillator)
gaps = np.diff(E)
assert np.all(gaps[:-1] > gaps[1:])                            # strictly decreasing spacing
assert gaps[0] > gaps[-1]

# 5. a heavier molecule holds MORE bound vibrational states
assert len(sj.solve_bound_states(800.0)) > len(sj.solve_bound_states(200.0))
# and higher levels deviate more from the harmonic ladder than the ground state
if len(E) >= 3:
    assert abs(E[2] - sj.harmonic_levels(mu, 3)[2]) > abs(E[0] - sj.harmonic_levels(mu, 3)[0])

# 6. kwarg bounds
for bad in (lambda: sj.lj_minimum(0),
            lambda: sj.harmonic_frequency(0),
            lambda: sj.solve_bound_states(200.0, r_lo=0),
            lambda: sj.solve_bound_states(-1.0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_schrodinger_lennard_jones: all checks passed")
