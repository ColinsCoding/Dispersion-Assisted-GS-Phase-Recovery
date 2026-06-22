"""Test the normal Zeeman effect: 2l+1 sublevels, the 3-line triplet, Bohr magneton."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import atomic as at

# 1. Bohr magneton has the known value ~ 9.274e-24 J/T
assert abs(at.bohr_magneton() - 9.2740e-24) < 1e-27

# 2. a level splits into 2l+1 equally spaced sublevels, symmetric about 0
for l in (1, 2, 5):
    levels = at.zeeman_sublevels(l, B=2.0)
    assert len(levels) == 2 * l + 1
    assert abs(levels.sum()) < 1e-30                      # symmetric -> sums to zero
    gaps = np.diff(levels)
    assert np.allclose(gaps, gaps[0])                     # equally spaced
    assert abs(gaps[0] - at.level_spacing(2.0)) < 1e-30   # spacing = mu_B B

# 3. the shift is m_l * mu_B * B; m_l = 0 is unshifted
assert at.zeeman_energy_shift(0, 3.0) == 0.0
assert abs(at.zeeman_energy_shift(1, 3.0) - at.bohr_magneton() * 3.0) < 1e-30
assert at.zeeman_energy_shift(-1, 3.0) == -at.zeeman_energy_shift(1, 3.0)

# 4. the NORMAL Zeeman triplet: exactly 3 lines, evenly spaced by the Larmor freq
lines = at.normal_zeeman_lines(500e12, B=1.0)
assert len(lines) == 3
d = at.larmor_frequency(1.0)
assert np.allclose(np.diff(lines), d)                    # spacing = mu_B B / h
assert abs(d - 13.996e9) < 0.05e9                        # ~14 GHz at 1 T

# 5. the splitting scales linearly with B (double B -> double the spacing)
assert abs(at.larmor_frequency(2.0) - 2 * at.larmor_frequency(1.0)) < 1e-3

# 6. symbolic shifts: 2l+1 expressions, the m_l = +l one is l*mu_B*B
import sympy as sp
mu_B, B = sp.symbols("mu_B B", positive=True)
sym = at.zeeman_shift_symbolic(3)
assert len(sym) == 7 and sym[-1][0] == 3
assert sp.simplify(sym[-1][1] - 3 * mu_B * B) == 0

print(f"TEST PASS  (mu_B={at.bohr_magneton():.4e} J/T; 2l+1 equal sublevels summing to "
      f"0; shift m_l mu_B B; normal triplet 3 lines spaced {d/1e9:.2f} GHz at 1 T; "
      f"linear in B)")
