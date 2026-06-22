"""Test what comes after the Bohr magneton: spin, the g-factor, Stern-Gerlach."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import atomic as at

# 1. electron spin g-factor ~ 2.0023 (Dirac=2 plus QED)
assert abs(at.electron_g_factor() - 2.00232) < 1e-4

# 2. the spin moment is ~1 Bohr magneton (g/2), from a HALF unit of spin
assert abs(at.spin_magnetic_moment() / at.MU_B - 1.00116) < 1e-3

# 3. Lande g-factors for the sodium-D terms (the textbook checks)
assert abs(at.lande_g_factor(0.5, 0, 0.5) - 2.0) < 1e-9        # 2S_1/2: pure spin
assert abs(at.lande_g_factor(0.5, 1, 0.5) - 2/3) < 1e-9        # 2P_1/2
assert abs(at.lande_g_factor(1.5, 1, 0.5) - 4/3) < 1e-9        # 2P_3/2
# pure orbital (s=0, j=l): g = 1, the NORMAL Zeeman case
assert abs(at.lande_g_factor(2, 2, 0) - 1.0) < 1e-9
assert at.lande_g_factor(0, 0, 0) == 0.0                       # j=0: no splitting

# 4. Stern-Gerlach: spin-1/2 gives TWO equal-and-opposite forces -> two beams
dBdz = 1e3                                                     # T/m
F_up = at.stern_gerlach_force(+0.5, dBdz)
F_dn = at.stern_gerlach_force(-0.5, dBdz)
assert np.isclose(F_up, -F_dn)                                # symmetric split
assert abs(abs(F_up) - at.electron_g_factor()*0.5*at.MU_B*dBdz) < 1e-30
# the deflection is discrete (two spots), not a continuous classical smear
forces = [at.stern_gerlach_force(ms, dBdz) for ms in (-0.5, 0.5)]
assert len(set(np.round(forces, 30))) == 2                    # exactly two values

# 5. anomalous Zeeman: levels with DIFFERENT g_J -> more than 3 lines.
#    2S_1/2 (g=2) -> 2P_1/2 (g=2/3): the upper splits by 2*g_J*m_j, giving 4 components.
B = 1.0
g_lower, g_upper = 2.0, 2/3
lower = [at.anomalous_zeeman_shift(mj, g_lower, B) for mj in (-0.5, 0.5)]
upper = [at.anomalous_zeeman_shift(mj, g_upper, B) for mj in (-0.5, 0.5)]
# allowed transitions Delta m_j in {-1,0,+1}; count distinct photon-energy shifts
shifts = set()
for u in upper:
    for d in lower:
        shifts.add(round((u - d) / (at.MU_B * B), 6))
assert len(shifts) > 3                                        # MORE than the normal triplet

print(f"TEST PASS  (g_s={at.electron_g_factor()}; spin moment ~1 mu_B; Lande g "
      f"2/2/3 for 2S/2P_1/2, 4/3 for 2P_3/2, 1 pure-orbital; Stern-Gerlach splits in "
      f"two; anomalous Zeeman gives {len(shifts)}>3 lines)")
