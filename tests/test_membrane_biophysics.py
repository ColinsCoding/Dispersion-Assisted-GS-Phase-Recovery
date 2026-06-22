"""Test membrane biophysics: Nernst, Goldman, bilayer capacitance, the RC clock."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import membrane_biophysics as mb

# 1. Nernst potentials match the textbook physiology values
assert abs(mb.nernst_potential(4, 140, +1) * 1e3 - (-95.0)) < 2.0      # E_K ~ -95 mV
assert abs(mb.nernst_potential(145, 12, +1) * 1e3 - 66.6) < 2.0        # E_Na ~ +67 mV
assert abs(mb.nernst_potential(110, 10, -1) * 1e3 - (-64.0)) < 2.0     # E_Cl ~ -64 mV (anion)
# zero gradient -> zero potential; flipping z flips sign
assert abs(mb.nernst_potential(100, 100, 1)) < 1e-12
assert np.isclose(mb.nernst_potential(140, 4, +1), -mb.nernst_potential(4, 140, +1))

# 2. Goldman resting potential is ~ -70 mV and lies BETWEEN E_K and E_Na
V = mb.goldman_potential({"K": 1.0, "Na": 0.04, "Cl": 0.45},
                         {"K": 5, "Na": 145, "Cl": 110}, {"K": 140, "Na": 12, "Cl": 10})
assert -80e-3 < V < -55e-3                                             # ~ -67 mV
assert mb.nernst_potential(5, 140, +1) < V < mb.nernst_potential(145, 12, +1)
# K-dominated: raising K permeability pulls resting toward E_K (more negative-ish);
# raising Na permeability pulls it toward E_Na (depolarizes) -> the action-potential idea
V_depol = mb.goldman_potential({"K": 1.0, "Na": 20.0, "Cl": 0.45},
                               {"K": 5, "Na": 145, "Cl": 110}, {"K": 140, "Na": 12, "Cl": 10})
assert V_depol > V                                                    # Na opening depolarizes

# 3. the bilayer capacitance is ~1 uF/cm^2 (0.005-0.012 F/m^2)
c = mb.specific_capacitance()
assert 0.005 < c < 0.012, c                                           # ~0.0085 F/m^2 = 0.85 uF/cm^2
assert np.isclose(mb.membrane_capacitance(1e-8), c * 1e-8)           # scales with area

# 4. the RC clock: tau = R_m C_m, and the charging curve reaches 63% at t=tau
tau = mb.membrane_time_constant(1e8, 100e-12)
assert abs(tau - 1e-2) < 1e-12                                        # 10 ms
V_t = mb.membrane_charging(tau, V_final=-0.055, tau=tau, V0=-0.070)
frac = (V_t - (-0.070)) / ((-0.055) - (-0.070))
assert abs(frac - (1 - 1/np.e)) < 1e-9                               # 63% of the way at t=tau
assert abs(mb.membrane_charging(50*tau, -0.055, tau, -0.070) - (-0.055)) < 1e-4   # settles

# 5. cable length constant scales as sqrt(diameter)
l1 = mb.length_constant(1.0, 1.0, 1e-6)
l4 = mb.length_constant(1.0, 1.0, 4e-6)
assert np.isclose(l4 / l1, 2.0)                                       # 4x diameter -> 2x lambda

print(f"TEST PASS  (E_K=-95mV, E_Na=+67mV, E_Cl=-64mV; resting {V*1e3:.0f} mV between them; "
      f"bilayer {c*1e2:.2f} uF/cm^2; tau=R C=10ms (63% at tau); lambda ~ sqrt(d))")
