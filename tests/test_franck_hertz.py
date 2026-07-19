"""Test franck_hertz: low-voltage electrons can't lose energy inelastically
(below threshold), and the simulated I-V curve's dip spacing matches
mercury's 4.9 eV excitation energy."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import franck_hertz as fh

rng = np.random.default_rng(0)

# 1. an electron that never reaches 4.9 eV total can't lose any energy
#    inelastically -- final energy should equal the accelerating voltage
low_energy = fh.simulate_electron(V_grid=2.0, V_excitation=4.9, n_segments=500,
                                   p_collision=0.2, rng=np.random.default_rng(1))
assert abs(low_energy - 2.0) < 1e-9

# 2. a high-voltage electron with collisions enabled loses SOME energy
#    (its final KE should be less than the accelerating voltage)
high_energy = fh.simulate_electron(V_grid=20.0, V_excitation=4.9, n_segments=2000,
                                    p_collision=0.3, rng=np.random.default_rng(2))
assert high_energy < 20.0

# 3. the I-V curve's dip spacing clusters near the 4.9V excitation energy
V_values = np.linspace(0, 30, 120)
I_values = fh.franck_hertz_iv_curve(V_values, n_electrons=600, n_segments=1000, seed=0)
dips, spacings = fh.find_dip_spacing(V_values, I_values)
assert len(spacings) >= 3
assert abs(spacings.mean() - 4.9) < 0.5

print("test_franck_hertz: all checks passed")
