"""Test the Bohr model (13.6 eV, hydrogen spectrum) and Moseley's law."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import atomic as at

# 1. the Rydberg energy is 13.606 eV
assert abs(at.rydberg_energy() - 13.6057) < 1e-3

# 2. Bohr levels: E1 = -13.6, E2 = -3.4, E_inf = 0; scale as Z^2/n^2
assert abs(at.bohr_energy_level(1) + 13.6057) < 1e-3
assert abs(at.bohr_energy_level(2) + 3.4014) < 1e-3
assert abs(at.bohr_energy_level(1, Z=2) - 4 * at.bohr_energy_level(1)) < 1e-9   # Z^2

# 3. hydrogen lines: Balmer H-alpha 656 nm, Lyman-alpha 122 nm
assert abs(at.hydrogen_line_wavelength(2, 3) * 1e9 - 656.3) < 1.0
assert abs(at.hydrogen_line_wavelength(1, 2) * 1e9 - 121.6) < 0.5
# the Balmer series limit (n->inf, n_low=2) is 364.6 nm
assert abs(at.hydrogen_line_wavelength(2, 10000) * 1e9 - 364.6) < 1.0

# 4. Moseley's law: K-alpha energies near the measured values
assert abs(at.moseley_kalpha_energy(29) / 1e3 - 8.0) < 0.3      # Cu ~ 8.0 keV
assert abs(at.moseley_kalpha_energy(42) / 1e3 - 17.2) < 0.5     # Mo ~ 17.2 keV

# 5. the defining feature: sqrt(frequency) is LINEAR in (Z-1)
Z = np.arange(20, 60)
sqrt_f = np.sqrt([at.moseley_kalpha_frequency(z) for z in Z])
slope, intercept = np.polyfit(Z - 1, sqrt_f, 1)
fit = slope * (Z - 1) + intercept
ss_res = np.sum((sqrt_f - fit) ** 2)
ss_tot = np.sum((sqrt_f - sqrt_f.mean()) ** 2)
assert 1 - ss_res / ss_tot > 0.99999                            # essentially perfect line
assert abs(intercept) < 1e-3 * sqrt_f.max()                     # passes through Z=1

# 6. the slope is sqrt((3/4) * R * c) -- a constant, not element-specific
R_c = at.rydberg_energy(unit="J") / at._H                       # Rydberg frequency
assert abs(slope - np.sqrt(0.75 * R_c)) / slope < 1e-6

print(f"TEST PASS  (Rydberg {at.rydberg_energy():.3f} eV; Balmer 656 nm / Lyman 122 nm; "
      f"Moseley Cu {at.moseley_kalpha_energy(29)/1e3:.1f} keV, Mo "
      f"{at.moseley_kalpha_energy(42)/1e3:.1f} keV; sqrt(f) linear in Z-1, R2>0.99999)")
