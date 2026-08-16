"""Test dgs.microplastic.materials: polymer refractive-index/density lookup."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.microplastic import materials as mat
from dgs.microplastic import physics as phy

# 1. basic lookups, case-insensitive on input
assert mat.refractive_index("PET") == 1.575
assert mat.refractive_index("pet") == 1.575
assert mat.density("PP") == 0.905
assert mat.medium_refractive_index("Water") == 1.333

# 2. unknown keys raise, not silently return None/garbage
try:
    mat.refractive_index("UNOBTANIUM")
    assert False, "should reject unknown polymer"
except KeyError:
    pass

# 3. every polymer has a RIC code in 1-7, and by_ric partitions them correctly
all_polys = mat.list_polymers()
assert len(all_polys) == len(mat.POLYMERS)
recovered = sorted(p for ric in range(1, 8) for p in mat.by_ric(ric))
assert recovered == sorted(all_polys)   # by_ric covers everyone, no overlap/miss

# 4. optical contrast is just n_polymer - n_medium, sign meaningful
dn = mat.optical_contrast("PS", "water")
assert np.isclose(dn, mat.refractive_index("PS") - mat.medium_refractive_index("water"))
assert dn > 0     # PS (n=1.59) is optically denser than water (n=1.333)

# 5. settling sign matches known physical behavior: PP floats, PET/PS sink in water
assert mat.settling_sign("PP", "water") == -1     # PP density 0.905 < water 1.000
assert mat.settling_sign("PET", "water") == 1     # PET density 1.380 > water 1.000
assert mat.settling_sign("PS", "water") == 1      # PS density 1.050 > water 1.000

# 6. polymer_complex_index feeds directly into month-1's physics.complex_index
n, kappa = mat.polymer_complex_index("PMMA")
assert n == mat.refractive_index("PMMA") and kappa == 0.0
n_tilde = phy.complex_index(n, kappa)
assert np.isclose(n_tilde.real, 1.491) and np.isclose(n_tilde.imag, 0.0)
try:
    mat.polymer_complex_index("PMMA", kappa=-0.1)
    assert False, "should reject negative kappa"
except ValueError:
    pass

# 7. plot_periodic_table renders without error and returns a populated figure
fig, ax = mat.plot_periodic_table(medium="water")
assert len(ax.patches) == len(mat.POLYMERS)   # one tile per polymer
import matplotlib.pyplot as plt
plt.close(fig)

print("TEST PASS  (lookup case-insensitive; unknown keys raise; RIC partition complete; "
      "optical contrast = n_poly - n_medium; settling sign matches known float/sink "
      "behavior for PP/PET/PS; polymer_complex_index feeds physics.complex_index; "
      "plot_periodic_table renders)")
