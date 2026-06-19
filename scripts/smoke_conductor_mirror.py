"""Smoke-test the perfect-conductor limit -> mirror: reflectivity, skin depth, tau."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import electrodynamics as ed

w = 2 * np.pi * 10e9            # 10 GHz microwave
sigma_cu = 5.96e7

# 1. reflectivity is a valid probability in [0, 1]
R = ed.conductor_reflectivity(w, sigma_cu)
assert 0.0 <= R <= 1.0

# 2. a good conductor is a near-perfect mirror at microwave (R ~ 1)
assert R > 0.999, R

# 3. the "Ohm = 0" / perfect-conductor limit: sigma -> infinity drives R -> 1,
#    skin depth -> 0, charge relaxation time -> 0
sigmas = np.array([1e2, 1e4, 1e6, 1e8, 1e10])
Rs = np.array([ed.conductor_reflectivity(w, s) for s in sigmas])
ds = np.array([ed.skin_depth(w, s) for s in sigmas])
taus = np.array([ed.charge_relaxation_time(ed._EPS0, s) for s in sigmas])
assert np.all(np.diff(Rs) > 0), Rs          # more conductive -> more reflective
assert np.all(np.diff(ds) < 0)               # more conductive -> thinner skin
assert np.all(np.diff(taus) < 0)             # more conductive -> faster relaxation
assert Rs[-1] > 0.9999 and ds[-1] < ds[0]    # near-perfect mirror in the limit

# 4. a poor conductor reflects less than a good one
assert ed.conductor_reflectivity(w, 1e2) < ed.conductor_reflectivity(w, 1e7)

print(f"SMOKE PASS  (copper R={R:.5f} @10GHz; sigma->inf: R->{Rs[-1]:.5f}, "
      f"skin depth {ds[0]*1e6:.1f}->{ds[-1]*1e9:.2e} (um->nm))")
