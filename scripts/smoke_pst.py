"""Smoke-test the Phase Stretch Transform edge detector (PhyCV, from scratch)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pst

# 1. kernels are well-formed
ph = pst.pst_phase_kernel(64, 64, warp=15, strength=0.48)
assert ph.shape == (64, 64) and np.isreal(ph).all()
assert abs(ph.max() - 0.48) < 1e-9 and ph.min() >= -1e-12         # normalized to strength
L = pst.localization_kernel(64, 64, sigma=0.12)
assert abs(L.flat[0] - 1.0) < 1e-12 and 0 < L.min() <= L.max() <= 1  # DC=1, low-pass

# 2. a uniform image has no edges -> PST response ~ 0 everywhere
flat = np.ones((128, 128))
assert np.abs(pst.pst(flat)).max() < 1e-9

# 3. a disk: PST lights up the rim, not the interior
yy, xx = np.mgrid[0:128, 0:128]
r = np.sqrt((xx - 64)**2 + (yy - 64)**2)
disk = (r < 30).astype(float)
p = np.abs(pst.pst(disk))
gy, gx = np.gradient(disk); rim = np.sqrt(gx**2 + gy**2) > 0.1
interior = r < 20
assert p[rim].mean() > 20 * p[interior].mean(), p[rim].mean() / max(p[interior].mean(), 1e-12)
assert p.shape == disk.shape

# 4. detected edge pixels actually sit on the circle (radius ~ 30)
edges = pst.pst_edges(disk, thresh=0.5)
assert edges.sum() > 0
edge_radii = r[edges == 1]
assert abs(edge_radii.mean() - 30) < 5, edge_radii.mean()         # edges on the rim (slight kernel ring-out)

# 5. on the rim (where there is signal), the phase response scales with `strength`
p_lo = np.abs(pst.pst(disk, strength=0.2))[rim].mean()
p_hi = np.abs(pst.pst(disk, strength=0.45))[rim].mean()
assert p_hi > p_lo, (p_lo, p_hi)
# and a different shape (square) is also detected on its boundary
sq = ((np.abs(xx - 64) < 25) & (np.abs(yy - 64) < 25)).astype(float)
assert pst.pst_edges(sq).sum() > 0

# 6. validation
for bad in (lambda: pst.pst_phase_kernel(8, 8, warp=0),
            lambda: pst.localization_kernel(8, 8, sigma=0)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad parameter")

print(f"SMOKE PASS  (rim/interior PST response = {p[rim].mean()/p[interior].mean():.0f}x; "
      f"{edges.sum()} edge px at mean radius {edge_radii.mean():.1f} ~ 30)")
