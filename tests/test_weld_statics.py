"""Test dgs.weld_statics: the eccentrically loaded fillet-weld group. Section
properties (L_w, Ix, Iy, J) match the closed form for two parallel welds, the
0.707 fillet throat, the combined direct+torsional shear at the critical point
against a hand calculation, the load-through-centroid degenerate case (torsion
vanishes), factor of safety, linearity, and the equilibrium reaction tie-in."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import weld_statics as ws

# Two vertical fillet welds, each L=100 mm, W=50 mm apart, centroid at origin.
L, W = 100.0, 50.0
segs = [((-W/2, -L/2), (-W/2, L/2)), ((W/2, -L/2), (W/2, L/2))]

# 1. section properties vs closed form (treat-as-line method)
props = ws.weld_properties(segs)
assert np.isclose(props["L_w"], 2 * L)
assert np.allclose(props["centroid"], (0.0, 0.0), atol=1e-9)
assert np.isclose(props["Ix"], L**3 / 6, rtol=1e-4)       # 166666.7
assert np.isclose(props["Iy"], L * W**2 / 2, rtol=1e-4)   # 125000
assert np.isclose(props["J"], L**3 / 6 + L * W**2 / 2, rtol=1e-4)  # 291666.7
assert np.isclose(props["J"], props["Ix"] + props["Iy"])   # J = Ix + Iy

# 2. fillet throat: the famous 0.707*leg for a 45-degree fillet; 90 deg -> leg
assert np.isclose(ws.fillet_throat(6.0), 6.0 * np.sqrt(2) / 2)   # 4.2426
assert np.isclose(ws.fillet_throat(8.0, angle_deg=90.0), 8.0)

# 3. THE worked problem: 10 kN down, 100 mm to the right of the centroid.
#    Hand calc: T = -1e6 N.mm, direct = 50 N/mm, resultant at corner = 218.6 N/mm
throat = ws.fillet_throat(6.0)
res = ws.eccentric_weld_stress(segs, P=(0.0, -10000.0),
                               load_point=(100.0, 0.0), throat=throat)
assert np.isclose(res["T"], -1.0e6, rtol=1e-6)
assert np.isclose(res["direct_shear_per_length"], 50.0, rtol=1e-6)
assert np.isclose(res["peak_force_per_length"], 218.6, rtol=2e-3)
assert np.isclose(res["shear_stress"], 218.6 / throat, rtol=2e-3)  # ~51.5 MPa
# critical point is one of the four corners farthest from the centroid
cx, cy = res["critical_point"]
assert np.isclose(abs(cx), W/2, atol=1e-6) and np.isclose(abs(cy), L/2, atol=1e-6)
# torsion strictly increases the peak above pure direct shear
assert res["peak_force_per_length"] > res["direct_shear_per_length"]

# 4. degenerate case: load THROUGH the centroid -> no torque, pure direct shear
res0 = ws.eccentric_weld_stress(segs, P=(0.0, -10000.0),
                                load_point=(0.0, 0.0), throat=throat)
assert np.isclose(res0["T"], 0.0, atol=1e-6)
assert np.isclose(res0["peak_force_per_length"], 50.0, rtol=1e-6)   # == direct
assert np.isclose(res0["shear_stress"], 50.0 / throat, rtol=1e-6)

# 5. factor of safety vs an E70xx-ish 96 MPa allowable
fos = ws.factor_of_safety(res["shear_stress"], allowable_stress=96.0)
assert np.isclose(fos, 96.0 / res["shear_stress"])
assert 1.8 < fos < 1.95

# 6. linearity: doubling the load doubles the stress (elastic method)
res2 = ws.eccentric_weld_stress(segs, P=(0.0, -20000.0),
                                load_point=(100.0, 0.0), throat=throat)
assert np.isclose(res2["shear_stress"], 2 * res["shear_stress"], rtol=1e-6)

# 7. the equilibrium tie-in: weld reacts -P and -T (sum_F=0, sum_M=0)
(Rx, Ry), M = ws.weld_reaction(P=(0.0, -10000.0), load_point=(100.0, 0.0),
                               centroid=(0.0, 0.0))
assert np.isclose(Rx, 0.0) and np.isclose(Ry, 10000.0)
assert np.isclose(M, 1.0e6)          # -T, balancing the applied moment

# 8. kwarg bounds
for bad in (lambda: ws.weld_properties([]),
            lambda: ws.fillet_throat(0),
            lambda: ws.fillet_throat(6, angle_deg=200),
            lambda: ws.eccentric_weld_stress(segs, (1, 2, 3), (0, 0), throat),
            lambda: ws.eccentric_weld_stress(segs, (0, -1), (0, 0), 0),
            lambda: ws.factor_of_safety(-1, 96),
            lambda: ws._sample([((0, 0), (0, 0))])):   # zero-length segment
    try:
        bad()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("test_weld_statics: all checks passed")
