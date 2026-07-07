"""Test dgs.gnss_positioning: the pseudorange model, the linearized geometry
matrix, exact recovery of position AND clock bias by Gauss-Newton from >=4
satellites (the 4th pays for the clock), graceful behavior under noise, and the
dilution-of-precision geometry factor (spread = sharp, bunched = smeared)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import gnss_positioning as gp

receiver_true = np.array([6.371e6, 0.0, 0.0])
bias_true = 100e-6 * gp.C_LIGHT                    # 100 us clock error -> ~30 km
good_dirs = np.array([[1, 0, 0], [0.5, 0.8, 0.3], [0.5, -0.8, 0.3],
                      [0.4, 0.3, -0.85], [0.6, -0.2, 0.75], [0.2, 0.6, -0.77]])
sats = gp._sats_from_directions(receiver_true, good_dirs, 26.56e6)
rho = np.array([gp.pseudorange(receiver_true, s, bias_true) for s in sats])

# 1. pseudorange = geometric range + clock bias
s0 = sats[0]
assert np.isclose(gp.pseudorange(receiver_true, s0, bias_true),
                  np.linalg.norm(receiver_true - s0) + bias_true)

# 2. geometry matrix: (n,4), unit LOS rows, ones in the bias column
G = gp.geometry_matrix(sats, receiver_true)
assert G.shape == (6, 4)
assert np.allclose(np.linalg.norm(G[:, :3], axis=1), 1.0)     # unit line-of-sight
assert np.allclose(G[:, 3], 1.0)                              # clock-bias partial

# 3. exact recovery of position and clock bias (6 satellites)
sol = gp.solve_position(sats, rho)
assert sol["converged"]
assert np.linalg.norm(sol["position"] - receiver_true) < 1e-3   # sub-mm
assert np.isclose(sol["clock_bias"], bias_true, atol=1e-3)
assert sol["residual_rms"] < 1e-6

# 4. exactly 4 satellites still solve (4 unknowns) -- the minimum
sats4 = sats[:4]
rho4 = rho[:4]
sol4 = gp.solve_position(sats4, rho4)
assert np.linalg.norm(sol4["position"] - receiver_true) < 1e-3
assert np.isclose(sol4["clock_bias"], bias_true, atol=1e-3)

# 5. fewer than 4, or mismatched lengths, is an error
try:
    gp.solve_position(sats[:3], rho[:3]); assert False
except ValueError:
    pass
try:
    gp.solve_position(sats, rho[:5]); assert False
except ValueError:
    pass

# 6. under measurement noise: the fix degrades gracefully, residual ~ noise
rng = np.random.default_rng(0)
sigma = 1.0                                                   # 1 m pseudorange noise
rho_noisy = rho + rng.normal(0, sigma, len(rho))
soln = gp.solve_position(sats, rho_noisy)
assert np.linalg.norm(soln["position"] - receiver_true) < 20.0   # meters, not km
assert 0.1 < soln["residual_rms"] < 5.0                          # order of the noise

# 7. dilution of precision: good spread is small, bunched is much larger
dop = gp.dilution_of_precision(sats, receiver_true)
assert dop["GDOP"] < 5 and dop["PDOP"] > 0 and dop["TDOP"] > 0
assert dop["GDOP"] >= dop["PDOP"]                            # GDOP includes the clock term
bunched = gp._sats_from_directions(
    receiver_true, [[1, 0, 0], [1, .1, 0], [1, 0, .1], [1, .1, .1]], 26.56e6)
assert gp.dilution_of_precision(bunched, receiver_true)["GDOP"] > 100 * dop["GDOP"]
# a truly degenerate geometry (all line-of-sight in a plane) has no fix
coplanar = gp._sats_from_directions(
    receiver_true, [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], 26.56e6)
try:
    gp.dilution_of_precision(coplanar, receiver_true); assert False
except ValueError:
    pass

# 8. kwarg bounds
try:
    gp.geometry_matrix(sats, sats[0]); assert False          # estimate on a satellite
except ValueError:
    pass

print("test_gnss_positioning: all checks passed")
