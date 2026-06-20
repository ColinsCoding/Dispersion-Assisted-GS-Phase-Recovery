"""Smoke-test Gaussian beam optics: waist, curvature, Gouy, q-parameter, ABCD."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import gaussian_beam as gb

w0, lam = 0.5e-3, 1.55e-6
zR = gb.rayleigh_range(w0, lam)

# 1. Rayleigh range definition
assert abs(zR - np.pi * w0**2 / lam) < 1e-12

# 2. spot size: w0 at the waist, w0*sqrt(2) at z=zR
assert abs(gb.beam_width(0, w0, lam) - w0) < 1e-15
assert abs(gb.beam_width(zR, w0, lam) - w0 * np.sqrt(2)) < 1e-12

# 3. curvature: flat (infinite R) at the waist, minimum R = 2 zR at z = zR
assert np.isinf(gb.radius_of_curvature(0, w0, lam))
assert abs(gb.radius_of_curvature(zR, w0, lam) - 2 * zR) < 1e-9
# R(z) is minimized at z=zR (sample around it)
zs = np.linspace(0.1 * zR, 5 * zR, 500)
assert abs(zs[np.argmin(gb.radius_of_curvature(zs, w0, lam))] - zR) < 0.02 * zR

# 4. Gouy phase: 0 at waist, pi/4 at zR, -> pi/2 far field (pi total through focus)
assert abs(gb.gouy_phase(0, w0, lam)) < 1e-15
assert abs(gb.gouy_phase(zR, w0, lam) - np.pi / 4) < 1e-12
assert abs(gb.gouy_phase(1e6 * zR, w0, lam) - np.pi / 2) < 1e-3

# 5. far field: width grows linearly with slope = divergence angle
theta = gb.divergence(w0, lam)
assert abs(gb.beam_width(1e4 * zR, w0, lam) / (1e4 * zR) - theta) < 1e-4

# 6. complex q: q(0) = i zR; recover (w, R) back from q
q0 = gb.q_parameter(0, w0, lam)
assert abs(q0 - 1j * zR) < 1e-12
w_rec, R_rec = gb.width_curvature_from_q(q0, lam)
assert abs(w_rec - w0) < 1e-9 and np.isinf(R_rec)

# 7. ABCD free-space propagation by d equals advancing q by d
d = 2 * zR
q_prop = gb.abcd_propagate(gb.q_parameter(0, w0, lam), 1, d, 0, 1)
assert abs(q_prop - gb.q_parameter(d, w0, lam)) < 1e-9
# and the recovered width matches the direct formula
w_prop, _ = gb.width_curvature_from_q(q_prop, lam)
assert abs(w_prop - gb.beam_width(d, w0, lam)) < 1e-9

# 8. validation
for bad in (lambda: gb.rayleigh_range(0, lam), lambda: gb.divergence(w0, -1)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad input")

print(f"SMOKE PASS  (zR={zR:.3f} m; w(zR)=w0*sqrt2; R_min=2zR; Gouy(zR)=45deg; "
      f"divergence={theta*1e3:.3f} mrad; ABCD == direct)")
