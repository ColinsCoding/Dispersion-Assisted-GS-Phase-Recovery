"""Test dgs/irrotational_solenoidal_polyglot.py: Griffiths' curl-of-
gradient and divergence-of-curl identities, proven exactly with SymPy,
cross-checked by torch autograd (exact) and a from-scratch C program
(finite differences), plus the wire-circulation subtlety (curl=0 pointwise
away from the wire, but nonzero loop circulation).

torch is py-3.12-only in this repo; MATLAB isn't used here but gcc is --
the C cross-check is skipped gracefully if gcc isn't found."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import math
import numpy as np
import sympy as sp
from dgs.irrotational_solenoidal_polyglot import (
    irrotational_field_proof, solenoidal_field_proof, wire_circulation_symbolic,
    torch_div_curl_at_points, wire_circulation_numeric, GCC_DEFAULT,
)

# 1. irrotational_field_proof: E~rhat/r^2 is exactly -grad(1/r), curl=0
irr = irrotational_field_proof()
assert irr["is_gradient_of_minus_1_over_r"] is True
assert irr["curl_is_zero"] is True

# 2. solenoidal_field_proof: B~phihat/s is exactly curl(-ln(s)zhat), div=0,
#    AND curl=0 away from the wire (the subtlety's setup)
sol = solenoidal_field_proof()
assert sol["is_curl_of_vector_potential"] is True
assert sol["div_is_zero"] is True
assert sol["curl_is_zero_away_from_wire"] is True

# 3. wire_circulation_symbolic: EXACTLY 2*pi, independent of radius
#    (checked at several symbolic/numeric radii, not just the default)
assert sp.simplify(wire_circulation_symbolic() - 2 * sp.pi) == 0
for a_val in (sp.Rational(1, 2), sp.Integer(1), sp.Integer(7)):
    assert sp.simplify(wire_circulation_symbolic(radius=a_val) - 2 * sp.pi) == 0

print("dgs.irrotational_solenoidal_polyglot: SymPy exact-proof checks passed")

# 4. torch (py 3.12 only): pointwise divergence/curl must be ~0 off the
#    singularity, for BOTH fields
import torch  # noqa: F401 -- this test file is torch-required (py 3.12)

rng = np.random.default_rng(0)
irr_pts = rng.uniform(-3, 3, size=(200, 3))
irr_pts = irr_pts[np.linalg.norm(irr_pts, axis=1) > 0.5]
irr_check = torch_div_curl_at_points("irrotational", irr_pts)
assert np.max(np.abs(irr_check["divergence"])) < 1e-8   # not claimed zero (it's 4*pi*delta), just informative
assert np.max(np.abs(irr_check["curl"])) < 1e-8, "irrotational field must have ~0 curl off-origin"

sol_pts = rng.uniform(-3, 3, size=(200, 3))
sol_pts[:, 2] = rng.uniform(-5, 5, 200)   # z can be anything (translation-invariant along the wire)
s = np.linalg.norm(sol_pts[:, :2], axis=1)
sol_pts = sol_pts[s > 0.5]
sol_check = torch_div_curl_at_points("solenoidal", sol_pts)
assert np.max(np.abs(sol_check["divergence"])) < 1e-8, "solenoidal field must have ~0 divergence off-wire"
assert np.max(np.abs(sol_check["curl"])) < 1e-8, "solenoidal field must ALSO have ~0 curl off-wire"

# 5. wire_circulation_numeric: matches the exact 2*pi, and is INDEPENDENT
#    of the loop radius -- the actual numeric confirmation of the subtlety
for radius in (0.3, 1.0, 2.5, 10.0):
    circ = wire_circulation_numeric(radius=radius, n_points=200_000)
    assert abs(circ - 2 * math.pi) < 1e-3, f"radius={radius}: circulation={circ}, expected 2*pi"

print("dgs.irrotational_solenoidal_polyglot: torch checks passed")

# 6. C (if gcc installed): finite-difference div/curl must agree with the
#    exact-zero claims and with torch's autograd values
if os.path.exists(GCC_DEFAULT):
    import tempfile
    from dgs.irrotational_solenoidal_polyglot import cross_validate_languages
    with tempfile.TemporaryDirectory() as tmp:
        result = cross_validate_languages(tmp)
    for kind in ("irrotational", "solenoidal"):
        assert result[kind]["max_abs_diff_divergence"] < 1e-6, result[kind]
        assert result[kind]["max_abs_diff_curl"] < 1e-6, result[kind]
        assert np.max(np.abs(result[kind]["c_divergence"])) < 1e-6
        assert np.max(np.abs(result[kind]["c_curl"])) < 1e-6
    print("dgs.irrotational_solenoidal_polyglot: C cross-check passed")
else:
    print(f"dgs.irrotational_solenoidal_polyglot: gcc not found at {GCC_DEFAULT}, skipped C checks")

print("all dgs.irrotational_solenoidal_polyglot tests passed")
