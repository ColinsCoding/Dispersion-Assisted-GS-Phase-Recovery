"""Test Griffiths Problem 1.16: div(r-hat/r^2) = 0 for r!=0 but flux = 4 pi -> delta."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import inverse_square as isq

# 1. the surprising calculation: div(r-hat/r^2) = 0 everywhere off the origin
assert isq.divergence_inverse_square() == 0

# 2. yet the flux through ANY sphere is 4 pi -- independent of the radius
for R in (0.3, 1.0, 7.0):
    assert abs(isq.flux_through_sphere(R) - 4 * np.pi) < 1e-3
# the flux does not depend on R (it is the SAME through every sphere)
assert abs(isq.flux_through_sphere(0.5) - isq.flux_through_sphere(10.0)) < 1e-6

# 3. zero divergence everywhere + nonzero flux  ->  the divergence is a delta at 0,
#    with strength 4 pi (the divergence theorem: volume integral = surface flux)
assert abs(isq.point_source_delta_strength() - 4 * np.pi) < 1e-9
assert abs(isq.flux_through_sphere(2.0) - isq.point_source_delta_strength()) < 1e-3

print("TEST PASS  (div(r-hat/r^2)=0 for r!=0; flux=4 pi through every sphere; "
      "so div = 4 pi delta^3(r) -- all the divergence at the origin)")
