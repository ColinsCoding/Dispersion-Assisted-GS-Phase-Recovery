"""Test DimensionalMeasurement: arithmetic propagates uncertainty exactly
the way dgs.error_propagation's own textbook rules do, units multiply/divide
correctly, incompatible-DIMENSION operations are rejected, and the honest
limitation (same-dimension, different-unit-SYSTEM mismatches are NOT
caught) is demonstrated rather than silently wrong."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sympy.physics import units as u
from dgs import units_error_calculator as uec
from dgs import dimensional_analysis as da
from dgs import error_propagation as ep

# 1. Griffiths 7.13 emf = B*h*v: the propagated sigma must match
# error_propagation.propagate()'s own unit-free result on the same numbers --
# adding units bookkeeping must not change the numeric uncertainty
B = uec.DimensionalMeasurement(0.5, 0.01, u.tesla)
h = uec.DimensionalMeasurement(2.0, 0.05, u.meters)
v = uec.DimensionalMeasurement(3.0, 0.1, u.meters / u.seconds)
emf = B * h * v

val_ref, sig_ref = ep.propagate(lambda p: p[0] * p[1] * p[2], [0.5, 2.0, 3.0], [0.01, 0.05, 0.1])
assert abs(emf.value - val_ref) < 1e-9
assert abs(emf.sigma - sig_ref) < 1e-9

# 2. emf's resulting unit really is dimensionally a Volt
assert emf.convert_check(u.volts) is True

# 3. incompatible DIMENSIONS are rejected on +/- (force vs energy)
force = uec.DimensionalMeasurement(10.0, 0.2, u.newtons)
energy = uec.DimensionalMeasurement(5.0, 0.1, u.joules)
try:
    force + energy
except ValueError:
    pass
else:
    raise AssertionError("should reject adding a force to an energy")

try:
    force - energy
except ValueError:
    pass
else:
    raise AssertionError("should reject subtracting an energy from a force")

# 4. SAME dimension, compatible units: addition succeeds and sigma adds in
# quadrature exactly like error_propagation.add_in_quadrature
f1 = uec.DimensionalMeasurement(10.0, 0.2, u.newtons)
f2 = uec.DimensionalMeasurement(3.0, 0.1, u.newtons)
total = f1 + f2
assert abs(total.value - 13.0) < 1e-9
assert abs(total.sigma - ep.add_in_quadrature(0.2, 0.1)) < 1e-9
assert da.dims_equal(total.unit, u.newtons)

# 5. power rule: (10 +/- 0.2 m)^2 -- relative sigma scales by |n|=2
length = uec.DimensionalMeasurement(10.0, 0.2, u.meters)
area = length ** 2
assert abs(area.value - 100.0) < 1e-9
assert abs(area.sigma - ep.power_rule(100.0, 10.0, 0.2, 2)) < 1e-9
assert da.dims_equal(area.unit, u.meters ** 2)

# 6. scalar (dimensionless) multiply/divide leaves the unit unchanged
doubled = length * 2
assert abs(doubled.value - 20.0) < 1e-9
assert abs(doubled.sigma - 0.4) < 1e-9
assert da.dims_equal(doubled.unit, u.meters)

halved = length / 2
assert abs(halved.value - 5.0) < 1e-9
assert abs(halved.sigma - 0.1) < 1e-9

# 7. Honest limitation, demonstrated rather than just claimed: N*s and
# lbf-equivalent*s (pounds * g) are the SAME dimension (force*time), so
# dims_equal correctly says they match -- this calculator does NOT (and by
# design cannot) catch the Mars-Climate-Orbiter class of unit-SYSTEM
# conversion-factor bug; that needs a numeric conversion check, not a
# dimensional-dependency comparison.
impulse_N_s = u.newtons * u.seconds
impulse_lbf_s = u.pounds * u.acceleration_due_to_gravity * u.seconds
assert da.dims_equal(impulse_N_s, impulse_lbf_s) is True

print("all dgs.units_error_calculator tests passed")
