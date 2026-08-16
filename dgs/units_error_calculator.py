"""A units + dimensional-analysis + error-propagation calculator.

dgs.dimensional_analysis checks whether two SymPy unit expressions reduce to
the SAME base SI dimensions. dgs.error_propagation.Measurement carries a
value with its uncertainty through arithmetic (add-in-quadrature for +/-,
relative-quadrature for */, |n|-scaling for **n). Neither module knows about
the other: a Measurement has no notion of "meters", and dims_equal has no
notion of uncertainty.

DimensionalMeasurement below is the missing combination: a value that
carries BOTH a unit (checked with dims_equal on every +/- so DIMENSIONALLY
incompatible quantities can't silently combine -- e.g. force + energy) AND
an uncertainty (propagated with error_propagation's own textbook rules,
reused here rather than re-derived).

One honest limitation, worth stating rather than glossing over: dims_equal
compares base SI DIMENSIONS (mass, length, time, ...), not unit SYSTEMS.
Newton-seconds and pound-force-seconds are both force*time -- dims_equal
correctly says they MATCH, because they are the same physical quantity.
The Mars Climate Orbiter loss wasn't a dimensional mismatch at all; it was
exactly this case, a scale-factor (unit-system) error that dims_equal
cannot catch by design (see dimensional_analysis.mars_climate_orbiter_case_study
for the actual numeric conversion-factor bug). Catching THAT class of bug
needs a numeric conversion check (SymPy's own `convert_to`), not a
dimensional-dependency comparison -- a genuinely different kind of mistake
from the force-vs-energy example this module actually catches below.
"""

from sympy.physics import units as u

from . import dimensional_analysis as da
from . import error_propagation as ep


class DimensionalMeasurement:
    """A value +/- sigma, tagged with a SymPy unit expression.

    Arithmetic checks units (dims_equal) as well as propagating uncertainty
    (error_propagation's add-in-quadrature / relative-quadrature / |n|-scaling
    rules) -- get either one wrong and the operation raises or the number is
    silently off, exactly the two failure modes a real lab calculation has
    to guard against.
    """

    __slots__ = ("value", "sigma", "unit")

    def __init__(self, value, sigma, unit):
        self.value = float(value)
        self.sigma = float(abs(sigma))
        self.unit = unit

    def __repr__(self):
        return f"{self.value:.6g} +/- {self.sigma:.3g}  [{self.unit}]"

    def _require_same_dimensions(self, other, op):
        if not da.dims_equal(self.unit, other.unit):
            raise ValueError(
                f"cannot {op} incompatible units: {self.unit} vs {other.unit} "
                f"-- these do not reduce to the same base SI dimensions."
            )

    def __add__(self, other):
        self._require_same_dimensions(other, "add")
        return DimensionalMeasurement(
            self.value + other.value,
            ep.add_in_quadrature(self.sigma, other.sigma),
            self.unit,
        )

    def __sub__(self, other):
        self._require_same_dimensions(other, "subtract")
        return DimensionalMeasurement(
            self.value - other.value,
            ep.add_in_quadrature(self.sigma, other.sigma),
            self.unit,
        )

    def __mul__(self, other):
        if isinstance(other, DimensionalMeasurement):
            value = self.value * other.value
            sigma = ep.product_rule(value, [(self.value, self.sigma), (other.value, other.sigma)])
            return DimensionalMeasurement(value, sigma, self.unit * other.unit)
        # scalar (dimensionless) multiply: unit unchanged
        value = self.value * other
        return DimensionalMeasurement(value, abs(other) * self.sigma, self.unit)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, DimensionalMeasurement):
            value = self.value / other.value
            sigma = ep.product_rule(value, [(self.value, self.sigma), (other.value, other.sigma)])
            return DimensionalMeasurement(value, sigma, self.unit / other.unit)
        value = self.value / other
        return DimensionalMeasurement(value, abs(self.sigma / other), self.unit)

    def __pow__(self, n):
        value = self.value ** n
        sigma = ep.power_rule(value, self.value, self.sigma, n)
        return DimensionalMeasurement(value, sigma, self.unit ** n)

    def convert_check(self, target_unit):
        """Is this measurement's unit dimensionally compatible with
        target_unit? (Doesn't do the numeric conversion -- SymPy's own
        `convert_to` does that once you know the two ARE compatible; this
        answers the prior question that actually caused the Mars Climate
        Orbiter loss: are they even the same kind of quantity?)"""
        return da.dims_equal(self.unit, target_unit)


if __name__ == "__main__":
    # Griffiths 7.13: emf = B*h*v -- same numbers as error_propagation.py's
    # own demo, now carrying real units through every step.
    B = DimensionalMeasurement(0.5, 0.01, u.tesla)
    h = DimensionalMeasurement(2.0, 0.05, u.meters)
    v = DimensionalMeasurement(3.0, 0.1, u.meters / u.seconds)

    emf = B * h * v
    print(f"emf = B*h*v = {emf}")
    print(f"  is emf's unit dimensionally a Volt? {emf.convert_check(u.volts)}")

    # cross-check the propagated sigma against error_propagation's own
    # unit-free propagate() on the same numbers -- the units bookkeeping
    # here must NOT change the numeric uncertainty result
    val_unitless, sig_unitless = ep.propagate(lambda p: p[0] * p[1] * p[2], [0.5, 2.0, 3.0], [0.01, 0.05, 0.1])
    print(f"  cross-check vs error_propagation.propagate(): {val_unitless:.4f} +/- {sig_unitless:.5f}")

    print("\n--- Catch a genuine DIMENSIONAL mismatch: force + energy ---\n")
    force = DimensionalMeasurement(10.0, 0.2, u.newtons)
    energy = DimensionalMeasurement(5.0, 0.1, u.joules)
    try:
        bad = force + energy
        print("BUG: this addition should have been rejected:", bad)
    except ValueError as e:
        print(f"Correctly rejected: {e}")

    print("\n--- What this calculator does NOT catch, and why (the actual Mars Climate Orbiter bug) ---\n")
    impulse_N_s = DimensionalMeasurement(1.0, 0.001, u.newtons * u.seconds)
    impulse_lbf_s = DimensionalMeasurement(1.0, 0.001, u.pounds * u.acceleration_due_to_gravity * u.seconds)
    same_dimension = da.dims_equal(impulse_N_s.unit, impulse_lbf_s.unit)
    print(f"N*s and (lbf-equivalent)*s dims_equal: {same_dimension}  "
          f"(both are force*time -- correctly the SAME dimension)")
    print("Adding these would NOT raise here, even though 1 N != 1 lbf numerically --")
    print("that's a unit-SYSTEM conversion-factor bug (see")
    print("dimensional_analysis.mars_climate_orbiter_case_study), a different failure")
    print("mode than the dimensional mismatch this calculator actually guards against.")
