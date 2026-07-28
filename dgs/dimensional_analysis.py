"""Dimensional analysis for electromagnetism, done for real with SymPy's
unit system -- not "the units obviously work out," but an actual symbolic
check that both sides of each equation reduce to the same base SI
dimensions (mass, length, time, current).

The motivating complaint this answers ("industry doesn't do dimensional
analysis"): the most famous real counter-example is NASA's 1999 Mars
Climate Orbiter, lost because one team's navigation software output thrust
impulse in pound-force-seconds while the receiving software assumed
newton-seconds -- a missed dimensional/unit check that a tool like this
would have caught immediately (the two values differ by exactly the
lbf->N conversion factor, ~4.448x, not some subtle error).
"""

from sympy import Rational, pi, N
from sympy.physics import units as u
from sympy.physics.units.systems.si import SI


def is_dimensionless(expr):
    """True iff expr's dimensional dependencies reduce to {} -- a genuine
    Buckingham-Pi dimensionless group, not just 'the same dimension as
    something else' (that's what dims_equal is for)."""
    dim = SI.get_dimensional_expr(expr)
    deps = SI.get_dimension_system().get_dimensional_dependencies(dim)
    return deps == {}


def dims_equal(expr1, expr2):
    """True iff two SymPy unit-expressions reduce to the SAME base SI
    dimensional dependencies (mass, length, time, current exponents) --
    not just the same dimension NAME (e.g. 'voltage' vs a raw combination
    that happens to also be volts)."""
    dim1 = SI.get_dimensional_expr(expr1)
    dim2 = SI.get_dimensional_expr(expr2)
    deps1 = SI.get_dimension_system().get_dimensional_dependencies(dim1)
    deps2 = SI.get_dimension_system().get_dimensional_dependencies(dim2)
    return deps1 == deps2


def check_coulombs_law():
    """F = k*q1*q2/r^2 must reduce to a force (Newtons)."""
    F_expr = u.coulomb_constant * u.coulombs * u.coulombs / u.meters ** 2
    return dims_equal(F_expr, u.newtons)


def check_gauss_law():
    """Gauss's law: (E field) x (area) = Q_enclosed / epsilon0.
    E has units of volts/meter, so E*A must match Q/epsilon0."""
    E_times_A = (u.volts / u.meters) * u.meters ** 2
    Q_over_eps0 = u.coulombs / u.vacuum_permittivity
    return dims_equal(E_times_A, Q_over_eps0)


def check_em_wave_speed():
    """c = 1/sqrt(mu0*epsilon0) must reduce to a velocity (m/s), the
    single most consequential dimensional check in electrodynamics --
    Maxwell's equations alone imply light is an EM wave BECAUSE this
    combination has units of speed."""
    c_expr = 1 / (u.vacuum_permeability * u.vacuum_permittivity) ** Rational(1, 2)
    return dims_equal(c_expr, u.meters / u.seconds)


def check_poynting_vector():
    """Poynting vector S = E x H must reduce to power/area (W/m^2):
    E [V/m] * H [A/m] = (V*A)/m^2 = W/m^2, since a volt-amp is a watt."""
    S_expr = (u.volts / u.meters) * (u.amperes / u.meters)
    power_per_area = u.watts / u.meters ** 2
    return dims_equal(S_expr, power_per_area)


def check_faraday_law():
    """Faraday's law: EMF = -d(Phi)/dt. Phi (magnetic flux) is in webers
    (volt-seconds); its time derivative must reduce to volts."""
    dPhi_dt = u.webers / u.seconds
    return dims_equal(dPhi_dt, u.volts)


def check_impedance_of_free_space():
    """eta0 = sqrt(mu0/epsilon0) must reduce to Ohms -- the constant that
    sets the ratio |E|/|H| for a plane wave in vacuum (~377 Ohm)."""
    eta0_expr = (u.vacuum_permeability / u.vacuum_permittivity) ** Rational(1, 2)
    return dims_equal(eta0_expr, u.ohms)


# ── Dimensionless ratios (Buckingham Pi groups) ───────────────────────────────
#
# A dimensional-EQUALITY check (dims_equal, above) answers "do both sides of
# an equation reduce to the same units?" A dimensionless RATIO is a different
# and arguably more powerful idea: certain combinations of physical
# quantities have NO units at all, and it's exactly that unit-free-ness that
# lets a single number characterize a whole physical regime -- independent of
# whether you measured it in meters or feet.

def check_reynolds_number():
    """Re = rho * v * L / mu must be dimensionless -- it's the ratio of
    inertial to viscous forces that separates laminar (Re << 1) from
    turbulent (Re >> 1) flow, regardless of the fluid or the unit system."""
    rho = u.kilogram / u.meter ** 3
    v = u.meter / u.second
    L = u.meter
    mu = u.pascal * u.second   # dynamic viscosity: Pa*s = kg/(m*s)
    Re_expr = rho * v * L / mu
    return is_dimensionless(Re_expr)


def fine_structure_constant():
    """alpha = e^2 / (4*pi*epsilon0*hbar*c): the dimensionless number that
    sets the strength of the electromagnetic interaction. Built from four
    SI constants that individually carry units (charge, permittivity,
    action, velocity) -- that they combine into something with NO units at
    all is the entire physical content of the result. Returns (alpha,
    is_dimensionless)."""
    alpha_expr = (
        u.elementary_charge ** 2
        / (4 * pi * u.vacuum_permittivity * u.hbar * u.speed_of_light)
    )
    factor, _dim = SI._collect_factor_and_dimension(alpha_expr)
    alpha = float(N(factor))
    return alpha, is_dimensionless(alpha_expr)


def dispersion_regime_parameter(D):
    """The temporal analog of the optical Fresnel number for gs_core's
    dispersion operator H(nu) = exp(i*pi*D*nu^2), nu in [-0.5, 0.5).

    The quadratic phase sweeps theta_max = pi*D*(0.5)^2 = pi*D/4 radians
    across the Nyquist edge. Far-field diffraction (Fraunhofer) requires
    a LARGE phase excursion across the aperture; the temporal analog
    (dispersive/real-time Fourier transform, see the stationary-phase
    argument in griffiths_1_42_1_43...ipynb) needs the same thing: the
    phase must sweep through many radians across the signal's normalized
    bandwidth for the stationary-phase point nu*=t/D to dominate.

    |D|/4 IS that dimensionless sweep, in units of pi radians -- and it's
    exactly the quantity gs_core._check_dispersion gates on (warns below
    |D|=100, documents |D|>=5000 as needed for GS convergence). Returns
    the dimensionless ratio |D|/4.
    """
    return abs(D) / 4.0


def run_all_checks():
    """Run every EM dimensional check in this module; returns a dict of
    name -> bool. A real dimensional-analysis gate would fail the build if
    any of these came back False."""
    return {
        "coulombs_law": check_coulombs_law(),
        "gauss_law": check_gauss_law(),
        "em_wave_speed": check_em_wave_speed(),
        "poynting_vector": check_poynting_vector(),
        "faraday_law": check_faraday_law(),
        "impedance_of_free_space": check_impedance_of_free_space(),
        "reynolds_number_is_dimensionless": check_reynolds_number(),
        "fine_structure_constant_is_dimensionless": fine_structure_constant()[1],
    }


def mars_climate_orbiter_case_study():
    """The real 1999 NASA/Lockheed Martin unit-mismatch failure: ground
    software computed small forces (impulse bit) in pound-force-seconds;
    the spacecraft's navigation software assumed newton-seconds, per the
    interface specification. Returns the actual numeric consequence: the
    ratio between the two unit systems, applied uncorrected, sent the
    orbiter into a trajectory that put it too close to Mars and it was
    lost. This is not a subtle rounding error -- it's exactly the
    lbf-to-N conversion factor, applied zero times instead of once."""
    lbf_to_N = 4.4482216152605   # exact SI definition-derived conversion factor
    reported_impulse_lbf_s = 1.0
    misinterpreted_as_N_s = reported_impulse_lbf_s   # the bug: no conversion applied
    correct_N_s = reported_impulse_lbf_s * lbf_to_N
    error_factor = correct_N_s / misinterpreted_as_N_s
    return {
        "lbf_to_N_conversion_factor": lbf_to_N,
        "value_as_sent_N_s": misinterpreted_as_N_s,
        "value_should_have_been_N_s": correct_N_s,
        "resulting_error_factor": error_factor,
    }


if __name__ == "__main__":
    print("Dimensional analysis of standard electrodynamics equations (SymPy, base SI dims):")
    results = run_all_checks()
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    assert all(results.values())

    print("\nWhy this matters in industry -- the Mars Climate Orbiter (1999):")
    mco = mars_climate_orbiter_case_study()
    print(f"  1 lbf*s = {mco['lbf_to_N_conversion_factor']:.6f} N*s")
    print(f"  ground software sent:      {mco['value_as_sent_N_s']:.4f} (labeled N*s, actually lbf*s)")
    print(f"  navigation software wanted: {mco['value_should_have_been_N_s']:.4f} N*s")
    print(f"  resulting error factor: {mco['resulting_error_factor']:.4f}x -- "
          f"a dimensional-analysis check like the ones above would have caught this instantly.")

    print("\nDimensionless ratios (Buckingham Pi groups):")
    alpha, alpha_ok = fine_structure_constant()
    print(f"  fine structure constant: alpha = {alpha:.10f}  (1/alpha = {1/alpha:.4f})  "
          f"dimensionless: {alpha_ok}")
    print(f"  dispersion regime parameter |D|/4 at D=-5000 (gs_core default): "
          f"{dispersion_regime_parameter(-5000):.1f}  (>>1 -> real-time-Fourier-transform regime)")
    print(f"  dispersion regime parameter |D|/4 at D=-10 (below gs_core's warn threshold): "
          f"{dispersion_regime_parameter(-10):.2f}  (~1 -> near-field, GS should stagnate)")
