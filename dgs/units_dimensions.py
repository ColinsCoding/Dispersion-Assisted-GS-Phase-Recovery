"""Units and dimensions as linear algebra: exponent vectors, the Buckingham
Pi theorem, and natural scales built from physical constants.

How this differs from the two modules already in the repo:

  dgs.dimensional_analysis  CHECKS that known EM equations balance (SymPy units).
  dgs.chem_units            CONVERTS quantities with a factor-label class `Q`.
  dgs.units_dimensions      DERIVES relations you were never given: which
                            combinations of variables are even allowed.

The one idea
------------
Every physical quantity has a dimension vector: its exponents over the seven
SI base dimensions (M, L, T, I, Theta, N, J). Multiplying quantities ADDS
vectors; raising to a power SCALES the vector. So

  "which products  x1^a1 * x2^a2 * ...  have the dimensions I want?"

is a LINEAR system  D a = b,  where column j of D is the dimension vector of
variable j. Two cases matter:

  b = 0   the solutions are the dimensionless groups. Their number is
          n - rank(D): the Buckingham Pi theorem. (Reynolds number, the
          fine-structure constant, drag coefficient.)
  b != 0  a unique solution gives a scaling law up to a pure number
          (pendulum period, Bohr radius, Planck length, Stefan-Boltzmann).

What dimensional analysis can NOT do, stated honestly: it never supplies
the dimensionless prefactor (the 2*pi in the pendulum period, the pi^2/60 in
the Stefan-Boltzmann constant), and it fails by construction when a
dimensionless group exists among the chosen variables (then any function of
that group may multiply the answer -- `solve_power_law` raises instead of
guessing).

Exact rational arithmetic (fractions + SymPy), so exponents like 1/2 and 2/5
come out exactly. py -3.13 safe; needs only sympy.
"""
import math
from fractions import Fraction

from sympy import Matrix, Rational, ilcm

BASE = ("M", "L", "T", "I", "Th", "N", "J")   # mass, length, time, current,
                                              # temperature, amount, luminous intensity


# ----------------------------------------------------------------------
# 1. Dimension vectors
# ----------------------------------------------------------------------

def dim(M=0, L=0, T=0, I=0, Th=0, N=0, J=0):
    """Dimension vector as a 7-tuple of exact Fractions, e.g. force = dim(M=1, L=1, T=-2)."""
    return tuple(Fraction(x) for x in (M, L, T, I, Th, N, J))


DIMENSIONLESS = dim()


def _check(d, name="dimension"):
    if not isinstance(d, tuple) or len(d) != len(BASE):
        raise ValueError(f"{name} must be a {len(BASE)}-tuple from dim(), got {d!r}")
    return d


def dim_mul(*dims):
    """Dimension of a product: exponent vectors add."""
    out = [Fraction(0)] * len(BASE)
    for d in dims:
        out = [a + b for a, b in zip(out, _check(d))]
    return tuple(out)


def dim_pow(d, power):
    """Dimension of x**power: the exponent vector is scaled."""
    p = Fraction(power)
    return tuple(a * p for a in _check(d))


def dim_div(num, den):
    """Dimension of a quotient: exponent vectors subtract."""
    return dim_mul(num, dim_pow(den, -1))


def dim_str(d):
    """Human-readable form, e.g. 'M L^2 T^-2'; '1' for a pure number."""
    parts = []
    for name, e in zip(BASE, _check(d)):
        if e == 0:
            continue
        parts.append(name if e == 1 else f"{name}^{e}")
    return " ".join(parts) if parts else "1"


def check_homogeneous(**terms):
    """Raise ValueError unless every named term has the same dimension.

    Terms that are added, subtracted or equated MUST match; this is the
    error-catching use of dimensions (see dgs.dimensional_analysis for the
    Mars Climate Orbiter case)."""
    if not terms:
        raise ValueError("need at least one term")
    names = list(terms)
    first = _check(terms[names[0]], names[0])
    for n in names[1:]:
        if _check(terms[n], n) != first:
            raise ValueError(
                f"dimension mismatch: {names[0]} is [{dim_str(first)}] but "
                f"{n} is [{dim_str(terms[n])}]")
    return first


# Named dimensions used across mechanics, circuits, optics and modern physics.
DIMS = {
    "mass": dim(M=1), "length": dim(L=1), "time": dim(T=1),
    "current": dim(I=1), "temperature": dim(Th=1), "amount": dim(N=1),
    "area": dim(L=2), "volume": dim(L=3), "frequency": dim(T=-1),
    "velocity": dim(L=1, T=-1), "acceleration": dim(L=1, T=-2),
    "density": dim(M=1, L=-3),
    "force": dim(M=1, L=1, T=-2),
    "pressure": dim(M=1, L=-1, T=-2),
    "energy": dim(M=1, L=2, T=-2),
    "power": dim(M=1, L=2, T=-3),
    "action": dim(M=1, L=2, T=-1),                 # h, hbar, angular momentum
    "viscosity": dim(M=1, L=-1, T=-1),             # dynamic viscosity mu
    "charge": dim(I=1, T=1),
    "voltage": dim(M=1, L=2, T=-3, I=-1),
    "resistance": dim(M=1, L=2, T=-3, I=-2),
    "capacitance": dim(M=-1, L=-2, T=4, I=2),
    "inductance": dim(M=1, L=2, T=-2, I=-2),
    "irradiance": dim(M=1, T=-3),                  # W/m^2
    "gvd": dim(L=-1, T=2),                         # beta_2 in s^2/m (fiber dispersion)
}


# ----------------------------------------------------------------------
# 2. The dimension matrix and the Buckingham Pi theorem
# ----------------------------------------------------------------------

def _validated(variables):
    if not isinstance(variables, dict) or not variables:
        raise ValueError("variables must be a non-empty dict {name: dim(...)}")
    for name, d in variables.items():
        _check(d, name)
    return variables


def dimension_matrix(variables):
    """D[i, j] = exponent of base dimension i in variable j (7 rows, n columns)."""
    _validated(variables)
    cols = list(variables.values())
    return Matrix(len(BASE), len(cols),
                  lambda i, j: Rational(cols[j][i].numerator, cols[j][i].denominator))


def count_pi_groups(variables):
    """Buckingham Pi count: (number of variables) - rank(dimension matrix)."""
    return len(variables) - dimension_matrix(variables).rank()


def buckingham_pi(variables):
    """Basis of dimensionless groups, as a list of {name: exponent} dicts.

    Each group is a null vector of the dimension matrix, cleared to the
    smallest integers with its first non-zero exponent positive. Variables
    with exponent 0 are omitted. The basis is not unique: any product of
    powers of Pi groups is also a Pi group (Re and 1/Re are the same physics).
    """
    names = list(_validated(variables))
    groups = []
    for vec in dimension_matrix(variables).nullspace():
        scale = ilcm(*[x.q for x in vec]) if len(vec) > 1 else vec[0].q
        ints = [int(x * scale) for x in vec]
        first = next(x for x in ints if x != 0)
        if first < 0:
            ints = [-x for x in ints]
        groups.append({n: e for n, e in zip(names, ints) if e != 0})
    return groups


def group_dimension(exponents, variables):
    """Dimension of prod(variable**exponent). Useful to verify a group really is dimensionless."""
    return dim_mul(*[dim_pow(variables[n], e) for n, e in exponents.items()])


# ----------------------------------------------------------------------
# 3. Scaling laws: the unique power law with a target dimension
# ----------------------------------------------------------------------

def solve_power_law(target, variables):
    """Exponents {name: Fraction} such that prod(var**exp) has dimension `target`.

    Raises ValueError if no power law of these variables has that dimension,
    or if the answer is not unique (a dimensionless group exists among the
    variables, so dimensional analysis alone cannot decide).
    """
    _check(target, "target")
    names = list(_validated(variables))
    D = dimension_matrix(variables)
    b = Matrix([Rational(x.numerator, x.denominator) for x in target])
    try:
        sol, params = D.gauss_jordan_solve(b)
    except ValueError:
        raise ValueError(
            f"no product of powers of {names} has dimension [{dim_str(target)}]") from None
    if len(params):
        raise ValueError(
            f"not unique: {len(params)} dimensionless group(s) exist among {names}; "
            "drop a variable or fix the group's value")
    return {n: Fraction(int(s.p), int(s.q)) for n, s in zip(names, sol)}


def power_law_str(exponents):
    """'L^1/2 g^-1/2' style rendering of an exponent dict."""
    parts = [n if e == 1 else f"{n}^{e}" for n, e in exponents.items() if e != 0]
    return " ".join(parts) if parts else "1"


# ----------------------------------------------------------------------
# 4. Physical constants and the natural scales they define
# ----------------------------------------------------------------------
# value in SI units, dimension. c, h, e, k_B, N_A are EXACT by the 2019 SI
# redefinition; G, m_e, m_p, eps0 are measured (CODATA 2018 values).
_H = 6.62607015e-34
_E = 1.602176634e-19
_EPS0 = 8.8541878128e-12

CONSTANTS = {
    "c":    (299792458.0,            DIMS["velocity"]),
    "h":    (_H,                     DIMS["action"]),
    "hbar": (_H / (2 * math.pi),     DIMS["action"]),
    "e":    (_E,                     DIMS["charge"]),
    "k_B":  (1.380649e-23,           dim(M=1, L=2, T=-2, Th=-1)),
    "N_A":  (6.02214076e23,          dim(N=-1)),
    "G":    (6.67430e-11,            dim(M=-1, L=3, T=-2)),
    "m_e":  (9.1093837015e-31,       DIMS["mass"]),
    "m_p":  (1.67262192369e-27,      DIMS["mass"]),
    "eps0": (_EPS0,                  dim(M=-1, L=-3, T=4, I=2)),
    # Coulomb coupling e^2 / (4 pi eps0): an energy times a length. Using it
    # instead of e and eps0 separately removes the current dimension, which
    # is what lets atomic scales come out of M, L, T alone.
    "ke2":  (_E ** 2 / (4 * math.pi * _EPS0), dim(M=1, L=3, T=-2)),
}

C_LIGHT = CONSTANTS["c"][0]
HBAR = CONSTANTS["hbar"][0]
E_CHARGE = _E
HBAR_C_MEV_FM = HBAR * C_LIGHT / (_E * 1e6) / 1e-15   # ~197.327 MeV fm


def scale(target, names):
    """Natural scale: the unique combination of the named CONSTANTS with dimension `target`.

    Returns (value_in_SI, exponents). Example: scale(DIMS['length'],
    ['hbar', 'G', 'c']) is the Planck length. The value carries NO
    dimensionless prefactor -- that is all dimensional analysis can give.
    """
    unknown = [n for n in names if n not in CONSTANTS]
    if unknown:
        raise ValueError(f"unknown constant(s) {unknown}; available: {sorted(CONSTANTS)}")
    exps = solve_power_law(target, {n: CONSTANTS[n][1] for n in names})
    value = math.prod(CONSTANTS[n][0] ** float(e) for n, e in exps.items())
    return value, exps


def evaluate_group(exponents, values):
    """Numerical value of prod(values[name]**exponent)."""
    missing = [n for n in exponents if n not in values]
    if missing:
        raise ValueError(f"no value supplied for {missing}")
    return math.prod(float(values[n]) ** float(e) for n, e in exponents.items())


# ----------------------------------------------------------------------
# 5. Units as measuring sticks: light-time and natural units
# ----------------------------------------------------------------------

def light_travel_time(distance_m, velocity_factor=1.0):
    """Seconds for a signal to cover `distance_m`.

    velocity_factor = v/c: 1.0 in vacuum, about 0.66 in common coaxial cable,
    about 0.68 in silica fiber (1/n with n ~ 1.47). A distance of 299 792 458 m
    is one light-second; 0.2998 m is one light-nanosecond ("about a foot").
    """
    if distance_m < 0:
        raise ValueError("distance must be non-negative")
    if not 0 < velocity_factor <= 1:
        raise ValueError("velocity_factor must be in (0, 1]")
    return distance_m / (velocity_factor * C_LIGHT)


def light_distance(seconds, velocity_factor=1.0):
    """Metres covered in `seconds`: the inverse of light_travel_time."""
    if seconds < 0:
        raise ValueError("time must be non-negative")
    if not 0 < velocity_factor <= 1:
        raise ValueError("velocity_factor must be in (0, 1]")
    return seconds * velocity_factor * C_LIGHT


def joule_to_ev(energy_J):
    """1 eV = e joules exactly (the charge of one electron through one volt)."""
    return energy_J / _E


def ev_to_joule(energy_eV):
    return energy_eV * _E


def energy_to_wavelength(energy_J):
    """Photon wavelength lambda = h c / E. In natural units 'energy = 1/length'."""
    if energy_J <= 0:
        raise ValueError("energy must be positive")
    return _H * C_LIGHT / energy_J


def mass_to_rest_energy_ev(mass_kg):
    """E = m c^2 in eV: why particle masses are quoted in MeV/c^2."""
    if mass_kg < 0:
        raise ValueError("mass must be non-negative")
    return mass_kg * C_LIGHT ** 2 / _E


if __name__ == "__main__":
    print("=== units_dimensions: dimensions as linear algebra ===\n")

    pend = {"L": DIMS["length"], "g": DIMS["acceleration"], "m": DIMS["mass"]}
    print("pendulum period  T ~", power_law_str(solve_power_law(DIMS["time"], pend)),
          "  (mass drops out; the 2*pi is not derivable)")

    flow = {"rho": DIMS["density"], "v": DIMS["velocity"],
            "L": DIMS["length"], "mu": DIMS["viscosity"]}
    print("pipe flow Pi group:", power_law_str(buckingham_pi(flow)[0]), " = Reynolds number")

    alpha = buckingham_pi({n: CONSTANTS[n][1] for n in ("ke2", "hbar", "c")})[0]
    val = evaluate_group(alpha, {n: CONSTANTS[n][0] for n in alpha})
    print(f"(ke2, hbar, c) Pi group: {power_law_str(alpha)} = 1/{1/val:.3f}  (fine-structure constant)")

    for label, target, names in [
        ("Planck length", DIMS["length"], ["hbar", "G", "c"]),
        ("Planck time", DIMS["time"], ["hbar", "G", "c"]),
        ("Bohr radius", DIMS["length"], ["hbar", "m_e", "ke2"]),
        ("Hartree energy", DIMS["energy"], ["hbar", "m_e", "ke2"]),
    ]:
        value, exps = scale(target, names)
        print(f"{label:<15} = {power_law_str(exps):<28} = {value:.4e} SI")

    print(f"\nhbar*c = {HBAR_C_MEV_FM:.3f} MeV fm")
    print(f"1 light-nanosecond = {light_distance(1e-9)*100:.2f} cm; "
          f"1 m of coax (0.66c) delays a pulse {light_travel_time(1.0, 0.66)*1e9:.2f} ns")
    print("\n=== DONE ===")
