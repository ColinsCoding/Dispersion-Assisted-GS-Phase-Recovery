"""Test dgs.units_dimensions: dimension-vector algebra, the Buckingham Pi count
and groups, unique scaling laws with exact rational exponents, natural scales
checked against their accepted values, and -- as important -- that the solver
REFUSES when no answer or no unique answer exists."""
import sys, pathlib, math
from fractions import Fraction as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import units_dimensions as ud
from dgs.units_dimensions import DIMS, CONSTANTS, dim

# 1. dimension-vector algebra: products add exponents, powers scale them
assert ud.dim_mul(DIMS["force"], DIMS["length"]) == DIMS["energy"]          # work = F * d
assert ud.dim_div(DIMS["energy"], DIMS["time"]) == DIMS["power"]
assert ud.dim_mul(DIMS["voltage"], DIMS["current"]) == DIMS["power"]        # P = V I
assert ud.dim_div(DIMS["voltage"], DIMS["current"]) == DIMS["resistance"]   # Ohm's law
assert ud.dim_mul(DIMS["resistance"], DIMS["capacitance"]) == DIMS["time"]  # RC time constant
assert ud.dim_pow(ud.dim_mul(DIMS["inductance"], DIMS["capacitance"]), F(1, 2)) == DIMS["time"]  # sqrt(LC)
assert ud.dim_div(DIMS["inductance"], DIMS["resistance"]) == DIMS["time"]   # L/R time constant
assert ud.dim_str(DIMS["energy"]) == "M L^2 T^-2"
assert ud.dim_str(ud.DIMENSIONLESS) == "1"

# 2. homogeneity: kinetic and potential energy may be added; energy and force may not
ke = ud.dim_mul(DIMS["mass"], ud.dim_pow(DIMS["velocity"], 2))
pe = ud.dim_mul(DIMS["mass"], DIMS["acceleration"], DIMS["length"])
assert ud.check_homogeneous(kinetic=ke, potential=pe) == DIMS["energy"]
try:
    ud.check_homogeneous(energy=DIMS["energy"], force=DIMS["force"]); assert False
except ValueError as err:
    assert "mismatch" in str(err)

# 3. pendulum: T ~ sqrt(L/g) exactly, and the mass exponent is exactly zero
pend = {"L": DIMS["length"], "g": DIMS["acceleration"], "m": DIMS["mass"]}
assert ud.solve_power_law(DIMS["time"], pend) == {"L": F(1, 2), "g": F(-1, 2), "m": F(0)}

# 4. Buckingham Pi count = n - rank, and every returned group really is dimensionless
flow = {"rho": DIMS["density"], "v": DIMS["velocity"], "L": DIMS["length"], "mu": DIMS["viscosity"]}
assert ud.count_pi_groups(flow) == 1
assert ud.buckingham_pi(flow) == [{"rho": 1, "v": 1, "L": 1, "mu": -1}]     # Reynolds number

drag = dict(F=DIMS["force"], **flow)
groups = ud.buckingham_pi(drag)
assert ud.count_pi_groups(drag) == len(groups) == 2                          # C_D and Re
for g in groups:
    assert ud.group_dimension(g, drag) == ud.DIMENSIONLESS

assert ud.count_pi_groups(pend) == 0 and ud.buckingham_pi(pend) == []        # nothing dimensionless

# 5. Taylor blast wave: R ~ (E t^2 / rho)^(1/5), fifth-root exponents come out exact
blast = {"E": DIMS["energy"], "rho": DIMS["density"], "t": DIMS["time"]}
assert ud.solve_power_law(DIMS["length"], blast) == {"E": F(1, 5), "rho": F(-1, 5), "t": F(2, 5)}

# 6. fiber optics tie-in: dispersion length L_D = T0^2 / beta_2
assert ud.solve_power_law(DIMS["length"], {"T0": DIMS["time"], "beta2": DIMS["gvd"]}) \
    == {"T0": F(2), "beta2": F(-1)}

# 7. the fine-structure constant is THE dimensionless group of (ke2, hbar, c)
names = ("ke2", "hbar", "c")
alpha_group = ud.buckingham_pi({n: CONSTANTS[n][1] for n in names})
assert alpha_group == [{"ke2": 1, "hbar": -1, "c": -1}]
alpha = ud.evaluate_group(alpha_group[0], {n: CONSTANTS[n][0] for n in names})
assert math.isclose(1 / alpha, 137.035999, rel_tol=1e-7)

# 8. natural scales against accepted values (prefactor is exactly 1 for these)
assert math.isclose(ud.scale(DIMS["length"], ["hbar", "m_e", "ke2"])[0], 5.29177210903e-11, rel_tol=1e-8)  # Bohr radius
hartree_J, _ = ud.scale(DIMS["energy"], ["hbar", "m_e", "ke2"])
assert math.isclose(ud.joule_to_ev(hartree_J), 27.211386, rel_tol=1e-7)      # = 2 * 13.6 eV
assert math.isclose(ud.scale(DIMS["length"], ["hbar", "m_e", "c"])[0], 3.8615926796e-13, rel_tol=1e-8)   # reduced Compton
assert math.isclose(ud.scale(DIMS["length"], ["hbar", "G", "c"])[0], 1.616255e-35, rel_tol=1e-5)         # Planck length
assert math.isclose(ud.scale(DIMS["time"], ["hbar", "G", "c"])[0], 5.391247e-44, rel_tol=1e-5)           # Planck time
assert math.isclose(ud.scale(DIMS["mass"], ["hbar", "G", "c"])[0], 2.176434e-8, rel_tol=1e-5)            # Planck mass

# 9. dimensional analysis gets the FORM of Stefan-Boltzmann, not the pi^2/60
sb_dim = dim(M=1, T=-3, Th=-4)                                               # W m^-2 K^-4
sb_scale, sb_exps = ud.scale(sb_dim, ["k_B", "hbar", "c"])
assert sb_exps == {"k_B": F(4), "hbar": F(-3), "c": F(-2)}
assert math.isclose(sb_scale * math.pi ** 2 / 60, 5.670374419e-8, rel_tol=1e-8)
assert not math.isclose(sb_scale, 5.670374419e-8, rel_tol=0.5)               # off by ~6x without the prefactor

# 10. the solver must REFUSE rather than guess
try:                                         # no time can be built from a length and a mass
    ud.solve_power_law(DIMS["time"], {"L": DIMS["length"], "m": DIMS["mass"]}); assert False
except ValueError as err:
    assert "no product" in str(err)
try:                                         # adding c admits alpha: Bohr / Compton / classical radius all fit
    ud.scale(DIMS["length"], ["hbar", "m_e", "ke2", "c"]); assert False
except ValueError as err:
    assert "not unique" in str(err)
try:
    ud.scale(DIMS["length"], ["hbar", "nonsense"]); assert False
except ValueError as err:
    assert "unknown constant" in str(err)
for bad in (lambda: ud.dim_mul((1, 2, 3)), lambda: ud.buckingham_pi({}),
            lambda: ud.evaluate_group({"x": 1}, {})):
    try:
        bad(); assert False
    except ValueError:
        pass

# 11. light-time and natural-unit conversions
assert math.isclose(ud.light_distance(1.0), 299792458.0)                     # one light-second
assert math.isclose(ud.light_travel_time(ud.light_distance(3.7)), 3.7)       # round trip
assert math.isclose(ud.light_distance(1e-9), 0.299792458)                    # light-nanosecond ~ 1 foot
assert math.isclose(ud.light_travel_time(1.0, 0.5), 2 * ud.light_travel_time(1.0))
assert math.isclose(ud.light_travel_time(3.844e8), 1.282, rel_tol=1e-3)      # Earth-Moon
assert math.isclose(ud.HBAR_C_MEV_FM, 197.3269804, rel_tol=1e-9)
assert math.isclose(ud.ev_to_joule(ud.joule_to_ev(2.5)), 2.5)
assert math.isclose(ud.energy_to_wavelength(ud.ev_to_joule(1.0)) * 1e9, 1239.84198, rel_tol=1e-8)  # 1240 eV nm
assert math.isclose(ud.mass_to_rest_energy_ev(CONSTANTS["m_e"][0]) / 1e6, 0.51099895, rel_tol=1e-8)
for bad in (lambda: ud.light_travel_time(-1.0), lambda: ud.light_travel_time(1.0, 1.5),
            lambda: ud.light_distance(1.0, 0.0), lambda: ud.energy_to_wavelength(0.0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_units_dimensions: all checks passed")
