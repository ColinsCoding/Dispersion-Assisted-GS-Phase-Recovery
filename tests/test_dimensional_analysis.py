"""Test dimensional analysis of standard electrodynamics equations, and
crucially confirm the checker actually REJECTS dimensionally wrong
equations -- a checker that only ever prints PASS has never been proven
to catch anything."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sympy.physics import units as u
from dgs import dimensional_analysis as da

# 1. every real EM equation in this module is dimensionally consistent
results = da.run_all_checks()
assert all(results.values()), results

# 2. the checker must REJECT genuinely wrong equations -- e.g. force
#    can NOT equal energy (they differ by a length), and voltage can NOT
#    equal current (they differ by resistance)
assert da.dims_equal(u.newtons, u.joules) == False
assert da.dims_equal(u.volts, u.amperes) == False
assert da.dims_equal(u.watts, u.newtons) == False

# 3. it correctly ACCEPTS equations that are dimensionally identical even
#    when built from totally different combinations of base units
#    (Ohm's law rearranged three ways all reduce to the same dimensions)
V_over_I = u.volts / u.amperes
assert da.dims_equal(V_over_I, u.ohms) == True
P_over_I2 = u.watts / u.amperes ** 2   # P = I^2*R -> R = P/I^2
assert da.dims_equal(P_over_I2, u.ohms) == True

# 4. Coulomb's law specifically: swapping Newtons for Joules must fail,
#    confirming check_coulombs_law isn't vacuously true
F_expr = u.coulomb_constant * u.coulombs * u.coulombs / u.meters ** 2
assert da.dims_equal(F_expr, u.newtons) == True
assert da.dims_equal(F_expr, u.joules) == False

# 5. the EM wave speed combination really does require BOTH permeability
#    and permittivity -- permeability alone does not reduce to a velocity
assert da.dims_equal(1 / u.vacuum_permeability, u.meters / u.seconds) == False

# 6. Mars Climate Orbiter case study: the numeric consequence is exactly
#    the lbf->N conversion factor, not some other arbitrary number
mco = da.mars_climate_orbiter_case_study()
assert abs(mco["resulting_error_factor"] - 4.4482216152605) < 1e-9
assert abs(mco["value_should_have_been_N_s"] - mco["lbf_to_N_conversion_factor"]) < 1e-9

print("all dgs.dimensional_analysis tests passed")
