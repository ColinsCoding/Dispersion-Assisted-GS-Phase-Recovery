"""Test dgs/molecular_ohms_law.py: macroscopic Ohm's law (V=IR) derived
down to compound-dependent Drude conductivity, reusing
dgs/solid_state_physics.py's drude_conductivity directly."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.molecular_ohms_law import (
    resistance_from_conductivity, ohms_law_voltage,
    compound_resistance_table, COMPOUND_LIBRARY,
)

# 1. resistance_from_conductivity: known values, R = L/(sigma*A)
R = resistance_from_conductivity(sigma=1.0, length_m=1.0, area_m2=1.0)
assert abs(R - 1.0) < 1e-12
R2 = resistance_from_conductivity(sigma=2.0, length_m=1.0, area_m2=1.0)
assert abs(R2 - 0.5) < 1e-12, "doubling sigma should halve R"

# 2. resistance_from_conductivity bounds
for bad_kwargs in [dict(sigma=0.0, length_m=1.0, area_m2=1.0),
                    dict(sigma=1.0, length_m=0.0, area_m2=1.0),
                    dict(sigma=1.0, length_m=1.0, area_m2=0.0)]:
    try:
        resistance_from_conductivity(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

# 3. ohms_law_voltage: V = I*R, exact
assert abs(ohms_law_voltage(current_A=2.0, resistance_ohm=5.0) - 10.0) < 1e-12
try:
    ohms_law_voltage(current_A=1.0, resistance_ohm=-1.0)
    raise AssertionError("expected ValueError for negative resistance")
except ValueError:
    pass

# 4. compound_resistance_table: copper must have the LOWEST resistance and
#    intrinsic silicon the HIGHEST, for the same fixed geometry -- the
#    actual physical claim this module makes
L, A = 0.01, 1e-6
rows = compound_resistance_table(L, A)
by_name = {r["compound"]: r for r in rows}
assert by_name["copper (Cu, metal)"]["resistance_ohm"] < by_name["intrinsic silicon (Si)"]["resistance_ohm"]
assert rows[0]["compound"] == "copper (Cu, metal)", "table should be sorted by resistance, lowest first"
assert rows[-1]["compound"] == "intrinsic silicon (Si)"

# 5. Copper's Drude conductivity should be close to its real textbook value
#    (~5.96e7 S/m), confirming the n_density constant matches
#    dgs/solid_state_physics.py's own demo value, not a made-up number
sigma_cu = by_name["copper (Cu, metal)"]["sigma_S_per_m"]
assert abs(sigma_cu - 5.96e7) / 5.96e7 < 0.02, f"expected ~5.96e7 S/m for copper, got {sigma_cu:.3e}"

# 6. Doping must increase resistivity's inverse (conductivity) by many
#    orders of magnitude: n-doped silicon should conduct far better than
#    intrinsic silicon at the SAME geometry -- the actual "molecular
#    manufacturing" claim
sigma_intrinsic = by_name["intrinsic silicon (Si)"]["sigma_S_per_m"]
sigma_doped = by_name["n-doped silicon (Si:P, representative)"]["sigma_S_per_m"]
assert sigma_doped / sigma_intrinsic > 1e5, "doping should increase conductivity by orders of magnitude"

# 7. compound_resistance_table works with a custom (non-default) compound dict too
custom = {"test material": {"n_density_m3": 1e28, "tau_s": 1e-14, "m_eff": COMPOUND_LIBRARY[
    "copper (Cu, metal)"]["m_eff"], "note": "test"}}
custom_rows = compound_resistance_table(L, A, compounds=custom)
assert len(custom_rows) == 1
assert custom_rows[0]["compound"] == "test material"

print("all dgs.molecular_ohms_law tests passed")
