"""Test intro-chem dimensional analysis: Q algebra, molar mass, Avogadro chains."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import chem_units as cu
from dgs.chem_units import Q, G, KG, M, S, MOL, K, L, ML, HR, MILE, ATM, J, CAL, TORR

# 1. factor-label classics: units cancel by exponent arithmetic
assert abs((60 * MILE / HR).to(M / S) - 26.8224) < 1e-4      # 60 mph
assert abs((G / ML).to(KG / M**3) - 1000.0) < 1e-9           # water density
assert abs((1 * ATM).to(TORR) - 760.0) < 1e-9                # pressure ladder
assert abs((1 * CAL).to(J) - 4.184) < 1e-12                  # calorie def
assert abs((1 * ML).to((0.01 * M) ** 3) - 1.0) < 1e-12       # 1 mL == 1 cm^3

# 2. dimension safety: mismatched add/convert raises; matched add works
for bad in (lambda: 3 * G + 2 * S,
            lambda: (5 * M).to(S),
            lambda: 1 * KG + 1.0):                            # Q + plain number
    try:
        bad()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
assert abs((1 * KG + 500 * G).to(G) - 1500.0) < 1e-9

# 3. gas constant is ONE quantity in two costumes; STP molar volume 22.414 L
assert abs(cu.R_GAS.to(J / (MOL * K)) - 8.31446) < 1e-5
assert abs(cu.R_GAS.to(L * ATM / (MOL * K)) - 0.0820574) < 1e-6
# note: RT/P is L *per mole* -- .to(L) alone raises, which is the point
assert abs((cu.R_GAS * (273.15 * K) / ATM).to(L / MOL) - 22.414) < 1e-3

# 4. formula parser incl. parentheses; textbook molar masses (g/mol)
assert cu.parse_formula("Al2(SO4)3") == {"Al": 2, "S": 3, "O": 12}
assert cu.parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
for formula, mm in (("H2O", 18.015), ("CO2", 44.009), ("NaCl", 58.44),
                    ("C6H12O6", 180.156), ("CaCO3", 100.086),
                    ("Al2(SO4)3", 342.13), ("NH3", 17.031)):
    assert abs(cu.molar_mass(formula).to(G / MOL) - mm) < 0.02, formula

# 5. bad formulas raise
for bad_f in ("Xx2O", "H2O)", "(H2O", "", "h2o"):
    try:
        cu.parse_formula(bad_f)
        raise AssertionError(f"expected ValueError for {bad_f!r}")
    except ValueError:
        pass

# 6. the Avogadro chain: 9 g H2O = 0.4996 mol = 3.009e23 molecules
n = cu.grams_to_moles(9.0, "H2O")
assert abs(n.to(MOL) - 0.49959) < 1e-4
assert abs(cu.grams_to_molecules(9.0, "H2O") - 3.0087e23) < 1e20
assert abs(cu.moles_to_molecules(1 * MOL) - 6.02214076e23) < 1e10

# 7. molarity and dilution: 58.44 g NaCl in 2 L = 0.500 M;
#    250 mL of 0.100 M from 0.500 M stock needs 50 mL
conc = cu.molarity(cu.grams_to_moles(58.44, "NaCl"), 2 * L)
assert abs(conc.to(MOL / L) - 0.500) < 1e-3
v1 = cu.dilution_v1(conc, 0.100 * MOL / L, 250 * ML)
assert abs(v1.to(ML) - 50.0) < 0.1
try:
    cu.dilution_v1(conc, 1.0 * MOL / L, 250 * ML)   # can't dilute UP
    raise AssertionError("expected ValueError")
except ValueError:
    pass

# 8. temperature is affine, not factor-label: offsets handled separately
assert abs(cu.c_to_k(25.0).to(K) - 298.15) < 1e-12
assert abs(cu.f_to_c(32.0) - 0.0) < 1e-12
assert abs(cu.f_to_c(98.6) - 37.0) < 1e-9
try:
    cu.c_to_k(-300.0)
    raise AssertionError("expected ValueError")
except ValueError:
    pass

# 9. guard rails on physical inputs
for bad in (lambda: cu.grams_to_moles(-1.0, "H2O"),
            lambda: cu.moles_to_molecules(-0.5)):
    try:
        bad()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

print(f"TEST PASS  (60 mph=26.82 m/s; R=8.314 J/molK=0.08206 Latm/molK; STP=22.414 L; "
      f"Al2(SO4)3={cu.molar_mass('Al2(SO4)3').to(G/MOL):.2f} g/mol; 9 g H2O=3.01e23 "
      f"molecules; dilution 50 mL; g+s raises)")
