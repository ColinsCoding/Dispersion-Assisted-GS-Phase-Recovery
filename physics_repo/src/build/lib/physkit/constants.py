"""CODATA 2018 physical constants in SI units.

Every value is a plain float in coherent SI (metre, kilogram, second, ampere, kelvin, mole, candela),
so expressions built from them are dimensionally consistent by construction. `table()` returns a
Pandas DataFrame for display in the notebooks.
"""

# Exact (SI-defining) constants
C = 299792458.0                 # speed of light in vacuum, m/s
H = 6.62607015e-34              # Planck constant, J s
HBAR = H / (2.0 * 3.141592653589793)   # reduced Planck constant, J s
E = 1.602176634e-19             # elementary charge, C
K_B = 1.380649e-23              # Boltzmann constant, J/K
N_A = 6.02214076e23             # Avogadro constant, 1/mol

# Measured constants
G = 6.67430e-11                 # Newtonian gravitation, m^3/(kg s^2)
EPS0 = 8.8541878128e-12         # vacuum permittivity, F/m
MU0 = 1.25663706212e-6          # vacuum permeability, N/A^2
M_E = 9.1093837015e-31          # electron mass, kg
M_P = 1.67262192369e-27         # proton mass, kg
A0 = 5.29177210903e-11          # Bohr radius, m
RYDBERG_EV = 13.605693122994    # Rydberg energy (H ionization), eV
R_INF = 10973731.568160         # Rydberg constant, 1/m
MU_B = 9.2740100783e-24         # Bohr magneton, J/T
ALPHA = 7.2973525693e-3         # fine-structure constant (dimensionless)

# Convenient conversions
EV = E                          # 1 eV in joules
NM = 1e-9                       # 1 nanometre in metres
COULOMB_K = 1.0 / (4.0 * 3.141592653589793 * EPS0)   # Coulomb constant, N m^2/C^2


def table():
    """Return a Pandas DataFrame of the constants: symbol, value, SI unit, description."""
    import pandas as pd
    rows = [
        ("c", C, "m/s", "speed of light"),
        ("h", H, "J s", "Planck constant"),
        ("hbar", HBAR, "J s", "reduced Planck constant"),
        ("e", E, "C", "elementary charge"),
        ("k_B", K_B, "J/K", "Boltzmann constant"),
        ("N_A", N_A, "1/mol", "Avogadro constant"),
        ("eps0", EPS0, "F/m", "vacuum permittivity"),
        ("m_e", M_E, "kg", "electron mass"),
        ("m_p", M_P, "kg", "proton mass"),
        ("a0", A0, "m", "Bohr radius"),
        ("Ry", RYDBERG_EV, "eV", "Rydberg energy"),
        ("mu_B", MU_B, "J/T", "Bohr magneton"),
        ("alpha", ALPHA, "1", "fine-structure constant"),
    ]
    return pd.DataFrame(rows, columns=["symbol", "value_SI", "unit", "description"])
