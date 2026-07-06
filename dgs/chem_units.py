"""Dimensional analysis for intro chemistry -- factor-label done as real algebra.

The factor-label method ("multiply by 1 until the units you don't want
cancel") is just arithmetic on quantities that carry a DIMENSION VECTOR:
exponents over the base dimensions (mass, length, time, amount, temperature).

    60 mile/hr * (1609.344 m/mile) * (1 hr/3600 s) = 26.82 m/s

Here that's literally `(60*MILE/HR).to(M/S)` -- the class multiplies the
numbers and ADDS the exponent vectors, and `.to()` refuses if the target
dimension doesn't match. Adding quantities with different dimensions raises
immediately: that is the whole content of dimensional analysis as an
error-catching tool (see dgs.dimensional_analysis for the EM/SymPy version
and the Mars Climate Orbiter story).

Chemistry workflows on top:
  * molar_mass("Al2(SO4)3") -- formula parser with parentheses
  * grams -> moles -> molecules  (Avogadro chain)
  * molarity and M1*V1 = M2*V2 dilution
  * gas constant R expressed in J/(mol K) vs L atm/(mol K) -- SAME quantity
  * temperature: Celsius/Fahrenheit are AFFINE maps (an offset, not a
    factor), which is exactly why the factor-label method cannot do them.

NumPy-free core (pure python), py -3.13 safe. Education.
"""

import re

_DIM_NAMES = ("mass", "length", "time", "amount", "temperature")


class Q:
    """A physical quantity: a number plus base-dimension exponents
    (mass, length, time, amount, temperature). * and / combine exponents;
    + and - demand identical dimensions; .to(unit) converts or refuses."""

    __slots__ = ("val", "dim")

    def __init__(self, val, dim=(0, 0, 0, 0, 0)):
        self.val = float(val)
        self.dim = tuple(dim)

    def _dimstr(self):
        parts = [f"{n}^{e}" for n, e in zip(_DIM_NAMES, self.dim) if e != 0]
        return " ".join(parts) if parts else "dimensionless"

    def __repr__(self):
        return f"Q({self.val:g}, {self._dimstr()})"

    # -- multiplication/division: exponents add/subtract (units "cancel") --
    def __mul__(self, other):
        if isinstance(other, Q):
            return Q(self.val * other.val,
                     tuple(a + b for a, b in zip(self.dim, other.dim)))
        return Q(self.val * other, self.dim)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Q):
            return Q(self.val / other.val,
                     tuple(a - b for a, b in zip(self.dim, other.dim)))
        return Q(self.val / other, self.dim)

    def __rtruediv__(self, other):
        return Q(other / self.val, tuple(-a for a in self.dim))

    def __pow__(self, n):
        return Q(self.val ** n, tuple(a * n for a in self.dim))

    # -- addition: only meaningful for matching dimensions --
    def _require_same_dim(self, other, op):
        if not isinstance(other, Q) or self.dim != other.dim:
            odim = other._dimstr() if isinstance(other, Q) else "plain number"
            raise ValueError(
                f"cannot {op} [{self._dimstr()}] and [{odim}] -- "
                "dimensional analysis forbids it")

    def __add__(self, other):
        self._require_same_dim(other, "add")
        return Q(self.val + other.val, self.dim)

    def __sub__(self, other):
        self._require_same_dim(other, "subtract")
        return Q(self.val - other.val, self.dim)

    def __neg__(self):
        return Q(-self.val, self.dim)

    def to(self, unit):
        """Express this quantity as a multiple of `unit`. The factor-label
        endgame: returns a plain number, but ONLY if dimensions match."""
        self._require_same_dim(unit, "convert between")
        return self.val / unit.val


# ----------------------------------------------------------------------
# Unit registry (values in SI base units)
# ----------------------------------------------------------------------

KG = Q(1, (1, 0, 0, 0, 0));  G = 1e-3 * KG;  MG = 1e-6 * KG
LB = 0.45359237 * KG
AMU = 1.66053906660e-27 * KG

M = Q(1, (0, 1, 0, 0, 0));  CM = 1e-2 * M;  MM = 1e-3 * M;  KM = 1e3 * M
NM = 1e-9 * M;  INCH = 0.0254 * M;  MILE = 1609.344 * M

S = Q(1, (0, 0, 1, 0, 0));  MIN = 60 * S;  HR = 3600 * S

MOL = Q(1, (0, 0, 0, 1, 0))
N_A = 6.02214076e23            # things per mole (exact, 2019 SI)

K = Q(1, (0, 0, 0, 0, 1))

L = 1e-3 * M ** 3;  ML = 1e-3 * L          # note: 1 mL == 1 cm^3
NEWTON = KG * M / S ** 2
J = NEWTON * M;  KJ = 1e3 * J;  CAL = 4.184 * J;  KCAL = 1e3 * CAL
PA = NEWTON / M ** 2;  ATM = 101325 * PA;  TORR = ATM / 760;  BAR = 1e5 * PA

R_GAS = 8.31446261815324 * J / (MOL * K)   # exact (2019 SI): k_B * N_A


# ----------------------------------------------------------------------
# Molar mass: parse "Al2(SO4)3" and sum atomic masses
# ----------------------------------------------------------------------

ATOMIC_MASS = {  # g/mol, IUPAC 2021 abridged
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Br": 79.904, "Ag": 107.87, "Sn": 118.71, "I": 126.90,
    "Ba": 137.33, "Au": 196.97, "Hg": 200.59, "Pb": 207.2, "U": 238.03,
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)|(\()|(\))(\d*)")


def parse_formula(formula):
    """'Al2(SO4)3' -> {'Al': 2, 'S': 3, 'O': 12}. Raises on unknown symbols
    or unbalanced parentheses."""
    if not formula or not isinstance(formula, str):
        raise ValueError("formula must be a non-empty string, e.g. 'H2O'")
    stack = [{}]
    pos = 0
    for m in _TOKEN.finditer(formula):
        if m.start() != pos:
            raise ValueError(f"cannot parse formula at '{formula[pos:]}'")
        pos = m.end()
        elem, count, open_p, close_p, group_count = m.groups()
        if elem:
            if elem not in ATOMIC_MASS:
                raise ValueError(f"unknown element '{elem}' in '{formula}'")
            n = int(count) if count else 1
            stack[-1][elem] = stack[-1].get(elem, 0) + n
        elif open_p:
            stack.append({})
        elif close_p:
            if len(stack) == 1:
                raise ValueError(f"unbalanced ')' in '{formula}'")
            group = stack.pop()
            mult = int(group_count) if group_count else 1
            for el, n in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + n * mult
    if pos != len(formula) or len(stack) != 1:
        raise ValueError(f"cannot parse formula '{formula}'")
    return stack[0]


def molar_mass(formula):
    """Molar mass as a Q in g/mol: molar_mass('H2O').to(G/MOL) == 18.015."""
    counts = parse_formula(formula)
    return sum(ATOMIC_MASS[el] * n for el, n in counts.items()) * G / MOL


# ----------------------------------------------------------------------
# The intro-chem conversion chains
# ----------------------------------------------------------------------

def grams_to_moles(grams, formula):
    """n = m / M -- the single most-used conversion in intro chem."""
    if grams < 0:
        raise ValueError("mass must be >= 0 g")
    return (grams * G) / molar_mass(formula)      # -> Q in mol


def moles_to_molecules(q_mol):
    """N = n * N_A. Accepts a Q in moles (or plain number of moles)."""
    n = q_mol.to(MOL) if isinstance(q_mol, Q) else float(q_mol)
    if n < 0:
        raise ValueError("moles must be >= 0")
    return n * N_A


def grams_to_molecules(grams, formula):
    """The full chain: g -> (divide by g/mol) -> mol -> (times N_A) -> count."""
    return moles_to_molecules(grams_to_moles(grams, formula))


def molarity(q_moles, q_volume):
    """M = n/V as a Q; read it out with .to(MOL/L)."""
    return q_moles / q_volume


def dilution_v1(M1, M2, V2):
    """M1*V1 = M2*V2 solved for V1: how much stock do you pipette?
    Pass Q quantities; returns a Q volume."""
    if M2.to(MOL / L) > M1.to(MOL / L):
        raise ValueError("dilution cannot INCREASE concentration (M2 > M1)")
    return M2 * V2 / M1


# ----------------------------------------------------------------------
# Temperature: the affine trap (offsets, so NOT factor-label)
# ----------------------------------------------------------------------

def c_to_k(celsius):
    """K = C + 273.15. An offset -- no conversion FACTOR exists, which is
    why temperature is the classic factor-label counterexample."""
    if celsius < -273.15:
        raise ValueError("below absolute zero")
    return (celsius + 273.15) * K


def f_to_c(fahrenheit):
    """C = (F - 32) * 5/9: an offset AND a factor (fully affine)."""
    return (fahrenheit - 32.0) * 5.0 / 9.0


if __name__ == "__main__":
    print("=== factor-label as algebra ===")
    print(f"  60 mph = {(60 * MILE / HR).to(M / S):.4f} m/s")
    print(f"  1 g/mL = {(G / ML).to(KG / M**3):.0f} kg/m^3")
    print(f"  R = {R_GAS.to(J / (MOL * K)):.4f} J/(mol K)"
          f" = {R_GAS.to(L * ATM / (MOL * K)):.6f} L atm/(mol K)  <- same Q")
    print(f"  molar volume at STP (273.15 K, 1 atm):"
          f" {(R_GAS * (273.15 * K) / ATM).to(L / MOL):.3f} L/mol")

    print("\n=== molar mass parser ===")
    for f in ("H2O", "CO2", "NaCl", "C6H12O6", "CaCO3", "Al2(SO4)3"):
        print(f"  {f:10s} {molar_mass(f).to(G / MOL):8.3f} g/mol")

    print("\n=== the Avogadro chain: 9.0 g of water ===")
    n = grams_to_moles(9.0, "H2O")
    print(f"  9.0 g / 18.015 g/mol = {n.to(MOL):.4f} mol"
          f" = {grams_to_molecules(9.0, 'H2O'):.3e} molecules")

    print("\n=== molarity + dilution ===")
    conc = molarity(grams_to_moles(58.44, "NaCl"), 2 * L)
    print(f"  58.44 g NaCl in 2 L -> {conc.to(MOL / L):.3f} M")
    v1 = dilution_v1(conc, 0.1 * MOL / L, 250 * ML)
    print(f"  to make 250 mL of 0.100 M: pipette {v1.to(ML):.1f} mL of stock")

    print("\n=== dimensional analysis catches nonsense ===")
    try:
        _ = 3 * G + 2 * S
    except ValueError as e:
        print(f"  3 g + 2 s -> ValueError: {e}")
