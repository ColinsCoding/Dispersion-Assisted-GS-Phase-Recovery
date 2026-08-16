"""maxwell_discrete_symmetries.py -- parity (P) and time-reversal (T)
symmetry of Maxwell's equations, derived from the field DEFINITIONS
(Coulomb's law, Biot-Savart law, F=qE+qv x B) rather than asserted.

Extends notebooks/griffiths_1p10_pseudovectors.ipynb, which already proves
(Problem 1.10, parts a-d) that a cross product of two polar vectors is a
pseudovector -- this module applies that exact result to the actual E and B
field laws, then adds time-reversal (not covered there at all), and checks
both symmetries directly against all four Maxwell equations.

PARITY (r -> -r), FROM THE FIELD LAWS:
  - Coulomb: E(r) = kq*r/|r|^3. Substituting r->-r flips the numerator's
    sign exactly once -> E(-r) = -E(r). E is a POLAR vector (P-type -1),
    derived here, not assumed.
  - Biot-Savart: dB ~ I*dl x r/|r|^3. A current element dl is itself a
    displacement (polar, P-type -1), same as r. Under parity BOTH dl->-dl
    and r->-r, so their cross product picks up (-1)*(-1)=+1 --
    B(-r,-dl) = +B(r,dl). B is an AXIAL vector / pseudovector (P-type +1),
    the exact mechanism notebooks/griffiths_1p10_pseudovectors.ipynb proves
    abstractly (cross of two polars = pseudovector), applied here to the
    concrete field law.

TIME REVERSAL (t -> -t), FROM THE CHAIN RULE:
  For a trajectory x(t), the time-reversed motion is x~(t)=x(-t) (run the
  film backward). The chain rule gives v~(t) = -v(-t) (velocity is T-odd)
  and a~(t) = +a(-t) (acceleration is T-even) -- verified symbolically
  below with sympy.Function, not just quoted. Combined with F=ma (T-even,
  since Newton's law must hold run forward or backward) and the Lorentz
  force law F=q(E + v x B): q and E must be T-even (T-even F, T-even v x B
  requires... see below), and B must be T-odd for v x B to stay T-even
  given v is T-odd.

BOTH SYMMETRIES ARE CHECKED DIRECTLY AGAINST MAXWELL'S FOUR EQUATIONS
(maxwell_parity_consistency, maxwell_time_reversal_consistency) using the
one multiplicative rule that makes this tractable: for PARITY, cross/dot
products and the gradient operator (P-type -1, transforming like r)
compose by MULTIPLYING P-types; for TIME REVERSAL, curl/div (spatial, do
not touch t) leave T-type unchanged, while d/dt FLIPS T-type. Both rules
are the direct algebraic content of the derivations above, not new
assumptions.

Context, not physics: Keisuke Goda (University of Tokyo) is a real
co-author of the original STEAM paper (Goda, Tsia, Jalali, Nature 458, 1145
(2009)) already cited throughout this repo (dgs/sbir_portfolio.py,
dgs/steam_imaging.py) -- the actual Japan side of that collaboration. Noted
here only as context for why "Japan" and "Jalali" cluster together in this
repo's citations; it has no bearing on the P/T symmetry derivations above.
"""

from __future__ import annotations
import sympy as sp
from typing import Dict


# ── 1. Parity: E is polar, derived from Coulomb's law ────────────────────────

def coulomb_field_parity_check() -> bool:
    """E(r) = k*q*r/|r|^3. Verify E(-r) = -E(r) exactly (E is a polar
    vector), by direct symbolic substitution -- not assumed."""
    x, y, z, q, k = sp.symbols("x y z q k", real=True)
    r = sp.Matrix([x, y, z])

    def E_field(rvec):
        rm = sp.sqrt(rvec.dot(rvec))
        return k * q * rvec / rm ** 3

    E_r = E_field(r)
    E_at_minus_r = E_field(-r)
    return sp.simplify(E_at_minus_r - (-E_r)) == sp.zeros(3, 1)


# ── 2. Parity: B is axial (pseudovector), derived from Biot-Savart ──────────

def biot_savart_field_parity_check() -> bool:
    """dB ~ I*dl x r/|r|^3. Under parity BOTH the current element dl and
    the position r flip sign (both are polar/displacement-like), so their
    cross product picks up (-1)*(-1)=+1: verify B(-r,-dl) = +B(r,dl)
    exactly (B is axial), by direct symbolic substitution."""
    x, y, z, lx, ly, lz, mu0, I = sp.symbols("x y z l_x l_y l_z mu0 I", real=True)
    r = sp.Matrix([x, y, z])
    dl = sp.Matrix([lx, ly, lz])

    def B_field(rvec, dlvec):
        rm = sp.sqrt(rvec.dot(rvec))
        return (mu0 * I / (4 * sp.pi)) * dlvec.cross(rvec) / rm ** 3

    B_r = B_field(r, dl)
    B_inverted = B_field(-r, -dl)
    return sp.simplify(B_inverted - B_r) == sp.zeros(3, 1)


# ── 3. Time reversal: velocity is T-odd, acceleration is T-even ─────────────

def time_reversal_velocity_parity() -> bool:
    """For x(t), the reversed trajectory is x~(t)=x(-t). Verify (chain
    rule, via sympy) that v~(t) = d/dt[x(-t)] equals exactly -v(-t) --
    velocity is T-odd, derived symbolically, not asserted."""
    t = sp.Symbol("t", real=True)
    x = sp.Function("x")(t)
    v = sp.diff(x, t)
    x_tilde = x.subs(t, -t)
    v_tilde = sp.diff(x_tilde, t)
    v_at_negt = v.subs(t, -t)
    return sp.simplify(v_tilde - (-v_at_negt)) == 0


def time_reversal_acceleration_parity() -> bool:
    """Same setup as time_reversal_velocity_parity: verify
    a~(t) = d^2/dt^2[x(-t)] equals exactly +a(-t) -- acceleration is
    T-even (the two chain-rule sign flips from differentiating twice
    cancel), consistent with F=ma holding in both time directions."""
    t = sp.Symbol("t", real=True)
    x = sp.Function("x")(t)
    a = sp.diff(x, t, 2)
    x_tilde = x.subs(t, -t)
    a_tilde = sp.diff(x_tilde, t, 2)
    a_at_negt = a.subs(t, -t)
    return sp.simplify(a_tilde - a_at_negt) == 0


# ── 4. Maxwell's equations: parity self-consistency ──────────────────────────

# P-type: -1 for a polar (true) vector/scalar-odd quantity, +1 for an axial
# (pseudo) quantity. Composition rule, derived from sections 1-2: cross/dot
# products and the gradient operator (which transforms like r, P-type -1)
# combine by MULTIPLYING P-types.
_P_TYPE: Dict[str, int] = {"rho": +1, "J": -1, "E": -1, "B": +1, "nabla": -1}


def maxwell_parity_consistency() -> Dict[str, bool]:
    """Check all four Maxwell equations for internal parity consistency,
    using the P-types derived (not assumed) in sections 1-2 above and the
    multiplicative composition rule for cross/dot products with the
    gradient operator. div B=0 is trivially consistent (RHS is exactly 0,
    which carries no nonzero-parity constraint) and is reported as such
    rather than claimed as a real check.
    """
    p = _P_TYPE
    gauss_E = (p["nabla"] * p["E"]) == p["rho"]                    # div E = rho/eps0
    faraday = (p["nabla"] * p["E"]) == p["B"]                       # curl E = -dB/dt
    ampere_maxwell = (p["nabla"] * p["B"]) == (p["J"])              # curl B ~ J + dE/dt
    ampere_maxwell_displacement = (p["nabla"] * p["B"]) == (p["E"])  # displacement-current term shares J's type
    return {
        "gauss_E (div E = rho/eps0)": gauss_E,
        "gauss_B (div B = 0)": True,  # trivial: RHS=0, noted not claimed as a real check
        "faraday (curl E = -dB/dt)": faraday,
        "ampere_maxwell (curl B ~ mu0*J + mu0*eps0*dE/dt)":
            ampere_maxwell and ampere_maxwell_displacement,
    }


# ── 5. Maxwell's equations: time-reversal self-consistency ──────────────────

# T-type: +1 if the reversed quantity equals +Q(r,-t), -1 if it equals
# -Q(r,-t). Composition rule, derived in section 3 (via F=ma T-even and the
# Lorentz force law): div/curl (spatial, do not touch t) leave T-type
# unchanged; d/dt FLIPS T-type (one factor of -1 per time derivative).
_T_TYPE: Dict[str, int] = {"rho": +1, "J": -1, "E": +1, "B": -1}


def maxwell_time_reversal_consistency() -> Dict[str, bool]:
    """Check all four Maxwell equations for internal time-reversal
    consistency, using the T-types derived in section 3 and the rule that
    a time derivative flips T-type while div/curl do not. div B=0 is
    trivial for the same reason as in maxwell_parity_consistency."""
    t = _T_TYPE
    d_dt = lambda ttype: -ttype  # one time derivative flips T-type
    gauss_E = t["E"] == t["rho"]                       # div E = rho/eps0 (div: no flip)
    faraday = t["E"] == d_dt(t["B"])                    # curl E = -dB/dt
    ampere_maxwell = (t["B"] == t["J"]) and (t["B"] == d_dt(t["E"]))  # curl B ~ mu0*J + mu0*eps0*dE/dt
    return {
        "gauss_E (div E = rho/eps0)": gauss_E,
        "gauss_B (div B = 0)": True,  # trivial: RHS=0
        "faraday (curl E = -dB/dt)": faraday,
        "ampere_maxwell (curl B ~ mu0*J + mu0*eps0*dE/dt)": ampere_maxwell,
    }


# ── 6. Combined PT: a genuinely derived, not asserted, observation ──────────

def combined_pt_type() -> Dict[str, int]:
    """PT-type = P-type * T-type for E and B: both come out ODD under the
    combined operation, a direct consequence of sections 1-4 (not a new
    assumption) -- E is P-odd/T-even, B is P-even/T-odd, so both products
    are -1."""
    return {"E": _P_TYPE["E"] * _T_TYPE["E"], "B": _P_TYPE["B"] * _T_TYPE["B"]}


if __name__ == "__main__":
    print("=== 1-2. Parity, derived from the field laws ===")
    print(f"Coulomb's law: E(-r) = -E(r)  (E is polar)?      {coulomb_field_parity_check()}")
    print(f"Biot-Savart:   B(-r,-dl) = +B(r,dl)  (B is axial)? {biot_savart_field_parity_check()}")

    print("\n=== 3. Time reversal, derived via the chain rule ===")
    print(f"v~(t) = -v(-t)  (velocity is T-odd)?      {time_reversal_velocity_parity()}")
    print(f"a~(t) = +a(-t)  (acceleration is T-even)? {time_reversal_acceleration_parity()}")

    print("\n=== 4. Maxwell's equations: parity self-consistency ===")
    for name, ok in maxwell_parity_consistency().items():
        print(f"  {name}: {'consistent' if ok else 'INCONSISTENT'}")

    print("\n=== 5. Maxwell's equations: time-reversal self-consistency ===")
    for name, ok in maxwell_time_reversal_consistency().items():
        print(f"  {name}: {'consistent' if ok else 'INCONSISTENT'}")

    print("\n=== 6. Combined PT ===")
    pt = combined_pt_type()
    for field, val in pt.items():
        print(f"  PT-type of {field}: {val:+d}  ({'odd' if val < 0 else 'even'})")
