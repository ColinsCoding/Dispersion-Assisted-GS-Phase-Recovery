"""Feynman Lectures I, Ch. 22 (Algebra): the direct operations (addition,
multiplication, raising to a power -- Eq. 22.1) and their inverses
(subtraction, division, root, logarithm -- Eq. 22.2), each rule VERIFIED
(symbolically where it holds identically, numerically where it only holds
for positive reals) rather than just transcribed from the table.

Plus the historical log-table trick (multiplication -> addition via
log/antilog lookup) that predates calculators, and its direct
computer-engineering descendant: a ROM-based lookup table (LUT), where
"how many address bits to index a table of depth N" is literally Feynman's
inverse-of-power (logarithm) operation applied with base 2.
"""
import math

import numpy as np
import sympy as sp

from dgs.taylor import taylor_coefficients

a, b, c = sp.symbols("a b c")


# ── Eq. 22.1: direct operations, the rules that hold IDENTICALLY ───────────
def verify_direct_operation_rules() -> dict:
    """(a)-(e), (i)-(k): commutativity, associativity, distributivity, and
    the identity elements -- these hold for ALL a,b,c, so check them as
    symbolic identities (difference simplifies to exactly 0), not numeric
    samples."""
    checks = {
        "(a) a+b=b+a": (a + b, b + a),
        "(b) a+(b+c)=(a+b)+c": (a + (b + c), (a + b) + c),
        "(c) ab=ba": (a * b, b * a),
        "(d) a(b+c)=ab+ac": (a * (b + c), a * b + a * c),
        "(e) (ab)c=a(bc)": ((a * b) * c, a * (b * c)),
        "(i) a+0=a": (a + 0, a),
        "(j) a*1=a": (a * 1, a),
        "(k) a^1=a": (a**1, a),
    }
    return {name: bool(sp.simplify(lhs - rhs) == 0) for name, (lhs, rhs) in checks.items()}


def verify_power_law_rules(a_val=2.0, b_val=3.0, c_val=5.0) -> dict:
    """(f)-(h): (ab)^c=a^c*b^c, a^b*a^c=a^(b+c), (a^b)^c=a^(bc) -- UNLIKE
    (a)-(e), these fail in general for negative a,b with non-integer c (that
    is precisely the "continuity and ordering" caveat Feynman flags
    immediately after the table), so they are checked numerically at a
    concrete positive sample, not asserted as blanket symbolic identities."""
    if a_val <= 0 or b_val <= 0:
        raise ValueError("power-law rules (f)-(h) require positive a, b")
    results = {
        "(f) (ab)^c=a^c*b^c": abs((a_val * b_val) ** c_val - a_val**c_val * b_val**c_val) < 1e-9,
        "(g) a^b*a^c=a^(b+c)": abs(a_val**b_val * a_val**c_val - a_val ** (b_val + c_val)) < 1e-9,
        "(h) (a^b)^c=a^(b*c)": abs((a_val**b_val) ** c_val - a_val ** (b_val * c_val)) < 1e-9,
    }
    return results


# ── Eq. 22.2: the four inverse operations ───────────────────────────────────
def solve_for_missing_addend(a_val, c_val):
    """a+b=c -> b=c-a."""
    b_sym = sp.Symbol("b")
    return sp.solve(sp.Eq(a_val + b_sym, c_val), b_sym)[0]


def solve_for_missing_factor(a_val, c_val):
    """ab=c -> b=c/a."""
    if a_val == 0:
        raise ValueError("division requires a != 0")
    b_sym = sp.Symbol("b")
    return sp.solve(sp.Eq(a_val * b_sym, c_val), b_sym)[0]


def solve_for_root(a_exp, c_val):
    """b^a=c -> b=c^(1/a), the a-th root of c ("what number, raised to the
    a-th power, equals c?")."""
    if c_val < 0:
        raise ValueError("solve_for_root assumes a real positive base here")
    b_sym = sp.Symbol("b", positive=True)
    return sp.solve(sp.Eq(b_sym**a_exp, c_val), b_sym)[0]


_A_SYM, _B_SYM, _C_SYM = sp.symbols("a b c", positive=True)
_LOG_FORMULA = sp.solve(sp.Eq(_A_SYM**_B_SYM, _C_SYM), _B_SYM)[0]  # log(c)/log(a)


def solve_for_logarithm(a_base, c_val):
    """a^b=c -> b=log_a(c) ("to what power must a be raised to get c?"),
    the OTHER inverse of raising to a power -- distinct from the root
    because a^b and b^a are not the same operation.

    Solves the equation ONCE with symbolic a,c (giving log(c)/log(a)) and
    substitutes numbers into that formula, rather than re-running sp.solve
    per call: sp.solve(Eq(a_base**b, c_val), b) silently returns an empty
    list whenever a_base is a float very close to 1 (e.g. 1.01) -- a real
    solver limitation, not a math failure, worked around by solving
    generically first."""
    if a_base <= 0 or a_base == 1 or c_val <= 0:
        raise ValueError("logarithm requires a base > 0, a != 1, and c > 0")
    return _LOG_FORMULA.subs({_A_SYM: a_base, _C_SYM: c_val})


def verify_root_and_log_are_distinct_inverses(a_val=2.0, c_val=8.0) -> dict:
    """CHECKED: for a^a_val=c_val i.e. root asks 'b^2=8 -> b=?' while log
    asks '2^b=8 -> b=?' -- two different numbers, confirming Feynman's
    point that root and logarithm are genuinely separate inverse
    operations, not the same rule in disguise."""
    root_result = float(solve_for_root(a_val, c_val))
    log_result = float(solve_for_logarithm(a_val, c_val))
    return {
        "root (b^2=8)": root_result,
        "log (2^b=8)": log_result,
        "distinct": abs(root_result - log_result) > 1e-9,
    }


# ── The log-table trick: multiplication -> addition ─────────────────────────
def build_log_lookup_table(x_min=1.0, x_max=10.0, n_entries=1000):
    """A discretized log10 lookup table, x -> log10(x) -- the historical
    'table of logarithms'/slide-rule method for turning multiplication into
    addition, decades before electronic calculators."""
    if x_min <= 0 or x_max <= x_min:
        raise ValueError("build_log_lookup_table requires 0 < x_min < x_max")
    xs = np.linspace(x_min, x_max, n_entries)
    logs = np.log10(xs)
    return xs, logs


def multiply_via_log_table(x1, x2, xs, logs):
    """Look up log10(x1), log10(x2) by table interpolation, ADD them
    (mantissa arithmetic replacing multiplication), then invert (10**sum) to
    recover x1*x2 -- exactly the pre-calculator log-table method, verified
    below against direct multiplication."""
    log1 = np.interp(x1, xs, logs)
    log2 = np.interp(x2, xs, logs)
    return 10 ** (log1 + log2)


def verify_log_table_multiplication(x1=2.5, x2=3.5, n_entries=100_000) -> dict:
    """CHECKED against direct multiplication x1*x2, to interpolation-limited
    precision (n_entries controls table resolution, hence accuracy)."""
    xs, logs = build_log_lookup_table(1.0, 10.0, n_entries)
    lut_result = multiply_via_log_table(x1, x2, xs, logs)
    direct_result = x1 * x2
    return {
        "lut_result": lut_result,
        "direct_result": direct_result,
        "relative_error": abs(lut_result - direct_result) / direct_result,
    }


# ── Computer-engineering descendant: hardware LUT addressing ────────────────
def lut_address_bits_for_depth(table_depth: int) -> int:
    """How many binary ADDRESS bits are needed to index a lookup table of
    `table_depth` entries: ceil(log2(table_depth)) -- Feynman's inverse-of-
    power (logarithm) operation, applied with base 2, is literally the
    hardware question "how many bits address this ROM?"."""
    if table_depth < 1:
        raise ValueError("table_depth must be >= 1")
    return math.ceil(float(solve_for_logarithm(2, table_depth)) - 1e-12) if table_depth > 1 else 0


def lut_quantization_error_bound(input_bits: int, function_range: float) -> float:
    """Worst-case quantization error for a `input_bits`-address LUT covering
    a function whose output spans `function_range`: half the step size,
    step = function_range / 2^input_bits -- the direct hardware cost of
    using a lookup table (finite address bits) instead of computing a
    transcendental function exactly."""
    if input_bits < 1:
        raise ValueError("input_bits must be >= 1")
    depth = 2**input_bits
    step = function_range / depth
    return step / 2


# ── Taylor-series correction to the Rule of 72 ──────────────────────────────
_R_SYM = sp.Symbol("r", positive=True)


def ln1pr_taylor_coefficients(order=4):
    """The Taylor coefficients of ln(1+r) about r=0, via dgs.taylor's generic
    machinery: 0, 1, -1/2, 1/3, -1/4, ... -- these ARE the terms the Rule of
    72 silently truncates after the first."""
    return taylor_coefficients(sp.log(1 + _R_SYM), _R_SYM, 0, order)


def doubling_time_taylor_correction(order=2):
    """The exact doubling time t=log(2)/log(1+r) diverges as r->0 (it's not
    a plain Taylor series in r, since the denominator vanishes there), so
    this is a LAURENT/asymptotic expansion, built by substituting
    ln1pr_taylor_coefficients's series for log(1+r) into log(2)/log(1+r)
    and re-expanding. Truncating at the leading 1/r term alone recovers the
    Rule-of-72-style approximation log(2)/r; the next term, log(2)/2, is the
    correction that shrinks the rule's error at higher rates."""
    t_exact = sp.log(2) / sp.log(1 + _R_SYM)
    return sp.series(t_exact, _R_SYM, 0, order).removeO()


def doubling_time_correction_check(rate_pct, order=2) -> dict:
    """CHECKED: exact doubling time vs. first-order-only (log(2)/r) vs. the
    Taylor-corrected formula, at a concrete rate -- confirms the correction
    genuinely narrows the gap rather than just adding an unverified term."""
    if rate_pct <= 0:
        raise ValueError("rate_pct must be > 0")
    r_val = rate_pct / 100
    t_exact = float(sp.log(2) / sp.log(1 + r_val))
    t_first_order = float(sp.log(2) / r_val)
    t_corrected = float(doubling_time_taylor_correction(order).subs(_R_SYM, r_val))
    return {
        "exact": t_exact,
        "first_order": t_first_order,
        "corrected": t_corrected,
        "first_order_error": abs(t_first_order - t_exact),
        "corrected_error": abs(t_corrected - t_exact),
    }


if __name__ == "__main__":
    print("=== Eq. 22.1 (a)-(e),(i)-(k): identities, verified symbolically ===")
    for name, ok in verify_direct_operation_rules().items():
        print(f"  {name}: {ok}")

    print("\n=== Eq. 22.1 (f)-(h): power laws, verified numerically (a=2,b=3,c=5) ===")
    for name, ok in verify_power_law_rules().items():
        print(f"  {name}: {ok}")

    print("\n=== Eq. 22.2: the four inverse operations, a=3, c=12 ===")
    print(f"  (a') a+b=12 -> b = {solve_for_missing_addend(3, 12)}")
    print(f"  (b') 3*b=12 -> b = {solve_for_missing_factor(3, 12)}")
    print(f"  (c') b^3=12 -> b = {float(solve_for_root(3, 12)):.6f} (cube root)")
    print(f"  (d') 3^b=12 -> b = {float(solve_for_logarithm(3, 12)):.6f} (log base 3)")

    print("\n=== Root vs. logarithm: genuinely different inverse operations ===")
    result = verify_root_and_log_are_distinct_inverses()
    print(f"  b^2=8 -> b={result['root (b^2=8)']:.6f}   2^b=8 -> b={result['log (2^b=8)']:.6f}"
          f"   distinct: {result['distinct']}")

    print("\n=== Log-table trick: 2.5 x 3.5 via lookup + addition ===")
    lut = verify_log_table_multiplication()
    print(f"  LUT result: {lut['lut_result']:.6f}   direct: {lut['direct_result']:.6f}"
          f"   relative error: {lut['relative_error']:.2e}")

    print("\n=== Computer-engineering LUT addressing ===")
    for depth in (16, 256, 1024, 4096):
        print(f"  table depth {depth:5d} -> {lut_address_bits_for_depth(depth)} address bits")
    err = lut_quantization_error_bound(input_bits=10, function_range=2.0)
    print(f"  10-bit LUT covering a range-2.0 function: worst-case error = {err:.6e}")

    print("\n=== Taylor-series correction to the Rule of 72 ===")
    print(f"  ln(1+r) coefficients: {ln1pr_taylor_coefficients()}")
    print(f"  doubling time, asymptotic series: {doubling_time_taylor_correction()}")
    for rate in (1, 6, 12):
        r = doubling_time_correction_check(rate)
        print(f"  rate={rate:2d}%  exact={r['exact']:.4f}  first-order={r['first_order']:.4f} "
              f"(err {r['first_order_error']:.4f})  corrected={r['corrected']:.4f} "
              f"(err {r['corrected_error']:.4f})")
