"""Engineering Economics: the interest-factor toolkit, an exact IRR solver,
and the Profitability Index.

Every factor below converts one cash-flow pattern into another under
compound interest at rate i per period, over n periods. All eight are
DERIVED from the same geometric-series sum (not memorized independently),
and each is cross-checked two ways: symbolically via SymPy, and
numerically by literally simulating the cash flows year-by-year and
summing.

  F/P  (single payment, compound amount):  F = P(1+i)^n
  P/F  (single payment, present worth):    P = F(1+i)^-n
  F/A  (uniform series, compound amount):  F = A[(1+i)^n - 1]/i
  A/F  (sinking fund):                     A = F*i / [(1+i)^n - 1]
  P/A  (uniform series, present worth):    P = A[(1+i)^n - 1] / [i(1+i)^n]
  A/P  (capital recovery):                 A = P*i(1+i)^n / [(1+i)^n - 1]
  P/G  (arithmetic gradient, present):     P = G*[(1+i)^n - i*n - 1] / [i^2 (1+i)^n]
  A/G  (arithmetic gradient, to annuity):  A = G*[1/i - n/((1+i)^n - 1)]

Also implements NPV, a real root-solving IRR (bisection -- the existing
notebooks/ml_econ_physics.ipynb's IRR was a coarse grid-search argmin of
|NPV|, accurate only to its grid spacing; this replaces that with an
exact solve, verified to actually zero NPV, not just minimize |NPV| over
a fixed grid) and the Profitability Index PI = PV(benefits)/PV(costs).
"""
import math


def _check_i_n(i, n):
    if i <= -1:
        raise ValueError("interest rate must be > -1 (per period)")
    if n < 0:
        raise ValueError("n (number of periods) must be non-negative")


# ----------------------------------------------------------------------
# Single payment factors
# ----------------------------------------------------------------------

def compound_amount_factor(P, i, n):
    """F/P: future worth of a single present sum P, compounded at i for n periods."""
    _check_i_n(i, n)
    return P * (1 + i) ** n


def present_worth_factor(F, i, n):
    """P/F: present worth of a single future sum F, discounted at i for n periods."""
    _check_i_n(i, n)
    return F * (1 + i) ** (-n)


# ----------------------------------------------------------------------
# Uniform series factors
# ----------------------------------------------------------------------

def uniform_series_compound_amount(A, i, n):
    """F/A: future worth of n equal end-of-period payments A.
    Derived from the geometric series sum_{k=0}^{n-1} (1+i)^k = [(1+i)^n - 1]/i."""
    _check_i_n(i, n)
    if n == 0:
        return 0.0
    if i == 0:
        return A * n
    return A * ((1 + i) ** n - 1) / i


def sinking_fund_factor(F, i, n):
    """A/F: the equal periodic deposit needed to accumulate F in n periods.
    Exact inverse of uniform_series_compound_amount."""
    _check_i_n(i, n)
    if n == 0:
        raise ValueError("n must be positive")
    if i == 0:
        return F / n
    return F * i / ((1 + i) ** n - 1)


def uniform_series_present_worth(A, i, n):
    """P/A: present worth of n equal end-of-period payments A.
    = (F/A) * (P/F): discount the accumulated future value back to present."""
    _check_i_n(i, n)
    if n == 0:
        return 0.0
    if i == 0:
        return A * n
    return A * ((1 + i) ** n - 1) / (i * (1 + i) ** n)


def capital_recovery_factor(P, i, n):
    """A/P: the equal periodic payment that recovers a present investment P
    plus interest over n periods. Exact inverse of uniform_series_present_worth."""
    _check_i_n(i, n)
    if n == 0:
        raise ValueError("n must be positive")
    if i == 0:
        return P / n
    return P * i * (1 + i) ** n / ((1 + i) ** n - 1)


# ----------------------------------------------------------------------
# Arithmetic gradient factors
# ----------------------------------------------------------------------

def gradient_present_worth(G, i, n):
    """P/G: present worth of an arithmetic gradient series 0, G, 2G, ..., (n-1)G
    (payment in period n. Cash flow at period t is (t-1)*G for t=1..n)."""
    _check_i_n(i, n)
    if n == 0:
        return 0.0
    if i == 0:
        return G * n * (n - 1) / 2.0
    return G * ((1 + i) ** n - i * n - 1) / (i ** 2 * (1 + i) ** n)


def gradient_to_uniform_series(G, i, n):
    """A/G: the equivalent uniform series A that replaces an arithmetic
    gradient series G. = (P/G) * (A/P), i.e. gradient_present_worth converted
    to an annuity via capital_recovery_factor -- verified as such in tests."""
    _check_i_n(i, n)
    if n == 0:
        return 0.0
    if i == 0:
        return G * (n - 1) / 2.0
    return G * (1.0 / i - n / ((1 + i) ** n - 1))


# ----------------------------------------------------------------------
# Cash-flow-series analysis: NPV, IRR (exact), Profitability Index
# ----------------------------------------------------------------------

def npv(cashflows, i):
    """Net present value of a cash-flow list [C0, C1, ..., Cn] (C0 typically
    the negative up-front investment) at periodic interest rate i."""
    if i <= -1:
        raise ValueError("interest rate must be > -1")
    return sum(cf / (1 + i) ** t for t, cf in enumerate(cashflows))


def irr(cashflows, lo=-0.9999, hi=10.0, tol=1e-14, max_iter=200):
    """Internal rate of return: the EXACT root of NPV(i)=0, found by bisection
    (robust -- doesn't need a derivative, and works for any bracket where
    NPV changes sign). Requires npv(lo) and npv(hi) to have opposite signs
    (standard investment-then-returns cash flow pattern); raises rather than
    silently returning a grid point if no sign change is found in [lo, hi].

    `tol` bounds the bracket WIDTH IN RATE SPACE (dimensionless), not the
    dollar-scale NPV residual -- those are different units, and conflating
    them (an earlier version of this function did) stops bisection too
    early on large cash flows, where a tiny rate error still corresponds to
    a not-quite-zero dollar residual. Bisection here always runs until the
    bracket itself is tol-wide, giving a residual NPV that shrinks with the
    cash flows' own derivative scale, not a fixed dollar amount.
    """
    f_lo, f_hi = npv(cashflows, lo), npv(cashflows, hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        raise ValueError(
            f"NPV does not change sign over [{lo}, {hi}] (NPV({lo})={f_lo:.4g}, "
            f"NPV({hi})={f_hi:.4g}) -- no IRR in this bracket, or multiple/no real roots"
        )
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = npv(cashflows, mid)
        if f_mid == 0 or (hi - lo) / 2 < tol:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return (lo + hi) / 2.0


def profitability_index(cashflows, i):
    """PI = PV(benefits) / PV(costs). Cash flows are split by sign: negative
    entries are costs (present-valued as positive magnitudes), positive
    entries are benefits. PI > 1 means the project's discounted benefits
    exceed its discounted costs -- equivalent to NPV > 0 at the same rate,
    verified as such in tests."""
    if i <= -1:
        raise ValueError("interest rate must be > -1")
    pv_costs = sum(-cf / (1 + i) ** t for t, cf in enumerate(cashflows) if cf < 0)
    pv_benefits = sum(cf / (1 + i) ** t for t, cf in enumerate(cashflows) if cf > 0)
    if pv_costs == 0:
        raise ValueError("no negative cash flows (costs) to form a ratio against")
    return pv_benefits / pv_costs


if __name__ == "__main__":
    print("=== ENGINEERING ECONOMICS: INTEREST FACTORS ===\n")

    P, i, n = 10000.0, 0.08, 5
    F = compound_amount_factor(P, i, n)
    print(f"F/P: ${P:,.0f} at {i:.0%} for {n} years -> F = ${F:,.2f}")
    print(f"P/F check: recovering P from F gives back ${present_worth_factor(F, i, n):,.2f}")

    A = 2000.0
    F_series = uniform_series_compound_amount(A, i, n)
    print(f"\nF/A: ${A:,.0f}/yr for {n} years at {i:.0%} -> F = ${F_series:,.2f}")
    print(f"A/F check: sinking fund to reach that F gives back ${sinking_fund_factor(F_series, i, n):,.2f}/yr")

    P_series = uniform_series_present_worth(A, i, n)
    print(f"\nP/A: ${A:,.0f}/yr for {n} years at {i:.0%} -> P = ${P_series:,.2f}")
    print(f"A/P check: capital recovery on that P gives back ${capital_recovery_factor(P_series, i, n):,.2f}/yr")

    G = 500.0
    print(f"\nP/G: gradient ${G:,.0f}/yr, {n} years at {i:.0%} -> P = ${gradient_present_worth(G, i, n):,.2f}")
    print(f"A/G: equivalent annuity = ${gradient_to_uniform_series(G, i, n):,.2f}/yr")

    print("\n=== NPV / IRR / PROFITABILITY INDEX (SBIR-style cash flow) ===")
    cashflows = [-275000, 80000, 90000, 95000, 100000, 85000]
    rate = 0.10
    print(f"NPV at {rate:.0%}: ${npv(cashflows, rate):,.2f}")
    r_irr = irr(cashflows)
    print(f"IRR (exact root of NPV=0): {r_irr:.4%}  (NPV there = {npv(cashflows, r_irr):.6f}, should be ~0)")
    pi = profitability_index(cashflows, rate)
    print(f"Profitability Index at {rate:.0%}: {pi:.4f}  ({'accept' if pi > 1 else 'reject'}: PI {'>' if pi>1 else '<='} 1)")

    print("\n=== DONE ===")
