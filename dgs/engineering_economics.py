"""
Engineering economics (ENGR 140): time value of money, NPV, IRR, depreciation, break-even.

In plain words
--------------
A dollar today is worth more than a dollar next year, because today's dollar can earn
interest i. Every engineering-economics problem moves cash flows to one point in time using
these factors (standard textbook notation, read "find P given F, at rate i, over n periods"):

    (P/F, i, n) = (1 + i)^-n                        single payment, future -> present
    (F/P, i, n) = (1 + i)^n
    (P/A, i, n) = ((1 + i)^n - 1) / (i (1 + i)^n)   uniform series A per period -> present
    (A/P, i, n) = i (1 + i)^n / ((1 + i)^n - 1)     capital recovery: present -> annual
    (F/A, i, n) = ((1 + i)^n - 1) / i
    (A/F, i, n) = i / ((1 + i)^n - 1)               sinking fund

Then:
* NPV (net present value) = sum of every cash flow times (P/F, i, t). Accept if NPV > 0 at the
  minimum attractive rate of return (MARR).
* IRR = the rate where NPV = 0.
* EAC (equivalent annual cost) = turn a purchase plus running costs into one yearly number,
  so it can be compared to a pay-per-use option.
* Break-even = the usage level where owning and renting cost the same per year.

Sources (IEEE style)
--------------------
[1] D. G. Newnan, T. G. Eschenbach, J. P. Lavelle, and N. A. Lewis, Engineering Economic
    Analysis, 14th ed. Oxford Univ. Press, 2019, chs. 3-5, 7, 11.
[2] W. G. Sullivan, E. M. Wicks, and C. P. Koelling, Engineering Economy, 17th ed. Pearson, 2018.
"""
import math


def _rate(i):
    if not i > -1:
        raise ValueError(f"interest rate i must be > -1 (as a decimal, e.g. 0.08), got {i}")


def _periods(n):
    if int(n) != n or n < 0:
        raise ValueError(f"number of periods n must be a non-negative integer, got {n}")


def p_given_f(i, n):
    """(P/F, i, n) = (1 + i)^-n."""
    _rate(i); _periods(n)
    return (1 + i) ** -n


def f_given_p(i, n):
    """(F/P, i, n) = (1 + i)^n."""
    _rate(i); _periods(n)
    return (1 + i) ** n


def p_given_a(i, n):
    """(P/A, i, n): present worth of A at the end of each of periods 1..n."""
    _rate(i); _periods(n)
    if i == 0:
        return float(n)
    return ((1 + i) ** n - 1) / (i * (1 + i) ** n)


def a_given_p(i, n):
    """(A/P, i, n), the capital-recovery factor."""
    _periods(n)
    if n == 0:
        raise ValueError("A/P is undefined for n = 0")
    return 1.0 / p_given_a(i, n)


def f_given_a(i, n):
    """(F/A, i, n)."""
    _rate(i); _periods(n)
    if i == 0:
        return float(n)
    return ((1 + i) ** n - 1) / i


def a_given_f(i, n):
    """(A/F, i, n), the sinking-fund factor."""
    _periods(n)
    if n == 0:
        raise ValueError("A/F is undefined for n = 0")
    return 1.0 / f_given_a(i, n)


def npv(cash_flows, i):
    """Net present value of cash_flows[t] at end of period t (t = 0 is now)."""
    _rate(i)
    return sum(cf * (1 + i) ** -t for t, cf in enumerate(cash_flows))


def irr(cash_flows, lo=-0.99, hi=10.0, tol=1e-12):
    """Internal rate of return by bisection; needs exactly one sign change of NPV in [lo, hi]."""
    f_lo, f_hi = npv(cash_flows, lo), npv(cash_flows, hi)
    if f_lo * f_hi > 0:
        raise ValueError("NPV does not change sign on [lo, hi]; no IRR bracketed")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f_mid = npv(cash_flows, mid)
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def equivalent_annual_cost(first_cost, annual_cost, salvage, i, n):
    """EAC = P (A/P) + annual cost - S (A/F): the yearly cost of owning for n years."""
    if first_cost < 0 or annual_cost < 0 or salvage < 0:
        raise ValueError("first_cost, annual_cost and salvage are costs/values and must be >= 0")
    return first_cost * a_given_p(i, n) + annual_cost - salvage * a_given_f(i, n)


def straight_line(first_cost, salvage, n):
    """Straight-line depreciation: list of (year, depreciation, end-of-year book value)."""
    _periods(n)
    if n == 0 or salvage > first_cost:
        raise ValueError("need n >= 1 and salvage <= first_cost")
    d = (first_cost - salvage) / n
    return [(t, d, first_cost - d * t) for t in range(1, n + 1)]


def declining_balance(first_cost, n, factor=2.0, salvage=0.0):
    """Declining-balance depreciation (factor 2 = double declining); never below salvage."""
    _periods(n)
    if n == 0 or factor <= 0:
        raise ValueError("need n >= 1 and factor > 0")
    rate = factor / n
    book, rows = float(first_cost), []
    for t in range(1, n + 1):
        d = min(book * rate, max(book - salvage, 0.0))
        book -= d
        rows.append((t, d, book))
    return rows


def breakeven_usage(own_eac, rent_rate_per_unit, own_cost_per_unit=0.0):
    """Usage per year where owning (EAC + variable cost) equals renting (rate * usage)."""
    if rent_rate_per_unit <= own_cost_per_unit:
        raise ValueError("renting is never more expensive per unit; no break-even")
    return own_eac / (rent_rate_per_unit - own_cost_per_unit)


def payback_period(cash_flows):
    """Simple (undiscounted) payback: first time the running total reaches >= 0, interpolated."""
    total = 0.0
    for t, cf in enumerate(cash_flows):
        prev, total = total, total + cf
        if t > 0 and total >= 0 > prev:
            return t - 1 + (-prev) / cf
    return math.inf
