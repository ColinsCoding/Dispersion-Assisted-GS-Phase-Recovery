#!/usr/bin/env python3
"""
engineering_economy.py — ENGR 140 (CSUS) engineering-economy solver.

Plain functions extracted from the Lecture 2-11 slide formulas (Alsharqawi):
time-value-of-money factors, worth measures (PE/FE/AE at MARR), IRR/ERR,
payback, SIR, capitalized worth, and book depreciation (SL/UOP/SOYD/DB/DDB).

Conventions match the course:
  P present, F future, A uniform end-of-period series, G arithmetic gradient,
  i interest per period (decimal), n periods, j geometric growth rate.
  Cash flow lists are indexed by period: cashflows[0] is t=0 (usually the
  negative investment), cashflows[t] the net flow at end of period t.

IRR is found as the roots of the NPW polynomial (numpy). If PyTorch is
importable, irr_torch() offers an autograd Newton solver as well — optional,
because torch cannot always load on low-memory machines.

Run `python engineering_economy.py` to self-verify against the lecture
examples (expected values straight from the slides).
"""

import numpy as np

# ── time value of money factors (Lecture 3) ──────────────────────────────────

def fp(i, n):
    """(F/P,i,n) single payment compound amount: F = P*fp(i,n)."""
    return (1.0 + i) ** n


def pf(i, n):
    """(P/F,i,n) single payment present worth: P = F*pf(i,n)."""
    return (1.0 + i) ** -n


def fa(i, n):
    """(F/A,i,n) uniform series compound amount: F = A*fa(i,n)."""
    return n if i == 0 else ((1.0 + i) ** n - 1.0) / i


def af(i, n):
    """(A/F,i,n) sinking fund: A = F*af(i,n)."""
    return 1.0 / fa(i, n)


def pa(i, n):
    """(P/A,i,n) uniform series present worth: P = A*pa(i,n)."""
    return n if i == 0 else ((1.0 + i) ** n - 1.0) / (i * (1.0 + i) ** n)


def ap(i, n):
    """(A/P,i,n) capital recovery (loan payment): A = P*ap(i,n)."""
    return 1.0 / pa(i, n)


def ag(i, n):
    """(A/G,i,n) arithmetic gradient to uniform series: A = G*ag(i,n)."""
    return 0.5 * (n - 1) if i == 0 else 1.0 / i - n / ((1.0 + i) ** n - 1.0)


def pg(i, n):
    """(P/G,i,n) arithmetic gradient present worth: P = G*pg(i,n)."""
    if i == 0:
        return 0.5 * n * (n - 1)
    return ((1.0 + i) ** n - i * n - 1.0) / (i * i * (1.0 + i) ** n)


def pa_geo(i, j, n):
    """(P/A1,j,i,n) geometric gradient present worth: P = A1*pa_geo(i,j,n).
    A1 is the first payment; payments grow by rate j per period."""
    if abs(i - j) < 1e-12:
        return n / (1.0 + i)
    return (1.0 - ((1.0 + j) / (1.0 + i)) ** n) / (i - j)


# ── worth measures (Lecture 5) ───────────────────────────────────────────────

def pe(cashflows, i):
    """Net Present Worth / Present Equivalent at rate i (MARR).
    Accept if pe > 0."""
    return sum(cf * pf(i, t) for t, cf in enumerate(cashflows))


def fe(cashflows, i):
    """Net Future Worth / Future Equivalent at end of last period."""
    n = len(cashflows) - 1
    return sum(cf * fp(i, n - t) for t, cf in enumerate(cashflows))


def ae(cashflows, i):
    """Net Annual Worth / Annual Equivalent."""
    n = len(cashflows) - 1
    return pe(cashflows, i) * ap(i, n)


def payback(cashflows, i=0.0):
    """Payback period in years; i > 0 gives the discounted payback
    (Lecture 5 cost-of-funds table method). Returns None if never paid back."""
    balance = cashflows[0]
    for t, cf in enumerate(cashflows[1:], start=1):
        prev = balance                  # deficit at end of previous period
        balance = balance * (1.0 + i) + cf
        if balance >= 0:
            # course convention: interpolate with the pre-interest deficit
            return t - 1 + (-prev / cf) if cf else float(t)
    return None


def irr(cashflows):
    """Internal rate of return: the i where pe(cashflows, i) = 0.
    Solves the NPW polynomial in x = 1/(1+i) with numpy roots.
    Returns a single rate for simple investments; a list if the cash flow is
    non-simple and has several sign-change roots (course: use another method).
    """
    coeffs = list(cashflows)            # sum cf_t * x^t = 0,  x = 1/(1+i)
    roots = np.roots(coeffs[::-1])
    rates = []
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 1e-9:
            rate = 1.0 / r.real - 1.0
            if rate > -0.999:
                rates.append(round(rate, 10))
    rates = sorted(set(rates))
    if not rates:
        return None
    return rates[0] if len(rates) == 1 else rates


def err(cashflows, marr):
    """External rate of return (Lecture 5): the rate i* where the future
    worth of outflows compounded at i* equals the future worth of inflows
    compounded at the MARR."""
    n = len(cashflows) - 1
    fe_in = sum(cf * fp(marr, n - t) for t, cf in enumerate(cashflows) if cf > 0)
    # fe_out(i*) = sum |cf| (1+i*)^(n-t) is monotonic in i* -> bisection
    def fe_out(rate):
        return sum(-cf * fp(rate, n - t) for t, cf in enumerate(cashflows) if cf < 0)
    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fe_out(mid) < fe_in:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sir(cashflows, i):
    """Saving/Investment Ratio: PE(inflows)/PE(outflows). Accept if > 1."""
    benefits = sum(cf * pf(i, t) for t, cf in enumerate(cashflows) if cf > 0)
    costs = sum(-cf * pf(i, t) for t, cf in enumerate(cashflows) if cf < 0)
    return benefits / costs if costs else float("inf")


def cw(A, i):
    """Capitalized Worth: present amount that yields A per period forever."""
    return A / i


# ── book depreciation (Lecture 8) ────────────────────────────────────────────
# Each function returns a list of (n, D_n, BV_n) rows, n = 1..N.

def depreciation_sl(P, S, N):
    """Straight line: D_n = (P - S)/N."""
    D = (P - S) / N
    return [(n, D, P - n * D) for n in range(1, N + 1)]


def depreciation_soyd(P, S, N):
    """Sum-of-years'-digits: D_n = (N-n+1)/SOYD * (P - S)."""
    soyd = N * (N + 1) / 2
    rows, bv = [], P
    for n in range(1, N + 1):
        D = (N - n + 1) / soyd * (P - S)
        bv -= D
        rows.append((n, D, bv))
    return rows


def depreciation_uop(P, S, units, total_units=None):
    """Units of production: D_n = (units_n/total)*(P - S)."""
    total = total_units if total_units is not None else sum(units)
    rows, bv = [], P
    for n, u in enumerate(units, start=1):
        D = u / total * (P - S)
        bv -= D
        rows.append((n, D, bv))
    return rows


def depreciation_db(P, S, N, rate=None, switch_to_sl=False):
    """Declining balance at `rate` (default 1/N; use 2/N for DDB).
    Handles both lecture cases: never depreciate below salvage (case 2), and
    optionally switch to SL when the SL charge exceeds the DB charge (case 1).
    """
    d = rate if rate is not None else 1.0 / N
    rows, bv = [], P
    switched = False
    sl_charge = 0.0
    for n in range(1, N + 1):
        D_db = d * bv
        if switch_to_sl and not switched:
            remaining = N - n + 1
            D_sl = (bv - S) / remaining
            if D_sl > D_db:
                switched = True
                sl_charge = D_sl
        D = sl_charge if switched else D_db
        if bv - D < S:                  # case 2: floor at salvage value
            D = bv - S
        bv -= D
        rows.append((n, D, bv))
    return rows


def depreciation_ddb(P, S, N, switch_to_sl=False):
    """Double declining balance: DB at rate 2/N."""
    return depreciation_db(P, S, N, rate=2.0 / N, switch_to_sl=switch_to_sl)


# ── optional torch backend ───────────────────────────────────────────────────

def irr_torch(cashflows, guess=0.1, iters=60):
    """IRR by autograd Newton iteration on the NPW (requires torch)."""
    import torch
    cf = torch.tensor(cashflows, dtype=torch.float64)
    t = torch.arange(len(cashflows), dtype=torch.float64)
    rate = torch.tensor(float(guess), dtype=torch.float64, requires_grad=True)
    for _ in range(iters):
        npw = (cf / (1.0 + rate) ** t).sum()
        (grad,) = torch.autograd.grad(npw, rate)
        with torch.no_grad():
            step = npw / grad
            rate -= step
        rate.requires_grad_(True)
        if abs(step.item()) < 1e-12:
            break
    return rate.item()


# ── self-verification against the lecture examples ───────────────────────────

def verify():
    checks = [
        # Lecture 3
        ("L3 ex5  F/P",        20000 * fp(0.10, 5),                    32210.2),
        ("L3 ex6  P/F",        30000 * pf(0.08, 5),                    20417.5),
        ("L3 ex9  F/A",        3000 * fa(0.07, 10),                    41449.2),
        ("L3 ex11 A/F",        5000 * af(0.07, 5),                       869.5),
        ("L3 ex12 P/A",        24000 * pa(0.10, 21),                  207569.0),
        ("L3 ex13 P/A+P/G",    3000 * pa(0.08, 5) + 1000 * pg(0.08, 5), 19350.5),
        ("L3 ex14 A/G",        3000 + 1000 * ag(0.08, 5),               4846.5),
        ("L3 ex15 geometric",  500 + 550 * pa_geo(0.08, 0.10, 9),       5437.9),
        # Lecture 5
        ("L5 ex5  PE",  pe([-650000] + [162500] * 8, 0.15),            79186.3),
        ("L5 ex6  PE",  pe([-75000, 24400, 27340, 55760], 0.15),        3553.5),
        ("L5 ex7  FE",  fe([-75000, 24400, 27340, 55760], 0.15),        5404.4),
        ("L5 ex9  AE",  ae([-75000, 24400, 27340, 55760], 0.15),        1556.5),
        ("L5 ex3  payback",
         payback([-85000, 15000, 25000, 35000, 45000, 45000, 35000]),      3.22),
        ("L5 ex4  disc payback",
         payback([-85000, 15000, 25000, 35000, 45000, 45000, 35000], 0.15), 4.17),
        ("L5 ex6  IRR %",  irr([-75000, 24400, 27340, 55760]) * 100,      17.45),
        ("L5 ex10A IRR %", irr([-1000, 0, 0, 0, 1500]) * 100,             10.67),
        ("L5 ex13 ERR %",  err([-650000] + [162500] * 8, 0.15) * 100,     16.66),
        ("L5 ex15 SIR",    sir([-650000] + [162500] * 8, 0.15),            1.12),
        ("L5 ex16 CW",     cw(750000 * af(0.05, 10), 0.05),           1192500.0),
        # Lecture 8
        ("L8 ex1  SL D",    depreciation_sl(10000, 2000, 5)[0][1],       1600.0),
        ("L8 ex3  SOYD D1", depreciation_soyd(10000, 2000, 5)[0][1],     2666.7),
        ("L8 ex4  DB D1",   depreciation_db(10000, 2000, 5)[0][1],       2000.0),
        ("L8 ex5  DB floor", depreciation_db(10000, 4500, 5)[3][1],       620.0),
        ("L8 ex4  DB->SL n3",
         depreciation_db(10000, 2000, 5, switch_to_sl=True)[2][1],       1466.7),
        ("L8 ex6  DDB D1",  depreciation_ddb(10000, 2000, 5)[0][1],      4000.0),
    ]
    all_ok = True
    for name, got, want in checks:
        ok = got is not None and abs(got - want) / max(abs(want), 1) < 0.005
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:22s} got {got:>12.2f}  expected {want:>12.2f}")
    print("\nALL LECTURE EXAMPLES PASS" if all_ok else "\nSOME CHECKS FAILED")
    return all_ok


if __name__ == "__main__":
    verify()
