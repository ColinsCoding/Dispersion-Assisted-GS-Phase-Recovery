"""Biostatistics: survival analysis, clinical trials, biological data.

SURVIVAL ANALYSIS CONNECTION TO PHYSICS:
  Survival function S(t) = P(T > t) = exp(-Lambda(t))  (exponential: Lambda=lambda*t)
  Exactly the attention span model: S(t) = exp(-t/tau).
  Kaplan-Meier: nonparametric estimate of S(t) from censored data.
  Hazard rate h(t) = -d/dt[ln S(t)] = f(t)/S(t).

  EXPONENTIAL DISTRIBUTION (memoryless):
    S(t) = exp(-lambda*t),  h(t) = lambda  (constant)
    -> radioactive decay, Poisson photon counting, simple attention span

  WEIBULL DISTRIBUTION:
    S(t) = exp(-(t/eta)^beta)
    h(t) = (beta/eta)*(t/eta)^(beta-1)
    beta < 1: decreasing hazard (infant mortality / early failure)
    beta = 1: constant hazard (exponential, memoryless)
    beta > 1: increasing hazard (aging / wear-out)

  LOG-RANK TEST: compare survival between two groups (treatment vs control).
  COX PROPORTIONAL HAZARDS: h(t|x) = h_0(t) * exp(beta*x).

EXTINCTION EVENT CONNECTION:
  Species extinction follows Weibull-like survival with:
    beta > 1 (risk increases with time as habitat shrinks)
    The 'hazard rate' = extinction risk per unit time
    Mass extinction = sudden increase in h(t) (step function in hazard)
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════════════
# §1  SURVIVAL DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════

def exponential_survival(t: np.ndarray, lam: float) -> Dict:
    """Exponential survival: S(t) = exp(-lambda*t), h(t) = lambda.

    Memoryless: P(T>s+t|T>s) = P(T>t).
    Mean survival time = 1/lambda.
    """
    S = np.exp(-lam * t)
    f = lam * S           # PDF
    H = lam * t           # cumulative hazard
    return {
        "t": t, "S": S, "f": f, "H": H,
        "h": np.full_like(t, lam),   # constant hazard
        "mean": 1/lam,
        "median": np.log(2)/lam,
        "distribution": "exponential",
    }


def weibull_survival(t: np.ndarray, eta: float, beta: float) -> Dict:
    """Weibull survival: S(t) = exp(-(t/eta)^beta).

    eta: scale (characteristic life), beta: shape.
    h(t) = (beta/eta)*(t/eta)^(beta-1)
    """
    t_safe = np.maximum(t, 1e-300)
    S = np.exp(-(t_safe/eta)**beta)
    f = (beta/eta) * (t_safe/eta)**(beta-1) * S
    H = (t_safe/eta)**beta
    h = (beta/eta) * (t_safe/eta)**(beta-1)
    mean = eta * float(sp.gamma(sp.Rational(1,1) + sp.Rational(1,1)/beta).evalf()) if beta > 0 else np.inf
    return {
        "t": t, "S": S, "f": f, "H": H, "h": h,
        "eta": eta, "beta": beta,
        "regime": ("infant_mortality" if beta < 1
                   else "constant_hazard" if beta == 1
                   else "aging_wearout"),
        "distribution": "weibull",
    }


# ════════════════════════════════════════════════════════════════════════════
# §2  KAPLAN-MEIER ESTIMATOR
# ════════════════════════════════════════════════════════════════════════════

def kaplan_meier(event_times: np.ndarray,
                  censored: Optional[np.ndarray] = None) -> Dict:
    """Kaplan-Meier nonparametric survival estimate.

    event_times : observed times (event OR censoring)
    censored    : boolean array; True if observation is censored (no event)

    KM estimator: S_hat(t) = prod_{t_i <= t} (1 - d_i/n_i)
    where d_i = events at t_i, n_i = at-risk count at t_i.

    Handles right-censoring (subject leaves study before event).
    """
    n = len(event_times)
    if censored is None:
        censored = np.zeros(n, dtype=bool)

    # Sort by time
    order = np.argsort(event_times)
    t_sorted = event_times[order]
    c_sorted  = censored[order]

    # Unique event times (not censored)
    event_mask = ~c_sorted
    t_events   = np.unique(t_sorted[event_mask])

    S_vals = [1.0]
    t_vals = [0.0]
    n_at_risk = n

    prev_t = 0.0
    at_risk = n
    for t_i in t_events:
        # Update at-risk count: remove those who left (censored or event) before t_i
        at_risk -= int(np.sum((t_sorted >= prev_t) & (t_sorted < t_i)))
        d_i = int(np.sum(t_sorted[event_mask] == t_i))
        if at_risk > 0:
            S_new = S_vals[-1] * (1 - d_i / at_risk)
        else:
            S_new = S_vals[-1]
        S_vals.append(S_new)
        t_vals.append(float(t_i))
        prev_t = t_i

    t_km = np.array(t_vals)
    S_km = np.array(S_vals)

    # Greenwood's formula for variance: Var[S(t)] = S(t)^2 * sum(d_i / n_i(n_i-d_i))
    # Approximate 95% CI using log-log transform (more stable for S near 0,1)
    S_km_safe = np.maximum(S_km, 1e-12)
    se_loglog = np.sqrt(np.cumsum(
        np.array([0] + [d_i / (at_risk * (at_risk - d_i + 1e-6))
                        for d_i in [int(np.sum(t_sorted[event_mask] == ti))
                                    for ti in t_events]])
    ))
    # 95% CI on log(-log(S))
    ci_width = 1.96 * se_loglog / (np.abs(np.log(S_km_safe)) + 1e-12)
    exp_w  = np.clip(np.exp(ci_width),  0, 1e6)
    exp_nw = np.clip(np.exp(-ci_width), 0, 1e6)
    ci_lower = S_km_safe ** exp_w
    ci_upper = S_km_safe ** exp_nw

    # Median survival: smallest t where S <= 0.5
    below_half = t_km[S_km <= 0.5]
    median_t   = float(below_half[0]) if len(below_half) > 0 else np.inf

    return {
        "t":        t_km,
        "S":        S_km,
        "CI_lower": np.clip(ci_lower, 0, 1),
        "CI_upper": np.clip(ci_upper, 0, 1),
        "median_t": median_t,
        "n_events": int(np.sum(~censored)),
        "n_censored": int(np.sum(censored)),
        "n_total":  n,
    }


# ════════════════════════════════════════════════════════════════════════════
# §3  LOG-RANK TEST
# ════════════════════════════════════════════════════════════════════════════

def log_rank_test(t1: np.ndarray, c1: np.ndarray,
                   t2: np.ndarray, c2: np.ndarray) -> Dict:
    """Log-rank test: compare survival curves of two groups.

    H0: S1(t) = S2(t) for all t  (same survival).
    Test statistic: chi-squared with 1 degree of freedom.

    Used in clinical trials: treatment vs placebo.
    """
    # Combine all event times
    all_times = np.unique(np.concatenate([t1[~c1], t2[~c2]]))

    n1 = len(t1); n2 = len(t2)
    O_minus_E_sq_over_V = 0.0
    O1_total = 0; E1_total = 0.0
    V_total  = 0.0

    for t in all_times:
        # At risk in each group at time t
        r1 = int(np.sum(t1 >= t))
        r2 = int(np.sum(t2 >= t))
        r  = r1 + r2
        if r == 0:
            continue
        # Deaths in each group at time t
        d1 = int(np.sum((t1 == t) & ~c1))
        d2 = int(np.sum((t2 == t) & ~c2))
        d  = d1 + d2
        # Expected deaths
        e1 = d * r1 / r
        # Variance (hypergeometric)
        v  = d * r1 * r2 * (r - d) / (r**2 * (r - 1)) if r > 1 else 0.0
        O1_total += d1
        E1_total += e1
        V_total  += v

    if V_total < 1e-12:
        return {"chi2": 0.0, "p_value": 1.0, "reject_H0": False,
                "O1": O1_total, "E1": E1_total}

    chi2 = (O1_total - E1_total)**2 / V_total

    # p-value from chi-squared(1) distribution
    # P(chi2 > x) = erfc(sqrt(x/2)) for 1 df
    from math import erfc
    p_val = float(erfc(np.sqrt(chi2 / 2)))

    return {
        "chi2":       chi2,
        "p_value":    p_val,
        "reject_H0":  p_val < 0.05,
        "O1":         O1_total,
        "E1":         E1_total,
        "V":          V_total,
        "df":         1,
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  EXTINCTION EVENT MODEL (BIOLOGICAL APPLICATION)
# ════════════════════════════════════════════════════════════════════════════

def species_extinction_model(n_species: int = 100,
                               baseline_hazard: float = 0.01,
                               extinction_event_t: float = 50.0,
                               extinction_multiplier: float = 50.0,
                               t_max: float = 100.0,
                               seed: int = 42) -> Dict:
    """Simulate species extinction with a mass extinction event.

    Each species has baseline Weibull hazard. At t=extinction_event_t,
    hazard multiplies by extinction_multiplier (habitat loss, asteroid, etc.).

    Baseline: h_0(t) = lambda_0 (exponential, constant background extinction)
    Mass extinction: h(t) = h_0 * M  for t > t_event

    This is a piecewise-constant hazard (step function) model.
    Used to fit: K-Pg boundary (65 Mya), Permian-Triassic (252 Mya).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, t_max, 500)

    # Piecewise survival: S(t) = exp(-integral_0^t h(s)ds)
    lam0 = baseline_hazard
    lam1 = lam0 * extinction_multiplier
    t_ev = extinction_event_t

    # Survival function
    H = np.where(t <= t_ev,
                  lam0 * t,
                  lam0 * t_ev + lam1 * (t - t_ev))
    S = np.exp(-H)

    # Simulate individual species extinction times
    # Sample from piecewise exponential using inverse CDF
    u = rng.uniform(0, 1, n_species)
    # S(t*) = u -> H(t*) = -ln(u)
    log_u = -np.log(u + 1e-15)
    # Invert piecewise H
    H_at_event = lam0 * t_ev
    ext_times = np.where(
        log_u <= H_at_event,
        log_u / lam0,
        t_ev + (log_u - H_at_event) / lam1
    )
    ext_times = np.minimum(ext_times, t_max)
    censored  = ext_times >= t_max

    # KM estimate from simulated data
    km = kaplan_meier(ext_times, censored)

    # Survival at event time and end
    S_at_event = float(np.exp(-lam0 * t_ev))
    S_at_end   = float(np.exp(-lam0*t_ev - lam1*(t_max-t_ev)))

    return {
        "t":                    t,
        "S_true":               S,
        "S_at_event":           S_at_event,
        "S_at_end":             S_at_end,
        "pct_extinct_baseline": (1 - S_at_end) * 100 / extinction_multiplier,
        "pct_extinct_total":    (1 - S_at_end) * 100,
        "extinction_event_t":   extinction_event_t,
        "extinction_multiplier": extinction_multiplier,
        "km":                   km,
        "ext_times":            ext_times,
        "n_survived":           int(np.sum(~censored == False)),
        "n_extinct":            int(np.sum(~censored)),
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  SYMPY: 5 BIOSTATISTICS EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def biostatistics_sympy_5() -> Dict:
    """5 key equations: survival, hazard, Weibull, Cox, KM product."""
    t, lam, eta, beta_w = sp.symbols("t lambda eta beta", positive=True)
    x, beta_cox = sp.symbols("x beta_cox", real=True)

    # 1. Exponential survival
    eq1 = sp.Eq(sp.Symbol("S(t)"),
                sp.exp(-lam * t))

    # 2. Hazard rate from survival
    S_sym = sp.Function("S")(t)
    eq2 = sp.Eq(sp.Symbol("h(t)"),
                -sp.diff(sp.log(S_sym), t))

    # 3. Weibull survival
    eq3 = sp.Eq(sp.Symbol("S_Weibull(t)"),
                sp.exp(-(t/eta)**beta_w))

    # 4. Cox proportional hazards
    h0 = sp.Function("h_0")(t)
    eq4 = sp.Eq(sp.Symbol("h(t|x)"),
                h0 * sp.exp(beta_cox * x))

    # 5. Kaplan-Meier product
    d_i, n_i = sp.symbols("d_i n_i", positive=True)
    eq5 = sp.Eq(sp.Symbol("S_KM(t)"),
                sp.Symbol("prod_{t_i<=t}") * (1 - d_i/n_i))

    return {
        "Exponential_survival": eq1,
        "Hazard_from_survival": eq2,
        "Weibull_survival":     eq3,
        "Cox_proportional":     eq4,
        "Kaplan_Meier":         eq5,
    }


if __name__ == "__main__":
    print("=== Exponential Survival (lambda=0.1) ===")
    t = np.linspace(0, 30, 300)
    es = exponential_survival(t, lam=0.1)
    print(f"  Mean survival: {es['mean']:.1f}  Median: {es['median']:.1f}")
    print(f"  S(10) = {np.interp(10, t, es['S']):.4f}  (theory: {np.exp(-1):.4f})")

    print("\n=== Weibull (beta=2 = aging) ===")
    wb = weibull_survival(t, eta=10.0, beta=2.0)
    print(f"  Regime: {wb['regime']}")
    print(f"  S(10) = {np.interp(10, t, wb['S']):.4f}  (theory: {np.exp(-1):.4f})")

    print("\n=== Kaplan-Meier Estimate ===")
    rng = np.random.default_rng(0)
    event_times = rng.exponential(10, 30)
    censored    = rng.random(30) < 0.3
    km = kaplan_meier(event_times, censored)
    print(f"  n={km['n_total']}  events={km['n_events']}  censored={km['n_censored']}")
    print(f"  Median survival: {km['median_t']:.2f}")

    print("\n=== Log-Rank Test: 2 groups ===")
    t1 = rng.exponential(10, 20);  c1 = np.zeros(20, bool)
    t2 = rng.exponential(5,  20);  c2 = np.zeros(20, bool)
    lr = log_rank_test(t1, c1, t2, c2)
    print(f"  chi2={lr['chi2']:.3f}  p={lr['p_value']:.4f}  reject={lr['reject_H0']}")

    print("\n=== Extinction Event Model ===")
    ext = species_extinction_model(n_species=200, extinction_multiplier=50)
    print(f"  S before event: {ext['S_at_event']:.3f}")
    print(f"  S at end:       {ext['S_at_end']:.4f}")
    print(f"  Total extinct:  {ext['pct_extinct_total']:.1f}%")
    print(f"  Survived (simulated): {ext['n_survived']}")
