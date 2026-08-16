"""A decision function for exponentially decaying signals: A*e^(-k*t),
its indefinite integral (where the +C actually matters, and where it
provably doesn't), and a classifier that branches samples by time
(left/right of a reference) and index parity (odd/even), with an "other"
category for anything that doesn't fit the decay model at all.

THE MODEL. A*e^(-k*t) is the same first-order-decay functional form as
half this repo already uses (RC discharge, radioactive decay, ATP
turnover in dgs.atp_molecular_energetics, the falling-mass/RL-
circuit/transistor isomorphism in dgs.virtual_work_transistor_kinematics)
-- here it's the SIGNAL a classifier has to make decisions about.

THE +C QUESTION. The indefinite integral is
    integral A*e^(-k*t) dt = -(A/k)*e^(-k*t) + C,
and C is genuinely arbitrary for the INDEFINITE integral -- but for any
DEFINITE integral (a real physical quantity, e.g. total energy delivered
between two times), C cancels exactly: F(t1)+C - (F(t0)+C) = F(t1)-F(t0).
cumulative_energy() computes that definite integral and
verify_C_cancels() checks NUMERICALLY, with two different arbitrary
choices of C, that the answer doesn't change -- the concrete resolution
of "which C do I use" for anything you'd actually measure.

THE CLASSIFIER. classify_sample() sorts a (sample index, time, observed
value) triple into one of five buckets:
  even_left, even_right, odd_left, odd_right  -- index parity x time side
  of a reference t0, for samples that DO fit the decay model, or
  other                                        -- for samples whose
  observed value deviates too far from A*e^(-k*t) (a residual/outlier
  check, not just a parity/side label) -- the model has to actually
  explain the data before it gets classified as fitting it.

py-3.13, NumPy + SymPy (SymPy only for the symbolic antiderivative check).
"""

from __future__ import annotations
import numpy as np
import sympy as sp


def exponential_decay(A: float, k: float, t):
    """A*e^(-k*t) -- the model signal."""
    if A <= 0 or k <= 0:
        raise ValueError("A and k must be positive")
    return A * np.exp(-k * np.asarray(t, dtype=float))


def decay_antiderivative(A: float, k: float, t, C: float = 0.0):
    """F(t) = -(A/k)*e^(-k*t) + C -- ANY antiderivative of A*e^(-k*t); C is
    a free choice for the indefinite integral."""
    if A <= 0 or k <= 0:
        raise ValueError("A and k must be positive")
    return -(A / k) * np.exp(-k * np.asarray(t, dtype=float)) + C


def verify_antiderivative_symbolic(A_val: float = 2.0, k_val: float = 0.5) -> dict:
    """SymPy check: d/dt[-(A/k)*e^(-k*t)+C] must equal A*e^(-k*t) exactly
    (C drops out of the derivative, as it must) -- CHECKED symbolically,
    not just trusted from the algebra in the docstring."""
    t_s, A_s, k_s, C_s = sp.symbols("t A k C", positive=False)
    F = -(A_s / k_s) * sp.exp(-k_s * t_s) + C_s
    dF_dt = sp.diff(F, t_s)
    target = A_s * sp.exp(-k_s * t_s)
    matches = sp.simplify(dF_dt - target) == 0
    if not matches:
        raise AssertionError(f"d/dt[F] = {dF_dt} does not simplify to {target}")
    return {"F": F, "dF_dt": dF_dt, "target": target, "matches": matches}


def cumulative_energy(A: float, k: float, t0: float, t1: float, C: float = 0.0) -> float:
    """Definite integral of A*e^(-k*t) from t0 to t1, via F(t1)-F(t0).
    Included C is passed through decay_antiderivative but MUST cancel --
    verified explicitly by verify_C_cancels(), not just assumed here."""
    return decay_antiderivative(A, k, t1, C) - decay_antiderivative(A, k, t0, C)


def verify_C_cancels(A: float, k: float, t0: float, t1: float,
                      C_values=(0.0, 5.0, -3.7, 1000.0), tol: float = 1e-9) -> dict:
    """CHECKED: cumulative_energy gives the IDENTICAL answer for several
    different (arbitrary) choices of C -- the concrete demonstration that
    the integration constant is unobservable in any definite integral."""
    results = [cumulative_energy(A, k, t0, t1, C) for C in C_values]
    spread = max(results) - min(results)
    if spread > tol:
        raise AssertionError(f"cumulative_energy varied with C: {dict(zip(C_values, results))}")
    return {"C_values": C_values, "results": results, "spread": spread, "energy": results[0]}


def fit_decay_params(t, y) -> dict:
    """Recover (A, k) from noisy samples by linearizing: ln(y) = ln(A) -
    k*t is linear in t, so a degree-1 least-squares fit (np.polyfit) gives
    slope=-k, intercept=ln(A) directly -- no nonlinear optimizer needed."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("y must be strictly positive to take its logarithm")
    if len(t) < 2:
        raise ValueError("need at least 2 points to fit a line")
    slope, intercept = np.polyfit(t, np.log(y), 1)
    return {"A_fit": float(np.exp(intercept)), "k_fit": float(-slope)}


def classify_sample(index: int, t: float, t0: float, A: float, k: float,
                     y_observed: float | None = None, residual_tol: float = 0.1) -> str:
    """Bucket ONE (index, time) sample into even_left/even_right/
    odd_left/odd_right (all relative to t0, "left"=t<t0), or "other" if
    y_observed is given and deviates from A*e^(-k*t) by more than
    residual_tol (relative). With no y_observed, the sample is always
    assumed to fit the model (classification is purely parity x side)."""
    parity = "odd" if index % 2 != 0 else "even"
    side = "left" if t < t0 else "right"
    if y_observed is not None:
        predicted = exponential_decay(A, k, t)
        rel_residual = abs(y_observed - predicted) / max(predicted, 1e-30)
        if rel_residual > residual_tol:
            return "other"
    return f"{parity}_{side}"


def classify_batch(indices, t_arr, t0: float, A: float, k: float,
                    y_observed=None, residual_tol: float = 0.1) -> dict:
    """Vectorized classify_sample over many (index, t[, y_observed])
    triples. Returns the per-sample labels and a count per category."""
    indices = np.asarray(indices, dtype=int)
    t_arr = np.asarray(t_arr, dtype=float)
    if len(indices) != len(t_arr):
        raise ValueError("indices and t_arr must have the same length")
    if y_observed is not None:
        y_observed = np.asarray(y_observed, dtype=float)
        if len(y_observed) != len(t_arr):
            raise ValueError("y_observed must match t_arr in length")

    labels = []
    for n, (idx, t) in enumerate(zip(indices, t_arr)):
        y = None if y_observed is None else float(y_observed[n])
        labels.append(classify_sample(int(idx), float(t), t0, A, k, y, residual_tol))
    labels = np.array(labels)
    categories = ("even_left", "even_right", "odd_left", "odd_right", "other")
    counts = {c: int(np.sum(labels == c)) for c in categories}
    return {"labels": labels, "counts": counts}


if __name__ == "__main__":
    A, k = 5.0, 0.3

    print("=== the model ===")
    t_demo = np.array([0.0, 1.0, 2.0, 5.0])
    print(f"  A*e^(-k*t) at t={t_demo}: {np.round(exponential_decay(A, k, t_demo), 4)}")

    print("\n=== the antiderivative, and why +C doesn't matter for a definite integral ===")
    check = verify_antiderivative_symbolic()
    print(f"  d/dt[-(A/k)e^(-kt)+C] == A*e^(-kt), symbolically verified: {check['matches']}")
    energy_check = verify_C_cancels(A, k, t0=0.0, t1=3.0)
    print(f"  cumulative_energy(0,3) for C in {energy_check['C_values']}: {np.round(energy_check['results'], 6)}")
    print(f"  spread across all choices of C: {energy_check['spread']:.2e}  (should be ~0)")

    print("\n=== recovering (A, k) from noisy data ===")
    rng = np.random.default_rng(0)
    t_fit = np.linspace(0, 10, 40)
    y_fit = exponential_decay(A, k, t_fit) * (1 + 0.03 * rng.standard_normal(40))
    fit = fit_decay_params(t_fit, y_fit)
    print(f"  true A={A}, k={k}  ->  fit A={fit['A_fit']:.3f}, k={fit['k_fit']:.3f}")

    print("\n=== classifying samples: parity x time-side x model-fit ===")
    t0 = 5.0
    indices = np.arange(20)
    t_samples = np.linspace(0, 10, 20)
    y_clean = exponential_decay(A, k, t_samples)
    y_clean[3] *= 3.0   # inject one outlier -> should land in "other"
    result = classify_batch(indices, t_samples, t0, A, k, y_observed=y_clean)
    print(f"  counts: {result['counts']}")
    print(f"  sample 3 (the injected outlier) classified as: {result['labels'][3]}")
