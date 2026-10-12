"""Phase II, Thrust A, prototyped: extending this repo's causality-
certificate methodology (Kramers-Kronig catching a phase-retrieval
ambiguity GS's own convergence metric cannot see) to quantum phase
estimation (QPE) -- reusing dgs.quantum_modulo.qpe_simulate directly,
not reimplementing it.

THE AMBIGUITY QPE ACTUALLY HAS: QPE measures an eigenphase modulo 1 --
it cannot tell phase 0.63 from 1.63 from 2.63. A SINGLE QPE run (what
this module calls the "naive" estimate) implicitly assumes the
unwrapped value is just its own mod-1 reading; this module verifies
directly (Monte Carlo, below and in the companion notebook) that this
assumption is correct only 1/K of the time for a value genuinely
uniform over K wrap periods -- a real, quantified ambiguity, not a
hypothetical one.

THE CERTIFICATE: exactly the same "measure the same physical quantity
a second, independent way and check consistency" idea Kramers-Kronig
provided for classical dispersion, borrowed here from a specific,
well-established real technique: multi-wavelength interferometry, where
a distance too long to measure unambiguously with one wavelength is
resolved by measuring it with a SECOND wavelength and combining the two
readings. Implemented as a second QPE run at a COARSER scale (measuring
theta/K mod 1 instead of theta mod 1 -- a "longer synthetic
wavelength"), which roughly locates theta within its full range; the
fine measurement then supplies high precision within whichever
K-sized bucket the coarse measurement points to.

A GENUINE, CHARACTERIZED FAILURE MODE (not swept under the rug): the
coarse measurement has its OWN wrap-around, right at theta=K. Trials
with theta close to K can have their coarse estimate snap to ~0 instead
of ~K, causing the wrong bucket to be selected. Confirmed directly (see
the companion notebook): every observed failure in a 200-trial run
occurred within about 0.03 of the boundary theta=K. This is exactly the
kind of edge effect real dual-wavelength interferometry also has to
manage (a synthetic wavelength has its own ambiguity range), not a bug
introduced by this simulation.
"""
import numpy as np

from dgs.quantum_modulo import qpe_simulate


def two_scale_phase_unwrap(theta_true, K, n_ancilla=5):
    """Simulate the two-QPE-measurement certificate for a value theta_true
    assumed to lie in [0, K). Returns the naive (fine-only, wrap-blind)
    estimate and the certified (two-scale) estimate, plus the raw QPE
    outputs for inspection.

    Reuses dgs.quantum_modulo.qpe_simulate for BOTH measurements --
    the only new logic here is how the two results are COMBINED."""
    if K < 1 or not float(K).is_integer():
        raise ValueError("K must be a positive integer (number of wrap periods)")
    K = int(K)
    if not (0 <= theta_true < K):
        raise ValueError(f"theta_true must be in [0, {K})")

    phase_fine_true = theta_true % 1.0
    phase_coarse_true = (theta_true / K) % 1.0

    qpe_fine = qpe_simulate(phase_fine_true, n_ancilla=n_ancilla)
    qpe_coarse = qpe_simulate(phase_coarse_true, n_ancilla=n_ancilla)

    phi_fine = qpe_fine["phi_est"]
    phi_coarse = qpe_coarse["phi_est"]
    coarse_guess = K * phi_coarse

    candidates = [phi_fine + m for m in range(K)]
    best_m = min(range(K), key=lambda m: abs(candidates[m] - coarse_guess))

    return {
        "theta_true": theta_true, "K": K, "n_ancilla": n_ancilla,
        "naive_estimate": phi_fine,          # what a single QPE run alone would report
        "certified_estimate": candidates[best_m],
        "selected_bucket": best_m,
        "coarse_guess": coarse_guess,
        "phi_fine": phi_fine, "phi_coarse": phi_coarse,
    }


def monte_carlo_unwrap_success_rates(K, n_ancilla=5, n_trials=200, seed=0):
    """Monte Carlo estimate of P(naive estimate correct) vs P(certified
    estimate correct), for theta_true drawn uniformly over [0, K) -- the
    same Monte Carlo discipline as
    notebooks/kk_certificate_monte_carlo.ipynb: many random trials,
    reported with counts (not just a single anecdote)."""
    rng = np.random.default_rng(seed)
    tol = (1.0 / (2 ** n_ancilla)) * 1.5   # allow for QPE's own quantization error

    naive_correct = 0
    certified_correct = 0
    failures = []
    for _ in range(n_trials):
        theta_true = rng.uniform(0, K)
        result = two_scale_phase_unwrap(theta_true, K, n_ancilla)
        naive_ok = abs(result["naive_estimate"] - theta_true) < tol
        certified_ok = abs(result["certified_estimate"] - theta_true) < tol
        naive_correct += naive_ok
        certified_correct += certified_ok
        if not certified_ok:
            failures.append(result)

    return {
        "K": K, "n_ancilla": n_ancilla, "n_trials": n_trials,
        "naive_correct": naive_correct, "certified_correct": certified_correct,
        "naive_success_rate": naive_correct / n_trials,
        "certified_success_rate": certified_correct / n_trials,
        "failures": failures,
    }


def wilson_ci(successes, n, z=1.96):
    """95% Wilson confidence interval -- reused verbatim from
    notebooks/kk_certificate_monte_carlo.ipynb's own helper, so the same
    statistical discipline applies here."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


if __name__ == "__main__":
    print("=== THRUST A PROTOTYPE: CAUSALITY-CERTIFICATE METHODOLOGY, APPLIED TO QPE ===\n")

    theta_true = 2.63
    K, n_ancilla = 5, 5
    result = two_scale_phase_unwrap(theta_true, K, n_ancilla)
    print(f"true theta: {theta_true} (in [0,{K}))")
    print(f"naive estimate (single QPE, wrap-blind):  {result['naive_estimate']:.4f}  "
          f"(off by {abs(result['naive_estimate']-theta_true):.4f})")
    print(f"certified estimate (two-scale):           {result['certified_estimate']:.4f}  "
          f"(off by {abs(result['certified_estimate']-theta_true):.4f})")

    print(f"\n=== MONTE CARLO: {200} TRIALS, theta ~ Uniform[0,{K}) ===\n")
    mc = monte_carlo_unwrap_success_rates(K, n_ancilla, n_trials=200, seed=0)
    ci_naive = wilson_ci(mc["naive_correct"], mc["n_trials"])
    ci_certified = wilson_ci(mc["certified_correct"], mc["n_trials"])
    print(f"naive success rate:     {mc['naive_correct']}/{mc['n_trials']} = "
          f"{mc['naive_success_rate']:.1%}  (95% CI: [{ci_naive[0]:.1%}, {ci_naive[1]:.1%}])")
    print(f"certified success rate: {mc['certified_correct']}/{mc['n_trials']} = "
          f"{mc['certified_success_rate']:.1%}  (95% CI: [{ci_certified[0]:.1%}, {ci_certified[1]:.1%}])")
    print(f"\nnumber of certified-estimate failures: {len(mc['failures'])}")
    for f in mc["failures"]:
        print(f"  theta_true={f['theta_true']:.4f}  certified={f['certified_estimate']:.4f}  "
              f"(distance from boundary K={K}: {K - f['theta_true']:.4f})")

    print("\n=== DONE ===")
