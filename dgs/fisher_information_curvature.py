"""fisher_information_curvature.py -- Fisher information IS the curvature of
log-probability, and that one fact threads together three things already in
this repo that otherwise read as separate topics: Dirac-delta concentration
(dgs/bayes_dirac_symmetry.py), Bayesian posterior updates
(dgs/bayes_inference.py), and the quantum Cramer-Rao bound already used for
GS phase estimation (dgs/uncertainty_qm.py's statistical_resolution).

    I(theta) = -E[d^2/dtheta^2 ln L(theta)]      (curvature of log-likelihood)
    Var(theta_hat) >= 1/I(theta)                  (Cramer-Rao bound)
    Dirac delta = lim_{sigma->0} Gaussian(sigma)  <=>  I(theta) -> infinity

**The connecting story.** Bayes' theorem updates a probability distribution
as data arrives; each update sharpens (increases the curvature of) the
posterior -- literally "concentration" in the everyday sense. Run it enough
times and the posterior concentrates toward a point, approaching
dgs/bayes_dirac_symmetry.py's Dirac-delta limit, with Fisher information
diverging. This is the same "narrower peak = more certain" intuition
dgs/uncertainty_qm.py already exploits for phase estimation
("convergence rate ~ Fisher information of the diversity measurement") --
this module makes that curvature connection explicit and numeric instead of
leaving it as a one-line docstring comment, reusing the existing Bayes/Dirac/
quantum machinery rather than re-deriving any of it.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

from dgs.bayes_dirac_symmetry import dirac_delta_as_gaussian_limit
from dgs.bayes_inference import gaussian_mean_posterior


# ── 1. Curvature of log-likelihood = Fisher information ────────────────────

def log_likelihood_curvature(log_L: np.ndarray, theta_grid: np.ndarray) -> np.ndarray:
    """d^2(ln L)/dtheta^2 at every grid point, via central finite differences
    (np.gradient applied twice) -- the literal curvature of the log-likelihood
    surface, before any statistical interpretation is attached to it."""
    log_L = np.asarray(log_L, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)
    if log_L.shape != theta_grid.shape:
        raise ValueError("log_L and theta_grid must have the same shape")
    if len(theta_grid) < 5:
        raise ValueError("need at least 5 grid points for a stable second derivative")
    d1 = np.gradient(log_L, theta_grid)
    d2 = np.gradient(d1, theta_grid)
    return d2


def fisher_information_numeric(log_L: np.ndarray, theta_grid: np.ndarray, theta_hat: float) -> float:
    """Fisher information I(theta_hat) = -curvature of ln L at theta_hat,
    read off log_likelihood_curvature by nearest grid point."""
    curvature = log_likelihood_curvature(log_L, theta_grid)
    idx = int(np.argmin(np.abs(theta_grid - theta_hat)))
    return float(-curvature[idx])


def gaussian_fisher_information(sigma: float) -> float:
    """Analytic reference: for a Gaussian likelihood of known sigma, the
    Fisher information for its mean is exactly I = 1/sigma^2 -- narrower
    (smaller sigma) means MORE curvature, MORE information, MORE certainty."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 1.0 / sigma ** 2


def cramer_rao_bound(fisher_information: float) -> float:
    """Var(theta_hat) >= 1/I(theta) -- the Cramer-Rao lower bound on the
    variance of any unbiased estimator, given its Fisher information."""
    if fisher_information <= 0:
        raise ValueError("fisher_information must be positive")
    return 1.0 / fisher_information


# ── 2. Dirac delta as the infinite-Fisher-information limit ────────────────

def dirac_delta_fisher_information_limit(sigma_values: np.ndarray) -> Dict:
    """As sigma -> 0, dgs/bayes_dirac_symmetry.py's Gaussian approximation of
    the Dirac delta becomes an infinitely sharp spike -- and its Fisher
    information 1/sigma^2 diverges to match. This is the precise sense in
    which 'perfect knowledge' (a Dirac-delta-certain outcome) corresponds to
    infinite Fisher information / zero Cramer-Rao variance: not an analogy,
    the same sigma controls both."""
    sigma_values = np.asarray(sigma_values, dtype=float)
    if np.any(sigma_values <= 0):
        raise ValueError("all sigma_values must be positive")
    fisher_info = 1.0 / sigma_values ** 2
    peak_heights = np.array([
        dirac_delta_as_gaussian_limit(np.array([0.0]), a=0.0, sigma=s)[0] for s in sigma_values
    ])
    return {"sigma_values": sigma_values, "fisher_information": fisher_info,
            "delta_peak_height": peak_heights}


# ── 3. Bayesian posterior concentration = growing Fisher information ───────

def bayesian_concentration_demo(n_values, sigma: float = 2.0, sigma0: float = 10.0,
                                 mu0: float = 0.0, mu_true: float = 3.0,
                                 rng_seed: int = 0) -> Dict:
    """Run dgs/bayes_inference.py's gaussian_mean_posterior with increasing
    sample size n and track the posterior PRECISION (1/variance) -- which,
    for a Gaussian posterior, IS the Fisher information accumulated so far.
    Demonstrates posterior concentration numerically: more data -> higher
    precision -> narrower posterior -> closer to the Dirac-delta limit above,
    not just asserted but the exact same 1/sigma^2 quantity computed both
    ways."""
    n_values = np.asarray(n_values, dtype=int)
    if np.any(n_values < 1):
        raise ValueError("all n_values must be >= 1")
    if sigma <= 0 or sigma0 <= 0:
        raise ValueError("sigma and sigma0 must be positive")
    rng = np.random.default_rng(rng_seed)
    full_data = mu_true + sigma * rng.standard_normal(int(n_values.max()))

    precisions, post_means, post_stds = [], [], []
    for n in n_values:
        post = gaussian_mean_posterior(full_data[:n], mu0, sigma0, sigma)
        precisions.append(post["precision"])
        post_means.append(post["mean"])
        post_stds.append(post["std"])

    return {
        "n_values": n_values,
        "posterior_precision": np.array(precisions),
        "posterior_mean": np.array(post_means),
        "posterior_std": np.array(post_stds),
        "mu_true": mu_true,
    }


if __name__ == "__main__":
    print("=== 1. Fisher information from curvature, vs the analytic Gaussian answer ===")
    theta = np.linspace(-5, 5, 2001)
    sigma_true = 1.5
    log_L = -0.5 * (theta / sigma_true) ** 2  # log of a Gaussian likelihood, up to a constant
    I_numeric = fisher_information_numeric(log_L, theta, theta_hat=0.0)
    I_analytic = gaussian_fisher_information(sigma_true)
    print(f"  numeric I (curvature at theta_hat=0): {I_numeric:.4f}")
    print(f"  analytic I = 1/sigma^2:                {I_analytic:.4f}")
    print(f"  Cramer-Rao bound on Var(theta_hat):    {cramer_rao_bound(I_numeric):.4f}  "
          f"(should be ~sigma^2={sigma_true**2:.4f})")

    print("\n=== 2. Dirac delta = infinite-Fisher-information limit ===")
    limit = dirac_delta_fisher_information_limit(np.array([1.0, 0.1, 0.01, 0.001]))
    for s, I, peak in zip(limit["sigma_values"], limit["fisher_information"], limit["delta_peak_height"]):
        print(f"  sigma={s:7.3f}  Fisher info={I:12.1f}  Dirac-limit peak height={peak:12.2f}")

    print("\n=== 3. Bayesian posterior concentration = growing Fisher information ===")
    demo = bayesian_concentration_demo(n_values=[1, 5, 20, 100, 500])
    for n, prec, mean, std in zip(demo["n_values"], demo["posterior_precision"],
                                   demo["posterior_mean"], demo["posterior_std"]):
        print(f"  n={n:4d}  precision(=Fisher info)={prec:8.3f}  "
              f"posterior mean={mean:.3f} (true={demo['mu_true']})  std={std:.3f}")

    print("\n=== 4. The quantum connection (dgs/uncertainty_qm.py, not re-derived) ===")
    from dgs.uncertainty_qm import statistical_resolution
    qm_result = statistical_resolution(N_photons=1000, n_measurements=500, phi_true=0.75)
    print("  dgs.uncertainty_qm.statistical_resolution already computes the SAME")
    print("  Cramer-Rao-bound/Fisher-information relationship for GS phase estimation --")
    print(f"  see its own 'GS_connection' note: {qm_result['GS_connection']}")
