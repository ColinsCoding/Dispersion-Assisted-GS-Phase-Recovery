"""Quantum receivers: discriminating two coherent states |alpha>, |-alpha>
(BPSK) below the standard quantum limit.

Mean photon number per pulse N = alpha^2 (alpha real). Equal priors.

    homodyne (SQL)    P = (1/2) erfc(sqrt(2N))          measure x = a + a^dag
    heterodyne        P = (1/2) erfc(sqrt(N))           both quadratures, +1 vacuum unit
    Kennedy           P = (1/2) e^{-4N}                 displace by -alpha, count photons
    optimised Kennedy P = (1/2)[1 - e^{-(a-b)^2} + e^{-(a+b)^2}],  best b > alpha
    Dolinar (adaptive, greedy per time slice)  ->  Helstrom as slices -> infinity
    Helstrom bound    P = (1/2)(1 - sqrt(1 - |<alpha|-alpha>|^2)),  <alpha|-alpha> = e^{-2N}

Each closed form is checked independently:
    homodyne / heterodyne  Monte-Carlo Gaussian quadrature samples
    Kennedy family          Monte-Carlo Poisson photon counts
    Helstrom                trace norm of (rho_+ - rho_-)/2 in a truncated Fock space
                            (an eigenvalue problem in a vector space)
    Dolinar                 exact recursion over all click histories + Monte Carlo

Squeezing: homodyne with a squeezed coherent state, total photons fixed:
    N = alpha^2 + sinh^2 r,  P = (1/2) erfc(sqrt(2) alpha e^{r}).
"""
from functools import lru_cache
from math import factorial

import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
from scipy.special import erfc


def _check_N(N):
    N = np.asarray(N, dtype=float)
    if np.any(N < 0):
        raise ValueError("mean photon number N must be non-negative")
    return N


# ── closed forms ─────────────────────────────────────────────────────

def homodyne_error(N):
    return 0.5 * erfc(np.sqrt(2 * _check_N(N)))


def heterodyne_error(N):
    return 0.5 * erfc(np.sqrt(_check_N(N)))


def kennedy_error(N):
    return 0.5 * np.exp(-4 * _check_N(N))


def displaced_counting_error(N, b):
    """Displace by -b, decide '+' on zero clicks."""
    a = np.sqrt(_check_N(N))
    return 0.5 * (1 - np.exp(-(a - b) ** 2) + np.exp(-(a + b) ** 2))


def optimized_kennedy(N):
    """Best displacement b and its error (numerical minimisation over b)."""
    a = float(np.sqrt(_check_N(N)))
    res = minimize_scalar(lambda b: displaced_counting_error(N, b),
                          bounds=(0.0, a + 3.0), method="bounded",
                          options={"xatol": 1e-12})
    return float(res.x), float(res.fun)


def optimized_kennedy_condition():
    """SymPy: dP/db = 0 via the chain rule -> (a-b) e^{-(a-b)^2} = (a+b) e^{-(a+b)^2}."""
    a, b = sp.symbols("alpha beta", positive=True)
    P = sp.Rational(1, 2) * (1 - sp.exp(-(a - b) ** 2) + sp.exp(-(a + b) ** 2))
    return sp.simplify(sp.diff(P, b))


def overlap(N):
    """<alpha|-alpha> = e^{-2N} (real, alpha real)."""
    return np.exp(-2 * _check_N(N))


def helstrom_error(N):
    return 0.5 * (1 - np.sqrt(1 - overlap(N) ** 2))


def kennedy_helstrom_ratio_limit():
    """SymPy limit of P_Kennedy / P_Helstrom as N -> oo (it is 2)."""
    N = sp.Symbol("N", positive=True)
    PK = sp.exp(-4 * N) / 2
    PH = (1 - sp.sqrt(1 - sp.exp(-4 * N))) / 2
    return sp.limit(PK / PH, N, sp.oo)


# ── vector-space check of Helstrom ───────────────────────────────────

def coherent_state(alpha, cutoff=60):
    """Fock-basis amplitudes e^{-|a|^2/2} a^n / sqrt(n!)."""
    if cutoff < 2:
        raise ValueError("cutoff must be >= 2")
    n = np.arange(cutoff)
    logfact = np.array([np.log(float(factorial(int(k)))) for k in n])
    with np.errstate(divide="ignore"):
        logamp = n * np.log(abs(alpha) + 0.0) - 0.5 * logfact if alpha != 0 else np.where(n == 0, 0.0, -np.inf)
    amp = np.exp(-abs(alpha) ** 2 / 2 + logamp) * np.sign(alpha) ** n if alpha != 0 else (n == 0).astype(float)
    return amp.astype(complex)


def helstrom_error_fock(N, cutoff=60):
    """(1/2)(1 - ||(rho_+ - rho_-)/2||_1) from eigenvalues of the density matrices."""
    a = float(np.sqrt(N))
    vp, vm = coherent_state(a, cutoff), coherent_state(-a, cutoff)
    gamma = 0.5 * (np.outer(vp, vp.conj()) - np.outer(vm, vm.conj()))
    tn = np.sum(np.abs(np.linalg.eigvalsh(gamma)))
    return 0.5 * (1 - tn)


def helstrom_measurement_check(N, cutoff=60):
    """Optimal measurement = projector onto the positive eigenspace of rho_+ - rho_-.
    Returns (error of that projective measurement, Helstrom closed form)."""
    a = float(np.sqrt(N))
    vp, vm = coherent_state(a, cutoff), coherent_state(-a, cutoff)
    gamma = np.outer(vp, vp.conj()) - np.outer(vm, vm.conj())
    w, V = np.linalg.eigh(gamma)
    Pp = V[:, w > 0] @ V[:, w > 0].conj().T              # decide '+'
    err = 0.5 * (1 - np.real(vp.conj() @ Pp @ vp)) + 0.5 * np.real(vm.conj() @ Pp @ vm)
    return float(err), float(helstrom_error(N))


# ── Monte Carlo of the physical receivers ────────────────────────────

def mc_homodyne(N, trials=200_000, seed=0):
    """x = a + a^dag sample: mean +-2 alpha, variance 1 (vacuum). Decide by sign."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=trials)
    x = 2 * np.sqrt(N) * s + rng.standard_normal(trials)
    return float(np.mean(np.sign(x) != s))


def mc_heterodyne(N, trials=200_000, seed=0):
    """Heterodyne x-output: mean +-sqrt(2) alpha... equivalently SNR halved: variance 2."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=trials)
    x = 2 * np.sqrt(N) * s + np.sqrt(2) * rng.standard_normal(trials)
    return float(np.mean(np.sign(x) != s))


def mc_displaced_counting(N, b, trials=200_000, dark=0.0, eta=1.0, seed=0):
    """Displace by -b, Poisson clicks with mean eta|±alpha - b|^2 + dark; '+' if zero."""
    if not 0 < eta <= 1 or dark < 0:
        raise ValueError("0 < eta <= 1, dark >= 0")
    rng = np.random.default_rng(seed)
    a = np.sqrt(N)
    s = rng.choice([-1, 1], size=trials)
    clicks = rng.poisson(eta * (s * a - b) ** 2 + dark)
    decide = np.where(clicks == 0, 1, -1)
    return float(np.mean(decide != s))


def displaced_counting_error_imperfect(N, b, eta=1.0, dark=0.0):
    """Closed form with detector efficiency eta and dark-count mean `dark`."""
    a = np.sqrt(_check_N(N))
    p0p = np.exp(-eta * (a - b) ** 2 - dark)
    p0m = np.exp(-eta * (a + b) ** 2 - dark)
    return 0.5 * (1 - p0p) + 0.5 * p0m


# ── Dolinar-type adaptive receiver ───────────────────────────────────

def _slice_click_probs(a, b):
    """P(no click | +), P(no click | -) for slice amplitude a, displacement -b."""
    return np.exp(-(a - b) ** 2), np.exp(-(a + b) ** 2)


def _greedy_b(p, a):
    """Displacement minimising the error if we had to decide right after this slice.
    Cached: within one receiver the reachable posteriors form a finite tree."""
    return _greedy_b_cached(round(float(p), 12), round(float(a), 12))


@lru_cache(maxsize=200_000)
def _greedy_b_cached(p, a):
    def after(b):
        n_p, n_m = _slice_click_probs(a, b)
        return (min(p * n_p, (1 - p) * n_m) + min(p * (1 - n_p), (1 - p) * (1 - n_m)))
    # after(b) has several local minima (null the '+' state, null the '-' state,
    # or a far displacement): grid-search globally, then refine locally.
    grid = np.linspace(-a - 4.0, a + 4.0, 801)
    vals = [after(g) for g in grid]
    k = int(np.argmin(vals))
    step = grid[1] - grid[0]
    res = minimize_scalar(after, bounds=(grid[k] - step, grid[k] + step), method="bounded",
                          options={"xatol": 1e-12})
    return float(res.x) if res.fun <= vals[k] else float(grid[k])


def dolinar_error(N, slices=8):
    """Exact error of the greedy adaptive receiver: the pulse is split into
    `slices` equal pieces (amplitude alpha/sqrt(M) each); before each piece the
    displacement is re-chosen from the current posterior; clicks update the
    posterior by Bayes' rule. Exact expectation over all 2^M click histories."""
    if slices < 1 or slices > 16:
        raise ValueError("slices must be in [1, 16]")
    a = float(np.sqrt(_check_N(N))) / np.sqrt(slices)

    @lru_cache(maxsize=None)
    def E(p_key, k):
        p = p_key / 1e12
        if k == slices or p <= 0 or p >= 1:
            return min(p, 1 - p)
        b = _greedy_b(p, a)
        n_p, n_m = _slice_click_probs(a, b)
        total = 0.0
        for lp, lm in ((n_p, n_m), (1 - n_p, 1 - n_m)):
            po = p * lp + (1 - p) * lm
            if po <= 0:
                continue
            total += po * E(round(p * lp / po * 1e12), k + 1)
        return total

    return float(E(round(0.5 * 1e12), 0))


def mc_dolinar(N, slices=8, trials=20_000, seed=0):
    """Monte Carlo of the same adaptive receiver."""
    rng = np.random.default_rng(seed)
    a = np.sqrt(N) / np.sqrt(slices)
    errs = 0
    for _ in range(trials):
        s = rng.choice([-1, 1])
        p = 0.5
        for _k in range(slices):
            b = _greedy_b(p, a)
            n_p, n_m = _slice_click_probs(a, b)
            click = rng.random() > (n_p if s > 0 else n_m)
            lp, lm = (1 - n_p, 1 - n_m) if click else (n_p, n_m)
            p = p * lp / (p * lp + (1 - p) * lm)
        errs += (1 if p >= 0.5 else -1) != s
    return errs / trials


# ── squeezing ────────────────────────────────────────────────────────

def squeezed_homodyne_error(N, r):
    """Coherent amplitude alpha with x-quadrature squeezed by e^{-2r}; photon
    budget N = alpha^2 + sinh^2 r. Error (1/2) erfc(sqrt(2) alpha e^{r})."""
    N = float(_check_N(N))
    s2 = np.sinh(r) ** 2
    if s2 > N:
        return np.nan
    return 0.5 * erfc(np.sqrt(2) * np.sqrt(N - s2) * np.exp(r))


def optimal_squeezing(N):
    """Maximise (N - sinh^2 r) e^{2r} over r (SymPy derivative, numeric root)."""
    r = sp.Symbol("r", real=True)
    f = (N - sp.sinh(r) ** 2) * sp.exp(2 * r)
    df = sp.lambdify(r, sp.diff(f, r), "numpy")
    rmax = float(np.arcsinh(np.sqrt(N)))
    from scipy.optimize import brentq
    r_opt = brentq(df, 0.0, rmax - 1e-12) if df(0.0) > 0 else 0.0
    return r_opt, float(squeezed_homodyne_error(N, r_opt))
