"""Continuous vs. discrete EM spectra (Knight Ch. 37.1-37.3), in SymPy.

CONTINUOUS -- blackbody (hot dense matter): Planck's law is a smooth function of
wavelength; integrating it over all wavelengths gives Stefan-Boltzmann,
    int_0^oo x^3/(e^x - 1) dx = pi^4/15   ->   sigma = 2 pi^5 k^4 / (15 c^2 h^3).

DISCRETE -- gas discharge tube (isolated atoms): hydrogen levels E_n = -13.6 eV/n^2,
lines 1/lambda = R (1/n_l^2 - 1/n_u^2); the Balmer series piles up at the series
limit lambda = 4/R as n_u -> oo (a SymPy limit).

BOTH -- an x-ray tube: bremsstrahlung is CONTINUOUS with a sharp short-wavelength
cutoff lambda_min = hc/(eV) (Duane-Hunt), with DISCRETE characteristic lines on top
(Moseley: K-alpha ~ (3/4) 13.6 eV (Z-1)^2). So "x rays are discrete/continuous" is
half true either way.

NOT AN EM SPECTRUM -- cathode rays are electrons (Thomson, 1897).
"""
import numpy as np
import sympy as sp

H = 6.62607015e-34
C = 299792458.0
KB = 1.380649e-23
EV = 1.602176634e-19
R_INF = 1.0973731568e7          # m^-1
RY_EV = 13.605693               # eV
HC_EV_NM = H * C / EV * 1e9     # 1239.84 eV nm


# ── continuous ───────────────────────────────────────────────────────

def bose_integral_by_series():
    """int_0^oo x^3/(e^x - 1) dx, which SymPy cannot do directly, done the Calc 2 way:
    1/(e^x - 1) = sum_{n>=1} e^{-n x}  ->  term-by-term  int x^3 e^{-n x} dx = 6/n^4
    ->  sum 6/n^4 = 6 zeta(4) = pi^4/15.  Returns (term, sum)."""
    x = sp.Symbol("x", positive=True)
    n = sp.Symbol("n", integer=True, positive=True)
    term = sp.integrate(x ** 3 * sp.exp(-n * x), (x, 0, sp.oo))
    total = sp.summation(term, (n, 1, sp.oo))
    return sp.simplify(term), sp.simplify(total)


def bose_integral_numeric():
    from scipy.integrate import quad
    with np.errstate(over="ignore"):            # expm1(huge) = inf -> integrand 0, correct
        return quad(lambda t: t ** 3 / np.expm1(t) if t > 0 else 0.0, 0, np.inf)[0]


def stefan_boltzmann_symbolic():
    """Return (the Bose integral = pi^4/15, sigma expression)."""
    _, I = bose_integral_by_series()
    h, c, k = sp.symbols("h c k", positive=True)
    sigma = sp.simplify(2 * sp.pi * k ** 4 / (c ** 2 * h ** 3) * I)   # pi * (2 k^4/(c^2 h^3)) * I
    return sp.simplify(I), sigma


def sigma_numeric():
    _, sigma = stefan_boltzmann_symbolic()
    h, c, k = sp.symbols("h c k", positive=True)
    return float(sigma.subs({h: H, c: C, k: KB}))


def planck_lambda(lam_nm, T):
    lam = np.asarray(lam_nm, float) * 1e-9
    with np.errstate(over="ignore"):
        return 2 * H * C ** 2 / lam ** 5 / np.expm1(H * C / (lam * KB * T))


# ── discrete ─────────────────────────────────────────────────────────

def hydrogen_level_eV(n):
    if n < 1 or int(n) != n:
        raise ValueError("n must be a positive integer")
    return -RY_EV / n ** 2


def hydrogen_line_nm(n_upper, n_lower):
    if not (1 <= n_lower < n_upper):
        raise ValueError("need 1 <= n_lower < n_upper")
    return 1e9 / (R_INF * (1 / n_lower ** 2 - 1 / n_upper ** 2))


def series_limit_symbolic(n_lower=2):
    """lim_{n_u -> oo} lambda = n_lower^2 / R (Balmer: 4/R ~ 364.5 nm)."""
    n, R = sp.symbols("n_u R", positive=True)
    lam = 1 / (R * (sp.Rational(1, n_lower ** 2) - 1 / n ** 2))
    return sp.limit(lam, n, sp.oo), lam


# ── x-ray tube: both ─────────────────────────────────────────────────

def duane_hunt_cutoff_nm(V):
    """Shortest bremsstrahlung wavelength: all of eV into one photon."""
    if V <= 0:
        raise ValueError("tube voltage must be positive")
    return HC_EV_NM / V


def kramers_spectrum(lam_nm, V):
    """Kramers' law per unit wavelength: I ~ (lam/lam_min - 1) / lam^2 for lam > lam_min."""
    lam = np.asarray(lam_nm, float)
    lmin = duane_hunt_cutoff_nm(V)
    return np.where(lam > lmin, (lam / lmin - 1) / lam ** 2, 0.0)


def moseley_kalpha_keV(Z):
    """K-alpha photon energy ~ (3/4) Ry (Z - 1)^2 (Moseley's law)."""
    if Z < 2:
        raise ValueError("Z must be >= 2")
    return 0.75 * RY_EV * (Z - 1) ** 2 / 1e3


# ── the quiz options, tested in a loop ───────────────────────────────

SPECTRUM_KIND = {
    "blackbody": {"continuous"},
    "gas discharge tube": {"discrete"},
    "x rays": {"continuous", "discrete"},          # bremsstrahlung + characteristic lines
    "cathode rays": set(),                          # electrons -- no EM spectrum
}

OPTIONS = [
    ("cathode rays", "continuous", "x rays", "discrete"),
    ("x rays", "continuous", "cathode rays", "discrete"),
    ("blackbody", "discrete", "gas discharge tube", "continuous"),
    ("blackbody", "continuous", "gas discharge tube", "discrete"),
]


def check_option(opt):
    """(fully correct?, reasons). A claim is right only if it is the source's ONLY kind."""
    s1, k1, s2, k2 = opt
    why = []
    for s, k in ((s1, k1), (s2, k2)):
        kinds = SPECTRUM_KIND[s]
        if not kinds:
            why.append(f"{s} are electrons, not EM radiation")
        elif k not in kinds:
            why.append(f"{s} is {', '.join(sorted(kinds))}, not {k}")
        elif len(kinds) > 1:
            why.append(f"{s} is BOTH continuous and discrete (only half true)")
    return not why, why
