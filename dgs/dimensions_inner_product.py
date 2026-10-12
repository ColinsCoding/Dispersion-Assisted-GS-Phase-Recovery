"""Looping over dimensions (integer AND fractional) and the complex inner
product -- Calc 2 + intro programming, tied to the plum-pudding atom.

1. BALL VOLUME IN d DIMENSIONS (the Calc 2 Gaussian trick):
       (int e^{-x^2} dx)^d = pi^{d/2}  =  S_{d-1} int_0^inf r^{d-1} e^{-r^2} dr = S_{d-1} Gamma(d/2)/2
   so the sphere "area" S_{d-1} = 2 pi^{d/2} / Gamma(d/2) and V_d = S_{d-1}/d = pi^{d/2}/Gamma(d/2+1).
   Gamma makes the formula meaningful for FRACTIONAL d; integer d is checked by
   Monte Carlo (throw random points in the cube, count the ones inside).

2. GAUSS'S LAW IN d DIMENSIONS for a uniform ball (Thomson's "pudding"):
       E(r) S_{d-1} r^{d-1} = Q_enc / eps,   Q_enc = Q (r/R)^d  for r < R
   -> E(r) = Q r / (eps S_{d-1} R^d): LINEAR in r for every d; outside ~ r^{1-d}.

3. COMPLEX INNER PRODUCT  <u, v> = sum conj(u_i) v_i   (conjugate the FIRST slot)
   <u, u> = sum |u_i|^2 >= 0 is a length; Cauchy-Schwarz |<u,v>| <= |u||v|.
   Pitfalls: numpy.dot does NOT conjugate (np.vdot does); MATLAB u' conjugates, u.' does not.
   Continuous version: <f, g> = int conj(f) g dx -- Fourier modes e^{ikx} are orthogonal.
"""
import numpy as np
import sympy as sp

d_sym, r_sym = sp.symbols("d r", positive=True)


# ── 1. volumes ───────────────────────────────────────────────────────

def ball_volume_symbolic():
    """Derive V_d via the Gaussian trick (radial integral by SymPy)."""
    S = sp.Symbol("S")                                       # unknown sphere area
    radial = sp.integrate(r_sym ** (d_sym - 1) * sp.exp(-r_sym ** 2), (r_sym, 0, sp.oo))
    S_val = sp.solve(sp.Eq(sp.pi ** (d_sym / 2), S * radial), S)[0]
    return sp.simplify(S_val), sp.simplify(sp.gammasimp(S_val / d_sym))


def ball_volume(d, R=1.0):
    """pi^{d/2} R^d / Gamma(d/2 + 1), any real d > 0."""
    from scipy.special import gamma
    d = np.asarray(d, float)
    if np.any(d <= 0) or R <= 0:
        raise ValueError("d and R must be positive")
    return np.pi ** (d / 2) * R ** d / gamma(d / 2 + 1)


def ball_volume_mc(d, n=400_000, seed=0):
    """Monte Carlo for integer d: fraction of the cube [-1,1]^d inside the unit ball."""
    if int(d) != d or d < 1:
        raise ValueError("Monte Carlo needs a positive integer dimension")
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(n, int(d)))
    inside = np.mean(np.sum(pts ** 2, axis=1) <= 1)
    return float(inside * 2.0 ** d)


def peak_dimension():
    """d maximising the unit-ball volume (d/dd ln V = 0): about 5.2569."""
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda d: -ball_volume(d), bounds=(1, 10), method="bounded",
                          options={"xatol": 1e-10})
    return float(res.x)


# ── 2. Gauss's law in d dimensions ───────────────────────────────────

def thomson_field_d(d=None):
    """Field inside / outside a uniform ball in d dimensions (symbolic)."""
    Q, eps, R, r = sp.symbols("Q epsilon R r", positive=True)
    d = d_sym if d is None else sp.nsimplify(d)          # keep 3 as Integer(3), 2.5 as 5/2
    S = 2 * sp.pi ** (d / 2) / sp.gamma(d / 2)
    inside = sp.simplify(Q * (r / R) ** d / (eps * S * r ** (d - 1)))
    outside = sp.simplify(Q / (eps * S * r ** (d - 1)))
    return inside, outside, (Q, eps, R, r)


# ── 3. complex inner product ─────────────────────────────────────────

def inner(u, v):
    """<u, v> = sum conj(u_i) v_i, written as an explicit loop (intro programming)."""
    u, v = list(u), list(v)
    if len(u) != len(v):
        raise ValueError("vectors must have the same length")
    total = 0j
    for a, b in zip(u, v):
        total += complex(a).conjugate() * complex(b)
    return total


def norm(u):
    return float(np.sqrt(inner(u, u).real))


def cauchy_schwarz_holds(u, v, tol=1e-12):
    return abs(inner(u, v)) <= norm(u) * norm(v) + tol


def fourier_inner_symbolic(k, m):
    """<e^{ikx}, e^{imx}> on [0, 2 pi] = 2 pi delta_km (integers k, m)."""
    x = sp.Symbol("x", real=True)
    return sp.integrate(sp.exp(-sp.I * k * x) * sp.exp(sp.I * m * x), (x, 0, 2 * sp.pi))
