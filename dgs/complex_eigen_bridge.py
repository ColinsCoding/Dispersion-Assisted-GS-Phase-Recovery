"""Complex analysis as the bridge from circuit theory to modern physics.

One thread, four stops -- each claim checked two independent ways:

1. CIRCUITS: poles = eigenvalues = residues.
   Series RLC in state form x = [q, i], x' = A x + b v. The poles of the
   transfer function are the eigenvalues of A. The impulse response is the
   inverse Laplace transform, i.e. a sum of residues of I(s) e^{st} -- computed
   here as a literal contour integral (dgs.complex_numerics) and checked
   against the matrix exponential expm(A t).

2. BESSEL FUNCTIONS FROM A CONTOUR (Schlafli / generating function):
       J_n(x) = (1 / 2 pi i) oint exp[(x/2)(t - 1/t)] t^{-n-1} dt,  |t| = 1.
   On a circle the trapezoid rule converges EXPONENTIALLY for analytic
   integrands -- a computer-engineering fact with a complex-analysis proof.
   And I_n(x) = i^{-n} J_n(i x): the modified Bessel function is J on the
   imaginary axis.

3. MODERN PHYSICS: Bessel zeros are energy levels.
   A particle in a 2-D circular box (the IBM "quantum corral") has
       E_mn = hbar^2 j_mn^2 / (2 m a^2).
   The zeros j_mn come from (a) SciPy, (b) eigenvalues of the discretised
   radial operator (dgs.bessel_linalg), (c) the argument principle counting
   them inside a contour.

4. A MACHINED PART: clamped aluminium disk.
   Plate modes satisfy  J_m(L) I_m'(L) - I_m(L) J_m'(L) = 0  (J and J-on-the-
   imaginary-axis together); f = (L^2 / 2 pi a^2) sqrt(D / rho h),
   D = E h^3 / 12(1 - nu^2). Plus a transparent machining-cost model whose
   every rate is a labelled assumption.
"""
import numpy as np
from scipy import special
from scipy.linalg import expm
from scipy.optimize import brentq

from dgs.complex_numerics import contour_integral, winding_number
from dgs.bessel_linalg import bessel_zeros as bessel_zeros_matrix

HBAR = 1.054571817e-34
M_E = 9.1093837015e-31
EV = 1.602176634e-19


# ── 1. circuits: poles = eigenvalues = residues ──────────────────────

def rlc_state_matrix(R, L, C):
    """A for x = [q, i]:  q' = i,  i' = -(q/C + R i)/L."""
    if min(R, L, C) <= 0:
        raise ValueError("R, L, C must be positive")
    return np.array([[0.0, 1.0], [-1.0 / (L * C), -R / L]])


def rlc_poles(R, L, C):
    """Roots of L C s^2 + R C s + 1 (the characteristic polynomial)."""
    return np.roots([L * C, R * C, 1.0])


def rlc_impulse_current_residues(R, L, C, t, n=256):
    """i(t) for v(t) = delta(t): inverse Laplace of I(s) = s C/(L C s^2 + R C s + 1).

    Computed as  sum over poles p of (1/2 pi i) oint_{|s-p|=r} I(s) e^{st} ds,
    i.e. the Bromwich integral closed to the left = the residue sum, with each
    residue a literal contour integral.

    Radius choice matters numerically: on the contour |e^{st}| varies by
    e^{+-r t}, so r is capped at ~1/t (and at a third of the pole spacing so
    each circle encloses one pole). One big circle around both poles fails at
    large t -- the integrand reaches e^{60} on its right edge.
    """
    poles = rlc_poles(R, L, C)
    sep = abs(poles[0] - poles[1])
    I = lambda s: s * C / (L * C * s ** 2 + R * C * s + 1.0)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = []
    for tt in t:
        r_t = 1.0 / tt if tt > 0 else np.inf
        if sep > 1e-6 * np.max(np.abs(poles)):   # np.roots splits a double root by ~sqrt(eps)
            centers, r = poles, min(sep / 3, r_t)
        else:                                   # (near-)repeated pole: one circle
            centers, r = [np.mean(poles)], min(0.5 * np.max(np.abs(poles)), r_t)
        total = sum(contour_integral(lambda s: I(s) * np.exp(s * tt), complex(c), r, n)
                    for c in centers)
        out.append(total / (2j * np.pi))
    return np.array(out).real


def rlc_impulse_current_expm(R, L, C, t):
    """Same response from the state matrix: x(t) = expm(A t) b, b = [0, 1/L]."""
    A = rlc_state_matrix(R, L, C)
    b = np.array([0.0, 1.0 / L])
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.array([(expm(A * tt) @ b)[1] for tt in t])


# ── 2. Bessel functions from a contour ───────────────────────────────

def bessel_j_contour(n, x, n_points=64):
    """J_n(x) via the Schlafli generating-function contour on |t| = 1."""
    if n_points < 4:
        raise ValueError("n_points must be >= 4")
    # t = e^{i theta}, dt = i t dtheta:  (1/2 pi i) oint ... dt
    #   = (1/2 pi) int_0^{2pi} exp(i x sin(theta) - i n theta) dtheta,
    # evaluated by the n_points trapezoid rule (= mean over equispaced theta)
    theta = 2 * np.pi * np.arange(n_points) / n_points
    return float(np.mean(np.exp(1j * (x * np.sin(theta) - n * theta))).real)


def bessel_i_from_j(n, x):
    """I_n(x) = i^{-n} J_n(i x): modified Bessel = J on the imaginary axis."""
    return (1j ** (-n) * special.jv(n, 1j * np.asarray(x, dtype=float))).real


# ── 3. quantum corral ────────────────────────────────────────────────

def bessel_zeros_scipy(m, k):
    return special.jn_zeros(m, k)


def count_zeros_argument_principle(m, x_center, radius, n=20000):
    """Number of zeros of J_m inside |z - x_center| < radius (argument principle)."""
    return winding_number(lambda z: special.jv(m, z), x_center, radius, n)


def corral_energies_eV(radius_m, m_eff=0.38, n_m=4, n_r=3):
    """E_mn = hbar^2 j_mn^2 / (2 m* a^2) for a hard-wall circular box, in eV.

    Defaults: Fe-on-Cu(111) quantum corral (Crommie, Lutz & Eigler 1993),
    surface-state effective mass 0.38 m_e. Returns list of (E, m, n),
    sorted, energies measured from the band bottom.
    """
    if radius_m <= 0 or m_eff <= 0:
        raise ValueError("radius and effective mass must be positive")
    out = []
    for m in range(n_m):
        for nn, j in enumerate(special.jn_zeros(m, n_r), start=1):
            E = HBAR ** 2 * j ** 2 / (2 * m_eff * M_E * radius_m ** 2) / EV
            out.append((E, m, nn))
    return sorted(out)


def corral_density(m, n, r_over_a, theta):
    """|psi_mn|^2 on a polar grid, cos(m theta) standing wave, unnormalised."""
    j = special.jn_zeros(m, n)[-1]
    R_ = np.where(r_over_a <= 1, special.jv(m, j * r_over_a), 0.0)
    return (R_ * np.cos(m * theta)) ** 2


# ── 4. clamped aluminium disk ────────────────────────────────────────

AL6061 = dict(E=68.9e9, rho=2700.0, nu=0.33)


def clamped_plate_lambdas(m, k=3):
    """Roots L of J_m(L) I_m'(L) - I_m(L) J_m'(L) = 0 (clamped circular plate)."""
    if m < 0 or k < 1:
        raise ValueError("m >= 0, k >= 1")
    g = lambda L: special.jv(m, L) * special.ivp(m, L) - special.iv(m, L) * special.jvp(m, L)
    xs = np.linspace(0.5, 40, 4000)
    gv = g(xs)
    roots = []
    for a, b, ga, gb in zip(xs[:-1], xs[1:], gv[:-1], gv[1:]):
        if ga * gb < 0:
            roots.append(brentq(g, a, b, xtol=1e-13))
            if len(roots) == k:
                break
    return np.array(roots)


def clamped_plate_frequencies(radius_m, thickness_m, m=0, k=3, material=AL6061):
    """Natural frequencies (Hz) of a clamped circular plate."""
    if radius_m <= 0 or thickness_m <= 0:
        raise ValueError("radius and thickness must be positive")
    E, rho, nu = material["E"], material["rho"], material["nu"]
    D = E * thickness_m ** 3 / (12 * (1 - nu ** 2))
    L = clamped_plate_lambdas(m, k)
    return L ** 2 / (2 * np.pi * radius_m ** 2) * np.sqrt(D / (rho * thickness_m))


def machining_cost(stock_volume_m3, removed_volume_m3, mrr_cm3_per_min=20.0,
                   shop_rate_per_hr=120.0, al_price_per_kg=8.0, setup_hr=1.0,
                   finishing_factor=1.5, rho=AL6061["rho"]):
    """Transparent CNC cost model. EVERY default is an assumption -- get quotes.

    material = stock mass * $/kg
    cutting  = removed volume / material-removal-rate * finishing_factor
    total    = material + (setup + cutting time) * shop rate
    """
    if min(stock_volume_m3, mrr_cm3_per_min, shop_rate_per_hr) <= 0:
        raise ValueError("stock volume, MRR, shop rate must be positive")
    if removed_volume_m3 < 0 or removed_volume_m3 > stock_volume_m3:
        raise ValueError("removed volume must be in [0, stock volume]")
    material = stock_volume_m3 * rho * al_price_per_kg
    cut_min = removed_volume_m3 * 1e6 / mrr_cm3_per_min * finishing_factor
    hours = setup_hr + cut_min / 60.0
    return {"material": material, "machine_hours": hours,
            "machine": hours * shop_rate_per_hr,
            "total": material + hours * shop_rate_per_hr}
