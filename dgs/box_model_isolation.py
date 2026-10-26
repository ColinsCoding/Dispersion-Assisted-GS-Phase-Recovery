"""Knight 40.4 interpreted, picking a quantum model for AMO/QIS work, and vibration isolation.

In plain words:

1. Interpreting the rigid box (Knight 40.4). psi_n = sqrt(2/L) sin(n pi x / L) has n - 1 nodes,
   places where the particle is NEVER found. The probability of finding it in [a, b] is an
   exact integral; as n grows that probability approaches the classical value (b - a)/L, the
   spacing of levels relative to their energy, (2n + 1)/n^2, goes to zero, and the spread
   Delta x approaches the classical L/sqrt(12): the correspondence principle (40.5).

2. Picking a model. A model is chosen for the questions you ask of it:
   * Which transitions can a laser address separately? Box gaps (2n+1)E1 are all different,
     so 64 box levels can act as a 64-level qudit (= 6 qubits); harmonic gaps are all equal
     (hbar omega), so a harmonic oscillator cannot be addressed level by level -- the reason
     qubits need anharmonicity.
   * Where are the lines? Remote sensing of CO by laser absorption (lidar, open-path
     spectroscopy) needs line positions to within the Doppler width (~0.005 cm^-1). The
     harmonic model misses the first overtone by ~80 cm^-1 -- thousands of Doppler widths --
     while the Morse/anharmonic model gets it. A wrong model means the laser sees nothing.
   * An atom in a trap V = U0 (1 - exp(-2|x/w|^p)): finite differences give the true levels, and
     the harmonic, harmonic + quartic (first-order perturbation) and box models are scored by the
     SHAPE of the spectrum -- the ratios E_n / E_1, which no size parameter can fake (the same
     trick as dgs.steam3d_qm_classifier). A Gaussian tweezer (p = 2) picks harmonic(+quartic); a
     flat-top trap with steep walls (large p) picks the box. The potential decides, not habit.

3. Vibration isolation (analytical mechanics). The atoms, cavities and interferometers above sit
   on a table that the floor shakes. Lagrangian L = T - V with a Rayleigh dissipation function
   gives the equations of motion; the transmissibility of one stage,
        T(r) = sqrt((1 + (2 zeta r)^2) / ((1 - r^2)^2 + (2 zeta r)^2)),  r = f / f0,
   isolates only above r = sqrt(2) and falls as 1/r^2; two stages fall as 1/r^4.

Units: box problems in units of L and E1 unless named; tweezer and isolation in SI.
NumPy, SciPy (eigh_tridiagonal), SymPy.
"""

import numpy as np
import sympy as sp
from scipy.linalg import eigh_tridiagonal

HBAR = 1.054571817e-34
K_B = 1.380649e-23
AMU = 1.66053906660e-27
C_CM = 2.99792458e10          # speed of light in cm/s (wavenumbers)


# ---------------------------------------------------------------- 1. interpreting the rigid box

x_s, L_s, a_s, b_s = sp.symbols("x L a b", positive=True)
n_s = sp.symbols("n", positive=True, integer=True)


def box_psi_sym(n=n_s, L=L_s):
    return sp.sqrt(2 / L) * sp.sin(n * sp.pi * x_s / L)


def box_probability_sym(n=n_s, a=a_s, b=b_s, L=L_s):
    """Exact P(a < x < b) for state n (SymPy)."""
    return sp.simplify(sp.integrate(box_psi_sym(n, L) ** 2, (x_s, a, b)))


def box_probability(n, a, b, L=1.0):
    """Closed form (b-a)/L - [sin(2 n pi b/L) - sin(2 n pi a/L)] / (2 n pi)."""
    if n < 1 or not 0 <= a <= b <= L:
        raise ValueError("need n >= 1 and 0 <= a <= b <= L")
    k = 2 * n * np.pi / L
    return (b - a) / L - (np.sin(k * b) - np.sin(k * a)) / k


def box_nodes(n, L=1.0):
    """Interior nodes x = j L / n, j = 1..n-1."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.arange(1, n) * L / n


def box_delta_x(n, L=1.0):
    """Position spread Delta x = L sqrt(1/12 - 1/(2 n^2 pi^2)) -> L/sqrt(12) classically."""
    n = np.asarray(n, float)
    if np.any(n < 1):
        raise ValueError("n must be >= 1")
    return L * np.sqrt(1 / 12 - 1 / (2 * n ** 2 * np.pi ** 2))


def relative_spacing(n):
    """(E_{n+1} - E_n) / E_n = (2n + 1)/n^2 for the box."""
    n = np.asarray(n, float)
    return (2 * n + 1) / n ** 2


def box_moments_sym():
    """<x>, <x^2>, Delta x for general n (SymPy)."""
    psi2 = box_psi_sym() ** 2
    mx = sp.simplify(sp.integrate(x_s * psi2, (x_s, 0, L_s)))
    mx2 = sp.simplify(sp.integrate(x_s ** 2 * psi2, (x_s, 0, L_s)))
    return mx, mx2, sp.simplify(sp.sqrt(mx2 - mx ** 2))


# ---------------------------------------------------------------- 2a. 64 levels as a qudit

def transition_gaps(model, n_levels=64):
    """Neighbour gaps E_{n+1} - E_n in units of the model's scale (E1 for box, hbar*omega for harmonic)."""
    if n_levels < 2:
        raise ValueError("need at least 2 levels")
    n = np.arange(1, n_levels + 1, dtype=float)
    if model == "box":
        E = n ** 2
    elif model == "harmonic":
        E = n - 0.5
    else:
        raise ValueError("model must be 'box' or 'harmonic'")
    return np.diff(E)


def min_gap_separation(gaps):
    """Smallest difference between any two transition frequencies (0 = some transitions coincide)."""
    g = np.sort(np.asarray(gaps, float))
    return float(np.min(np.diff(g))) if g.size > 1 else np.inf


# ---------------------------------------------------------------- 2b. remote-sensing line positions (CO)

CO = dict(we=2169.81358, wexe=13.28831, mass_amu=27.9949)   # cm^-1 (spectroscopic constants, from memory)


def vib_band_origins(we, wexe, v_max=3):
    """Band origins v=0 -> v (cm^-1): harmonic v*we vs anharmonic v*we - v(v+1)*wexe."""
    v = np.arange(1, v_max + 1)
    return v * we, v * we - v * (v + 1) * wexe


def doppler_hwhm_cm(nu_cm, mass_amu, T=296.0):
    """Doppler half width at half maximum (cm^-1): nu sqrt(2 k T ln2 / (m c^2))."""
    m = mass_amu * AMU
    return nu_cm * np.sqrt(2 * K_B * T * np.log(2) / (m * (C_CM / 100) ** 2))


# ---------------------------------------------------------------- 2c. atom in a Gaussian tweezer: which model?

def tweezer_levels(U0, w, mass, n_levels=20, n_grid=4000, span=3.0, order=2):
    """Exact (finite-difference) levels of V = U0 (1 - exp(-2 |x/w|^order)) from the bottom (J).
    order = 2 is a Gaussian optical tweezer; large order is a flat-top trap with steep walls."""
    if U0 <= 0 or w <= 0 or mass <= 0 or order <= 0:
        raise ValueError("U0, w, mass, order must be positive")
    x = np.linspace(-span * w, span * w, n_grid)
    dx = x[1] - x[0]
    V = U0 * (1 - np.exp(-2 * np.abs(x / w) ** order))
    c = HBAR ** 2 / (2 * mass * dx ** 2)
    return eigh_tridiagonal(V + 2 * c, -c * np.ones(n_grid - 1), select="i",
                            select_range=(0, n_levels - 1), eigvals_only=True)


def box_width(w, order=2):
    """Width of the well at half depth: 2 w (ln2 / 2)^(1/order) -- the box model's L."""
    return 2 * w * (np.log(2) / 2) ** (1 / order)


def tweezer_models(U0, w, mass, n_levels=20, order=2):
    """Model predictions for the same levels. Harmonic and harmonic + first-order quartic exist only
    for order = 2 (a well with curvature at the bottom); the box uses the half-depth width."""
    n = np.arange(n_levels)
    L = box_width(w, order)
    out = dict(box=HBAR ** 2 * np.pi ** 2 * (n + 1) ** 2 / (2 * mass * L ** 2))
    if order == 2:
        omega = np.sqrt(4 * U0 / (mass * w ** 2))
        harm = HBAR * omega * (n + 0.5)
        # V ~ U0 (2 x^2/w^2 - 2 x^4/w^4 + ...): first-order shift -2 U0/w^4 <x^4>,
        # <n|x^4|n> = (hbar / 2 m omega)^2 * 3 (2 n^2 + 2 n + 1)
        l2 = HBAR / (2 * mass * omega)
        out.update(omega=omega, harmonic=harm,
                   harmonic_plus_quartic=harm - 2 * U0 / w ** 4 * l2 ** 2 * 3 * (2 * n ** 2 + 2 * n + 1))
    return out


def pick_model(U0, w, mass, n_levels, order=2):
    """Score each model by the SHAPE of its spectrum: RMS relative error of the ratios E_n / E_1
    (independent of the trap's size or depth, the one parameter you would fit). Returns
    (best, scores, fitted) where fitted gives the model's size parameter matched to the exact E_1:
    the box width L, or hbar*omega for the harmonic models."""
    if n_levels < 2:
        raise ValueError("need at least 2 levels to compare shapes")
    exact = tweezer_levels(U0, w, mass, n_levels, order=order)
    ratios = exact / exact[0]
    n = np.arange(n_levels)
    shapes = {"box": (n + 1) ** 2 / 1.0, "harmonic": 2 * n + 1.0}
    models = tweezer_models(U0, w, mass, n_levels, order)
    if "harmonic_plus_quartic" in models:
        q = models["harmonic_plus_quartic"]
        shapes["harmonic_plus_quartic"] = q / q[0]
    scores = {k: float(np.sqrt(np.mean((v[1:] / ratios[1:] - 1) ** 2))) for k, v in shapes.items()}
    fitted = dict(box_L=float(HBAR * np.pi / np.sqrt(2 * mass * exact[0])),
                  harmonic_hbar_omega=float(2 * exact[0]))
    return min(scores, key=scores.get), scores, fitted


# ---------------------------------------------------------------- 3. vibration isolation

def transmissibility(r, zeta):
    """Single-stage displacement transmissibility |X/U| for base excitation."""
    if zeta < 0:
        raise ValueError("zeta must be >= 0")
    r = np.asarray(r, float)
    return np.sqrt((1 + (2 * zeta * r) ** 2) / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2))


def two_stage_sym():
    """Euler-Lagrange equations for ground u -> (k1, c1) -> m1 -> (k2, c2) -> m2, and |X2/U|(s)."""
    t = sp.symbols("t", real=True)
    m1, m2, k1, k2, c1, c2, s = sp.symbols("m_1 m_2 k_1 k_2 c_1 c_2 s", positive=True)
    x1, x2, u = (sp.Function(nm)(t) for nm in ("x_1", "x_2", "u"))
    T = (m1 * x1.diff(t) ** 2 + m2 * x2.diff(t) ** 2) / 2
    V = (k1 * (x1 - u) ** 2 + k2 * (x2 - x1) ** 2) / 2
    D = (c1 * (x1.diff(t) - u.diff(t)) ** 2 + c2 * (x2.diff(t) - x1.diff(t)) ** 2) / 2
    Lg = T - V
    eqs = []
    for q in (x1, x2):
        eqs.append(sp.simplify(sp.diff(Lg.diff(q.diff(t)), t) - Lg.diff(q) + D.diff(q.diff(t))))
    X1, X2, U = sp.symbols("X_1 X_2 U")
    sub = {x1.diff(t, 2): s ** 2 * X1, x2.diff(t, 2): s ** 2 * X2, x1.diff(t): s * X1, x2.diff(t): s * X2,
           u.diff(t): s * U, x1: X1, x2: X2, u: U}
    lap = [sp.expand(e.subs(sub)) for e in eqs]
    sol = sp.solve(lap, [X1, X2], dict=True)[0]
    H = sp.simplify(sol[X2] / U)
    return dict(equations=eqs, H=H, symbols=dict(m1=m1, m2=m2, k1=k1, k2=k2, c1=c1, c2=c2, s=s))


def two_stage_transmissibility(f, f1, f2, zeta1, zeta2, mass_ratio=1.0):
    """|X2/U| at frequencies f for two stages with uncoupled frequencies f1, f2 (m2 = mass_ratio * m1)."""
    f = np.asarray(f, float)
    m1, m2 = 1.0, mass_ratio
    k1, k2 = m1 * (2 * np.pi * f1) ** 2, m2 * (2 * np.pi * f2) ** 2
    c1, c2 = 2 * zeta1 * np.sqrt(k1 * m1), 2 * zeta2 * np.sqrt(k2 * m2)
    s = 2j * np.pi * f
    # [[m1 s^2 + (c1+c2) s + k1 + k2, -(c2 s + k2)], [-(c2 s + k2), m2 s^2 + c2 s + k2]] X = [(c1 s + k1) U, 0]
    a11 = m1 * s ** 2 + (c1 + c2) * s + k1 + k2
    a12 = -(c2 * s + k2)
    a22 = m2 * s ** 2 + c2 * s + k2
    det = a11 * a22 - a12 ** 2
    return np.abs(-a12 * (c1 * s + k1) / det)


def required_f0_single(T_needed, r_floor=1.0):
    """Undamped single stage: natural frequency (relative to the disturbance) giving T = T_needed: r = sqrt(1 + 1/T)."""
    if not 0 < T_needed < 1:
        raise ValueError("T_needed must be in (0, 1)")
    return r_floor / np.sqrt(1 + 1 / T_needed)
