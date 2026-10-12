"""Wien's law, symbolic and numeric, and the 502 nm wave itself (Knight Ch. 37
adaptive question: Sun at 5780 K peaks at 502 nm; where does it peak at 7000 K?).

1. SYMBOLIC (SymPy): maximise Planck's B_lambda. With x = hc/(lambda k T),
   dB/dlambda = 0  <=>  x = 5 (1 - e^{-x})  <=>  x = 5 + W(-5 e^{-5})  (Lambert W),
   so lambda_max T = b = hc / (k x) = 2.898e-3 m K, and lambda_2 = lambda_1 T_1 / T_2.
2. NUMERIC: numpy grid argmax of dgs.blackbody.planck_radiance; torch autograd
   Newton iteration on d(ln B)/d(lambda) = 0 (CUDA if available); pandas table
   testing every multiple-choice option.
3. THE WAVE: E(z,t) = E0 cos(kz - wt) along a polarisation direction.
   k = 2 pi / lambda, f = c / lambda, photon energy hc/lambda; amplitude from an
   irradiance I via I = (1/2) c eps0 E0^2 (labelled assumption: solar constant).
   Projection onto a polariser at angle theta: amplitude E0 cos(theta), intensity
   I cos^2(theta) (Malus) -- an inner product, as in dimensions_inner_product.py.
   Sunlight is UNPOLARISED: averaging over random directions gives I/2.
4. POLYGON: the tip of E traces the polarisation ellipse; sampled as an N-gon its
   shoelace area -> pi a b, the area SymPy gives for the ellipse.
"""
import numpy as np
import sympy as sp

from dgs.blackbody import planck_radiance
from dgs.jones_calculus import jones_linear, jones_matrix_linear_polarizer

H = 6.62607015e-34
C = 299792458.0
KB = 1.380649e-23
EPS0 = 8.8541878128e-12
EV = 1.602176634e-19
SOLAR_CONSTANT = 1361.0            # W/m^2 at 1 AU, whole spectrum (assumption for the amplitude)

OPTIONS_NM = [608, 414, 502, 580, 700]


# ── 1. symbolic ──────────────────────────────────────────────────────

def wien_root_symbolic():
    """Solve x = 5(1 - e^{-x}) for the non-trivial root via Lambert W."""
    x = sp.Symbol("x", positive=True)
    sols = sp.solve(sp.Eq(x, 5 * (1 - sp.exp(-x))), x)
    root = [s for s in sols if s.is_positive and sp.N(s) > 1][0]
    return root, x


def wien_condition_symbolic():
    """d/dlambda of B_lambda = (2hc^2/lambda^5)/(e^{hc/(lambda k T)} - 1), set to 0,
    rewritten in x = hc/(lambda k T): returns the simplified condition expression."""
    lam, T, h, c, k = sp.symbols("lambda T h c k", positive=True)
    B = 2 * h * c ** 2 / lam ** 5 / (sp.exp(h * c / (lam * k * T)) - 1)
    dB = sp.diff(B, lam)
    x = sp.Symbol("x", positive=True)
    cond = sp.simplify((dB * lam ** 6 * (sp.exp(h * c / (lam * k * T)) - 1) ** 2 / (2 * h * c ** 2))
                       .subs(h * c / (lam * k * T), x))
    cond = sp.simplify(cond.subs(sp.exp(h * c / (lam * k * T)), sp.exp(x)).subs(h * c, x * lam * k * T))
    return sp.factor(cond), x


def wien_constant():
    root, _ = wien_root_symbolic()
    return H * C / (KB * float(sp.N(root, 20)))


def scale_peak_symbolic():
    l1, T1, T2 = sp.symbols("lambda_1 T_1 T_2", positive=True)
    l2 = sp.solve(sp.Eq(l1 * T1, sp.Symbol("lambda_2") * T2), sp.Symbol("lambda_2"))[0]
    return l2, (l1, T1, T2)


def scale_peak(lam1_nm, T1, T2):
    if min(lam1_nm, T1, T2) <= 0:
        raise ValueError("wavelength and temperatures must be positive")
    return lam1_nm * T1 / T2


# ── 2. numeric ───────────────────────────────────────────────────────

def peak_numpy(T, lo=100.0, hi=3000.0, n=200_001):
    lam = np.linspace(lo, hi, n)
    return float(lam[np.argmax(planck_radiance(lam, T))])


def peak_torch(T, lam0_nm=500.0, iters=30, device=None):
    """Newton's method on g(lam) = d ln B / d lam, derivatives by torch autograd."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lam = torch.tensor(lam0_nm * 1e-9, dtype=torch.float64, device=dev, requires_grad=True)
    hc_k = H * C / KB
    for _ in range(iters):
        lnB = -5 * torch.log(lam) - torch.log(torch.expm1(hc_k / (lam * T)))
        (g,) = torch.autograd.grad(lnB, lam, create_graph=True)
        (gp,) = torch.autograd.grad(g, lam)
        with torch.no_grad():
            lam -= g / gp
    return float(lam.detach().cpu()) * 1e9, str(dev)


def options_table(lam1_nm=502.0, T1=5780.0, T2=7000.0, options=OPTIONS_NM):
    """pandas DataFrame: each option, its implied lambda*T, error vs the law, verdict."""
    import pandas as pd
    target = scale_peak(lam1_nm, T1, T2)
    rows = []
    for o in options:
        rows.append({
            "option (nm)": o,
            "lambda*T (m K)": o * 1e-9 * T2,
            "error vs 502*5780/7000 (%)": 100 * (o - target) / target,
            "bluer than 502 nm?": o < lam1_nm,
            "verdict": "correct" if abs(o - target) < 1.0 else "wrong",
        })
    return pd.DataFrame(rows), target


# ── 3. the wave ──────────────────────────────────────────────────────

def wave_parameters(lam_nm=502.0, irradiance=SOLAR_CONSTANT):
    lam = lam_nm * 1e-9
    if lam <= 0 or irradiance <= 0:
        raise ValueError("wavelength and irradiance must be positive")
    return {
        "k (rad/m)": 2 * np.pi / lam,
        "frequency (THz)": C / lam / 1e12,
        "omega (rad/s)": 2 * np.pi * C / lam,
        "period (fs)": lam / C * 1e15,
        "photon energy (eV)": H * C / lam / EV,
        "E0 (V/m)": np.sqrt(2 * irradiance / (C * EPS0)),
        "B0 (uT)": np.sqrt(2 * irradiance / (C * EPS0)) / C * 1e6,
        "photons / (m^2 s) if all at this lam": irradiance / (H * C / lam),
    }


def field(z, t, lam_nm=502.0, E0=1.0, phase=0.0):
    lam = lam_nm * 1e-9
    k, w = 2 * np.pi / lam, 2 * np.pi * C / lam
    return E0 * np.cos(k * np.asarray(z) - w * np.asarray(t) + phase)


def polarizer_projection(input_angle_deg, polarizer_angle_deg, E0=1.0):
    """Amplitude and intensity after a polariser, via Jones calculus (dgs.jones_calculus)."""
    Ein = E0 * np.asarray(jones_linear(input_angle_deg), complex)
    Eout = np.asarray(jones_matrix_linear_polarizer(polarizer_angle_deg), complex) @ Ein
    amp = float(np.linalg.norm(Eout))
    return amp, amp ** 2


def unpolarized_through_polarizer(n=200_000, seed=0):
    """Average transmitted fraction for random input polarisation angles (-> 1/2)."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, np.pi, n)
    return float(np.mean(np.cos(th) ** 2))


# ── 4. polygon of the polarisation ellipse ───────────────────────────

def ellipse_polygon(Ex0, Ey0, delta, n=64):
    """Vertices of the E-field tip over one period: Ex = Ex0 cos wt, Ey = Ey0 cos(wt + delta)."""
    if n < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    wt = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([Ex0 * np.cos(wt), Ey0 * np.cos(wt + delta)])


def shoelace_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ellipse_area_symbolic():
    """Area enclosed by (Ex0 cos t, Ey0 cos(t + delta)) = pi Ex0 Ey0 |sin delta| (Green's theorem)."""
    t, a, b, dl = sp.symbols("t E_x0 E_y0 delta", positive=True)
    x, y = a * sp.cos(t), b * sp.cos(t + dl)
    area = sp.Rational(1, 2) * sp.integrate(x * sp.diff(y, t) - y * sp.diff(x, t), (t, 0, 2 * sp.pi))
    return sp.Abs(sp.simplify(area))            # sign = sense of rotation (handedness)


IEEE_CITATION = (
    'R. D. Knight, "Foundations of Modern Physics," ch. 37 in Physics for Scientists and '
    "Engineers: A Strategic Approach with Modern Physics, 5th ed. Hoboken, NJ, USA: Pearson, 2022."
)
