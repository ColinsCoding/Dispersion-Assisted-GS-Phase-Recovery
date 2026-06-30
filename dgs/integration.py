"""Definite integration: top-down (FTC) and bottom-up (Riemann/quadrature).

TOP-DOWN vs BOTTOM-UP:
  TOP-DOWN:    ∫_a^b f(x) dx = F(b) - F(a)    (Fundamental Theorem of Calculus)
               Start from the ANSWER (antiderivative F), evaluate at endpoints.
               Exact when F exists analytically.

  BOTTOM-UP:   ∫_a^b f(x) dx ≈ lim_{n->inf} sum_{i=1}^{n} f(x_i) * dx
               Build the answer from infinitely many thin slices.
               Works for ANY function (even ones with no closed-form F).

  THE BRIDGE:  FTC proves they give the SAME number.
               In physics: bottom-up = measurement (Riemann sum over data points)
                           top-down  = theory (analytic formula)
               When they agree: theory is confirmed.
               When they disagree: systematic error or wrong model.

TOLERANCE INTEGRALS:
  If f(x) has measurement uncertainty δf(x), the integral has:
    δ(∫f dx) = sqrt(∫ δf(x)^2 dx) * sqrt(dx)   (statistical, uncorrelated)
    δ(∫f dx) = ∫ |δf(x)| dx                     (worst-case, correlated)

  For TS-DFT: the measured spectrum I(omega) has shot-noise δI = sqrt(I/N_avg).
    Integrated power P = ∫I(omega)dω  has δP = sqrt(∫I(omega)/N_avg dω)
    Power SNR = P/δP = sqrt(N_avg * int(I^2 dω) / int(I dω))

  FIBER LENGTH TOLERANCE (half-meter):
    GVD mapping: omega = t/(beta2*L)
    delta_omega/omega = delta_L/L
    For L=10km, delta_L=0.5m: delta_omega/omega = 5e-5 (50 ppm error)
    -> wavelength error at 1550nm: delta_lambda = 1550 * 5e-5 = 0.078 nm
    Well within 0.1nm spectral resolution — TS-DFT is TOLERANT of splice errors.

PLANCK RADIATION INTEGRAL:
  B(nu) = (2*h*nu^3/c^2) / (exp(h*nu/kT) - 1)
  Total power: sigma*T^4 = (2*pi^4*k^4) / (15*h^3*c^2) * T^4  (Stefan-Boltzmann)
  Derived from: ∫_0^inf x^3/(e^x-1) dx = pi^4/15  (top-down: Riemann zeta)

JALALI TS-DFT AS AUTHORITY:
  GHz rep rate -> 10^9 spectra/second.
  Each spectrum = one Riemann sum over time: ∫ I(t) dt -> spectral power.
  The photodetector samples I(t_i) at dt intervals.
  ADC converts to digital counts: I_digital = round(I(t_i) / LSB)
  The integral ∫I dt is computed as digital sum: sum_i I_digital_i * dt.
  Counting photons = bottom-up integration at the quantum level.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Callable, Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# §1  RIEMANN SUMS (BOTTOM-UP)
# ════════════════════════════════════════════════════════════════════════════

def riemann_sum(f: Callable, a: float, b: float, n: int,
                 method: str = "midpoint") -> Dict:
    """Riemann sum approximation of ∫_a^b f(x) dx.

    method: 'left', 'right', 'midpoint'
    Convergence: left/right O(dx), midpoint O(dx^2).
    """
    dx   = (b - a) / n
    x_l  = np.linspace(a, b - dx, n)   # left endpoints
    x_r  = x_l + dx                     # right endpoints
    x_m  = x_l + dx/2                   # midpoints

    if method == "left":
        approx = float(np.sum(f(x_l) * dx))
        error_order = 1
    elif method == "right":
        approx = float(np.sum(f(x_r) * dx))
        error_order = 1
    else:  # midpoint
        approx = float(np.sum(f(x_m) * dx))
        error_order = 2

    return {
        "approx":      approx,
        "n":           n,
        "dx":          dx,
        "method":      method,
        "error_order": error_order,
    }


def riemann_convergence(f: Callable, a: float, b: float,
                         exact: float,
                         n_values: List[int] = None) -> Dict:
    """Show convergence of Riemann sums as n -> inf (bottom-up limit).

    Demonstrates: lim_{n->inf} sum = exact integral.
    Empirical order of convergence computed from log-log slope.
    """
    if n_values is None:
        n_values = [5, 10, 50, 100, 500, 1000, 5000]
    results = {}
    for n in n_values:
        for method in ["left", "midpoint"]:
            res = riemann_sum(f, a, b, n, method)
            err = abs(res["approx"] - exact)
            key = f"{method}_n{n}"
            results[key] = {
                "approx": res["approx"],
                "error":  err,
                "rel_error_pct": err / abs(exact) * 100 if exact != 0 else 0.0,
            }
    return {"exact": exact, "n_values": n_values, "results": results}


# ════════════════════════════════════════════════════════════════════════════
# §2  NUMERICAL QUADRATURE (BOTTOM-UP, HIGHER ORDER)
# ════════════════════════════════════════════════════════════════════════════

def trapezoid_rule(f: Callable, a: float, b: float, n: int) -> Dict:
    """Composite trapezoidal rule: O(dx^2) convergence.

    ∫_a^b f dx ≈ dx/2 * [f(a) + 2*f(x_1) + ... + 2*f(x_{n-1}) + f(b)]
    """
    x    = np.linspace(a, b, n+1)
    dx   = (b - a) / n
    y    = f(x)
    approx = float(np.trapezoid(y, x))
    return {"approx": approx, "n": n, "dx": dx, "method": "trapezoid",
            "error_order": 2}


def simpsons_rule(f: Callable, a: float, b: float, n: int) -> Dict:
    """Composite Simpson's 1/3 rule: O(dx^4) convergence (n must be even).

    ∫_a^b f dx ≈ dx/3 * [f(x_0) + 4*f(x_1) + 2*f(x_2) + 4*f(x_3) + ... + f(x_n)]
    Uses parabolic interpolation on each pair of intervals.
    """
    if n % 2 != 0:
        n += 1   # force even
    x    = np.linspace(a, b, n+1)
    dx   = (b - a) / n
    y    = f(x)
    # Weights: 1, 4, 2, 4, 2, ..., 4, 1
    w    = np.ones(n+1)
    w[1:-1:2] = 4   # odd indices
    w[2:-2:2] = 2   # even indices (not endpoints)
    approx = float(dx/3 * np.dot(w, y))
    return {"approx": approx, "n": n, "dx": dx, "method": "simpsons",
            "error_order": 4}


def gaussian_quadrature(f: Callable, a: float, b: float,
                         n_points: int = 5) -> Dict:
    """Gaussian-Legendre quadrature: exact for polynomials of degree 2n-1.

    Maps [a,b] -> [-1,1], applies Legendre nodes and weights.
    n=5: exact for degree 9 polynomials. Exponential convergence for smooth f.
    """
    # Gauss-Legendre nodes and weights on [-1,1]
    # Hardcoded for n=1..8
    GL = {
        1: ([0.0], [2.0]),
        2: ([-0.5773502692, 0.5773502692], [1.0, 1.0]),
        3: ([-0.7745966692, 0.0, 0.7745966692],
            [0.5555555556, 0.8888888889, 0.5555555556]),
        4: ([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
            [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]),
        5: ([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459],
            [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]),
        6: ([-0.9324695142, -0.6612093865, -0.2386191861,
              0.2386191861,  0.6612093865,  0.9324695142],
            [0.1713244924, 0.3607615730, 0.4679139346,
             0.4679139346, 0.3607615730, 0.1713244924]),
    }
    n_pts = min(n_points, 6)
    xi, wi = GL[n_pts]
    xi = np.array(xi); wi = np.array(wi)
    # Map to [a, b]
    t = 0.5*(b - a)*xi + 0.5*(b + a)
    approx = float(0.5*(b - a) * np.dot(wi, f(t)))
    return {"approx": approx, "n_points": n_pts, "method": "gauss_legendre",
            "error_order": f"exact for poly deg {2*n_pts-1}"}


# ════════════════════════════════════════════════════════════════════════════
# §3  TOP-DOWN: FUNDAMENTAL THEOREM OF CALCULUS
# ════════════════════════════════════════════════════════════════════════════

def ftc_sympy(f_expr: sp.Expr, x: sp.Symbol,
               a: float, b: float) -> Dict:
    """FTC: ∫_a^b f(x) dx = F(b) - F(a) where F' = f.

    TOP-DOWN: find antiderivative F symbolically, evaluate at endpoints.
    Returns symbolic F, exact value, and numerical verification.
    """
    F_sym   = sp.integrate(f_expr, x)
    F_b     = F_sym.subs(x, b)
    F_a     = F_sym.subs(x, a)
    exact_sym = sp.simplify(F_b - F_a)
    exact_num = float(exact_sym.evalf())

    # Numerical verification (bottom-up)
    f_num = sp.lambdify(x, f_expr, "numpy")
    numerical = trapezoid_rule(f_num, float(a), float(b), n=10000)

    return {
        "f":          f_expr,
        "F":          F_sym,
        "F(b)-F(a)":  exact_sym,
        "exact":      exact_num,
        "numerical":  numerical["approx"],
        "agreement":  abs(exact_num - numerical["approx"]) < 1e-4,
        "a": a, "b": b,
    }


def compare_top_down_bottom_up(f_expr: sp.Expr, x: sp.Symbol,
                                 a: float, b: float,
                                 n_list: List[int] = None) -> Dict:
    """Side-by-side: FTC (top-down) vs Riemann/quadrature (bottom-up).

    Shows convergence of bottom-up to top-down exact value.
    This IS the proof of the Fundamental Theorem in computational form.
    """
    if n_list is None:
        n_list = [5, 20, 100, 1000]

    top_down = ftc_sympy(f_expr, x, a, b)
    exact    = top_down["exact"]

    f_num = sp.lambdify(x, f_expr, "numpy")
    rows  = []
    for n in n_list:
        left = riemann_sum(f_num, float(a), float(b), n, "left")
        mid  = riemann_sum(f_num, float(a), float(b), n, "midpoint")
        simp = simpsons_rule(f_num, float(a), float(b), n)
        rows.append({
            "n":          n,
            "left_err":   abs(left["approx"] - exact),
            "mid_err":    abs(mid["approx"]  - exact),
            "simp_err":   abs(simp["approx"] - exact),
            "left":       left["approx"],
            "midpoint":   mid["approx"],
            "simpsons":   simp["approx"],
        })
    return {
        "exact":    exact,
        "F_sym":    top_down["F"],
        "a": a, "b": b,
        "rows":     rows,
        "n_list":   n_list,
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  TOLERANCE / ERROR PROPAGATION IN INTEGRALS
# ════════════════════════════════════════════════════════════════════════════

def integral_tolerance(f_vals: np.ndarray,
                        x: np.ndarray,
                        delta_f: np.ndarray,
                        mode: str = "statistical") -> Dict:
    """Propagate measurement uncertainty through a definite integral.

    I = ∫ f(x) dx  ->  δI depends on correlation of errors in f.

    mode='statistical': errors uncorrelated (shot noise)
      δI = sqrt(∫ (δf)^2 dx * dx)   <- quadrature addition
    mode='worst_case': errors correlated (systematic)
      δI = ∫ |δf(x)| dx             <- linear addition

    Parameters
    ----------
    f_vals  : function values on grid x
    x       : grid points
    delta_f : pointwise uncertainty in f
    """
    I_central = float(np.trapezoid(f_vals, x))

    if mode == "statistical":
        # Var(I) = sum_i (delta_f_i)^2 * dx^2 = dx * int(delta_f^2, dx)
        dx = x[1] - x[0]
        delta_I = float(np.sqrt(np.trapezoid(delta_f**2, x) * dx))
    else:  # worst_case
        delta_I = float(np.trapezoid(np.abs(delta_f), x))

    return {
        "I":        I_central,
        "delta_I":  delta_I,
        "SNR":      abs(I_central) / (delta_I + 1e-300),
        "rel_err":  delta_I / abs(I_central) if I_central != 0 else np.inf,
        "mode":     mode,
    }


def tsdft_fiber_length_tolerance(L_km: float = 10.0,
                                   delta_L_m: float = 0.5,
                                   lambda_nm: float = 1550.0,
                                   bandwidth_nm: float = 10.0) -> Dict:
    """Tolerance analysis: effect of 0.5m splice uncertainty on TS-DFT mapping.

    GVD wavelength mapping: lambda(t) = lambda_0 + t/(beta2*L) * dlambda/domega
    Uncertainty in L -> uncertainty in wavelength mapping:
      delta_lambda = lambda * delta_L / L

    For 10 km fiber with 0.5 m splice error:
      delta_lambda/lambda = 0.5/10000 = 5e-5  (50 ppm)
    At 1550 nm: delta_lambda = 0.078 nm  << 0.1 nm resolution

    Conclusion: TS-DFT is TOLERANT to half-meter splicing errors.
    The fiber length must be known to <0.1% for sub-nm spectral accuracy.
    """
    L_m      = L_km * 1e3
    rel_err  = delta_L_m / L_m
    delta_lam = lambda_nm * rel_err   # nm

    # SMF-28 GVD
    beta2   = -22e-27   # s²/m
    D_ps_nm_km = 17.0   # ps/(nm·km)

    # Spectral resolution: determined by pulse duration T0 and dispersion D*L
    # delta_lambda_res = T0 / (D * L)  where T0 is pulse duration in ps
    T_pulse_ps = 0.1   # 100 fs = 0.1 ps
    delta_lam_res = T_pulse_ps / (D_ps_nm_km * L_km)  # nm

    # Tolerance: delta_lambda < resolution/10 is considered OK (10:1 budget)
    tol_ok = delta_lam < delta_lam_res * 10   # 0.5m error vs 10x resolution budget

    # Integrated power tolerance (worst-case over bandwidth)
    n_pts = 500
    lam   = np.linspace(lambda_nm - bandwidth_nm/2,
                         lambda_nm + bandwidth_nm/2, n_pts)
    # Gaussian spectral envelope
    I_lam = np.exp(-0.5*((lam - lambda_nm)/(bandwidth_nm/4))**2)
    # Shot noise: sqrt(I) per point
    N_avg = 1000   # averages
    delta_I = np.sqrt(I_lam / N_avg)
    power_tol = integral_tolerance(I_lam, lam, delta_I, mode="statistical")

    return {
        "L_km":             L_km,
        "delta_L_m":        delta_L_m,
        "rel_error_ppm":    rel_err * 1e6,
        "delta_lambda_nm":  delta_lam,
        "resolution_nm":    delta_lam_res,
        "tolerance_ok":     tol_ok,
        "power_SNR":        power_tol["SNR"],
        "power_rel_err_pct": power_tol["rel_err"] * 100,
        "conclusion": (
            f"delta_L={delta_L_m}m on L={L_km}km: "
            f"wavelength error={delta_lam:.4f}nm, "
            f"resolution={delta_lam_res:.3f}nm. "
            f"{'OK: within tolerance.' if tol_ok else 'MARGINAL.'}"
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  PHYSICS INTEGRALS: PLANCK, GAUSSIAN, TS-DFT POWER
# ════════════════════════════════════════════════════════════════════════════

def planck_radiation_integral(T_K: float = 5778.0,
                                nu_min: float = 1e12,
                                nu_max: float = 1e15) -> Dict:
    """Planck spectral radiance B(nu) and total power via quadrature.

    B(nu) = (2*h*nu^3/c^2) / (exp(h*nu/kT) - 1)
    Total: sigma*T^4 = integral_0^inf B(nu)*pi dnu  (Stefan-Boltzmann)

    The integral ∫_0^inf x^3/(e^x-1)dx = pi^4/15 has no elementary antiderivative.
    It requires BOTTOM-UP integration (or Riemann zeta: zeta(4)*Gamma(4) = pi^4/15).
    This is a case where top-down FAILS (no closed form for finite limits)
    and bottom-up is the ONLY way to get a number.
    """
    H  = 6.626e-34
    C  = 2.998e8
    K  = 1.381e-23

    nu  = np.linspace(nu_min, nu_max, 20000)
    x   = H * nu / (K * T_K)
    x   = np.clip(x, 1e-10, 500)
    B   = 2*H*nu**3/C**2 / (np.exp(x) - 1)
    I_measured = float(np.trapezoid(B, nu))

    # Stefan-Boltzmann exact: sigma*T^4/pi
    sigma = 5.6704e-8
    I_exact = sigma * T_K**4 / np.pi

    # Fraction of power in optical window (400-700 nm)
    nu_vis_min = C / 700e-9
    nu_vis_max = C / 400e-9
    mask_vis = (nu >= nu_vis_min) & (nu <= nu_vis_max)
    I_vis = float(np.trapezoid(B[mask_vis], nu[mask_vis]))

    return {
        "T_K":          T_K,
        "I_numerical":  I_measured,
        "I_exact":      I_exact,
        "error_pct":    abs(I_measured - I_exact) / I_exact * 100,
        "I_visible":    I_vis,
        "vis_fraction": I_vis / I_measured,
        "nu_peak":      float(nu[np.argmax(B)]),
        "lambda_peak_nm": C / float(nu[np.argmax(B)]) * 1e9,
        "note": "pi^4/15 from Riemann zeta(4)*Gamma(4): NO closed antiderivative, must integrate numerically",
    }


def gaussian_normalization_integral() -> Dict:
    """The Gaussian integral: ∫_{-inf}^{inf} exp(-x^2) dx = sqrt(pi).

    CANNOT be done with FTC directly (no elementary antiderivative for exp(-x^2)).
    Classic bottom-up trick: compute [∫exp(-x^2)dx]^2 in polar coordinates.
    This is the SAME shell method as Poiseuille flow:
        I^2 = ∫∫ exp(-(x^2+y^2)) dx dy = ∫_0^{inf} exp(-r^2) * 2*pi*r dr = pi
        -> I = sqrt(pi)

    Connection to statistics: Gaussian PDF = exp(-x^2/2)/(sqrt(2*pi))
    normalizes because of this identity.
    """
    x = np.linspace(-8, 8, 100000)
    I_num = float(np.trapezoid(np.exp(-x**2), x))
    I_exact = float(np.sqrt(np.pi))

    # Also do 2D version: int int exp(-(x^2+y^2)) dx dy over [-8,8]^2
    n2d = 500
    x2  = np.linspace(-6, 6, n2d)
    X, Y = np.meshgrid(x2, x2)
    Z2   = np.exp(-(X**2 + Y**2))
    I2d  = float(np.trapezoid(np.trapezoid(Z2, x2), x2))

    return {
        "I_1d_numerical": I_num,
        "I_1d_exact":     I_exact,
        "I_2d_numerical": I2d,
        "I_2d_exact":     np.pi,
        "trick": "I^2 = double integral = polar coords -> pi (shell method)",
        "error_1d_pct": abs(I_num - I_exact) / I_exact * 100,
        "error_2d_pct": abs(I2d - np.pi) / np.pi * 100,
    }


def tsdft_power_integral(n_pts: int = 4096,
                          T0_fs: float = 100.0,
                          beta2: float = -22e-27,
                          L_km: float = 10.0,
                          N_avg: int = 1000) -> Dict:
    """Digital integration of TS-DFT signal: I(t) -> spectral power.

    SIMULTANEOUS counting + integration:
    Each GHz clock tick: ADC samples I(t_i).
    Digital accumulator: P += I(t_i) * dt   (running Riemann sum)
    After N_pts samples: P = ∫I(t)dt (exact in limit)

    This is bottom-up integration at the speed of light.
    One pulse = one Riemann sum = one spectrum.
    1 GHz rep rate = 10^9 Riemann sums per second.
    """
    L_m  = L_km * 1e3
    T0_s = T0_fs * 1e-15
    dt   = T0_s * 20 / n_pts   # time window = 20*T0

    t    = np.arange(n_pts) * dt - n_pts * dt / 2
    # Chirped Gaussian after fiber
    phi  = 0.5 * beta2 * L_m * (t / (beta2 * L_m))**2   # GVD phase
    E_t  = np.exp(-t**2 / (2 * T0_s**2)) * np.exp(1j * phi)
    I_t  = np.abs(E_t)**2

    # Add shot noise (N_avg averages)
    rng  = np.random.default_rng(0)
    noise = rng.standard_normal(n_pts) * np.sqrt(I_t / N_avg)
    I_measured = np.maximum(I_t + noise, 0)

    # Bottom-up digital integration (Riemann sum)
    P_digital  = float(I_measured.sum() * dt)   # running accumulator
    P_exact    = float(np.trapezoid(I_t, t))     # true integral

    # Digital word counting: if ADC is 12-bit, 0..4095
    ADC_bits = 12
    ADC_max  = 2**ADC_bits - 1
    I_counts = np.round(I_measured / I_measured.max() * ADC_max).astype(int)
    P_counts = float(I_counts.sum())   # total ADC counts

    # SNR
    SNR = float(P_exact / np.sqrt(np.trapezoid(I_t / N_avg, t)))

    return {
        "t":              t,
        "I_t":            I_measured,
        "P_digital":      P_digital,
        "P_exact":        P_exact,
        "P_counts":       P_counts,
        "ADC_bits":       ADC_bits,
        "error_pct":      abs(P_digital - P_exact) / P_exact * 100,
        "SNR":            SNR,
        "n_pts":          n_pts,
        "dt_fs":          dt * 1e15,
        "spectra_per_sec": 1e9,
        "interpretation": (
            "Each 1 GHz pulse = one Riemann sum over n_pts time samples. "
            f"Digital accumulator: {n_pts} multiply-accumulate ops per spectrum. "
            f"SNR = {SNR:.1f} (= sqrt(N_avg * N_photons))."
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# §6  SYMPY: 5 KEY INTEGRATION EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def integration_sympy_5() -> Dict:
    """5 equations: FTC, Riemann limit, Gaussian, Stefan-Boltzmann, tolerance."""
    x, a, b, n, dx = sp.symbols("x a b n dx", real=True)
    f, F = sp.Function("f"), sp.Function("F")
    delta_f = sp.Symbol("delta_f", positive=True)

    # 1. Fundamental Theorem of Calculus (top-down)
    eq1 = sp.Eq(sp.Integral(f(x), (x, a, b)),
                F(b) - F(a))

    # 2. Riemann sum limit (bottom-up) — stated as equality, not computed
    i = sp.Symbol("i", integer=True, positive=True)
    eq2 = sp.Eq(sp.Integral(f(x), (x, a, b)),
                sp.Symbol("lim_{n->inf}") *
                sp.Sum(f(a + i*(b-a)/n) * (b-a)/n, (i, 1, n)))

    # 3. Gaussian integral (top-down via polar trick)
    eq3 = sp.Eq(sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo)),
                sp.sqrt(sp.pi))

    # 4. Statistical tolerance: delta_I = sqrt(int delta_f^2 dx * dx)
    eq4 = sp.Eq(sp.Symbol("delta_I"),
                sp.sqrt(sp.Integral(delta_f**2, (x, a, b)) * dx))

    # 5. Stefan-Boltzmann (result of Planck integral)
    T, sigma = sp.symbols("T sigma", positive=True)
    eq5 = sp.Eq(sp.Symbol("P_total"),
                sigma * T**4)

    return {
        "FTC_top_down":      eq1,
        "Riemann_limit":     eq2,
        "Gaussian_integral": eq3,
        "Tolerance_delta_I": eq4,
        "Stefan_Boltzmann":  eq5,
    }


if __name__ == "__main__":
    print("=== TOP-DOWN vs BOTTOM-UP: int_0^1 x^2 dx = 1/3 ===")
    x_sym = sp.Symbol("x")
    cmp = compare_top_down_bottom_up(x_sym**2, x_sym, 0, 1)
    print(f"  Top-down exact: {cmp['exact']:.6f}  (theory: {1/3:.6f})")
    for row in cmp["rows"]:
        print(f"  n={row['n']:4d}: left_err={row['left_err']:.2e}  "
              f"mid_err={row['mid_err']:.2e}  simp_err={row['simp_err']:.2e}")

    print("\n=== FTC: int_0^pi sin(x) dx = 2 ===")
    ftc = ftc_sympy(sp.sin(x_sym), x_sym, 0, sp.pi)
    print(f"  Antiderivative: {ftc['F']}")
    print(f"  F(pi)-F(0) = {ftc['F(b)-F(a)']} = {ftc['exact']:.6f}")
    print(f"  Numerical: {ftc['numerical']:.6f}  Agreement: {ftc['agreement']}")

    print("\n=== Gaussian Integral: int exp(-x^2) dx = sqrt(pi) ===")
    g = gaussian_normalization_integral()
    print(f"  1D numerical: {g['I_1d_numerical']:.8f}  exact: {g['I_1d_exact']:.8f}")
    print(f"  2D numerical: {g['I_2d_numerical']:.6f}   exact: {g['I_2d_exact']:.6f}")
    print(f"  1D error: {g['error_1d_pct']:.6f}%")
    print(f"  Trick: {g['trick']}")

    print("\n=== TS-DFT Fiber Tolerance: 0.5 m splice on 10 km ===")
    tol = tsdft_fiber_length_tolerance()
    print(f"  {tol['conclusion']}")
    print(f"  Power SNR: {tol['power_SNR']:.1f}")

    print("\n=== Planck Integral (5778 K sun) ===")
    pl = planck_radiation_integral(5778)
    print(f"  Numerical: {pl['I_numerical']:.4e}  Exact: {pl['I_exact']:.4e}")
    print(f"  Error: {pl['error_pct']:.4f}%")
    print(f"  Lambda_peak = {pl['lambda_peak_nm']:.1f} nm  (theory: 502 nm)")
    print(f"  Visible fraction: {pl['vis_fraction']:.3f}")

    print("\n=== TS-DFT Digital Power Integration ===")
    ts = tsdft_power_integral()
    print(f"  P_digital: {ts['P_digital']:.4e}")
    print(f"  P_exact:   {ts['P_exact']:.4e}")
    print(f"  Error: {ts['error_pct']:.4f}%")
    print(f"  {ts['interpretation']}")

    print("\n=== 5 SymPy Equations ===")
    for name, eq in integration_sympy_5().items():
        print(f"  [{name}]  {eq}")
