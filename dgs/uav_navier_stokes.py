"""Navier-Stokes with SymPy, applied to a delivery UAV -- plus how to audit an
AI model's flow prediction, plus the link to Jalali-lab dispersion.

Incompressible Navier-Stokes (2-D, constant rho, mu; nu = mu/rho):
    div u = u_x + v_y = 0
    u_t + u u_x + v u_y = -p_x/rho + nu (u_xx + u_yy)
    v_t + u v_x + v v_y = -p_y/rho + nu (v_xx + v_yy)

Sections
  1. ns_residuals         -- SymPy residuals of any candidate (u, v, p)
     nondimensional_ns    -- substitute x = L x*, u = U u*, ... -> only Re survives
  2. Exact solutions, verified symbolically: Couette, plane Poiseuille,
     Stokes' first problem (impulsive plate), Taylor-Green vortex.
     blasius             -- boundary-layer ODE f''' + f f''/2 = 0 by shooting.
  3. UAV: actuator-disk (momentum) theory derived in SymPy; hover and Glauert
     forward-flight power; energy per mile vs. a sidewalk robot.
  4. "AI result" audit: fit a polynomial surrogate to Taylor-Green data, then
     check the physics residual it leaves behind. Adding a continuity penalty
     (linear in the coefficients) is a least-squares physics-informed fit.
  5. Applied EM link: the viscous diffusion Gaussian and the dispersed optical
     Gaussian pulse are the same function with  2 nu t  <->  -i beta2 z.
     That imaginary "viscosity" is the time-stretch dispersion H(f)=exp(j pi D f^2).
"""
from itertools import product

import numpy as np
import sympy as sp

x, y, t = sp.symbols("x y t", real=True)
rho, mu, nu = sp.symbols("rho mu nu", positive=True)


# ── 1. residuals and nondimensionalisation ───────────────────────────

def ns_residuals(u, v, p, nu_=nu, rho_=rho):
    """Return (continuity, x-momentum, y-momentum) residuals, simplified.

    u, v, p are SymPy expressions in x, y, t. An exact solution gives (0, 0, 0).
    """
    cont = sp.diff(u, x) + sp.diff(v, y)
    mx = (sp.diff(u, t) + u * sp.diff(u, x) + v * sp.diff(u, y)
          + sp.diff(p, x) / rho_ - nu_ * (sp.diff(u, x, 2) + sp.diff(u, y, 2)))
    my = (sp.diff(v, t) + u * sp.diff(v, x) + v * sp.diff(v, y)
          + sp.diff(p, y) / rho_ - nu_ * (sp.diff(v, x, 2) + sp.diff(v, y, 2)))
    return tuple(sp.simplify(r) for r in (cont, mx, my))


def nondimensional_ns():
    """Scale x-momentum by L, U, L/U, rho U^2 and show only 1/Re remains.

    Returns (scaled_equation_lhs, Re_symbol). The returned expression is the
    x-momentum residual in starred variables divided by U^2/L.
    """
    L, U = sp.symbols("L U", positive=True)
    xs, ys, ts = sp.symbols("x_s y_s t_s", real=True)
    us, vs, ps = [sp.Function(n)(xs, ys, ts) for n in ("u_s", "v_s", "p_s")]
    Re = sp.Symbol("Re", positive=True)
    # dimensional fields written via starred ones: u = U u*(x/L, y/L, tU/L)
    sub = {x: L * xs, y: L * ys, t: L / U * ts}
    d_x = lambda f: sp.diff(f, xs) / L
    d_y = lambda f: sp.diff(f, ys) / L
    d_t = lambda f: sp.diff(f, ts) * U / L
    u_, v_, p_ = U * us, U * vs, rho * U ** 2 * ps
    mx = (d_t(u_) + u_ * d_x(u_) + v_ * d_y(u_) + d_x(p_) / rho
          - (mu / rho) * (d_x(d_x(u_)) + d_y(d_y(u_))))
    scaled = sp.expand(mx / (U ** 2 / L))
    scaled = scaled.subs(mu, rho * U * L / Re)
    return sp.expand(scaled), Re


def reynolds(U, L, nu_value=1.5e-5):
    """Re = U L / nu  (default nu: air at ~20 C, m^2/s)."""
    U, L = np.asarray(U, dtype=float), np.asarray(L, dtype=float)
    if np.any(U < 0) or np.any(L <= 0) or nu_value <= 0:
        raise ValueError("U >= 0, L > 0, nu > 0 required")
    return U * L / nu_value


# ── 2. exact solutions ───────────────────────────────────────────────

def couette(U=sp.Symbol("U"), h=sp.Symbol("h", positive=True)):
    """Plate at y = h moving at U, plate at y = 0 fixed, no pressure gradient."""
    return U * y / h, sp.Integer(0), sp.Integer(0)


def plane_poiseuille(G=sp.Symbol("G", positive=True),
                     h=sp.Symbol("h", positive=True)):
    """Channel |y| < h driven by dp/dx = -G:  u = G (h^2 - y^2) / (2 mu).

    Written with mu = rho nu so ns_residuals (which uses nu and rho) sees one
    consistent viscosity.
    """
    return G * (h ** 2 - y ** 2) / (2 * rho * nu), sp.Integer(0), -G * x


def stokes_first_problem(U=sp.Symbol("U")):
    """Plate impulsively set moving at U:  u = U erfc(y / (2 sqrt(nu t)))."""
    return U * sp.erfc(y / (2 * sp.sqrt(nu * t))), sp.Integer(0), sp.Integer(0)


def taylor_green(nu_=nu, rho_=rho):
    """Decaying vortex array, an exact NONLINEAR solution.

    u =  sin x cos y F,  v = -cos x sin y F,  p = rho/4 (cos 2x + cos 2y) F^2,
    F = exp(-2 nu t).
    """
    F = sp.exp(-2 * nu_ * t)
    return (sp.sin(x) * sp.cos(y) * F, -sp.cos(x) * sp.sin(y) * F,
            rho_ / 4 * (sp.cos(2 * x) + sp.cos(2 * y)) * F ** 2)


def blasius(eta_max=10.0, n=2001):
    """Solve f''' + f f''/2 = 0, f(0)=f'(0)=0, f'(inf)=1 by shooting on f''(0).

    Returns dict with eta, f, fp (= u/U), fpp0, and delta99 (eta at u/U = 0.99).
    In dimensional form  delta99 = delta99_eta * x / sqrt(Re_x).
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    rhs = lambda e, s: [s[1], s[2], -0.5 * s[0] * s[2]]

    def miss(a):
        sol = solve_ivp(rhs, (0, eta_max), [0, 0, a], rtol=1e-10, atol=1e-12)
        return sol.y[1, -1] - 1.0

    a = brentq(miss, 0.1, 1.0, xtol=1e-12)
    eta = np.linspace(0, eta_max, n)
    sol = solve_ivp(rhs, (0, eta_max), [0, 0, a], t_eval=eta, rtol=1e-10, atol=1e-12)
    fp = sol.y[1]
    k = np.argmax(fp >= 0.99)
    d99 = np.interp(0.99, fp[k - 1:k + 1], eta[k - 1:k + 1])
    return {"eta": eta, "f": sol.y[0], "fp": fp, "fpp0": a, "delta99": d99}


# ── 3. UAV rotor: actuator-disk theory ──────────────────────────────

def actuator_disk_derivation():
    """Momentum theory for a hovering rotor, derived symbolically.

    Control volume around the slipstream (steady, inviscid -- the integral form
    of NS with mu -> 0 outside the thin blade boundary layers):
        mass:      mdot = rho A v_i
        momentum:  T    = mdot w            (w = far-wake velocity)
        energy:    P    = mdot w^2 / 2
        and P = T v_i (power = thrust x velocity at the disk)
    Solving gives w = 2 v_i, v_i = sqrt(T / (2 rho A)), P = T^(3/2) / sqrt(2 rho A).
    """
    T, A, vi, w = sp.symbols("T A v_i w", positive=True)
    mdot = rho * A * vi
    eqs = [sp.Eq(T, mdot * w), sp.Eq(T * vi, mdot * w ** 2 / 2)]
    sol = sp.solve(eqs, [w, vi], dict=True)[0]
    P = sp.simplify((T * vi).subs(sol))
    return {"w": sp.simplify(sol[w]), "v_i": sp.simplify(sol[vi]), "P": P}


def hover_power(mass_kg, rotor_diam_m, n_rotors=4, rho_air=1.225,
                figure_of_merit=0.7):
    """Shaft power (W) to hover: ideal actuator-disk power / figure of merit."""
    if min(mass_kg, rotor_diam_m, rho_air) <= 0 or n_rotors < 1:
        raise ValueError("mass, diameter, rho positive; n_rotors >= 1")
    if not 0 < figure_of_merit <= 1:
        raise ValueError("figure_of_merit must be in (0, 1]")
    T = mass_kg * 9.81 / n_rotors
    A = np.pi * rotor_diam_m ** 2 / 4
    return n_rotors * T ** 1.5 / np.sqrt(2 * rho_air * A) / figure_of_merit


def forward_flight_power(V, mass_kg, rotor_diam_m, n_rotors=4, rho_air=1.225,
                         figure_of_merit=0.7, CdA=0.05):
    """Shaft power (W) in level forward flight at airspeed V (m/s).

    Glauert's induced-velocity relation v_i = v_h^2 / sqrt(V^2 + v_i^2) (edgewise
    rotor, small tilt), solved by fixed point, plus parasite drag 1/2 rho CdA V^3.
    At V = 0 this reduces exactly to hover_power.
    """
    if V < 0 or CdA < 0:
        raise ValueError("V and CdA must be non-negative")
    T = mass_kg * 9.81 / n_rotors
    A = np.pi * rotor_diam_m ** 2 / 4
    vh = np.sqrt(T / (2 * rho_air * A))
    vi = vh
    for _ in range(200):
        vi_new = vh ** 2 / np.sqrt(V ** 2 + vi ** 2)
        if abs(vi_new - vi) < 1e-12:
            break
        vi = 0.5 * (vi + vi_new)
    induced = n_rotors * T * vi / figure_of_merit
    return induced + 0.5 * rho_air * CdA * V ** 3


def energy_per_mile_wh(power_w, speed_ms, overhead_s=0.0, overhead_power_w=0.0):
    """Wh per mile of travel, plus fixed per-trip energy (e.g. takeoff/landing)."""
    if speed_ms <= 0:
        raise ValueError("speed_ms must be positive")
    return (power_w * 1609.344 / speed_ms + overhead_power_w * overhead_s) / 3600.0


# ── 4. auditing an "AI" flow prediction ──────────────────────────────

def _exponents(deg):
    return [(a, b, c) for a, b, c in product(range(deg + 1), repeat=3)
            if a + b + c <= deg]


def _basis(xi, eta, tau, exps, dx=0, dy=0, dt=0):
    """Matrix of d^(dx,dy,dt) of monomials xi^a eta^b tau^c at sample points."""
    cols = []
    for a, b, c in exps:
        col = np.ones_like(xi)
        for var, e, d in ((xi, a, dx), (eta, b, dy), (tau, c, dt)):
            if d > e:
                col = np.zeros_like(xi)
                break
            coef = np.prod(np.arange(e - d + 1, e + 1)) if d else 1.0
            col = col * coef * var ** (e - d)
        cols.append(col)
    return np.stack(cols, axis=1)


def taylor_green_numeric(X, Y, T, nu_value=0.1, rho_value=1.0):
    F = np.exp(-2 * nu_value * T)
    return (np.sin(X) * np.cos(Y) * F, -np.cos(X) * np.sin(Y) * F,
            rho_value / 4 * (np.cos(2 * X) + np.cos(2 * Y)) * F ** 2)


def fit_surrogate(deg=5, n_data=400, n_colloc=2000, lam_div=0.0, noise=0.0,
                  nu_value=0.1, rho_value=1.0, seed=0):
    """Least-squares polynomial surrogate of Taylor-Green on [0,pi]^2 x [0,2].

    lam_div > 0 adds rows  sqrt(lam_div) * (u_x + v_y) = 0  at collocation
    points -- a physics-informed penalty that is LINEAR in the coefficients,
    so the fit stays a single least-squares solve (u and v coefficients are
    solved jointly). Returns dict with data RMSE and physics residual RMS
    (continuity and both momentum equations) on held-out points.
    """
    if deg < 1 or n_data < 1 or n_colloc < 1:
        raise ValueError("deg, n_data, n_colloc must be positive")
    if lam_div < 0 or noise < 0:
        raise ValueError("lam_div and noise must be non-negative")
    rng = np.random.default_rng(seed)
    hx, cx, ht, ct = np.pi / 2, np.pi / 2, 1.0, 1.0      # x = cx + hx*xi etc.
    exps = _exponents(deg)
    m = len(exps)

    def sample(n):
        return (rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n),
                rng.uniform(0, 2, n))

    def scaled(X, Y, T):
        return (X - cx) / hx, (Y - cx) / hx, (T - ct) / ht

    Xd, Yd, Td = sample(n_data)
    ud, vd, pd = taylor_green_numeric(Xd, Yd, Td, nu_value, rho_value)
    ud, vd, pd = (f + noise * rng.standard_normal(n_data) for f in (ud, vd, pd))
    B = _basis(*scaled(Xd, Yd, Td), exps)
    Z = np.zeros_like(B)
    rows = [np.hstack([B, Z]), np.hstack([Z, B])]
    rhs = [ud, vd]
    if lam_div > 0:
        Xc, Yc, Tc = sample(n_colloc)
        s = scaled(Xc, Yc, Tc)
        Bx, By = _basis(*s, exps, dx=1) / hx, _basis(*s, exps, dy=1) / hx
        w = np.sqrt(lam_div * n_data / n_colloc)
        rows.append(w * np.hstack([Bx, By]))
        rhs.append(np.zeros(n_colloc))
    cuv = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    cu, cv = cuv[:m], cuv[m:]
    cp = np.linalg.lstsq(B, pd, rcond=None)[0]

    # held-out evaluation
    Xe, Ye, Te = sample(3000)
    s = scaled(Xe, Ye, Te)
    D = lambda c, **k: _basis(*s, exps, **k) @ c / (hx ** (k.get("dx", 0) + k.get("dy", 0))
                                                    * ht ** k.get("dt", 0))
    u, v, p = D(cu), D(cv), D(cp)
    ue, ve, pe = taylor_green_numeric(Xe, Ye, Te, nu_value, rho_value)
    ux, uy, ut = D(cu, dx=1), D(cu, dy=1), D(cu, dt=1)
    vx, vy, vt = D(cv, dx=1), D(cv, dy=1), D(cv, dt=1)
    lap_u, lap_v = D(cu, dx=2) + D(cu, dy=2), D(cv, dx=2) + D(cv, dy=2)
    px, py = D(cp, dx=1), D(cp, dy=1)
    cont = ux + vy
    mx = ut + u * ux + v * uy + px / rho_value - nu_value * lap_u
    my = vt + u * vx + v * vy + py / rho_value - nu_value * lap_v
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    return {
        "n_coeffs": 3 * m,
        "data_rmse": rms(np.concatenate([u - ue, v - ve])),
        "p_rmse": rms(p - pe),
        "continuity_rms": rms(cont),
        "momentum_rms": rms(np.concatenate([mx, my])),
        "coef_u": cu, "coef_v": cv, "coef_p": cp, "exps": exps,
    }


# ── 5. applied EM: dispersion is diffusion with imaginary viscosity ──

def diffusion_gaussian(sigma0=sp.Symbol("sigma_0", positive=True)):
    """Spreading shear layer: u_t = nu u_yy with Gaussian initial profile.

    u = sigma0 / sqrt(sigma0^2 + 2 nu t) * exp(-y^2 / (2 (sigma0^2 + 2 nu t)))
    """
    s2 = sigma0 ** 2 + 2 * nu * t
    return sigma0 / sp.sqrt(s2) * sp.exp(-y ** 2 / (2 * s2))


def dispersed_pulse(T0=sp.Symbol("T_0", positive=True)):
    """Gaussian pulse in a dispersive fiber (Agrawal sign convention):

        i A_z = (beta2 / 2) A_TT,
        A = T0 / sqrt(T0^2 - i beta2 z) * exp(-T^2 / (2 (T0^2 - i beta2 z))).

    Same function as diffusion_gaussian with  y -> T,  sigma0 -> T0,
    2 nu t -> -i beta2 z.  Returns (A, (z, T, beta2)).
    """
    z, T = sp.symbols("z T", real=True)
    beta2 = sp.Symbol("beta_2", real=True)
    q = T0 ** 2 - sp.I * beta2 * z
    return T0 / sp.sqrt(q) * sp.exp(-T ** 2 / (2 * q)), (z, T, beta2)


def check_analogy():
    """Return (diffusion residual, dispersion residual, substitution check)."""
    u = diffusion_gaussian()
    r_diff = sp.simplify(sp.diff(u, t) - nu * sp.diff(u, y, 2))
    A, (z, T, beta2) = dispersed_pulse()
    r_disp = sp.simplify(sp.I * sp.diff(A, z) - beta2 / 2 * sp.diff(A, T, 2))
    sigma0 = sp.Symbol("sigma_0", positive=True)
    T0 = sp.Symbol("T_0", positive=True)
    mapped = u.subs({y: T, sigma0: T0}).subs(nu * t, -sp.I * beta2 * z / 2)
    mapped = mapped.subs(t, -sp.I * beta2 * z / (2 * nu))
    same = sp.simplify(mapped - A)
    return r_diff, r_disp, same
