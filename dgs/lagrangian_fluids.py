"""Lagrangian coordinates, Burgers' equation and Cole-Hopf: the 1D (x, t)
skeleton of Navier-Stokes, worked symbolically and numerically.

    momentum conservation   d(rho u)/dt + d(rho u^2 + p - mu u_x)/dx = 0   (Griffiths 8.2 analogue)
    viscous Burgers         u_t + u u_x = nu u_xx                          (1D NS, no pressure)
    Cole-Hopf               u = -2 nu phi_x / phi,  phi_t = nu phi_xx      (homogeneous heat eq.)
    Lagrangian map          x = X + u0(X) t,  J = dx/dX = 1 + u0'(X) t     (J -> 0: shock)
    relativistic dust       u^mu d_mu u^nu = 0  =>  v_t + v v_x = 0        (same characteristics)
    Madelung                NLSE  <=>  compressible Euler + quantum pressure (optical fluid proxy)
    blood vs water          Reynolds and Womersley numbers, Womersley pulsatile profile
    respiratory droplets    Stokes settling speed and fall time in still air

The numerical Burgers solver here (central differences, conservative flux,
RK4, periodic) is the reference that the torch (dgs/torch/burgers_batched.py),
CUDA (hardware/cuda_burgers/burgers.cu) and MATLAB (learn/matlab/burgers_cole_hopf.m)
versions are checked against; all of them are checked against the exact
Cole-Hopf solution.

The physiological and aerosol numbers are textbook-typical values for a
physics exercise. They are not medical or public-health guidance.
"""
import numpy as np
import sympy as sp

x, t, X = sp.symbols("x t X", real=True)
nu = sp.symbols("nu", positive=True)
c = sp.symbols("c", positive=True)


# ── 1. Griffiths check: momentum conservation has one shape ──────────

def momentum_conservation_identity():
    """Conservative form minus (rho * material acceleration + forces) minus
    u * (continuity). Must be identically 0: the conservative equation
        (rho u)_t + (rho u^2 + p - mu u_x)_x = 0
    is Newton's law rho Du/Dt = -p_x + mu u_xx once mass is conserved.
    Griffiths writes EM momentum the same way: d g/dt = div T - f."""
    rho, u, p = (sp.Function(n)(x, t) for n in ("rho", "u", "p"))
    mu = sp.symbols("mu", positive=True)
    conservative = sp.diff(rho * u, t) + sp.diff(rho * u**2 + p - mu * sp.diff(u, x), x)
    newton = rho * (sp.diff(u, t) + u * sp.diff(u, x)) + sp.diff(p, x) - mu * sp.diff(u, x, 2)
    continuity = sp.diff(rho, t) + sp.diff(rho * u, x)
    return sp.simplify(sp.expand(conservative - newton - u * continuity))


# ── 2. Cole-Hopf: Burgers -> homogeneous heat equation ──────────────

def burgers_operator(u, viscosity=nu):
    return sp.diff(u, t) + u * sp.diff(u, x) - viscosity * sp.diff(u, x, 2)


def cole_hopf_identity():
    """For ANY positive phi(x, t), with u = -2 nu phi_x / phi:
        Burgers[u] = -2 nu d/dx [ (phi_t - nu phi_xx) / phi ].
    Returns the difference of the two sides (must be 0). So phi solving the
    linear, homogeneous heat equation makes u solve nonlinear Burgers."""
    phi = sp.Function("phi")(x, t)
    u = -2 * nu * sp.diff(phi, x) / phi
    rhs = -2 * nu * sp.diff((sp.diff(phi, t) - nu * sp.diff(phi, x, 2)) / phi, x)
    return sp.simplify(burgers_operator(u) - rhs)


def cole_hopf_initial_phi(u0_expr):
    """phi0(x) = exp(-(1/(2 nu)) int_0^x u0(s) ds)."""
    s = sp.symbols("s", real=True)
    return sp.exp(-sp.integrate(u0_expr.subs(x, s), (s, 0, x)) / (2 * nu))


def taylor_shock(U=sp.Symbol("U", positive=True)):
    """Steady viscous shock u = -U tanh(U x / (2 nu)): jump 2U, width ~ 2 nu / U,
    max slope U^2 / (2 nu). Returns (u, Burgers residual)."""
    u = -U * sp.tanh(U * x / (2 * nu))
    return u, sp.simplify(burgers_operator(u))


# ── 3. Lagrangian coordinates and the 1D blowup ─────────────────────

def lagrangian_burgers_check():
    """Inviscid Burgers along particle paths: Du/Dt = 0, so each particle keeps
    u0(X) and moves on the straight line x = X + u0(X) t. With X(x, t) defined
    implicitly by that map, u(x, t) = u0(X) satisfies u_t + u u_x = 0 wherever
    J = 1 + u0'(X) t != 0. Returns (residual, u_x)."""
    u0 = sp.Function("u0")
    J = 1 + sp.diff(u0(X), X) * t
    # implicit differentiation of x = X + u0(X) t:  X_x = 1/J,  X_t = -u0/J
    X_x, X_t = 1 / J, -u0(X) / J
    u_x = sp.diff(u0(X), X) * X_x
    u_t = sp.diff(u0(X), X) * X_t
    return sp.simplify(u_t + u0(X) * u_x), sp.simplify(u_x)


def breaking_time(u0_expr):
    """t* = -1 / min u0'(X): the first time J = 0 somewhere (characteristics
    cross, u_x -> -infinity). None if u0 is nowhere decreasing."""
    d = sp.diff(u0_expr.subs(x, X), X)
    crit = sp.solve(sp.diff(d, X), X)
    cands = [d.subs(X, cpt) for cpt in crit] if crit else []
    f = sp.lambdify(X, d, "numpy")
    grid = np.linspace(0, 1, 20001)
    m = float(np.min(f(grid)))
    if cands:
        m = min([m] + [float(cv) for cv in cands if cv.is_real])
    return None if m >= 0 else -1.0 / m


# ── 4. special relativity: the Lagrangian derivative is u^mu d_mu ───

def four_velocity_derivative_check():
    """u^mu d_mu f = gamma (f_t + v f_x) = gamma Df/Dt  (1+1D, x^0 = c t).
    d/d tau along a worldline is the relativistic material derivative.
    Returns the difference (must be 0)."""
    v = sp.Function("v")(x, t)
    f = sp.Function("f")(x, t)
    gam = 1 / sp.sqrt(1 - v**2 / c**2)
    u0, u1 = gam * c, gam * v
    lhs = u0 * sp.diff(f, t) / c + u1 * sp.diff(f, x)
    return sp.simplify(lhs - gam * (sp.diff(f, t) + v * sp.diff(f, x)))


def relativistic_dust_reduces_to_burgers():
    """Pressureless dust: u^mu d_mu u^1 = 0 with u^1 = gamma v. Because gamma v
    depends on v alone, this equals gamma * d(gamma v)/dv * (v_t + v v_x), and
    d(gamma v)/dv = gamma^3 > 0. So the velocity obeys inviscid Burgers EXACTLY,
    at any speed below c. Returns (expression / (gamma^4), d(gamma v)/dv)."""
    v = sp.Function("v")(x, t)
    w = sp.symbols("w", real=True)
    gam = lambda q: 1 / sp.sqrt(1 - q**2 / c**2)
    g = gam(v)
    eq = g * (sp.diff(g * v, t) + v * sp.diff(g * v, x))
    burg = sp.diff(v, t) + v * sp.diff(v, x)
    dgv = sp.simplify(sp.diff(gam(w) * w, w))
    return sp.simplify(eq / (g**4 * burg)), sp.simplify(dgv - gam(w)**3)


def time_dilation_fraction(speed_m_s):
    """gamma - 1 for a speed, computed stably as v^2/(2c^2) + 3v^4/(8c^4)."""
    b2 = (np.asarray(speed_m_s, float) / 299792458.0) ** 2
    return b2 / 2 + 3 * b2**2 / 8


# ── 5. Madelung: the nonlinear Schrodinger equation is a fluid ──────

def madelung_check():
    """NLSE  i psi_t = -1/2 psi_xx + g |psi|^2 psi  with psi = A e^{iS}, rho = A^2,
    u = S_x. Returns (continuity residual, phase-equation residual): both are
    the NLSE itself, rewritten, and simplify to 0:
        rho_t + (rho u)_x = 0
        S_t + u^2/2 + g rho - A_xx / (2A) = 0     (x-derivative: Euler with
        pressure g rho^2 / 2 plus the 'quantum pressure' term)."""
    g = sp.symbols("g", real=True)
    A = sp.Function("A")(x, t)
    S = sp.Function("S")(x, t)
    psi = A * sp.exp(sp.I * S)
    E = sp.expand(sp.simplify((sp.I * sp.diff(psi, t) + sp.diff(psi, x, 2) / 2
                               - g * A**2 * psi) * sp.exp(-sp.I * S)))
    # A, S real: every term is either real or carries one explicit factor I
    parts = sp.collect(E, sp.I, evaluate=False)
    re, im = parts.get(sp.S.One, 0), parts.get(sp.I, 0)
    rho, u = A**2, sp.diff(S, x)
    continuity = sp.diff(rho, t) + sp.diff(rho * u, x)
    phase = sp.diff(S, t) + u**2 / 2 + g * rho - sp.diff(A, x, 2) / (2 * A)
    # NLSE imaginary part = A_t + A_x S_x + A S_xx / 2 = continuity / (2A)
    # NLSE real part      = -A * phase
    return (sp.simplify(continuity - 2 * A * im), sp.simplify(phase + re / A))


# ── 6. numerical Burgers: reference solver and exact solution ───────

def u0_sine(xg):
    return np.sin(2 * np.pi * xg)


def cole_hopf_exact(xg, time, viscosity, n_fine=4096):
    """Exact periodic viscous Burgers solution for u0 = sin(2 pi x) on [0,1):
    phi0 = exp(-(1 - cos 2 pi x) / (4 pi nu)), evolve each Fourier mode of the
    heat equation exactly, then u = -2 nu phi_x / phi (spectral derivative).
    Accurate to roundoff for nu >~ 0.005; returned on the grid xg."""
    if viscosity <= 0 or time < 0:
        raise ValueError("need viscosity > 0 and time >= 0")
    xf = np.arange(n_fine) / n_fine
    phi0 = np.exp(-(1 - np.cos(2 * np.pi * xf)) / (4 * np.pi * viscosity))
    k = 2 * np.pi * np.fft.fftfreq(n_fine, d=1.0 / n_fine)
    P = np.fft.fft(phi0) * np.exp(-viscosity * k**2 * time)
    phi = np.real(np.fft.ifft(P))
    phix = np.real(np.fft.ifft(1j * k * P))
    uf = -2 * viscosity * phix / phi
    # spectral interpolation onto xg (xg need not lie on the fine grid)
    U = np.fft.fft(uf) / n_fine
    xg = np.asarray(xg, float)
    kk = np.fft.fftfreq(n_fine, d=1.0 / n_fine)
    keep = np.abs(kk) < n_fine // 2
    return np.real(np.exp(2j * np.pi * np.outer(xg, kk[keep])) @ U[keep])


def burgers_rhs(u, dx, viscosity):
    """-(u^2/2)_x + nu u_xx with 2nd-order central differences, periodic."""
    up, um = np.roll(u, -1, axis=-1), np.roll(u, 1, axis=-1)
    return -(up**2 - um**2) / (4 * dx) + viscosity * (up - 2 * u + um) / dx**2


def stable_dt(n, viscosity, umax=1.0, safety=0.4):
    dx = 1.0 / n
    return safety * min(dx / umax, dx**2 / (2 * viscosity))


def solve_burgers(n=512, viscosity=0.01, t_end=0.5, dt=None, u0=u0_sine):
    """Reference solver: RK4 in time. Returns (x, u(t_end), n_steps)."""
    if n < 16 or viscosity <= 0 or t_end <= 0:
        raise ValueError("need n >= 16, viscosity > 0, t_end > 0")
    dx = 1.0 / n
    xg = np.arange(n) * dx
    dt = stable_dt(n, viscosity) if dt is None else dt
    steps = int(np.ceil(t_end / dt))
    dt = t_end / steps
    u = u0(xg).astype(float)
    for _ in range(steps):
        k1 = burgers_rhs(u, dx, viscosity)
        k2 = burgers_rhs(u + dt / 2 * k1, dx, viscosity)
        k3 = burgers_rhs(u + dt / 2 * k2, dx, viscosity)
        k4 = burgers_rhs(u + dt * k3, dx, viscosity)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xg, u, steps


def characteristics(n_lines=41, t_max=0.3, u0=u0_sine):
    """Straight characteristic lines x = X + u0(X) t (unwrapped)."""
    Xs = np.linspace(0, 1, n_lines)
    ts = np.linspace(0, t_max, 50)
    return Xs, ts, Xs[None, :] + u0(Xs)[None, :] * ts[:, None]


# ── 7. blood vs water ────────────────────────────────────────────────

FLUIDS = {  # kinematic viscosity, m^2/s (textbook-typical)
    "water 20 C": 1.00e-6,
    "water 37 C": 0.69e-6,
    "blood 37 C (high shear)": 3.5e-3 / 1060.0,
}

VESSELS = {  # radius m, mean speed m/s (textbook-typical adult values)
    "aorta": (12.5e-3, 0.20),
    "femoral artery": (4.0e-3, 0.10),
    "arteriole": (25e-6, 5e-3),
    "capillary": (4e-6, 0.5e-3),
}


def reynolds_womersley(radius_m, speed_m_s, kinematic_viscosity, heart_rate_hz=1.2):
    """Re = U (2R) / nu and Womersley alpha = R sqrt(omega / nu)."""
    if radius_m <= 0 or speed_m_s <= 0 or kinematic_viscosity <= 0 or heart_rate_hz <= 0:
        raise ValueError("all inputs must be positive")
    omega = 2 * np.pi * heart_rate_hz
    return speed_m_s * 2 * radius_m / kinematic_viscosity, radius_m * np.sqrt(omega / kinematic_viscosity)


def womersley_profile(r_over_R, alpha, phase):
    """Oscillatory pipe flow driven by -dp/dx = A cos(omega t). Returns u
    normalised by the steady Poiseuille centreline speed A R^2 / (4 mu):
        u = Re{ 4/(i alpha^2) [1 - J0(i^{3/2} alpha r/R) / J0(i^{3/2} alpha)] e^{i phase} }.
    alpha -> 0 gives the quasi-steady Poiseuille profile (1 - (r/R)^2) cos(phase)."""
    from scipy.special import jv
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    z = 1j**1.5 * alpha
    r = np.asarray(r_over_R, float)
    shape = 1 - jv(0, z * r) / jv(0, z)
    return np.real(4 / (1j * alpha**2) * shape * np.exp(1j * phase))


def womersley_ode_residual(alpha, n=2001):
    """Check the profile solves  i alpha^2 w = 4 + w'' + w'/r  (normalised
    momentum equation) by finite differences. Returns max |residual| on
    0.05 <= r <= 0.95."""
    from scipy.special import jv
    z = 1j**1.5 * alpha
    r = np.linspace(0, 1, n)
    w = 4 / (1j * alpha**2) * (1 - jv(0, z * r) / jv(0, z))
    h = r[1] - r[0]
    d1 = np.gradient(w, h, edge_order=2)
    d2 = np.gradient(d1, h, edge_order=2)
    res = 1j * alpha**2 * w - 4 - d2 - d1 / np.where(r == 0, 1, r)
    m = (r >= 0.05) & (r <= 0.95)
    return float(np.max(np.abs(res[m])))


# ── 8. respiratory droplets: Stokes settling ─────────────────────────

AIR_MU = 1.81e-5      # Pa s, 20 C
AIR_RHO = 1.20        # kg/m^3


def stokes_settling(diameter_m, rho_particle=1000.0, g=9.81, mu=AIR_MU, rho_air=AIR_RHO):
    """Terminal speed v = (rho_p - rho_a) g d^2 / (18 mu) and the particle
    Reynolds number Re = rho_a v d / mu. Stokes' law needs Re << 1.
    Ignores evaporation and slip correction (Cunningham, matters below ~1 um)."""
    d = np.asarray(diameter_m, float)
    if np.any(d <= 0):
        raise ValueError("diameter must be positive")
    v = (rho_particle - rho_air) * g * d**2 / (18 * mu)
    return v, rho_air * v * d / mu


def stokes_drag_symbolic():
    """Force balance  3 pi mu d v = (rho_p - rho_a) g pi d^3 / 6  solved for v."""
    mu_, d, v, g, rp, ra = sp.symbols("mu d v g rho_p rho_a", positive=True)
    return sp.solve(sp.Eq(3 * sp.pi * mu_ * d * v, (rp - ra) * g * sp.pi * d**3 / 6), v)[0]
