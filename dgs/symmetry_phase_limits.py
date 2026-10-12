"""Symmetry, phase, and limits: three ideas that run through E&M, quantum
mechanics, and quantum information -- each claim checked in SymPy.

SYMMETRY (what you can change without changing anything measurable)
  * E&M gauge: A -> A + grad chi, V -> V - d chi/dt leaves E, B unchanged
    (reuses dgs.gauge_invariance).
  * Charge conservation: the divergence of Ampere-Maxwell plus Gauss's law
    forces d rho/dt + div J = 0.
  * QM: a GLOBAL phase changes nothing; a LOCAL phase e^{i q chi/hbar} is
    allowed only if A shifts too -- the Schrodinger equation with minimal
    coupling (p - qA) is invariant under the pair. Same chi as in E&M.
  * Noether, operator form: [H, p] = 0 for a free particle -> momentum is
    conserved; d<O>/dt = (i/hbar) <[H, O]>.

PHASE (the part of a wave you cannot see directly)
  * Phase velocity omega/k vs group velocity d omega/dk; they differ when the
    medium disperses.
  * QM: |psi|^2 ignores the global phase, but RELATIVE phase makes interference.
  * QIS: global phase is unobservable; relative phase is the information.

LIMITS (how to build things and how to check answers)
  * Dipole: +q and -q separated by d, with q -> infinity, d -> 0, p = q d fixed.
  * Point charge as the limit of a shrinking Gaussian (the delta function).
  * Correspondence: level spacing / energy -> 0 as n -> infinity.
  * Gates: lim_{n->inf} (1 - i theta sigma / n)^n = exp(-i theta sigma):
    finite rotations are built from infinitesimal ones.
  * Homework sanity: test an answer in limits where you already know the result.

DISCOVERY (the "physics-AI" figure): projectile data x(t') for many throws
compresses to exactly two latent numbers (x0, v) -- visible as two nonzero
singular values of the data matrix. A neural autoencoder learns the same
compression nonlinearly. Education.
"""
import numpy as np
import sympy as sp

from dgs import gauge_invariance as gi

x, y, z, t = sp.symbols("x y z t", real=True)
hbar, m, q = sp.symbols("hbar m q", positive=True)


# ── symmetry ─────────────────────────────────────────────────────────

def em_gauge_check():
    """E and B before minus after a gauge transformation, for generic potentials
    and gauge function (via dgs.gauge_invariance). Returns (dE, dB) -- both zero."""
    V = sp.Function("V")(x, y, z, t)
    A = sp.Matrix([sp.Function(f"A{i}")(x, y, z, t) for i in "xyz"])
    chi = sp.Function("chi")(x, y, z, t)
    V2, A2 = gi.gauge_transform(V, A, chi)
    dE = sp.simplify(gi.E_field(V2, A2) - gi.E_field(V, A))
    dB = sp.simplify(gi.B_field(A2) - gi.B_field(A))
    return dE, dB


def charge_conservation_from_maxwell():
    """div(curl B) = 0 applied to curl B = mu0 J + mu0 eps0 dE/dt, with div E = rho/eps0:
    0 = mu0 (div J + d rho/dt). Returns the residual div J + d rho/dt expressed through
    Maxwell's equations (should be 0 identically)."""
    mu0, eps0 = sp.symbols("mu_0 epsilon_0", positive=True)
    E = sp.Matrix([sp.Function(f"E{i}")(x, y, z, t) for i in "xyz"])
    B = sp.Matrix([sp.Function(f"B{i}")(x, y, z, t) for i in "xyz"])
    curl = lambda F: sp.Matrix([sp.diff(F[2], y) - sp.diff(F[1], z),
                                sp.diff(F[0], z) - sp.diff(F[2], x),
                                sp.diff(F[1], x) - sp.diff(F[0], y)])
    div = lambda F: sp.diff(F[0], x) + sp.diff(F[1], y) + sp.diff(F[2], z)
    J = (curl(B) / mu0 - eps0 * E.diff(t))           # solve Ampere-Maxwell for J
    rho = eps0 * div(E)                               # Gauss's law
    return sp.simplify(div(J) + sp.diff(rho, t))


def global_phase_check():
    """|e^{i a} psi|^2 - |psi|^2 = 0 for any constant a."""
    a = sp.symbols("a", real=True)
    re, im = sp.symbols("re im", real=True)
    psi = re + sp.I * im
    return sp.simplify(sp.Abs(sp.exp(sp.I * a) * psi) ** 2 - sp.Abs(psi) ** 2)


def local_gauge_schrodinger_check():
    """1D Schrodinger equation with minimal coupling:
        i hbar d_t psi = (1/2m) (-i hbar d_x - q A)^2 psi + q V psi.
    Transform psi' = e^{i q chi / hbar} psi, A' = A + d_x chi, V' = V - d_t chi.
    Returns  e^{-i q chi/hbar} (Schrodinger operator on primed fields) - (operator on originals):
    zero means the equation is gauge invariant."""
    psi = sp.Function("psi")(x, t)
    A = sp.Function("A")(x, t)
    V = sp.Function("V")(x, t)
    chi = sp.Function("chi")(x, t)

    def residual(ps, Ax, Vx):
        Dpsi = -sp.I * hbar * sp.diff(ps, x) - q * Ax * ps
        kinetic = (-sp.I * hbar * sp.diff(Dpsi, x) - q * Ax * Dpsi) / (2 * m)
        return sp.I * hbar * sp.diff(ps, t) - kinetic - q * Vx * ps

    phase = sp.exp(sp.I * q * chi / hbar)
    r2 = residual(phase * psi, A + sp.diff(chi, x), V - sp.diff(chi, t))
    return sp.simplify(sp.expand(r2 / phase - residual(psi, A, V)))


def global_only_breaks_without_A():
    """Same check but WITHOUT shifting A and V: a local phase alone is NOT a symmetry.
    Returns the (nonzero) leftover."""
    psi = sp.Function("psi")(x, t)
    chi = sp.Function("chi")(x, t)

    def residual(ps):
        return sp.I * hbar * sp.diff(ps, t) + hbar**2 / (2 * m) * sp.diff(ps, x, 2)

    phase = sp.exp(sp.I * q * chi / hbar)
    return sp.simplify(sp.expand(residual(phase * psi) / phase - residual(psi)))


def noether_commutators():
    """[H, p] f for a free particle (0) and for H with potential V(x) (-(i hbar) V' f... nonzero).
    Returns (free, with_potential) acting on a test function f(x)."""
    f = sp.Function("f")(x)
    V = sp.Function("V")(x)
    p = lambda g: -sp.I * hbar * sp.diff(g, x)
    H_free = lambda g: -hbar**2 / (2 * m) * sp.diff(g, x, 2)
    H_pot = lambda g: H_free(g) + V * g
    free = sp.simplify(H_free(p(f)) - p(H_free(f)))
    pot = sp.simplify(H_pot(p(f)) - p(H_pot(f)))
    return free, pot


# ── phase ────────────────────────────────────────────────────────────

def phase_group_velocity_symbolic():
    """For omega(k) = c k / n(k): v_p = omega/k, v_g = d omega/dk. Returns both and
    their difference, which vanishes only when n is constant (no dispersion)."""
    k, c = sp.symbols("k c", positive=True)
    n = sp.Function("n")(k)
    w = c * k / n
    vp, vg = sp.simplify(w / k), sp.simplify(sp.diff(w, k))
    return vp, vg, sp.simplify(vg - vp)


def relative_phase_interference(phis):
    """Two equal paths with relative phase phi: |1 + e^{i phi}|^2 / 4 = cos^2(phi/2)."""
    phis = np.asarray(phis, float)
    return np.abs(1 + np.exp(1j * phis)) ** 2 / 4


def qubit_global_vs_relative_symbolic():
    """Bloch vector of (|0> + e^{i phi}|1>)/sqrt2 multiplied by a global e^{i gamma}:
    depends on phi, not on gamma."""
    g, ph = sp.symbols("gamma phi", real=True)
    psi = sp.exp(sp.I * g) * sp.Matrix([1, sp.exp(sp.I * ph)]) / sp.sqrt(2)
    rho = sp.simplify(psi * psi.H)
    bx = sp.simplify(2 * sp.re(sp.expand_complex(rho[0, 1])))
    by = sp.simplify(-2 * sp.im(sp.expand_complex(rho[0, 1])))
    bz = sp.simplify(rho[0, 0] - rho[1, 1])
    return sp.Matrix([bx, by, bz])


# ── limits ───────────────────────────────────────────────────────────

def dipole_limit_symbolic():
    """Potential of +q at z = d/2 and -q at z = -d/2, at distance r, angle theta.
    Set q = p/d and take d -> 0: V -> p cos(theta) / (4 pi eps0 r^2)."""
    r, th, d, p, eps0 = sp.symbols("r theta d p epsilon_0", positive=True)
    rp = sp.sqrt(r**2 + (d / 2) ** 2 - r * d * sp.cos(th))
    rm = sp.sqrt(r**2 + (d / 2) ** 2 + r * d * sp.cos(th))
    V = (p / d) / (4 * sp.pi * eps0) * (1 / rp - 1 / rm)
    return sp.simplify(sp.limit(V, d, 0))


def gaussian_delta_limit_symbolic():
    """A unit-area Gaussian of width s: area stays 1 while the peak height -> infinity
    as s -> 0, and int g(x) f(x) dx -> f(0) (checked for f = cos x)."""
    s = sp.symbols("s", positive=True)
    g = sp.exp(-x**2 / (2 * s**2)) / sp.sqrt(2 * sp.pi * s**2)
    area = sp.simplify(sp.integrate(g, (x, -sp.oo, sp.oo)))
    sift = sp.simplify(sp.integrate(g * sp.cos(x), (x, -sp.oo, sp.oo)))
    return area, sift, sp.limit(sift, s, 0), sp.limit(g.subs(x, 0), s, 0)


def correspondence_limit_symbolic():
    """Infinite well E_n ~ n^2: (E_{n+1} - E_n) / E_n -> 0 as n -> infinity, so the
    discrete ladder looks continuous -- classical physics re-emerges."""
    n = sp.symbols("n", positive=True, integer=True)
    E = n**2
    ratio = sp.simplify(((n + 1) ** 2 - E) / E)
    return ratio, sp.limit(ratio, n, sp.oo)


def infinitesimal_rotation_limit():
    """lim_{n->inf} (1 - i theta / n)^n = e^{-i theta}: each eigenvalue of
    (I - i theta sigma_z / n)^n. Returns the scalar limit and the matrix exponential."""
    th = sp.symbols("theta", real=True)
    nn = sp.symbols("n", positive=True)
    lim_scalar = sp.limit((1 - sp.I * th / nn) ** nn, nn, sp.oo)
    sz = sp.diag(1, -1)
    U = sp.simplify((-sp.I * th * sz).exp())
    return lim_scalar, U


def rotation_power_numeric(theta, n, sigma=None):
    """(I - i theta sigma / n)^n numerically, and its distance from expm(-i theta sigma)."""
    from scipy.linalg import expm
    sigma = np.array([[0, 1], [1, 0]], complex) if sigma is None else np.asarray(sigma, complex)
    step = np.eye(2) - 1j * theta * sigma / n
    approx = np.linalg.matrix_power(step, n)
    exact = expm(-1j * theta * sigma)
    return approx, float(np.linalg.norm(approx - exact))


def check_limits(expr, var, cases):
    """Homework sanity tool: evaluate the limit of `expr` as var -> value for each
    (value, expected) pair and report whether it matches."""
    out = []
    for val, expected in cases:
        got = sp.simplify(sp.limit(expr, var, val))
        out.append((val, got, sp.simplify(got - expected) == 0))
    return out


# ── discovery: two latent numbers behind projectile data ─────────────

def projectile_observations(n_throws=200, n_times=30, g=9.81, noise=0.0, rng=0):
    """Rows: x(t'_j) = x0 + v t'_j - g t'^2 / 2 for random (x0, v) per throw."""
    rng = np.random.default_rng(rng)
    tt = np.linspace(0, 1, n_times)
    x0 = rng.uniform(-1, 1, n_throws)
    v = rng.uniform(0, 5, n_throws)
    X = x0[:, None] + v[:, None] * tt[None, :] - 0.5 * g * tt[None, :] ** 2
    return X + noise * rng.normal(size=X.shape), x0, v, tt


def latent_dimension(X, rel_tol=1e-8):
    """Singular values of the mean-centered data; number above rel_tol x largest."""
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    return int(np.sum(s > rel_tol * s[0])), s


def predict_from_two_numbers(X, tt, t_query, g=9.81):
    """Decode: fit (x0, v) per row by least squares on the known law, then answer
    the 'question' x(t_query) -- the encoder/decoder of the figure, done exactly."""
    Aphys = np.column_stack([np.ones_like(tt), tt])
    coeff, *_ = np.linalg.lstsq(Aphys, (X + 0.5 * g * tt**2).T, rcond=None)
    x0, v = coeff
    return x0 + v * t_query - 0.5 * g * t_query**2, x0, v
