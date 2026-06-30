"""Mathematical Methods of Physics — upper-division toolkit.

Covers the Arfken/Boas curriculum:
  §1  Calculus 2/3: series, partial derivatives, multiple integrals
  §2  Linear algebra: eigenvalues, SVD, determinants, orthogonality
  §3  Ordinary differential equations: 1st-order, 2nd-order, Frobenius
  §4  Lagrangian / Hamiltonian mechanics (classical particle)
  §5  Complex analysis: Cauchy-Riemann, residue theorem, contour integrals
  §6  Modular arithmetic, min/max, floor/ceil — number theory helpers

Each section has:
  - Numerical implementations (numpy)
  - SymPy symbolic derivations (init_printing-ready)
  - 5 key equations per section
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Callable, Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# §1  CALCULUS 2 / 3
# ════════════════════════════════════════════════════════════════════════════

def taylor_series_sympy(f_expr, x_sym, x0, n_terms: int = 6) -> Dict:
    """Taylor series of f around x0 to n_terms terms.

    Returns symbolic series, remainder, and radius of convergence estimate.
    """
    series = sp.series(f_expr, x_sym, x0, n=n_terms)
    poly   = series.removeO()
    return {
        "series":     series,
        "poly":       poly,
        "terms":      [sp.diff(f_expr, x_sym, k).subs(x_sym, x0) /
                       sp.factorial(k) * (x_sym - x0)**k for k in range(n_terms)],
        "f0":         f_expr.subs(x_sym, x0),
    }


def partial_derivatives_sympy(f_expr,
                               vars_list: List[sp.Symbol]) -> Dict:
    """All first and second partial derivatives of f(x1, x2, ...).

    Returns gradient, Hessian matrix, Laplacian.
    """
    grad = sp.Matrix([sp.diff(f_expr, v) for v in vars_list])
    n = len(vars_list)
    H = sp.Matrix([[sp.diff(f_expr, vi, vj) for vj in vars_list]
                   for vi in vars_list])
    laplacian = sum(sp.diff(f_expr, v, 2) for v in vars_list)
    return {
        "gradient":   grad,
        "hessian":    H,
        "laplacian":  laplacian,
        "det_H":      H.det() if n <= 4 else None,
    }


def double_integral_sympy(f_expr,
                           x_sym: sp.Symbol, x_lims: Tuple,
                           y_sym: sp.Symbol, y_lims: Tuple) -> Dict:
    """Symbolic double integral int_{y0}^{y1} int_{x0}^{x1} f dx dy."""
    inner = sp.integrate(f_expr, (x_sym, *x_lims))
    result = sp.integrate(inner, (y_sym, *y_lims))
    return {
        "inner_integral": inner,
        "result":         sp.simplify(result),
        "integrand":      f_expr,
    }


def greens_theorem_check(P_expr, Q_expr,
                          x_sym: sp.Symbol,
                          y_sym: sp.Symbol) -> Dict:
    """Green's theorem: line integral = double integral of curl.

    int_C (P dx + Q dy) = int int_D (dQ/dx - dP/dy) dA

    Returns the curl (dQ/dx - dP/dy) symbolically.
    """
    dQ_dx = sp.diff(Q_expr, x_sym)
    dP_dy = sp.diff(P_expr, y_sym)
    curl   = sp.simplify(dQ_dx - dP_dy)
    exact  = sp.simplify(curl) == 0  # exact if curl=0 everywhere
    return {
        "dQ_dx":  dQ_dx,
        "dP_dy":  dP_dy,
        "curl":   curl,
        "exact":  exact,
        "theorem": sp.Eq(sp.Symbol("oint_C(P dx + Q dy)"),
                         sp.Integral(curl, (x_sym, -sp.oo, sp.oo),
                                           (y_sym, -sp.oo, sp.oo))),
    }


def calculus_sympy_5() -> Dict:
    """5 key calculus 2/3 equations."""
    x, y, z, n = sp.symbols("x y z n")
    # 1. Taylor: e^x = sum x^n/n!
    eq1 = sp.Eq(sp.exp(x), sp.Sum(x**n / sp.factorial(n), (n, 0, sp.oo)))
    # 2. Gradient magnitude (magnitude of del f)
    f = sp.Function("f")
    eq2 = sp.Eq(sp.Symbol("|grad f|^2"),
                sp.diff(f(x,y), x)**2 + sp.diff(f(x,y), y)**2)
    # 3. Laplacian in 2D
    eq3 = sp.Eq(sp.Symbol("lap_f"),
                sp.diff(f(x,y), x, 2) + sp.diff(f(x,y), y, 2))
    # 4. Green's theorem
    P, Q = sp.Function("P"), sp.Function("Q")
    eq4 = sp.Eq(sp.Symbol("oint_C"),
                sp.Integral(sp.diff(Q(x,y), x) - sp.diff(P(x,y), y),
                            (x, sp.Symbol("a"), sp.Symbol("b")),
                            (y, sp.Symbol("c"), sp.Symbol("d"))))
    # 5. Change of variables: polar Jacobian
    r, theta = sp.symbols("r theta", positive=True)
    eq5 = sp.Eq(sp.Symbol("dA"), r * sp.Symbol("dr") * sp.Symbol("dtheta"))
    return {
        "Taylor_exp":       eq1,
        "Gradient_sq":      eq2,
        "Laplacian_2D":     eq3,
        "Greens_theorem":   eq4,
        "Polar_Jacobian":   eq5,
    }


# ════════════════════════════════════════════════════════════════════════════
# §2  LINEAR ALGEBRA
# ════════════════════════════════════════════════════════════════════════════

def eigenvalue_decomposition(A: np.ndarray) -> Dict:
    """Full eigendecomposition of A with stability info.

    Returns eigenvalues, eigenvectors, condition number, determinant,
    and whether A is Hermitian (eigvals real) or defective.
    """
    vals, vecs = np.linalg.eig(A)
    cond = float(np.linalg.cond(A))
    det  = float(np.linalg.det(A))
    rank = int(np.linalg.matrix_rank(A))
    hermitian = bool(np.allclose(A, A.conj().T, atol=1e-10))
    if hermitian:
        vals_h, vecs_h = np.linalg.eigh(A)
        return {
            "eigenvalues":    vals_h,
            "eigenvectors":   vecs_h,
            "hermitian":      True,
            "condition":      cond,
            "determinant":    det,
            "rank":           rank,
            "trace_check":    abs(np.sum(vals_h) - np.trace(A)) < 1e-8,
            "det_check":      abs(np.prod(vals_h) - det) < 1e-6 * (abs(det)+1),
        }
    return {
        "eigenvalues":    vals,
        "eigenvectors":   vecs,
        "hermitian":      False,
        "condition":      cond,
        "determinant":    det,
        "rank":           rank,
        "trace_check":    abs(np.sum(vals) - np.trace(A)) < 1e-8,
    }


def svd_analysis(A: np.ndarray) -> Dict:
    """SVD: A = U * S * V^H.

    Returns singular values, rank, condition number, pseudo-inverse.
    Connects to PCA: left singular vectors = principal components.
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(s > 1e-10 * s[0]))
    A_pinv = np.linalg.pinv(A)
    A_reconstructed = U @ np.diag(s) @ Vh
    return {
        "U":          U,
        "s":          s,
        "Vh":         Vh,
        "rank":       rank,
        "condition":  float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
        "A_pinv":     A_pinv,
        "recon_error": float(np.linalg.norm(A - A_reconstructed)),
        "energy_1":   float(s[0]**2 / np.sum(s**2)),  # variance in first PC
    }


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthonormalization of column vectors.

    Input: (n, k) matrix of k column vectors in R^n.
    Output: (n, k) orthonormal basis Q with Q^T Q = I_k.
    """
    Q = np.zeros_like(vectors, dtype=float)
    for j in range(vectors.shape[1]):
        v = vectors[:, j].copy().astype(float)
        for i in range(j):
            v -= np.dot(Q[:, i], vectors[:, j]) * Q[:, i]
        norm = np.linalg.norm(v)
        Q[:, j] = v / norm if norm > 1e-14 else v
    return Q


def matrix_exponential(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Matrix exponential exp(A*t) via eigendecomposition.

    For diagonalizable A = V*D*V^{-1}:  exp(At) = V*exp(Dt)*V^{-1}

    Used in ODE solutions: d/dt x = A*x -> x(t) = exp(At)*x(0)
    """
    vals, vecs = np.linalg.eig(A)
    Vinv = np.linalg.inv(vecs)
    return (vecs @ np.diag(np.exp(vals * t)) @ Vinv).real


def linalg_sympy_5() -> Dict:
    """5 key linear algebra equations."""
    A = sp.MatrixSymbol("A", 3, 3)
    lam = sp.Symbol("lambda")
    v   = sp.MatrixSymbol("v", 3, 1)
    # 1. Eigenvalue equation
    eq1 = sp.Eq(A * v, lam * v)
    # 2. Determinant = product of eigenvalues
    eq2 = sp.Eq(sp.Symbol("det(A)"), sp.Symbol("lambda_1 * lambda_2 * lambda_3"))
    # 3. Trace = sum of eigenvalues
    eq3 = sp.Eq(sp.Symbol("tr(A)"), sp.Symbol("lambda_1 + lambda_2 + lambda_3"))
    # 4. SVD
    U, S, Vh = sp.MatrixSymbol("U", 3, 3), sp.MatrixSymbol("Sigma", 3, 3), sp.MatrixSymbol("V^H", 3, 3)
    eq4 = sp.Eq(A, U * S * Vh)
    # 5. Gram-Schmidt orthonormality
    Q = sp.MatrixSymbol("Q", 3, 3)
    I = sp.Identity(3)
    eq5 = sp.Eq(Q.T * Q, I)
    return {
        "Eigenvalue_eqn":    eq1,
        "Det_eigenvalues":   eq2,
        "Trace_eigenvalues": eq3,
        "SVD":               eq4,
        "GramSchmidt_ortho": eq5,
    }


# ════════════════════════════════════════════════════════════════════════════
# §3  ORDINARY DIFFERENTIAL EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def ode_second_order_sympy(p_coeff, q_coeff, r_rhs,
                            x_sym: sp.Symbol,
                            y_sym) -> Dict:
    """Solve y'' + p(x)*y' + q(x)*y = r(x) symbolically via dsolve.

    Returns general solution, particular solution (if r≠0),
    characteristic equation (if constant coefficients).
    """
    ode = y_sym(x_sym).diff(x_sym, 2) + p_coeff * y_sym(x_sym).diff(x_sym) + q_coeff * y_sym(x_sym) - r_rhs
    try:
        sol = sp.dsolve(ode, y_sym(x_sym))
    except Exception:
        sol = None
    # Characteristic equation for constant coefficients
    m = sp.Symbol("m")
    char_eq = None
    if not p_coeff.has(x_sym) and not q_coeff.has(x_sym):
        char_eq = sp.Eq(m**2 + p_coeff * m + q_coeff, 0)
        char_roots = sp.solve(char_eq, m)
    else:
        char_roots = []
    return {
        "ode":        ode,
        "solution":   sol,
        "char_eq":    char_eq,
        "char_roots": char_roots,
    }


def harmonic_oscillator_ivp(omega0: float, zeta: float,
                              t_span: Tuple[float, float],
                              x0: float = 1.0, v0: float = 0.0,
                              n_pts: int = 1000) -> Dict:
    """Damped harmonic oscillator: x'' + 2*zeta*omega0*x' + omega0^2*x = 0.

    Exact analytic solution via characteristic roots.
    zeta = 0:        undamped (simple harmonic)
    0 < zeta < 1:    underdamped (oscillatory decay)
    zeta = 1:        critically damped (fastest decay, no oscillation)
    zeta > 1:        overdamped (pure exponential)
    """
    t = np.linspace(*t_span, n_pts)
    alpha = zeta * omega0
    omega_d = omega0 * np.sqrt(abs(1 - zeta**2))

    if zeta < 1:
        # Underdamped: x(t) = e^{-alpha*t} * (A*cos(omega_d*t) + B*sin(omega_d*t))
        A = x0
        B = (v0 + alpha * x0) / omega_d
        x = np.exp(-alpha * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        regime = "underdamped"
    elif abs(zeta - 1) < 1e-10:
        # Critically damped: x(t) = (A + B*t)*e^{-omega0*t}
        A = x0; B = v0 + omega0 * x0
        x = (A + B * t) * np.exp(-omega0 * t)
        regime = "critically_damped"
    else:
        # Overdamped: x(t) = A*e^{r1*t} + B*e^{r2*t}
        r1 = -alpha + omega_d
        r2 = -alpha - omega_d
        B  = (v0 - r1 * x0) / (r2 - r1)
        A  = x0 - B
        x  = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        regime = "overdamped"

    # Energy: E = 0.5*(v^2 + omega0^2*x^2)  (undamped conserved, damped decays)
    v = np.gradient(x, t)
    E = 0.5 * (v**2 + omega0**2 * x**2)

    return {
        "t": t, "x": x, "v": v, "E": E,
        "omega0": omega0, "zeta": zeta, "omega_d": omega_d,
        "regime": regime,
        "Q_factor": 0.5 / zeta if zeta > 0 else np.inf,
    }


def frobenius_bessel_sympy(nu: int) -> Dict:
    """Bessel equation via Frobenius method (series solution around x=0).

    x^2*y'' + x*y' + (x^2 - nu^2)*y = 0

    Frobenius: assume y = sum_{k=0}^inf a_k * x^{k+s}
    Indicial equation: s^2 - nu^2 = 0 -> s = ±nu
    Recurrence: a_k = -a_{k-2} / (k*(k+2*nu))
    """
    x = sp.Symbol("x", positive=True)
    y = sp.Function("y")
    bessel_eq = (x**2 * y(x).diff(x, 2) +
                 x * y(x).diff(x) +
                 (x**2 - nu**2) * y(x))
    # Indicial equation
    s = sp.Symbol("s")
    indicial = sp.Eq(s**2 - nu**2, 0)
    indicial_roots = sp.solve(indicial, s)
    # J_nu(x) = sum_{k=0}^inf (-1)^k / (k! Gamma(k+nu+1)) * (x/2)^{2k+nu}
    k = sp.Symbol("k", integer=True, nonneg=True)
    J_nu = sp.Sum((-1)**k / (sp.factorial(k) * sp.gamma(k + nu + 1)) *
                  (x / 2)**(2*k + nu), (k, 0, sp.oo))
    return {
        "bessel_eq":      bessel_eq,
        "indicial":       indicial,
        "indicial_roots": indicial_roots,
        "J_nu_series":    sp.Eq(sp.Symbol(f"J_{nu}(x)"), J_nu),
    }


def ode_sympy_5() -> Dict:
    """5 key ODE equations."""
    x = sp.Symbol("x")
    y = sp.Function("y")
    omega, zeta, t_s = sp.symbols("omega_0 zeta t", positive=True)
    # 1. Damped harmonic oscillator ODE
    eq1 = sp.Eq(y(t_s).diff(t_s, 2) + 2*zeta*omega*y(t_s).diff(t_s) + omega**2*y(t_s), 0)
    # 2. Characteristic roots
    m = sp.Symbol("m")
    eq2 = sp.Eq(m, -zeta*omega + omega*sp.sqrt(zeta**2 - 1))
    # 3. Q factor
    eq3 = sp.Eq(sp.Symbol("Q"), sp.Rational(1, 2) / zeta)
    # 4. Bessel indicial equation
    s, nu = sp.Symbol("s"), sp.Symbol("nu")
    eq4 = sp.Eq(s**2 - nu**2, 0)
    # 5. Variation of parameters (particular solution)
    W = sp.Symbol("W")  # Wronskian
    eq5 = sp.Eq(sp.Symbol("y_p(x)"),
                sp.Symbol("y1") * sp.Integral(-sp.Symbol("y2*r/W"), x) +
                sp.Symbol("y2") * sp.Integral(sp.Symbol("y1*r/W"), x))
    return {
        "Damped_HO":         eq1,
        "Char_roots":        eq2,
        "Q_factor":          eq3,
        "Bessel_indicial":   eq4,
        "Variation_params":  eq5,
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  LAGRANGIAN / HAMILTONIAN MECHANICS
# ════════════════════════════════════════════════════════════════════════════

def lagrangian_particle_sympy(potential_expr,
                               coord_syms: List[sp.Symbol],
                               m_sym: sp.Symbol) -> Dict:
    """Euler-Lagrange equations for a particle with kinetic energy T=m*v^2/2.

    T = (m/2) * sum(q_i_dot^2)
    V = potential_expr(q_i)
    L = T - V
    EOM: d/dt(dL/dq_dot) - dL/dq = 0

    coord_syms: list of generalized coordinates [q1, q2, ...]
    Returns EOM for each coordinate symbolically.
    """
    t = sp.Symbol("t")
    qdot_syms = [sp.Function(str(q))(t).diff(t) for q in coord_syms]
    q_funcs   = [sp.Function(str(q))(t) for q in coord_syms]

    T = m_sym / 2 * sum(qd**2 for qd in qdot_syms)
    V = potential_expr.subs(list(zip(coord_syms, q_funcs)))
    L = T - V

    eoms = []
    for q_f, qd_f in zip(q_funcs, qdot_syms):
        dL_dqdot = sp.diff(L, qd_f)
        dL_dq    = sp.diff(L, q_f)
        eom      = sp.Eq(sp.diff(dL_dqdot, t) - dL_dq, 0)
        eoms.append(sp.simplify(eom))

    # Hamiltonian via Legendre transform: H = sum(p*qdot) - L
    p_syms = [sp.Symbol(f"p_{q}") for q in coord_syms]
    H_expr = sum(p*qd for p, qd in zip(p_syms, qdot_syms)) - L
    # Substitute qdot = p/m (from p = m*qdot for this T)
    for p_s, qd_f, q_f in zip(p_syms, qdot_syms, q_funcs):
        H_expr = H_expr.subs(qd_f, p_s / m_sym)
    H_simplified = sp.simplify(H_expr)

    return {
        "T":        T,
        "V":        V,
        "L":        L,
        "EOM":      eoms,
        "H":        H_simplified,
        "p_syms":   p_syms,
    }


def phase_space_trajectory(omega0: float, x0: float, p0: float,
                            t_max: float = 10.0, n_pts: int = 500) -> Dict:
    """Phase space orbit for harmonic oscillator H = p^2/(2m) + m*omega0^2*x^2/2.

    Elliptic orbits in (x, p) plane. Hamilton's equations:
      dx/dt = dH/dp = p/m
      dp/dt = -dH/dx = -m*omega0^2*x

    Energy ellipse: p^2/(2mE) + x^2/(2E/m*omega0^2) = 1
    """
    m = 1.0
    t  = np.linspace(0, t_max, n_pts)
    E  = 0.5 * p0**2 / m + 0.5 * m * omega0**2 * x0**2
    # Exact solution: x(t) = A*cos(omega*t + phi)
    A   = np.sqrt(2 * E / (m * omega0**2))
    phi = np.arctan2(-p0 / (m * omega0), x0)
    x_t = A * np.cos(omega0 * t + phi)
    p_t = -m * omega0 * A * np.sin(omega0 * t + phi)
    return {
        "t": t, "x": x_t, "p": p_t,
        "E": E, "A": A, "omega0": omega0,
        "H_check": np.allclose(
            0.5*p_t**2/m + 0.5*m*omega0**2*x_t**2, E, rtol=1e-10),
    }


def mechanics_sympy_5() -> Dict:
    """5 key Lagrangian/Hamiltonian equations."""
    t, m, q, p, omega = sp.symbols("t m q p omega_0", positive=True)
    V = sp.Function("V")
    # 1. Lagrangian
    eq1 = sp.Eq(sp.Symbol("L"), m*sp.Symbol("qdot")**2/2 - V(q))
    # 2. Euler-Lagrange
    L_s = sp.Function("L")
    eq2 = sp.Eq(sp.Symbol("d/dt(dL/dqdot) - dL/dq"), 0)
    # 3. Canonical momentum
    eq3 = sp.Eq(p, m * sp.Symbol("qdot"))
    # 4. Hamiltonian (Legendre transform)
    eq4 = sp.Eq(sp.Symbol("H"), p**2/(2*m) + V(q))
    # 5. Hamilton's equations (two coupled 1st-order ODEs)
    eq5 = sp.Eq(sp.Symbol("dq_dt"), sp.Symbol("partial_H_partial_p"))
    return {
        "Lagrangian":        eq1,
        "Euler_Lagrange":    eq2,
        "Canon_momentum":    eq3,
        "Hamiltonian":       eq4,
        "Hamilton_eqns":     eq5,
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  COMPLEX ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def cauchy_riemann_check(f_sympy, z_sym: sp.Symbol,
                          x_sym: sp.Symbol, y_sym: sp.Symbol) -> Dict:
    """Check Cauchy-Riemann equations for f(z) = u(x,y) + i*v(x,y).

    C-R conditions (necessary for analyticity):
      du/dx = dv/dy
      du/dy = -dv/dx

    If C-R holds everywhere: f is entire (analytic on all C).
    """
    # Substitute z = x + i*y; force real=True so re/im can simplify
    x_r = sp.Symbol(str(x_sym), real=True)
    y_r = sp.Symbol(str(y_sym), real=True)
    f_xy = f_sympy.subs(z_sym, x_r + sp.I * y_r)
    f_xy = sp.expand(f_xy)
    u = sp.re(f_xy)
    v = sp.im(f_xy)
    x_sym, y_sym = x_r, y_r   # use real-flagged symbols for derivatives
    # C-R equations
    du_dx = sp.diff(u, x_sym)
    du_dy = sp.diff(u, y_sym)
    dv_dx = sp.diff(v, x_sym)
    dv_dy = sp.diff(v, y_sym)
    cr1 = sp.simplify(du_dx - dv_dy)   # should be 0
    cr2 = sp.simplify(du_dy + dv_dx)   # should be 0
    analytic = (cr1 == 0 and cr2 == 0)
    return {
        "u": u, "v": v,
        "du_dx": du_dx, "dv_dy": dv_dy,
        "du_dy": du_dy, "dv_dx": dv_dx,
        "CR1":   cr1,   # du/dx - dv/dy (should be 0)
        "CR2":   cr2,   # du/dy + dv/dx (should be 0)
        "analytic": analytic,
    }


def residue_theorem(f_expr, z_sym: sp.Symbol,
                     poles: List[complex]) -> Dict:
    """Compute residues at given poles for contour integration.

    Residue theorem: int_C f(z) dz = 2*pi*i * sum(residues inside C)

    For simple pole at z0: Res[f, z0] = lim_{z->z0} (z-z0)*f(z)
    For pole of order n:   Res[f, z0] = 1/(n-1)! * lim_{z->z0} d^{n-1}/dz^{n-1}[(z-z0)^n*f(z)]
    """
    residues = {}
    for z0 in poles:
        z0_sym = sp.sympify(z0)
        try:
            res = sp.residue(f_expr, z_sym, z0_sym)
            residues[z0] = res
        except Exception:
            residues[z0] = None
    total_residue = sum(r for r in residues.values() if r is not None)
    contour_integral = sp.simplify(2 * sp.pi * sp.I * total_residue)
    return {
        "residues":         residues,
        "sum_residues":     total_residue,
        "contour_integral": contour_integral,
    }


def cauchy_integral_formula(f_expr, z_sym: sp.Symbol,
                              z0, n: int = 0) -> sp.Expr:
    """Cauchy integral formula: int_C f(z)/(z-z0)^{n+1} dz = 2pi*i/n! f^{(n)}(z0).

    n=0: standard Cauchy formula
    n>0: derivatives version
    """
    z0_sym = sp.sympify(z0)
    f_deriv = sp.diff(f_expr, z_sym, n)
    result = 2 * sp.pi * sp.I / sp.factorial(n) * f_deriv.subs(z_sym, z0_sym)
    return sp.simplify(result)


def partial_fractions_complex(f_expr, z_sym: sp.Symbol) -> Dict:
    """Partial fraction decomposition of f(z) = P(z)/Q(z).

    Finds poles and residues for contour integration via residue theorem.
    """
    pf = sp.apart(f_expr, z_sym)
    poles_found = sp.solve(sp.denom(sp.together(f_expr)), z_sym)
    return {
        "partial_fractions": pf,
        "poles":             poles_found,
        "f_original":        f_expr,
    }


def complex_analysis_sympy_5() -> Dict:
    """5 key complex analysis equations."""
    z, x, y = sp.Symbol("z"), sp.Symbol("x"), sp.Symbol("y")
    f, n_s = sp.Function("f"), sp.Symbol("n")
    # 1. Cauchy-Riemann (du/dx = dv/dy)
    u, v = sp.Function("u"), sp.Function("v")
    eq1 = sp.Eq(sp.diff(u(x,y), x), sp.diff(v(x,y), y))
    # 2. Residue theorem
    eq2 = sp.Eq(sp.Symbol("oint_C f(z) dz"),
                2 * sp.pi * sp.I * sp.Symbol("sum_k Res[f, z_k]"))
    # 3. Cauchy integral formula
    z0 = sp.Symbol("z_0")
    eq3 = sp.Eq(sp.Symbol("oint_C f(z)/(z-z0) dz"), 2*sp.pi*sp.I*f(z0))
    # 4. Laurent series (general form with negative powers)
    a = sp.IndexedBase("a")
    eq4 = sp.Eq(f(z),
                sp.Sum(a[n_s] * (z - z0)**n_s, (n_s, -sp.oo, sp.oo)))
    # 5. Euler's formula
    theta = sp.Symbol("theta", real=True)
    eq5 = sp.Eq(sp.exp(sp.I * theta),
                sp.cos(theta) + sp.I * sp.sin(theta))
    return {
        "Cauchy_Riemann":     eq1,
        "Residue_theorem":    eq2,
        "Cauchy_integral":    eq3,
        "Laurent_series":     eq4,
        "Euler_formula":      eq5,
    }


# ════════════════════════════════════════════════════════════════════════════
# §6  MODULAR ARITHMETIC, MIN/MAX, NUMBER THEORY
# ════════════════════════════════════════════════════════════════════════════

def modular_arithmetic(a: int, b: int, m: int) -> Dict:
    """Modular arithmetic: a mod m, b mod m, and operations.

    Includes: addition, subtraction, multiplication, power, inverse.
    Extended Euclidean algorithm for gcd and modular inverse.
    """
    def ext_gcd(a, b):
        if b == 0: return a, 1, 0
        g, x, y = ext_gcd(b, a % b)
        return g, y, x - (a // b) * y

    g, x, _ = ext_gcd(a % m, m)
    a_inv = (x % m) if g == 1 else None   # modular inverse (only if gcd=1)

    return {
        "a_mod_m":    a % m,
        "b_mod_m":    b % m,
        "add":        (a + b) % m,
        "sub":        (a - b) % m,
        "mul":        (a * b) % m,
        "pow_b":      pow(a, b, m),    # a^b mod m (fast exponentiation)
        "gcd_am":     g,
        "a_inverse":  a_inv,           # a^{-1} mod m (if gcd(a,m)=1)
        "coprime":    g == 1,
    }


def softmax_and_extrema(x: np.ndarray, temperature: float = 1.0) -> Dict:
    """Soft min/max (log-sum-exp) and hard min/max.

    Softmax:   sigma_i = exp(x_i/T) / sum_j exp(x_j/T)
    Log-sum-exp: log(sum exp(x_i)) (numerically stable)

    As T->0: softmax -> argmax (hard max)
    As T->inf: softmax -> uniform distribution

    Connection to physics: Boltzmann distribution P_i = exp(-E_i/kT) / Z
    """
    x = np.asarray(x, dtype=float)
    # Numerically stable softmax
    x_shifted = (x - x.max()) / temperature
    exp_x = np.exp(x_shifted)
    softmax = exp_x / exp_x.sum()
    log_sum_exp = x.max() + np.log(exp_x.sum())
    # Soft min: negate
    x_neg = -x / temperature
    exp_neg = np.exp(x_neg - x_neg.max())
    softmin = exp_neg / exp_neg.sum()
    return {
        "softmax":       softmax,
        "softmin":       softmin,
        "log_sum_exp":   float(log_sum_exp),
        "argmax":        int(np.argmax(x)),
        "argmin":        int(np.argmin(x)),
        "max":           float(x.max()),
        "min":           float(x.min()),
        "temperature":   temperature,
    }


def chinese_remainder_theorem(remainders: List[int],
                               moduli: List[int]) -> Dict:
    """Chinese Remainder Theorem: find x such that x ≡ r_i (mod m_i).

    Requires all m_i pairwise coprime.
    x = sum_i (r_i * M_i * y_i) mod M
    where M = prod(m_i), M_i = M/m_i, y_i = M_i^{-1} mod m_i
    """
    M = 1
    for m in moduli:
        M *= m
    x = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        # Find yi = Mi^{-1} mod m via extended Euclidean
        g, yi, _ = _ext_gcd(Mi, m)
        if g != 1:
            return {"error": f"moduli not coprime: gcd({Mi},{m})={g}"}
        x += r * Mi * yi
    return {
        "x":         x % M,
        "M":         M,
        "verify":    all((x % M) % m == r for r, m in zip(remainders, moduli)),
    }


def _ext_gcd(a, b):
    if b == 0: return a, 1, 0
    g, x, y = _ext_gcd(b, a % b)
    return g, y, x - (a // b) * y


def number_theory_sympy_5() -> Dict:
    """5 key number theory / modular arithmetic equations."""
    a, b, m, k = sp.symbols("a b m k", integer=True)
    # 1. Fermat's little theorem
    p = sp.Symbol("p")   # prime
    eq1 = sp.Eq(sp.Mod(a**p, p), sp.Mod(a, p))
    # 2. Euler's theorem
    phi = sp.Function("phi")
    eq2 = sp.Eq(sp.Mod(a**phi(m), m), 1)
    # 3. Extended Euclidean: gcd = linear combination
    x, y = sp.symbols("x y")
    eq3 = sp.Eq(sp.Symbol("gcd(a,b)"), a*x + b*y)
    # 4. Chinese Remainder Theorem
    eq4 = sp.Eq(sp.Symbol("x mod M"),
                sp.Symbol("sum_i r_i * M_i * (M_i^-1 mod m_i)"))
    # 5. Softmax / Boltzmann
    T = sp.Symbol("T", positive=True)
    E_i = sp.IndexedBase("E")
    Z   = sp.Sum(sp.exp(-E_i[k]/T), (k, 0, sp.Symbol("N")))
    eq5 = sp.Eq(sp.Symbol("P_i"), sp.exp(-E_i[sp.Symbol("i")]/T) / Z)
    return {
        "Fermat_little":     eq1,
        "Euler_theorem":     eq2,
        "Extended_Euclid":   eq3,
        "CRT":               eq4,
        "Boltzmann_softmax": eq5,
    }


# ════════════════════════════════════════════════════════════════════════════
# ALL 5-EQUATION COLLECTIONS
# ════════════════════════════════════════════════════════════════════════════

def all_math_methods_sympy() -> Dict:
    """All 5-equation sets across all math methods sections."""
    return {
        "calculus23":   calculus_sympy_5(),
        "linear_alg":  linalg_sympy_5(),
        "odes":        ode_sympy_5(),
        "mechanics":   mechanics_sympy_5(),
        "complex":     complex_analysis_sympy_5(),
        "number_thy":  number_theory_sympy_5(),
    }


if __name__ == "__main__":
    sp.init_printing(use_latex=False)

    print("=" * 60)
    print("MATHEMATICAL METHODS OF PHYSICS — SYMPY EQUATIONS")
    print("=" * 60)
    sections = all_math_methods_sympy()
    for section, eqs in sections.items():
        print(f"\n--- {section.upper()} ---")
        for name, eq in eqs.items():
            print(f"  [{name}]  {eq}")

    print("\n" + "=" * 60)
    print("NUMERICAL DEMOS")
    print("=" * 60)

    # Linear algebra: random symmetric matrix eigendecomposition
    A = np.array([[4., 2., 0.], [2., 3., 1.], [0., 1., 2.]])
    eig = eigenvalue_decomposition(A)
    print(f"\nEigenvalues of A:  {eig['eigenvalues'].round(4)}")
    print(f"Trace check:       {eig['trace_check']}")
    print(f"Det check:         {eig['det_check']}")

    # ODE: damped oscillator sweep
    print("\nDamped oscillator regimes (omega0=10, x0=1, v0=0):")
    for zeta, label in [(0.1, "under"), (1.0, "critical"), (2.0, "over")]:
        res = harmonic_oscillator_ivp(10.0, zeta, (0, 5))
        print(f"  zeta={zeta}: {label}, Q={res['Q_factor']:.2f}, "
              f"omega_d={res['omega_d']:.3f}")

    # Phase space
    traj = phase_space_trajectory(1.0, 1.0, 0.0)
    print(f"\nPhase space (SHO): Energy={traj['E']:.4f}, "
          f"H conserved: {traj['H_check']}")

    # Complex analysis: residue of 1/(z^2+1) at z=i
    z = sp.Symbol("z")
    f_demo = 1 / (z**2 + 1)
    res_out = residue_theorem(f_demo, z, [sp.I, -sp.I])
    print(f"\nResidues of 1/(z^2+1): {res_out['residues']}")
    print(f"Contour integral (upper half-plane, pole at z=i): "
          f"{res_out['residues'][sp.I] * 2*sp.pi*sp.I}")

    # Modular arithmetic
    mod = modular_arithmetic(7, 5, 11)
    print(f"\n7 mod 11 = {mod['a_mod_m']}, "
          f"7^5 mod 11 = {mod['pow_b']}, "
          f"7^-1 mod 11 = {mod['a_inverse']}")

    # CRT: x ≡ 2(mod 3), x ≡ 3(mod 5), x ≡ 2(mod 7)
    crt = chinese_remainder_theorem([2, 3, 2], [3, 5, 7])
    print(f"CRT: x = {crt['x']}, verify: {crt['verify']}")

    # Softmax
    sm = softmax_and_extrema(np.array([1.0, 2.0, 3.0, 0.5]))
    print(f"\nSoftmax([1,2,3,0.5]): {sm['softmax'].round(4)}")
    print(f"Argmax: {sm['argmax']}, log-sum-exp: {sm['log_sum_exp']:.4f}")
