"""Vector (two-component) Schrodinger equation: analytic, real/imaginary and discrete residuals.

In plain words: many real systems carry an internal two-level label on top of position:
spin up/down, two electronic states of a molecule, the two levels of a trapped-ion or
atom qubit, two coupled waveguide modes. The wavefunction is then a 2-vector
        Psi(x, t) = [psi1(x, t), psi2(x, t)]
and the potential is a 2x2 HERMITIAN matrix V(x, t) = [[V11, W], [W*, V22]]: the diagonal
entries are each state's own potential (real), the off-diagonal W couples the states.

The residual of a candidate Psi is what is left over when you plug it into the equation:
        R   = i hbar dPsi/dt + (hbar^2 / 2m) d^2Psi/dx^2 - V Psi          (time-dependent)
        R_E = -(hbar^2 / 2m) d^2Psi/dx^2 + V Psi - E Psi                  (stationary)
R = 0 means Psi is a solution. Writing psi_j = u_j + i v_j and W = a + i b splits the two
complex residuals into FOUR real PDE residuals, the form a real-valued solver works with.

Three different things are called "residual" and are kept separate here:
  1. analytic residual   - SymPy, exact; 0 for a true solution (rabi_oscillator_solution);
  2. discretized residual - finite differences on a grid; O(dx^2 + dt^2) even for a true
                            solution, and converges to the analytic forcing for a
                            manufactured (non-)solution;
  3. quantum-simulation backend - CUDA-Q / cuQuantum (optional) evolve a QUBIT state or
     contract tensor networks. They do not evaluate the classical PDE residual; here they
     only reproduce the x-independent internal (Rabi) part of the exact solution.

SI units: x [m], t [s], hbar [J s], m [kg], V, E, g [J], psi [m^-1/2], residuals [J m^-1/2].
Numerics use hbar = m = omega = 1. NumPy + SymPy; SciPy used when available.
"""

import numpy as np
import sympy as sp

try:
    import scipy
    import scipy.sparse as sps
    import scipy.sparse.linalg as spsl
    import scipy.linalg as sla
    HAVE_SCIPY = True
except ImportError:      # pragma: no cover - CPU fallback
    scipy = None
    HAVE_SCIPY = False

x, t = sp.symbols("x t", real=True)
hbar, m, omega = sp.symbols("hbar m omega", positive=True)
g = sp.symbols("g", real=True)
E = sp.symbols("E", real=True)


# ---------------------------------------------------------------- symbolic building blocks

def hermitian_potential(V11, V22, a, b):
    """V = [[V11, a + i b], [a - i b, V22]] with real V11, V22, a, b: Hermitian by construction."""
    return sp.Matrix([[V11, a + sp.I * b], [a - sp.I * b, V22]])


def is_hermitian(V):
    """True when V - V^dagger simplifies to the zero matrix."""
    return sp.simplify(V - V.H) == sp.zeros(*V.shape)


def vector_residual(Psi, V, hb=hbar, mass=m):
    """R = i hbar dPsi/dt + (hbar^2/2m) d^2Psi/dx^2 - V Psi (a 2x1 SymPy Matrix)."""
    if Psi.shape != (2, 1) or V.shape != (2, 2):
        raise ValueError("Psi must be 2x1 and V must be 2x2")
    return sp.I * hb * Psi.diff(t) + hb ** 2 / (2 * mass) * Psi.diff(x, 2) - V * Psi


def stationary_residual(Psi, V, energy=E, hb=hbar, mass=m):
    """R_E = -(hbar^2/2m) d^2Psi/dx^2 + V Psi - E Psi."""
    if Psi.shape != (2, 1) or V.shape != (2, 2):
        raise ValueError("Psi must be 2x1 and V must be 2x2")
    return -hb ** 2 / (2 * mass) * Psi.diff(x, 2) + V * Psi - energy * Psi


def simplify_matrix(M):
    return M.applyfunc(lambda e: sp.simplify(sp.expand(e)))


def general_fields():
    """Real fields u_j, v_j (psi_j = u_j + i v_j) and real potential entries V11, V22, a, b."""
    names = ["u1", "v1", "u2", "v2", "V11", "V22", "a", "b"]
    return {n: sp.Function(n, real=True)(x, t) for n in names}


def general_residual():
    """Symbolic R for arbitrary fields; returns (Psi, V, R, fields)."""
    f = general_fields()
    Psi = sp.Matrix([f["u1"] + sp.I * f["v1"], f["u2"] + sp.I * f["v2"]])
    V = hermitian_potential(f["V11"], f["V22"], f["a"], f["b"])
    return Psi, V, vector_residual(Psi, V), f


def split_real_imag(expr):
    """Split an expression built from REAL symbols/functions into (real, imag) by collecting I.

    Valid because the only imaginary unit in the expression is the explicit I; this is the
    'explicit real-valued fields' route (sp.re / sp.im cannot see that u_j(x, t) derivatives
    are real)."""
    parts = sp.collect(sp.expand(expr), sp.I, evaluate=False)
    extra = set(parts) - {sp.S.One, sp.I}
    if extra:
        raise ValueError(f"unexpected terms after collecting I: {extra}")
    return sp.simplify(parts.get(sp.S.One, 0)), sp.simplify(parts.get(sp.I, 0))


def real_imag_residuals():
    """The four real PDE residuals [Re R1, Im R1, Re R2, Im R2] for the general system."""
    _, _, R, f = general_residual()
    out = []
    for j in range(2):
        re_, im_ = split_real_imag(R[j])
        out += [re_, im_]
    return out, R, f


# ---------------------------------------------------------------- exact and stationary solutions

def ground_state():
    """Harmonic ground state phi0 = (m w / pi hbar)^{1/4} exp(-m w x^2 / 2 hbar), E0 = hbar w / 2."""
    return (m * omega / (sp.pi * hbar)) ** sp.Rational(1, 4) * sp.exp(-m * omega * x ** 2 / (2 * hbar)), hbar * omega / 2


def coupled_oscillator_potential(coupling=g):
    """V = (1/2) m w^2 x^2 * I + coupling * sigma_x (time independent, Hermitian)."""
    h = m * omega ** 2 * x ** 2 / 2
    return sp.Matrix([[h, coupling], [coupling, h]])


def rabi_oscillator_solution():
    """Exact solution of the coupled system: phi0 e^{-i E0 t/hbar} [cos(g t/hbar), -i sin(g t/hbar)].

    Position stays in the oscillator ground state while the internal state Rabi-flops."""
    phi0, E0 = ground_state()
    env = phi0 * sp.exp(-sp.I * E0 * t / hbar)
    return sp.Matrix([env * sp.cos(g * t / hbar), -sp.I * env * sp.sin(g * t / hbar)]), coupled_oscillator_potential()


def stationary_pair():
    """Psi_pm = phi0 [1, +-1]/sqrt(2) with E_pm = hbar w / 2 +- g."""
    phi0, E0 = ground_state()
    return [(phi0 / sp.sqrt(2) * sp.Matrix([1, s]), E0 + s * g) for s in (1, -1)]


def norm_density(Psi):
    """Psi^dagger Psi = |psi1|^2 + |psi2|^2."""
    return sp.simplify((Psi.H * Psi)[0, 0])


def total_probability(Psi):
    """Integral of Psi^dagger Psi over the real line."""
    return sp.simplify(sp.integrate(norm_density(Psi), (x, -sp.oo, sp.oo)))


# ---------------------------------------------------------------- manufactured solution (hbar = m = 1)

MS_COUPLING = sp.Rational(3, 10) + sp.I * sp.Rational(1, 5)   # W = 0.3 + 0.2 i
MS_DETUNING = sp.Rational(1, 2)


def manufactured():
    """A smooth, deliberately NON-solution Psi_m, a Hermitian V with complex coupling, and the
    analytic forcing F = R[Psi_m] (hbar = m = 1). A correct discretization of R applied to
    Psi_m must converge to F."""
    Psi = sp.Matrix([
        sp.exp(-x ** 2 / 2) * (sp.cos(t) + sp.I * x * sp.sin(2 * t)),
        sp.Rational(1, 2) * sp.exp(-(x - 1) ** 2 / 2) * sp.exp(-sp.I * t),
    ])
    V = hermitian_potential(x ** 2 / 2, x ** 2 / 2 + MS_DETUNING, sp.re(MS_COUPLING), sp.im(MS_COUPLING))
    F = simplify_matrix(vector_residual(Psi, V, hb=sp.S.One, mass=sp.S.One))
    return Psi, V, F


def lambdify_vector(M):
    """Vectorized numpy function (x, t) -> complex array of shape (2, ...)."""
    fs = [sp.lambdify((x, t), M[j], "numpy") for j in range(M.shape[0])]
    return lambda xv, tv: np.array([np.broadcast_to(np.asarray(f(xv, tv), complex), np.shape(xv)) for f in fs])


def potential_numeric(V):
    """Numeric V(x) of shape (2, 2, n); each entry lambdified separately (constants broadcast)."""
    fs = [[sp.lambdify(x, V[i, j], "numpy") for j in range(2)] for i in range(2)]
    return lambda xv: np.array([[np.broadcast_to(np.asarray(fs[i][j](xv), complex), np.shape(xv))
                                 for j in range(2)] for i in range(2)])


def discrete_residual(psi_func, V_func, xg, t0, dt, hb=1.0, mass=1.0):
    """Finite-difference residual at time t0 on interior points of grid xg:
    i hbar (Psi(t+dt) - Psi(t-dt)) / 2dt + hbar^2/2m (Psi_{k+1} - 2 Psi_k + Psi_{k-1}) / dx^2 - V Psi.
    Returns (x_interior, R) with R of shape (2, n-2)."""
    xg = np.asarray(xg, float)
    if xg.ndim != 1 or xg.size < 3:
        raise ValueError("xg must be a 1D grid with >= 3 points")
    if dt <= 0:
        raise ValueError("dt must be positive")
    dx = xg[1] - xg[0]
    P = psi_func(xg, t0)
    Pt = (psi_func(xg, t0 + dt) - psi_func(xg, t0 - dt)) / (2 * dt)
    Pxx = (P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) / dx ** 2
    Vm = V_func(xg[1:-1])
    VP = np.einsum("ijk,jk->ik", Vm, P[:, 1:-1])
    return xg[1:-1], 1j * hb * Pt[:, 1:-1] + hb ** 2 / (2 * mass) * Pxx - VP


def rms(R):
    """RMS norm over grid points of the vector residual: sqrt(mean_k(|R1_k|^2 + |R2_k|^2))."""
    R = np.asarray(R)
    return float(np.sqrt(np.mean(np.sum(np.abs(R) ** 2, axis=0))))


def convergence_orders(errors, hs):
    """Observed orders log(e_i/e_{i+1}) / log(h_i/h_{i+1})."""
    e, h = np.asarray(errors, float), np.asarray(hs, float)
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


# ---------------------------------------------------------------- discrete stationary problem

def block_hamiltonian(xg, V11, V22, W, hb=1.0, mass=1.0):
    """2N x 2N finite-difference Hamiltonian [[T+V11, W], [W*, T+V22]], Dirichlet walls.
    Sparse (SciPy) when available, dense NumPy otherwise."""
    xg = np.asarray(xg, float)
    n = xg.size
    if n < 3:
        raise ValueError("need at least 3 grid points")
    dx = xg[1] - xg[0]
    c = hb ** 2 / (2 * mass * dx ** 2)
    v11, v22 = np.broadcast_to(V11, n).astype(float), np.broadcast_to(V22, n).astype(float)
    w = np.broadcast_to(W, n).astype(complex)
    if HAVE_SCIPY:
        T = sps.diags([-c * np.ones(n - 1), 2 * c * np.ones(n), -c * np.ones(n - 1)], [-1, 0, 1])
        return sps.bmat([[T + sps.diags(v11), sps.diags(w)], [sps.diags(w.conj()), T + sps.diags(v22)]]).tocsc()
    T = np.diag(2 * c * np.ones(n)) - c * np.eye(n, k=1) - c * np.eye(n, k=-1)
    return np.block([[T + np.diag(v11), np.diag(w)], [np.diag(w.conj()), T + np.diag(v22)]])


def lowest_states(H, k=4):
    """Lowest k eigenpairs (deterministic): SciPy shift-invert eigsh, or NumPy eigh."""
    if HAVE_SCIPY and sps.issparse(H):
        v0 = np.ones(H.shape[0])
        vals, vecs = spsl.eigsh(H, k=k, sigma=-1e3, which="LM", v0=v0)
    else:
        Hd = H.toarray() if hasattr(H, "toarray") else H
        vals, vecs = np.linalg.eigh(Hd)
    order = np.argsort(vals)[:k]
    return vals[order], vecs[:, order]


def stationary_discrete_residuals(H, vals, vecs):
    """||H Psi - E Psi|| / ||Psi|| per eigenpair (the discrete R_E)."""
    R = H @ vecs - vecs * vals[None, :]
    return np.linalg.norm(R, axis=0) / np.linalg.norm(vecs, axis=0)


# ---------------------------------------------------------------- internal two-level dynamics + optional GPU backends

def two_level_propagator(coupling, time, hb=1.0):
    """U = exp(-i coupling sigma_x t / hbar) (SciPy expm when available, closed form otherwise)."""
    H = np.array([[0, coupling], [coupling, 0]], complex)
    if HAVE_SCIPY:
        return sla.expm(-1j * H * time / hb)
    th = coupling * time / hb
    return np.array([[np.cos(th), -1j * np.sin(th)], [-1j * np.sin(th), np.cos(th)]])


def rabi_z_expectation_cpu(coupling, times, hb=1.0):
    """<sigma_z>(t) starting in [1, 0]: cos(2 g t / hbar), computed from the propagator."""
    Z = np.diag([1.0, -1.0])
    out = []
    for tt in np.asarray(times, float):
        s = two_level_propagator(coupling, tt, hb) @ np.array([1, 0], complex)
        out.append(float(np.real(s.conj() @ Z @ s)))
    return np.array(out)


def cudaq_available():
    try:
        import cudaq  # noqa: F401
        return True
    except ImportError:
        return False


def rabi_z_expectation_cudaq(coupling, times, hb=1.0):   # pragma: no cover - needs cudaq
    """Same <sigma_z>(t) with CUDA-Q: exp(-i g t X / hbar) = rx(2 g t / hbar). Returns None if
    cudaq is not installed."""
    try:
        import cudaq
    except ImportError:
        return None

    @cudaq.kernel
    def rabi(theta: float):
        q = cudaq.qubit()
        rx(theta, q)  # noqa: F821  (CUDA-Q kernel builtin)

    return np.array([cudaq.observe(rabi, cudaq.spin.z(0), 2 * coupling * tt / hb).expectation()
                     for tt in np.asarray(times, float)])


def cuquantum_available():
    try:
        import cuquantum  # noqa: F401
        return True
    except ImportError:
        return False


def apply_propagator_cuquantum(U, psi):   # pragma: no cover - needs cuquantum + GPU
    """Contract U_ij psi_j with cuQuantum's tensor-network contract. None if unavailable."""
    try:
        try:
            from cuquantum.tensornet import contract
        except ImportError:
            from cuquantum import contract
    except ImportError:
        return None
    return np.asarray(contract("ij,j->i", np.asarray(U, complex), np.asarray(psi, complex)))


# ---------------------------------------------------------------- isolation table

def isolation_rows():
    """One row per isolated term/condition/backend, each with a check computed live."""
    Psi_g, V_g, R_g, f = general_residual()
    re_im, _, _ = real_imag_residuals()
    Psi_x, V_x = rabi_oscillator_solution()
    R_x = simplify_matrix(vector_residual(Psi_x, V_x))
    phi0, E0 = ground_state()
    k = sp.symbols("k", real=True)

    def ok(b):
        return "PASS" if b else "FAIL"

    plane = sp.exp(sp.I * k * x) * sp.Matrix([1, 0])
    td = sp.exp(-sp.I * E * t / hbar) * phi0 * sp.Matrix([1, 0])
    r1_ok = all(sp.simplify(sp.expand(re_im[2 * j] + sp.I * re_im[2 * j + 1] - R_g[j])) == 0 for j in range(2))
    st = [simplify_matrix(stationary_residual(P, coupled_oscillator_potential(), En)) for P, En in stationary_pair()]
    wrong = simplify_matrix(stationary_residual(stationary_pair()[0][0], coupled_oscillator_potential(), E0))
    norm = total_probability(Psi_x)
    rows = [
        ("C1", "psi1(x,t) = u1 + i v1", "component (internal state 1)", "m^-1/2",
         ok(R_g[0].has(f["u1"]) and R_g[1].has(f["u1"])) + ": enters R1 directly and R2 via W*"),
        ("C2", "psi2(x,t) = u2 + i v2", "component (internal state 2)", "m^-1/2",
         ok(R_g[1].has(f["u2"]) and R_g[0].has(f["u2"])) + ": enters R2 directly and R1 via W"),
        ("D1", "i hbar dPsi/dt", "time derivative", "J m^-1/2",
         ok(sp.simplify(sp.I * hbar * td.diff(t) - E * td) == sp.zeros(2, 1)) + ": i hbar d/dt e^{-iEt/hbar} = E"),
        ("D2", "(hbar^2/2m) d^2Psi/dx^2", "kinetic (dispersion) term", "J m^-1/2",
         ok(sp.simplify(hbar ** 2 / (2 * m) * plane.diff(x, 2) + hbar ** 2 * k ** 2 / (2 * m) * plane) == sp.zeros(2, 1))
         + ": plane wave gives -hbar^2 k^2/2m"),
        ("K1", "V11 psi1", "diagonal potential, state 1", "J m^-1/2", ok(V_g[0, 0].is_real is not False) + ": V11 real"),
        ("K2", "V22 psi2", "diagonal potential, state 2", "J m^-1/2", ok(V_g[1, 1].is_real is not False) + ": V22 real"),
        ("K3", "W psi2 in R1, W = a + i b", "coupling term", "J m^-1/2", ok(R_g[0].has(f["a"]) and R_g[0].has(f["b"])) + ": a, b in R1"),
        ("K4", "W* psi1 in R2", "Hermitian partner of coupling", "J m^-1/2", ok(is_hermitian(V_g)) + ": V - V^dagger = 0"),
        ("P1", "Re R1, Im R1, Re R2, Im R2", "four real PDE residuals", "J m^-1/2", ok(r1_ok) + ": Re + i Im reconstructs R"),
        ("X1", "R[phi0 e^{-iE0t/hbar} (cos gt/hbar, -i sin gt/hbar)]", "exact (Rabi) solution", "J m^-1/2",
         ok(R_x == sp.zeros(2, 1)) + ": analytic residual = 0"),
        ("S1", "R_E for phi0 (1, +-1)/sqrt2, E = hbar w/2 +- g", "stationary residual", "J m^-1/2",
         ok(all(s == sp.zeros(2, 1) for s in st) and wrong != sp.zeros(2, 1)) + ": 0 for E+-, nonzero for E0"),
        ("IC", "Psi(x,0) = phi0 (1, 0)", "initial condition", "m^-1/2",
         ok(sp.simplify(Psi_x.subs(t, 0) - phi0 * sp.Matrix([1, 0])) == sp.zeros(2, 1)) + ": t=0 substitution"),
        ("BC", "Psi -> 0 as |x| -> oo (Dirichlet on the grid)", "boundary condition", "m^-1/2",
         ok(all(sp.limit(sp.Abs(e), x, sp.oo) == 0 for e in Psi_x.subs(t, 0))) + ": limit x->oo is 0"),
        ("N1", "integral Psi^dagger Psi dx = 1", "normalization", "1",
         ok(norm == 1) + f": integral = {norm}"),
        ("N2", "d/dt integral Psi^dagger Psi dx = 0", "probability conservation", "s^-1",
         ok(sp.simplify(sp.diff(norm, t)) == 0) + ": V Hermitian => norm conserved"),
        ("B1", "FD block Hamiltonian, eigsh / expm", "CPU backend (NumPy/SciPy)", "-",
         f"SciPy {scipy.__version__} sparse" if HAVE_SCIPY else "NumPy dense fallback"),
        ("B2", "rx(2gt/hbar) kernel, observe Z", "optional CUDA-Q backend (qubit only)", "-",
         "installed" if cudaq_available() else "not installed: skipped, CPU reference used"),
        ("B3", "contract('ij,j->i', U, psi)", "optional cuQuantum backend (tensor network)", "-",
         "installed" if cuquantum_available() else "not installed: skipped, not needed for PDE residual"),
    ]
    return [dict(id=r[0], equation_or_term=r[1], role=r[2], units=r[3], verification=r[4]) for r in rows]
