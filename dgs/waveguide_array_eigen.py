"""Eigenvectors in photonics: coupled waveguides from two guides to arrays,
with the quantum-information view (quantum walks, perfect state transfer,
Hong-Ou-Mandel). SymPy derivations, NumPy/SciPy checks.

Coupled-mode equations for N evanescently coupled single-mode waveguides
(amplitude a_n, propagation constant beta, coupling kappa_n between n, n+1):
    -i da/dz = H a,   H = beta I + tridiag(kappa)      ->   a(z) = expm(i H z) a(0)
Solving = diagonalising H: the EIGENVECTORS of H are the array's supermodes,
the EIGENVALUES their propagation constants. Light launched into one guide is
a superposition of supermodes that dephase as z grows.

    two guides       eigenvalues beta +- kappa, eigenvectors (1, +-1)/sqrt2,
                     power in guide 2 = sin^2(kappa z)
    uniform array    v_k(n) = sin(n k pi/(N+1)), lambda_k = beta + 2 kappa cos(k pi/(N+1))
    infinite array   a_n(z) = i^n J_n(2 kappa z)  (discrete diffraction), rms width sqrt2 kappa z
    engineered       kappa_n = kappa0 sqrt(n(N-n)): equally spaced eigenvalues ->
                     PERFECT transfer guide 1 -> guide N at z = pi/(2 kappa0)
    two photons      coincidence probability = |perm U|^2 = cos^2(2 kappa z):
                     zero at the 50:50 point (Hong-Ou-Mandel dip)
"""
import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.special import jv


# ── matrices ─────────────────────────────────────────────────────────

def coupling_matrix(N, kappa=1.0, beta=0.0, couplings=None):
    """H = beta I + tridiagonal coupling. `couplings` (length N-1) overrides kappa."""
    if N < 2:
        raise ValueError("need at least 2 waveguides")
    k = np.full(N - 1, float(kappa)) if couplings is None else np.asarray(couplings, float)
    if k.shape != (N - 1,):
        raise ValueError("couplings must have length N-1")
    return beta * np.eye(N) + np.diag(k, 1) + np.diag(k, -1)


def propagate_eig(H, a0, z):
    """a(z) = V e^{i Lambda z} V^T a0 (H real symmetric)."""
    lam, V = np.linalg.eigh(H)
    return V @ (np.exp(1j * lam * z) * (V.T @ np.asarray(a0, complex)))


def propagate_expm(H, a0, z):
    return expm(1j * H * z) @ np.asarray(a0, complex)


# ── SymPy derivations ────────────────────────────────────────────────

def two_guide_symbolic():
    """Eigen-decomposition of [[beta, kappa], [kappa, beta]] and the power transfer."""
    b, k, z = sp.symbols("beta kappa z", positive=True)
    H = sp.Matrix([[b, k], [k, b]])
    ev = H.eigenvects()
    U = sp.simplify((sp.I * H * z).exp())
    a = U * sp.Matrix([1, 0])
    P2 = sp.simplify(sp.expand_complex(a[1] * sp.conjugate(a[1])))
    return {"eigenvects": ev, "U": U, "P2": sp.simplify(P2.rewrite(sp.cos))}


def uniform_array_eigen_check():
    """Show (H v)_n = 2 kappa cos(t) v_n for v_n = sin(n t), any interior n, and
    that the boundary condition v_0 = v_{N+1} = 0 forces t = k pi/(N+1)."""
    n, t, k = sp.symbols("n t kappa")
    v = lambda m: sp.sin(m * t)
    residual = sp.simplify(sp.expand_trig(k * (v(n - 1) + v(n + 1)) - 2 * k * sp.cos(t) * v(n)))
    return residual


def uniform_array_eigen_numeric(N, kappa=1.0):
    """Analytic eigenpairs (sorted by eigenvalue) for comparison with numpy."""
    k = np.arange(1, N + 1)
    lam = 2 * kappa * np.cos(k * np.pi / (N + 1))
    n = np.arange(1, N + 1)
    V = np.sin(np.outer(n, k) * np.pi / (N + 1))
    V /= np.linalg.norm(V, axis=0)
    order = np.argsort(lam)
    return lam[order], V[:, order]


def bessel_identity_check():
    """SymPy: a_n = i^n J_n(2 kappa z) satisfies -i a_n' = kappa (a_{n-1} + a_{n+1})."""
    n = sp.Symbol("n", integer=True)
    k, z = sp.symbols("kappa z", positive=True)
    a = lambda m: sp.I ** m * sp.besselj(m, 2 * k * z)
    res = -sp.I * sp.diff(a(n), z) - k * (a(n - 1) + a(n + 1))
    return sp.simplify(sp.expand(res.rewrite(sp.besselj)))


def discrete_diffraction(n, kappa_z):
    """Infinite array, launch in guide 0: |a_n|^2 = J_n(2 kappa z)^2."""
    return jv(n, 2 * kappa_z) ** 2


def rms_width_infinite(kappa_z):
    """sqrt(sum n^2 J_n(2kz)^2) = sqrt(2) kappa z -- ballistic, not diffusive."""
    return np.sqrt(2) * np.asarray(kappa_z, float)


# ── perfect state transfer ───────────────────────────────────────────

def pst_couplings(N, kappa0=1.0):
    """kappa_n = kappa0 sqrt(n (N - n)), n = 1..N-1 (Christandl et al. 2004)."""
    n = np.arange(1, N)
    return kappa0 * np.sqrt(n * (N - n))


def pst_eigenvalues_symbolic(N):
    """SymPy eigenvalues of the engineered array (kappa0 = 1): -(N-1), -(N-3), ..., N-1."""
    k = [sp.sqrt(m * (N - m)) for m in range(1, N)]
    H = sp.zeros(N, N)
    for m, c in enumerate(k):
        H[m, m + 1] = H[m + 1, m] = c
    return sorted(H.eigenvals(multiple=True))


def transfer_fidelity(N, z, couplings=None, kappa=1.0):
    """|a_N(z)|^2 after launching in guide 1."""
    H = coupling_matrix(N, kappa, couplings=couplings)
    a0 = np.zeros(N); a0[0] = 1
    return float(np.abs(propagate_eig(H, a0, z)[-1]) ** 2)


# ── two photons: Hong-Ou-Mandel ──────────────────────────────────────

def coupler_unitary_symbolic():
    k, z = sp.symbols("kappa z", positive=True)
    return sp.Matrix([[sp.cos(k * z), sp.I * sp.sin(k * z)],
                      [sp.I * sp.sin(k * z), sp.cos(k * z)]]), k, z


def coincidence_probability_symbolic():
    """One photon in each input: P(one photon in each output) = |perm U|^2."""
    U, k, z = coupler_unitary_symbolic()
    perm = U[0, 0] * U[1, 1] + U[0, 1] * U[1, 0]
    return sp.simplify(sp.trigsimp(perm * sp.conjugate(perm))), k, z


def coincidence_distinguishable_symbolic():
    """Classical (distinguishable) particles: |U11 U22|^2 + |U12 U21|^2."""
    U, k, z = coupler_unitary_symbolic()
    P = sp.Abs(U[0, 0] * U[1, 1]) ** 2 + sp.Abs(U[0, 1] * U[1, 0]) ** 2
    return sp.simplify(sp.trigsimp(P)), k, z


def mc_coincidences(kappa_z, trials=100_000, seed=0, indistinguishable=True):
    """Sample output patterns for one photon in each input of the coupler."""
    rng = np.random.default_rng(seed)
    c, s = np.cos(kappa_z), np.sin(kappa_z)
    if indistinguishable:
        p_coinc = (c ** 2 - s ** 2) ** 2
    else:
        p_coinc = c ** 4 + s ** 4
    return float(np.mean(rng.random(trials) < p_coinc)), p_coinc
