"""Black holes as quantum-information objects: Hawking temperature, entropy, evaporation,
and the Page curve -- each derived in SymPy or simulated with qubits.

  * TEMPERATURE. Schwarzschild f(r) = 1 - r_s/r. Surface gravity
    kappa = c^2 f'(r_s)/2 = c^4/(4GM). Requiring the Euclidean (imaginary-time)
    metric to be smooth at the horizon forces a period beta = 2 pi c/kappa in
    imaginary time, i.e. a temperature T_H = hbar kappa / (2 pi c k_B)
    = hbar c^3 / (8 pi G M k_B).
  * ENTROPY. The first law  d(Mc^2) = T dS  integrates to the Bekenstein-Hawking
    entropy S = k_B c^3 A / (4 G hbar): one quarter of the horizon area in Planck
    units. Divided by k_B ln 2 it is a count of BITS.
  * EVAPORATION. Stefan-Boltzmann (photons only, no greybody factors):
    dM/dt = -hbar c^4 / (15360 pi G^2 M^2), lifetime t = 5120 pi G^2 M^3 / (hbar c^4).
  * INFORMATION. If evaporation is unitary, the entanglement entropy of the
    radiation must rise and then FALL (the Page curve), turning over when half
    the qubits are out. Hawking's semiclassical calculation instead gives a
    curve that keeps rising -- the information paradox. A random-state qubit
    model reproduces Page's curve exactly; a Hayden-Preskill run shows a
    "diary" thrown into an old black hole coming back out almost immediately.

The qubit models are toy models (Haar-random states and unitaries standing in
for black-hole dynamics), the standard teaching device of Page (1993) and
Hayden & Preskill (2007). They show the information-theoretic logic, not
quantum gravity. Education.
"""
import numpy as np
import sympy as sp

# CODATA / SI exact where defined
G = 6.67430e-11
C = 299792458.0
HBAR = 1.054571817e-34
K_B = 1.380649e-23
M_SUN = 1.98892e30
YEAR = 3.15576e7
SIGMA_SB = 5.670374419e-8

Gs, Ms, cs, hb, kB, r = sp.symbols("G M c hbar k_B r", positive=True)


# ── SymPy derivations ────────────────────────────────────────────────

def surface_gravity_symbolic():
    """kappa = c^2 f'(r_s) / 2 for f = 1 - r_s/r, r_s = 2GM/c^2."""
    rs = 2 * Gs * Ms / cs**2
    f = 1 - rs / r
    return sp.simplify(cs**2 * sp.diff(f, r).subs(r, rs) / 2)


def euclidean_period_symbolic():
    """Near the horizon write r = r_s + x and keep first order: f ~ f'(r_s) x.
    Proper radial distance rho = int dx / sqrt(f) = 2 sqrt(x / f'(r_s)).
    The Euclidean metric f c^2 dtau^2 + dx^2/f becomes
        (f'(r_s) c / 2)^2 rho^2 dtau^2 + d rho^2,
    which is flat polar coordinates (no cone tip) only if
        (f'(r_s) c / 2) tau has period 2 pi.
    Returns (rho(x), angular-coefficient, beta = period of tau)."""
    x, fp, tau = sp.symbols("x fprime tau", positive=True)
    rho = sp.integrate(1 / sp.sqrt(fp * x), (x, 0, x))
    x_of_rho = sp.solve(sp.Eq(sp.Symbol("rho", positive=True), rho), x)[0]
    g_tt = (fp * x_of_rho) * cs**2                         # f c^2 at first order
    coeff = sp.sqrt(sp.simplify(g_tt / sp.Symbol("rho", positive=True) ** 2))
    beta = sp.simplify(2 * sp.pi / coeff)
    return rho, coeff, beta


def hawking_temperature_symbolic():
    """T = hbar / (k_B beta) with beta the Euclidean period, f'(r_s) = 1/r_s."""
    _, _, beta = euclidean_period_symbolic()
    rs = 2 * Gs * Ms / cs**2
    beta = beta.subs(sp.Symbol("fprime", positive=True), 1 / rs)
    return sp.simplify(hb / (kB * beta))


def entropy_from_first_law():
    """Integrate dS = c^2 dM / T_H from 0 to M and compare with k_B c^3 A/(4 G hbar)."""
    T = hawking_temperature_symbolic()
    m = sp.symbols("m", positive=True)
    S = sp.integrate((cs**2 / T).subs(Ms, m), (m, 0, Ms))
    A = 4 * sp.pi * (2 * Gs * Ms / cs**2) ** 2
    bh = kB * cs**3 * A / (4 * Gs * hb)
    return sp.simplify(S), sp.simplify(bh), sp.simplify(S - bh)


def evaporation_symbolic():
    """dM/dt = -sigma A T^4 / c^2 with sigma = pi^2 k_B^4 / (60 hbar^3 c^2).
    Returns (dM/dt, lifetime from M0 to 0)."""
    t, M0 = sp.symbols("t M_0", positive=True)
    T = hawking_temperature_symbolic()
    sigma = sp.pi**2 * kB**4 / (60 * hb**3 * cs**2)
    A = 4 * sp.pi * (2 * Gs * Ms / cs**2) ** 2
    dMdt = sp.simplify(-sigma * A * T**4 / cs**2)
    m = sp.symbols("m", positive=True)
    lifetime = sp.simplify(sp.integrate(-1 / dMdt.subs(Ms, m), (m, 0, M0)))
    return dMdt, lifetime


def unruh_temperature_symbolic():
    """T_U = hbar a / (2 pi c k_B): an accelerated observer sees a thermal bath.
    With a = kappa this is exactly the Hawking formula."""
    a = sp.symbols("a", positive=True)
    return hb * a / (2 * sp.pi * cs * kB)


# ── numbers ──────────────────────────────────────────────────────────

def _pos(name, v):
    if np.any(np.asarray(v, float) <= 0):
        raise ValueError(f"{name} must be positive")


def schwarzschild_radius(M):
    _pos("M", M)
    return 2 * G * np.asarray(M, float) / C**2


def hawking_temperature(M):
    _pos("M", M)
    return HBAR * C**3 / (8 * np.pi * G * np.asarray(M, float) * K_B)


def mass_for_temperature(T):
    _pos("T", T)
    return HBAR * C**3 / (8 * np.pi * G * K_B * np.asarray(T, float))


def entropy_bits(M):
    """S / (k_B ln 2) = 4 pi G M^2 / (hbar c ln 2)."""
    _pos("M", M)
    return 4 * np.pi * G * np.asarray(M, float) ** 2 / (HBAR * C * np.log(2))


def evaporation_power(M):
    _pos("M", M)
    return HBAR * C**6 / (15360 * np.pi * G**2 * np.asarray(M, float) ** 2)


def lifetime_s(M):
    _pos("M", M)
    return 5120 * np.pi * G**2 * np.asarray(M, float) ** 3 / (HBAR * C**4)


def mass_for_lifetime(t_s):
    _pos("t_s", t_s)
    return (np.asarray(t_s, float) * HBAR * C**4 / (5120 * np.pi * G**2)) ** (1 / 3)


def unruh_acceleration_for(T):
    _pos("T", T)
    return 2 * np.pi * C * K_B * np.asarray(T, float) / HBAR


def holographic_bits(radius_m):
    """Maximum information in a sphere: entropy of a black hole of that radius."""
    _pos("radius_m", radius_m)
    M = np.asarray(radius_m, float) * C**2 / (2 * G)
    return entropy_bits(M)


# ── qubit models: Page curve and Hayden-Preskill ─────────────────────

def haar_state(n_qubits, rng=None):
    rng = np.random.default_rng(rng)
    v = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return v / np.linalg.norm(v)


def haar_unitary(dim, rng=None):
    """QR of a complex Gaussian matrix with phase fix (Mezzadri 2007)."""
    rng = np.random.default_rng(rng)
    Z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))


def entanglement_entropy_bits(state, n_first, n_total):
    """Von Neumann entropy (bits) of the first n_first qubits of a pure state,
    from the Schmidt coefficients (SVD of the reshaped amplitude vector)."""
    if not 0 <= n_first <= n_total:
        raise ValueError("need 0 <= n_first <= n_total")
    M = np.asarray(state).reshape(2**n_first, 2 ** (n_total - n_first))
    p = np.linalg.svd(M, compute_uv=False) ** 2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def page_entropy_bits(n_a, n_b):
    """Page's average entanglement entropy for a random pure state on a
    2^n_a x 2^n_b bipartition (exact formula, Page 1993), in bits.
    For m <= n: S = sum_{k=n+1}^{mn} 1/k - (m-1)/(2n)  (nats)."""
    m, n = sorted((2**n_a, 2**n_b))
    s = np.sum(1.0 / np.arange(n + 1, m * n + 1)) - (m - 1) / (2 * n)
    return float(s / np.log(2))


def page_curve(n_total=12, n_samples=20, rng=0):
    """Radiation entropy vs number of emitted qubits k for random states.
    Returns (k, simulated mean, Page formula, Hawking's thermal guess = k)."""
    rng = np.random.default_rng(rng)
    ks = np.arange(n_total + 1)
    sim = np.zeros(len(ks))
    for _ in range(n_samples):
        psi = haar_state(n_total, rng)
        sim += [entanglement_entropy_bits(psi, k, n_total) for k in ks]
    sim /= n_samples
    page = np.array([page_entropy_bits(k, n_total - k) if 0 < k < n_total else 0.0 for k in ks])
    return ks, sim, page, ks.astype(float)


def mutual_information_bits(state, A, B, n_total):
    """I(A:B) = S(A) + S(B) - S(AB) for qubit index lists A, B of a pure state."""
    def S(idx):
        idx = sorted(idx)
        rest = [i for i in range(n_total) if i not in idx]
        psi = np.moveaxis(np.asarray(state).reshape([2] * n_total), idx + rest, list(range(n_total)))
        return entanglement_entropy_bits(psi.reshape(-1), len(idx), n_total)
    return S(A) + S(B) - S(list(A) + list(B))


def hayden_preskill(n_bh=5, n_emit_max=4, rng=0):
    """Old black hole B (n_bh qubits) maximally entangled with early radiation E.
    Alice's diary D (1 qubit) is maximally entangled with a reference R and
    thrown in. A Haar-random unitary scrambles B+D; then k qubits are emitted.
    Returns k and I(R : E + emitted) in bits; 2 bits = diary fully recoverable.

    Qubit order: R, D, B_1..B_n, E_1..E_n (all Bell pairs to start)."""
    rng = np.random.default_rng(rng)
    n = 2 + 2 * n_bh
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    # R-D Bell pair, then B_i-E_i Bell pairs, reorder to R, D, B..., E...
    psi = bell
    for _ in range(n_bh):
        psi = np.kron(psi, bell)
    order = [0, 1] + [2 + 2 * i for i in range(n_bh)] + [3 + 2 * i for i in range(n_bh)]
    psi = np.moveaxis(psi.reshape([2] * n), order, list(range(n))).reshape(-1)
    # scramble D + B (qubits 1 .. n_bh+1)
    U = haar_unitary(2 ** (n_bh + 1), rng)
    psi = psi.reshape(2, 2 ** (n_bh + 1), 2**n_bh)
    psi = np.einsum("ij,ajb->aib", U, psi).reshape(-1)
    E = list(range(2 + n_bh, n))
    out = []
    for k in range(n_emit_max + 1):
        emitted = list(range(1, 1 + k))            # the first k of the scrambled D+B qubits
        out.append(mutual_information_bits(psi, [0], E + emitted, n))
    return np.arange(n_emit_max + 1), np.array(out)


# ── DoD / federal QIS timeline (public, high-confidence milestones only) ──

QIS_TIMELINE = [
    (2002, "ARDA-sponsored Quantum Information Science and Technology Roadmap (v1, LANL-led)", "roadmap"),
    (2004, "QIST Roadmap version 2", "roadmap"),
    (2018, "National Quantum Initiative Act signed (December)", "law"),
    (2022, "USD(R&E) critical technology areas list names quantum science", "DoD policy"),
    (2022, "NSM-10: national security memo on quantum computing and migration to post-quantum cryptography", "policy"),
    (2022, "DARPA US2QC: underexplored systems for utility-scale quantum computing", "DARPA"),
    (2024, "NIST publishes first post-quantum cryptography standards (FIPS 203, 204, 205)", "standards"),
    (2024, "DARPA Quantum Benchmarking Initiative announced", "DARPA"),
]
