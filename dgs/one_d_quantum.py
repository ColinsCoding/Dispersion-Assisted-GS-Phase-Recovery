"""
One-dimensional quantum mechanics (Knight, Physics for Scientists and Engineers, Ch. 40).

In plain words
--------------
Everything in this chapter is one equation, the time-independent Schrodinger equation

    -(hbar^2 / 2m) psi''(x) + U(x) psi(x) = E psi(x),

solved for different potential-energy curves U(x). Draw U(x) as an **energy diagram**, draw
each allowed energy E_n as a horizontal line on it, and draw psi_n(x) sitting on its line.
Most chapter questions can be read straight off that picture:

* where E > U the particle oscillates (wave-like, shorter wavelength where E - U is larger);
* where E < U the wave function decays with penetration distance eta = hbar / sqrt(2m(U - E));
* psi_n has n - 1 nodes;
* a photon emitted or absorbed between two lines has lambda = hc / |E_i - E_j|.

Models (Knight conventions; n starts at 1 everywhere, including the oscillator)
-------------------------------------------------------------------------------
* rigid box, width L:     E_n = n^2 h^2 / (8 m L^2),  psi_n = sqrt(2/L) sin(n pi x / L)
* finite well, depth U0:  levels below U0 from the even/odd matching conditions
* harmonic oscillator:    E_n = (n - 1/2) hbar omega   (Griffiths writes (n + 1/2), n from 0)
* barrier, width w:       P_tunnel ~ exp(-2w / eta), plus the exact rectangular-barrier result
* any U(x):               ``solve_schrodinger`` (finite differences)

The CS / ML view: ``solve_schrodinger`` turns the differential equation into a symmetric
tridiagonal matrix H and calls an eigensolver. The energies are eigenvalues and the wave
functions are eigenvectors, the same linear algebra as PCA.

The quantum-information view: a qubit is two lines on an energy diagram. ``qubit_from_levels``
gives the qubit frequency f01 = (E2 - E1)/h and the anharmonicity alpha. If alpha = 0 (the
harmonic oscillator), a pulse tuned to 1 -> 2 also drives 2 -> 3, so the two levels can't be
isolated. ``rabi_population`` shows that a resonant pi pulse is a NOT gate.

Units: lengths in nm, energies in eV, mass in units of the electron mass m_e.

Sources (IEEE style)
--------------------
[1] R. D. Knight, Physics for Scientists and Engineers: A Strategic Approach, 5th ed.
    Pearson, 2022, ch. 40.
[2] D. J. Griffiths and D. F. Schroeter, Introduction to Quantum Mechanics, 3rd ed.
    Cambridge Univ. Press, 2018, secs. 2.2-2.6 (finite well, oscillator, barrier transmission).
[3] E. Tiesinga et al., "CODATA recommended values of the fundamental physical constants:
    2018," Rev. Mod. Phys., vol. 93, 025010, 2021.
"""
import math

import numpy as np

HBAR_C = 197.3269804          # eV nm
HC = 2 * math.pi * HBAR_C     # 1239.84198 eV nm
M_E_C2 = 510998.95            # eV
H_EV_S = 4.135667696e-15      # Planck constant, eV s
C_NM = HBAR_C ** 2 / (2 * M_E_C2)   # hbar^2 / (2 m_e) = 0.0380998 eV nm^2


def _kin(m):
    """hbar^2 / (2m) in eV nm^2 for mass m (units of m_e)."""
    if not m > 0:
        raise ValueError(f"mass must be > 0 (units of m_e), got {m}")
    return C_NM / m


def _check_n(n):
    if int(n) != n or n < 1:
        raise ValueError(f"quantum number n must be an integer >= 1 (Knight convention), got {n}")
    return int(n)


def _check_pos(name, val):
    if not val > 0:
        raise ValueError(f"{name} must be > 0, got {val}")


# --- 40.3-40.5 rigid box ------------------------------------------------------------------
def box_energy(n, L, m=1.0):
    """E_n = n^2 h^2 / (8 m L^2) in eV for a box of width L nm."""
    n = _check_n(n)
    _check_pos("L", L)
    return n ** 2 * math.pi ** 2 * _kin(m) / L ** 2


def box_psi(x, n, L):
    """Normalized psi_n(x) = sqrt(2/L) sin(n pi x / L) on [0, L], zero outside (nm^-1/2)."""
    n = _check_n(n)
    _check_pos("L", L)
    x = np.asarray(x, dtype=float)
    inside = (x >= 0) & (x <= L)
    return np.where(inside, math.sqrt(2 / L) * np.sin(n * math.pi * x / L), 0.0)


def box_probability(n, L, a, b):
    """Probability of finding the particle in [a, b] (clipped to [0, L]), from the exact integral."""
    n = _check_n(n)
    _check_pos("L", L)
    a, b = max(0.0, min(a, b)), min(float(L), max(a, b))
    if b <= a:
        return 0.0
    k2 = 2 * n * math.pi / L
    return (b - a) / L - (math.sin(k2 * b) - math.sin(k2 * a)) / (k2 * L)


# --- 40.6 finite well ---------------------------------------------------------------------
def penetration_depth(E, U0, m=1.0):
    """eta = hbar / sqrt(2m(U0 - E)) in nm; requires E < U0."""
    if not E < U0:
        raise ValueError(f"penetration depth needs E < U0, got E={E}, U0={U0}")
    return math.sqrt(_kin(m) / (U0 - E))


def _bisect(f, lo, hi, tol=1e-13):
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def finite_well_levels(U0, L, m=1.0):
    """Bound-state energies (eV above the well bottom) of a well of depth U0 eV and width L nm.

    Uses z = kL/2 and z0 = (L/2) sqrt(U0 / (hbar^2/2m)); even states satisfy z tan z = sqrt(z0^2 - z^2),
    odd states -z cot z = sqrt(z0^2 - z^2). There is always at least one bound state.
    """
    _check_pos("U0", U0)
    _check_pos("L", L)
    C = _kin(m)
    z0 = 0.5 * L * math.sqrt(U0 / C)
    root = lambda z: math.sqrt(max(z0 ** 2 - z ** 2, 0.0))
    even = lambda z: z * math.tan(z) - root(z)
    odd = lambda z: -z / math.tan(z) - root(z)
    eps = 1e-12
    zs, j = [], 0
    while j * math.pi < z0:
        lo, hi = j * math.pi + eps, min(j * math.pi + math.pi / 2 - eps, z0)
        zs.append(_bisect(even, lo, hi))
        lo = j * math.pi + math.pi / 2 + eps
        if lo < z0:
            zs.append(_bisect(odd, lo, min((j + 1) * math.pi - eps, z0)))
        j += 1
    return np.array(sorted(C * (2 * z / L) ** 2 for z in zs))


# --- 40.8 harmonic oscillator -------------------------------------------------------------
def ho_energy(n, hbar_omega):
    """E_n = (n - 1/2) hbar omega in eV, n = 1, 2, 3, ... (Knight)."""
    n = _check_n(n)
    _check_pos("hbar_omega", hbar_omega)
    return (n - 0.5) * hbar_omega


def ho_length(hbar_omega, m=1.0):
    """Oscillator length b = sqrt(hbar / (m omega)) in nm; the classical turning point of level n is sqrt(2n - 1) b."""
    _check_pos("hbar_omega", hbar_omega)
    return math.sqrt(2 * _kin(m) / hbar_omega)


def ho_psi(x, n, hbar_omega, m=1.0):
    """Normalized oscillator wave function, Knight index n (Hermite order n - 1)."""
    n = _check_n(n)
    b = ho_length(hbar_omega, m)
    k = n - 1
    xi = np.asarray(x, dtype=float) / b
    coef = np.zeros(k + 1)
    coef[k] = 1.0
    norm = 1.0 / math.sqrt(2 ** k * math.factorial(k) * b * math.sqrt(math.pi))
    return norm * np.polynomial.hermite.hermval(xi, coef) * np.exp(-xi ** 2 / 2)


# --- 40.10 tunneling ----------------------------------------------------------------------
def tunneling_probability(E, U0, w, m=1.0, exact=False):
    """Barrier of height U0 eV and width w nm.

    exact=False: Knight's estimate exp(-2w/eta) (needs E < U0).
    exact=True:  rectangular-barrier transmission, valid for any E > 0.
    """
    _check_pos("E", E)
    _check_pos("U0", U0)
    _check_pos("w", w)
    C = _kin(m)
    if not exact:
        return math.exp(-2 * w / penetration_depth(E, U0, m))
    if E < U0:
        kap = math.sqrt((U0 - E) / C)
        return 1 / (1 + U0 ** 2 * math.sinh(kap * w) ** 2 / (4 * E * (U0 - E)))
    if E > U0:
        kp = math.sqrt((E - U0) / C)
        return 1 / (1 + U0 ** 2 * math.sin(kp * w) ** 2 / (4 * E * (E - U0)))
    return 1 / (1 + U0 * w ** 2 / (4 * C))


# --- 40.2 / 40.7 / 40.9 any potential -----------------------------------------------------
def solve_schrodinger(x, U, m=1.0, n_states=6):
    """Finite-difference eigenstates of -(hbar^2/2m) psi'' + U psi = E psi on the grid x (nm).

    psi = 0 is imposed just outside both ends, so make the grid wide enough that the states
    you want have decayed there. Returns (E [eV], psi [n_states, len(x)]), with each psi
    normalized so that sum |psi|^2 dx = 1 and its first lobe positive.
    """
    x = np.asarray(x, dtype=float)
    U = np.broadcast_to(np.asarray(U, dtype=float), x.shape)
    if x.ndim != 1 or len(x) < 3:
        raise ValueError("x must be a 1-D grid with at least 3 points")
    dx = np.diff(x)
    if not np.allclose(dx, dx[0]) or dx[0] <= 0:
        raise ValueError("x must be uniformly spaced and increasing")
    if int(n_states) != n_states or not 1 <= n_states <= len(x):
        raise ValueError(f"n_states must be an integer in [1, {len(x)}], got {n_states}")
    dx = dx[0]
    t = _kin(m) / dx ** 2
    H = np.diag(U + 2 * t) - t * np.eye(len(x), k=1) - t * np.eye(len(x), k=-1)
    E, vec = np.linalg.eigh(H)
    psi = vec[:, :n_states].T / math.sqrt(dx)
    for p in psi:
        first = np.argmax(np.abs(p) > 0.01 * np.abs(p).max())
        if p[first] < 0:
            p *= -1
    return E[:n_states], psi


def count_nodes(psi, rel_tol=1e-3):
    """Number of sign changes of psi, ignoring the tails where |psi| < rel_tol * max|psi|."""
    p = np.asarray(psi, dtype=float)
    p = p[np.abs(p) > rel_tol * np.abs(p).max()]
    return int(np.sum(np.sign(p[1:]) != np.sign(p[:-1])))


def local_wavelength(E, U, m=1.0):
    """de Broglie wavelength h / sqrt(2m(E - U)) in nm where E > U, NaN in forbidden regions."""
    K = np.asarray(E - np.asarray(U, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(K > 0, 2 * math.pi * np.sqrt(_kin(m) / np.where(K > 0, K, 1.0)), np.nan)


# --- energy diagrams, spectra, qubits -----------------------------------------------------
def transition_wavelengths(levels_eV, max_excitation=None):
    """All downward transitions between levels, as rows (upper, lower, photon_eV, lambda_nm).

    If max_excitation is given, keep only levels within that energy of the lowest one (what a
    collision or photon of that energy can reach from the ground state).
    """
    lv = np.sort(np.asarray(levels_eV, dtype=float))
    if max_excitation is not None:
        lv = lv[lv - lv[0] <= max_excitation + 1e-12]
    rows = []
    for i in range(len(lv)):
        for j in range(i):
            dE = lv[i] - lv[j]
            if dE > 0:
                rows.append((i + 1, j + 1, dE, HC / dE))
    return rows


def qubit_from_levels(levels_eV):
    """Qubit made from the two lowest levels: f01 [Hz], anharmonicity alpha = E23 - E12 [eV]."""
    lv = np.sort(np.asarray(levels_eV, dtype=float))
    if len(lv) < 3:
        raise ValueError("need at least 3 levels to define an anharmonicity")
    return {"f01_Hz": (lv[1] - lv[0]) / H_EV_S,
            "alpha_eV": (lv[2] - lv[1]) - (lv[1] - lv[0])}


def rabi_population(omega_rabi, t, detuning=0.0):
    """Upper-level population of a driven two-level system starting in the lower level.

    P(t) = (Omega^2 / Omega_eff^2) sin^2(Omega_eff t / 2), Omega_eff = sqrt(Omega^2 + Delta^2)
    (angular frequencies, consistent units). At zero detuning, t = pi / Omega flips the state: a NOT gate.
    """
    _check_pos("omega_rabi", omega_rabi)
    oe = math.hypot(omega_rabi, detuning)
    return (omega_rabi / oe) ** 2 * np.sin(oe * np.asarray(t, dtype=float) / 2) ** 2


def energy_diagram(ax, x, U, E, psi, scale=None, U_max=None, labels=True):
    """Draw U(x), each E_n as a line, and psi_n sitting on its line (Knight-style energy diagram)."""
    x = np.asarray(x, dtype=float)
    U = np.broadcast_to(np.asarray(U, dtype=float), x.shape)
    E = np.asarray(E, dtype=float)
    if scale is None:
        gap = np.min(np.diff(E)) if len(E) > 1 else max(E[0], 1e-9)
        scale = 0.4 * gap / max(np.abs(psi).max(), 1e-300)
    ax.plot(x, U, "k", lw=2, label="U(x)")
    for n, (En, p) in enumerate(zip(E, psi), start=1):
        ax.axhline(En, color="0.75", lw=0.8, zorder=0)
        ax.plot(x, En + scale * p, lw=1.4)
        if labels:
            ax.text(1.0, En, f" n={n}: {En:.3g} eV", va="center", fontsize=8,
                    transform=ax.get_yaxis_transform(), clip_on=False)
    top = U_max if U_max is not None else E[-1] + 0.6 * (E[-1] - E[0] + 1e-12)
    ax.set_ylim(min(U.min(), E[0]) - 0.1 * (top - U.min()), top)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("energy (eV)")
    return ax
