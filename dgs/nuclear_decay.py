"""Radioactive decay -- Griffiths Introduction to Nuclear Physics + Modern Physics.

GRIFFITHS CHAIN RULE FOR DECAY:
  The chain rule here is NOT calculus chain rule -- it is the BATEMAN EQUATIONS:
  a series of coupled first-order linear ODEs describing a decay chain:

    A -> B -> C -> D -> ... (stable)

  Each nuclide feeds the next:
    dN_A/dt = -lambda_A * N_A
    dN_B/dt = +lambda_A * N_A - lambda_B * N_B
    dN_C/dt = +lambda_B * N_B - lambda_C * N_C

  This IS the chain rule in the differential equation sense:
  the solution to dN_B/dt depends on the solution to dN_A/dt (chained ODEs).

  Griffiths calls this the "radioactive series" (Griffiths Modern Physics, Ch 12).
  The standard decay rate: A = lambda * N  (A = activity in Bq = decays/s)
  Half-life: t_{1/2} = ln(2) / lambda

SECULAR EQUILIBRIUM:
  When t_{1/2}(parent) >> t_{1/2}(daughter), the daughter reaches a steady state
  where its activity equals the parent's activity:
    A_daughter = A_parent  (secular equilibrium)
  This is why 226Ra and 222Rn activities are equal in old uranium ore.

U-238 DECAY SERIES (the most important in nuclear physics):
  238U (4.47 Gyr) -> 234Th (24.1 d) -> 234Pa (1.17 min) -> 234U (245 kyr)
  -> 230Th (75.4 kyr) -> 226Ra (1600 yr) -> 222Rn (3.82 d) -> 218Po (3.05 min)
  -> 214Pb (26.8 min) -> 214Bi (19.7 min) -> 214Po (163 us) -> 210Pb (22.3 yr)
  -> 210Bi (5.01 d) -> 210Po (138 d) -> 206Pb (STABLE)

GRIFFITHS CHAPTER CONNECTIONS:
  Ch 3 (Quantum Mechanics) -> |psi|^2 = probability per unit volume
  Ch 12 (Nuclear Physics) -> decay constant lambda = 1/tau, half-life t_{1/2}
  Q value:    Q = (M_parent - M_daughters) * c^2  (energy released per decay)
  Alpha decay: tunneling through Coulomb barrier (Gamow factor)
  Beta decay:  weak interaction; W boson mediates n -> p + e^- + nu_bar_e
  Gamma decay: EM transition between nuclear energy levels (no mass change)
"""
import numpy as np
import sympy as sp


# ── physical constants ────────────────────────────────────────────────
LN2 = np.log(2)
AVOGADRO = 6.02214076e23
U_TO_KG = 1.66053906660e-27   # 1 atomic mass unit in kg
C_M_PER_S = 2.99792458e8
C2_J_PER_U = U_TO_KG * C_M_PER_S**2   # 1 u * c^2 in Joules = 931.5 MeV

# ── U-238 decay chain (abridged to key nuclides) ─────────────────────
# Each entry: (symbol, A, Z, half_life_s, decay_mode, Q_MeV)
U238_CHAIN = [
    ("238U",  238, 92, 4.47e9 * 3.156e7, "alpha",  4.27),
    ("234Th", 234, 90, 24.1   * 86400,   "beta-",  0.27),
    ("234Pa", 234, 91, 1.17   * 60,      "beta-",  2.19),
    ("234U",  234, 92, 245e3  * 3.156e7, "alpha",  4.86),
    ("230Th", 230, 90, 75.4e3 * 3.156e7, "alpha",  4.77),
    ("226Ra", 226, 88, 1600   * 3.156e7, "alpha",  4.87),
    ("222Rn", 222, 86, 3.82   * 86400,   "alpha",  5.59),
    ("218Po", 218, 84, 3.05   * 60,      "alpha",  6.11),
    ("214Pb", 214, 82, 26.8   * 60,      "beta-",  1.02),
    ("214Bi", 214, 83, 19.7   * 60,      "beta-",  3.27),
    ("214Po", 214, 84, 163.7e-6,         "alpha",  7.83),
    ("210Pb", 210, 82, 22.3   * 3.156e7, "beta-",  0.06),
    ("210Bi", 210, 83, 5.01   * 86400,   "beta-",  1.16),
    ("210Po", 210, 84, 138    * 86400,   "alpha",  5.41),
    ("206Pb", 206, 82, None,             "stable", 0.0),
]


def half_life(lambda_per_s):
    """Convert decay constant to half-life.

    t_{1/2} = ln(2) / lambda
    Unit: same as 1/lambda (seconds if lambda in s^-1).
    """
    if lambda_per_s <= 0:
        raise ValueError("lambda must be positive")
    return LN2 / lambda_per_s


def decay_constant(t_half_s):
    """Convert half-life to decay constant lambda = ln(2) / t_{1/2}."""
    if t_half_s <= 0:
        raise ValueError("half-life must be positive")
    return LN2 / t_half_s


def activity(N_atoms, lambda_per_s):
    """Activity A = lambda * N  [Becquerel = decays/s].

    1 Curie = 3.7e10 Bq (originally defined as activity of 1 gram of 226Ra).
    """
    if lambda_per_s < 0 or N_atoms < 0:
        raise ValueError("N and lambda must be non-negative")
    A_Bq = lambda_per_s * N_atoms
    return {"A_Bq": A_Bq, "A_Ci": A_Bq / 3.7e10, "A_mCi": A_Bq / 3.7e7}


def n_atoms_from_mass(mass_g, A_mass):
    """Number of atoms from mass.

    N = (mass_g / A_mass) * N_Avogadro
    """
    if mass_g < 0 or A_mass <= 0:
        raise ValueError("mass must be non-negative, A_mass positive")
    return (mass_g / A_mass) * AVOGADRO


def single_decay(N0, lambda_per_s, t_s):
    """Single-species radioactive decay: N(t) = N0 * exp(-lambda * t).

    Returns N(t), fraction remaining, activity at t.
    """
    if lambda_per_s < 0:
        raise ValueError("lambda must be non-negative (0 for stable)")
    N_t = N0 * np.exp(-lambda_per_s * np.asarray(t_s, dtype=float))
    A_t = lambda_per_s * N_t
    return {
        "N_t": N_t,
        "fraction_remaining": N_t / N0 if N0 > 0 else np.zeros_like(N_t),
        "A_Bq": A_t,
        "t_half_s": half_life(lambda_per_s) if lambda_per_s > 0 else None,
    }


# ── Bateman equations (decay chain) ──────────────────────────────────

def bateman_two_step(N_A0, N_B0, lambda_A, lambda_B, t_arr):
    """Analytic solution to a two-step decay chain A -> B -> C(stable).

    dN_A/dt = -lambda_A * N_A
    dN_B/dt = +lambda_A * N_A - lambda_B * N_B

    Solution (Bateman 1910):
      N_A(t) = N_A0 * exp(-lambda_A * t)
      N_B(t) = N_A0 * lambda_A / (lambda_B - lambda_A)
               * [exp(-lambda_A * t) - exp(-lambda_B * t)]
               + N_B0 * exp(-lambda_B * t)

    This IS the chain rule of ODEs: N_B depends on the full history of N_A.
    """
    if lambda_A < 0 or lambda_B < 0:
        raise ValueError("decay constants must be non-negative")
    t = np.asarray(t_arr, dtype=float)
    N_A = N_A0 * np.exp(-lambda_A * t)

    if abs(lambda_B - lambda_A) < 1e-30:
        # Degenerate case: lambda_A ~ lambda_B (use L'Hopital)
        N_B = (N_A0 * lambda_A * t * np.exp(-lambda_A * t) +
               N_B0 * np.exp(-lambda_B * t))
    else:
        N_B = (N_A0 * lambda_A / (lambda_B - lambda_A) *
               (np.exp(-lambda_A * t) - np.exp(-lambda_B * t)) +
               N_B0 * np.exp(-lambda_B * t))

    A_A = lambda_A * N_A
    A_B = lambda_B * N_B
    return {
        "t": t, "N_A": N_A, "N_B": N_B,
        "A_A_Bq": A_A, "A_B_Bq": A_B,
        "N_C": N_A0 + N_B0 - N_A - N_B,   # conservation: total nucleons
    }


def secular_equilibrium_time(lambda_A, lambda_B, tol=0.01):
    """Estimate time to reach secular equilibrium (A_B / A_A -> 1).

    Secular equilibrium is reached when the daughter's transient
    exp(-lambda_B * t) << 1, i.e., t >> 1/lambda_B (several daughter half-lives).
    At equilibrium: A_B = A_A (activities equal).
    Rule of thumb: t_eq ~ 7 * t_{1/2}(daughter) (99% equilibrium).
    """
    if lambda_B <= 0:
        raise ValueError("lambda_B must be positive")
    t_half_B = half_life(lambda_B)
    t_eq_s = 7 * t_half_B
    return {
        "t_equilibrium_s": t_eq_s,
        "t_equilibrium_days": t_eq_s / 86400,
        "t_half_daughter_s": t_half_B,
        "ratio_at_equilibrium": lambda_A / lambda_B if lambda_B > 0 else None,
        "secular_eq_condition": "t_{1/2}(parent) >> t_{1/2}(daughter)",
    }


def bateman_chain_numerical(lambdas, N0_arr, t_arr):
    """Numerical solution to an n-nuclide decay chain using RK4.

    Solves dN_i/dt = lambda_{i-1}*N_{i-1} - lambda_i*N_i
    for a chain of n nuclides. Last nuclide may be stable (lambda_n=0).

    lambdas: list of decay constants [lambda_1, ..., lambda_n] (lambda_n=0 for stable)
    N0_arr:  initial populations
    """
    n = len(lambdas)
    if len(N0_arr) != n:
        raise ValueError("lambdas and N0_arr must have same length")
    lambdas = np.asarray(lambdas, dtype=float)
    t = np.asarray(t_arr, dtype=float)

    def dNdt(N, _t):
        dN = np.zeros(n)
        for i in range(n):
            if i > 0:
                dN[i] += lambdas[i-1] * N[i-1]
            dN[i] -= lambdas[i] * N[i]
        return dN

    dt = t[1] - t[0] if len(t) > 1 else t[-1] / 100
    N = np.zeros((len(t), n))
    N[0] = np.asarray(N0_arr, dtype=float)
    for j in range(1, len(t)):
        h = t[j] - t[j-1]
        Nj = N[j-1]
        k1 = dNdt(Nj, t[j-1])
        k2 = dNdt(Nj + h/2 * k1, t[j-1] + h/2)
        k3 = dNdt(Nj + h/2 * k2, t[j-1] + h/2)
        k4 = dNdt(Nj + h * k3, t[j])
        N[j] = Nj + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        N[j] = np.maximum(N[j], 0)   # no negative populations

    activities = lambdas[np.newaxis, :] * N
    return {"t": t, "N": N, "A_Bq": activities}


# ── Q-value and decay energy ──────────────────────────────────────────

def q_value_alpha(M_parent_u, M_daughter_u, M_He4_u=4.002602):
    """Q value for alpha decay: Q = (M_parent - M_daughter - M_alpha) * c^2.

    Q > 0: exothermic (energy released -> kinetic energy of products).
    Q < 0: endothermic (not spontaneous).
    M in atomic mass units (u). 1 u*c^2 = 931.494 MeV.
    """
    Q_u = M_parent_u - M_daughter_u - M_He4_u
    Q_MeV = Q_u * 931.494
    Q_J = Q_u * C2_J_PER_U
    return {"Q_MeV": Q_MeV, "Q_J": Q_J, "spontaneous": Q_MeV > 0}


def q_value_beta_minus(M_parent_u, M_daughter_u):
    """Q value for beta-minus decay: Q = (M_parent - M_daughter) * c^2.

    For atomic masses (not nuclear): the electron masses cancel.
    n -> p + e^- + anti-nu_e  (mediated by W^- boson, weak interaction)
    """
    Q_u = M_parent_u - M_daughter_u
    Q_MeV = Q_u * 931.494
    return {"Q_MeV": Q_MeV, "Q_J": Q_u * C2_J_PER_U, "spontaneous": Q_MeV > 0}


def gamow_factor(Z_daughter, Z_alpha=2, E_MeV=5.0, R_fm=7.0):
    """Gamow tunneling factor for alpha decay (WKB approximation).

    G = sqrt(2*mu*V_C / hbar^2) * integral
    Simplified Gamow formula:
      G = Z_d * Z_alpha * e^2 / (hbar * v_alpha)
          ~ 2*pi * Z_d * Z_alpha * e^2 / (4*pi*eps0 * hbar * c * beta_alpha)
    Result: P_tunnel ~ exp(-2*G)

    The Geiger-Nuttall law: log(lambda) ~ a + b/sqrt(Q)
    This is why alpha decay half-lives span 20 orders of magnitude.
    """
    alpha_fine = 1/137.036   # fine structure constant
    m_alpha_MeV = 3727.379   # rest mass of alpha in MeV/c^2
    v_over_c = np.sqrt(2 * E_MeV / m_alpha_MeV)   # non-relativistic
    G = np.pi * Z_daughter * Z_alpha * alpha_fine / v_over_c
    P_tunnel = np.exp(-2 * G)
    return {
        "G_gamow": G,
        "P_tunnel": P_tunnel,
        "log10_P": np.log10(P_tunnel) if P_tunnel > 0 else -np.inf,
        "Z_daughter": Z_daughter, "E_MeV": E_MeV,
    }


# ── U-238 chain overview ──────────────────────────────────────────────

def u238_chain_overview():
    """Return the U-238 -> Pb-206 decay chain with half-lives and activities."""
    rows = []
    for symbol, A, Z, t_half, mode, Q in U238_CHAIN:
        if t_half is not None:
            lam = LN2 / t_half
            t_half_yr = t_half / 3.156e7
        else:
            lam = 0.0
            t_half_yr = None
        rows.append({
            "symbol": symbol, "A": A, "Z": Z,
            "t_half_s": t_half, "t_half_yr": t_half_yr,
            "lambda_per_s": lam, "decay_mode": mode, "Q_MeV": Q,
        })
    return rows


def secular_equilibrium_activities(N_U238_atoms):
    """Activities of all daughters assuming secular equilibrium with U-238.

    In secular equilibrium: A_i = A_parent for all i.
    This is the asymptotic state of ancient uranium ore.
    """
    chain = u238_chain_overview()
    lam_parent = chain[0]["lambda_per_s"]
    A_parent_Bq = lam_parent * N_U238_atoms
    results = []
    for row in chain:
        results.append({
            "symbol": row["symbol"],
            "A_Bq": A_parent_Bq if row["lambda_per_s"] > 0 else 0,
            "secular_eq": True,
        })
    return {"A_parent_Bq": A_parent_Bq, "daughters": results}


# ── modern physics field tree ─────────────────────────────────────────

PHYSICS_FIELD_TREE = {
    "Classical Mechanics": {
        "prereqs": [],
        "enables": ["Classical EM", "Thermodynamics", "Quantum Mechanics"],
        "key_tools": ["Newton F=ma", "Lagrangian", "Hamiltonian", "action principle"],
    },
    "Linear Algebra": {
        "prereqs": [],
        "enables": ["Quantum Mechanics", "ML/AI", "Signal Processing", "Quantum Info"],
        "key_tools": ["eigenvectors", "Hermitian matrices", "inner product", "SVD"],
    },
    "Fourier Analysis": {
        "prereqs": ["Linear Algebra"],
        "enables": ["Signal Processing", "Quantum Mechanics", "Optics", "FNO"],
        "key_tools": ["DFT", "convolution theorem", "delta function", "Parseval"],
    },
    "Classical EM": {
        "prereqs": ["Classical Mechanics", "Fourier Analysis"],
        "enables": ["Optics", "Special Relativity", "Quantum Mechanics"],
        "key_tools": ["Maxwell equations", "Poynting vector", "wave equation"],
    },
    "Thermodynamics": {
        "prereqs": ["Classical Mechanics"],
        "enables": ["Statistical Mechanics", "Nuclear Physics"],
        "key_tools": ["dU=TdS-PdV", "partition function", "entropy"],
    },
    "Statistical Mechanics": {
        "prereqs": ["Thermodynamics", "Fourier Analysis"],
        "enables": ["Quantum Statistics", "Solid State Physics"],
        "key_tools": ["Boltzmann dist", "Maxwell-Boltzmann", "Fermi-Dirac", "Bose-Einstein"],
    },
    "Special Relativity": {
        "prereqs": ["Classical EM"],
        "enables": ["Nuclear Physics", "Particle Physics", "QFT"],
        "key_tools": ["Lorentz transform", "4-vectors", "E=mc^2", "invariant interval"],
    },
    "Quantum Mechanics": {
        "prereqs": ["Classical Mechanics", "Linear Algebra", "Fourier Analysis"],
        "enables": ["Nuclear Physics", "Atomic Physics", "Quantum Info", "Solid State"],
        "key_tools": ["Schrodinger eq", "Dirac notation", "operators", "commutators"],
    },
    "Atomic Physics": {
        "prereqs": ["Quantum Mechanics", "Classical EM"],
        "enables": ["Optics/Lasers", "Nuclear Physics", "Chemistry"],
        "key_tools": ["Bohr model", "Rydberg", "selection rules", "fine structure"],
    },
    "Nuclear Physics": {
        "prereqs": ["Quantum Mechanics", "Special Relativity"],
        "enables": ["Particle Physics", "Medical Physics", "Nuclear Engineering"],
        "key_tools": ["Bateman equations", "Q-value", "Gamow factor", "shell model"],
    },
    "Solid State Physics": {
        "prereqs": ["Quantum Mechanics", "Statistical Mechanics"],
        "enables": ["Semiconductor Physics", "Photonics", "Superconductivity"],
        "key_tools": ["Kronig-Penney", "Bloch theorem", "band structure", "Fermi surface"],
    },
    "Optics": {
        "prereqs": ["Classical EM", "Fourier Analysis"],
        "enables": ["Photonics", "Jones Calculus", "Holography", "Phase Retrieval"],
        "key_tools": ["Huygens-Fresnel", "diffraction", "Jones matrix", "coherence"],
    },
    "Signal Processing": {
        "prereqs": ["Fourier Analysis"],
        "enables": ["DSP", "ML/AI", "Communications", "Phase Retrieval"],
        "key_tools": ["Z-transform", "Nyquist", "windowing", "matched filter"],
    },
    "Phase Retrieval / GS": {
        "prereqs": ["Optics", "Signal Processing", "Fourier Analysis"],
        "enables": ["Dispersion-Assisted GS", "Optical AI", "SEALS"],
        "key_tools": ["GS iteration", "diversity measurement", "FNO", "PINN"],
    },
    "ML/AI": {
        "prereqs": ["Linear Algebra", "Signal Processing", "Statistical Mechanics"],
        "enables": ["Photonic AI", "FNO", "Unsupervised PR"],
        "key_tools": ["backprop", "attention", "FNO", "diffusion models"],
    },
    "Quantum Info": {
        "prereqs": ["Quantum Mechanics", "Linear Algebra"],
        "enables": ["Topological QC", "QKD", "Quantum ML"],
        "key_tools": ["qubits", "Bell states", "Chern number", "Kitaev chain"],
    },
}


def physics_field_tree(topic=None):
    """Return the dependency tree for modern physics topics.

    If topic is given, return what that topic needs (prereqs) and enables.
    If topic is None, return full tree.
    """
    if topic is not None:
        if topic not in PHYSICS_FIELD_TREE:
            available = list(PHYSICS_FIELD_TREE.keys())
            raise ValueError(f"Unknown topic '{topic}'. Available: {available}")
        node = PHYSICS_FIELD_TREE[topic]
        return {"topic": topic, **node}
    return PHYSICS_FIELD_TREE


def topological_sort_physics():
    """Topological sort of the physics field tree (learning order).

    Returns topics sorted so all prerequisites appear before the topic.
    This is the NON-LINEAR Feynman order: learn by dependency, not chapter.
    """
    tree = PHYSICS_FIELD_TREE
    visited = set()
    order = []

    def visit(topic):
        if topic in visited:
            return
        visited.add(topic)
        if topic in tree:
            for prereq in tree[topic]["prereqs"]:
                visit(prereq)
        order.append(topic)

    for topic in tree:
        visit(topic)
    return order


# ── SymPy decay chain ─────────────────────────────────────────────────

def nuclear_decay_sympy_5():
    """Five key equations in nuclear decay (SymPy)."""
    t_s, N_s = sp.symbols('t N', positive=True)
    lam_A, lam_B = sp.symbols('lambda_A lambda_B', positive=True)
    N_A0, N_B0 = sp.symbols('N_{A0} N_{B0}', positive=True)
    Q_s, m_s, c_s = sp.symbols('Q m c', positive=True)

    N_A_sol = N_A0 * sp.exp(-lam_A * t_s)
    N_B_bateman = (N_A0 * lam_A / (lam_B - lam_A) *
                   (sp.exp(-lam_A * t_s) - sp.exp(-lam_B * t_s)) +
                   N_B0 * sp.exp(-lam_B * t_s))

    return {
        "single_decay": sp.Eq(sp.Function('N')(t_s), N_A0 * sp.exp(-lam_A * t_s)),
        "half_life": sp.Eq(sp.Symbol('t_{1/2}'), sp.log(2) / lam_A),
        "Bateman_N_A": sp.Eq(sp.Symbol('N_A(t)'), N_A_sol),
        "Bateman_N_B": sp.Eq(sp.Symbol('N_B(t)'), N_B_bateman),
        "Q_value": sp.Eq(Q_s, (m_s - sp.Symbol('m_daughters')) * c_s**2),
    }


if __name__ == "__main__":
    print("=== Single decay: 14C (t_{1/2}=5730 yr) ===")
    lam_C14 = LN2 / (5730 * 3.156e7)
    t = np.array([0, 5730, 11460, 17190]) * 3.156e7
    r = single_decay(1e12, lam_C14, t)
    for i, ti in enumerate([0, 1, 2, 3]):
        print(f"  t={ti} half-lives: N/N0={r['fraction_remaining'][i]:.4f}")

    print("\n=== Bateman 2-step: 226Ra -> 222Rn ===")
    lam_Ra = LN2 / (1600 * 3.156e7)
    lam_Rn = LN2 / (3.82 * 86400)
    t_arr = np.linspace(0, 10 * 86400, 500)
    b = bateman_two_step(1e15, 0, lam_Ra, lam_Rn, t_arr)
    print(f"  At t=10 days: A_Ra={b['A_A_Bq'][-1]:.2e} Bq, A_Rn={b['A_B_Bq'][-1]:.2e} Bq")
    print(f"  Secular eq check: A_Rn/A_Ra = {b['A_B_Bq'][-1]/b['A_A_Bq'][-1]:.4f} (->1)")

    print("\n=== U-238 chain (first 5 nuclides) ===")
    chain = u238_chain_overview()
    for row in chain[:5]:
        t_hr = (row['t_half_s'] / 3600) if row['t_half_s'] else None
        print(f"  {row['symbol']:8s}  {row['decay_mode']:6s}  "
              f"Q={row['Q_MeV']:.2f} MeV  "
              f"t_{'{1/2}'}={t_hr:.2e} h" if t_hr else f"  {row['symbol']} STABLE")

    print("\n=== Gamow factor for 210Po alpha (E=5.4 MeV, Z_d=82) ===")
    g = gamow_factor(Z_daughter=82, E_MeV=5.41)
    print(f"  G={g['G_gamow']:.2f}, P_tunnel={g['P_tunnel']:.2e}")

    print("\n=== Modern physics learning order (topological sort) ===")
    order = topological_sort_physics()
    for i, topic in enumerate(order, 1):
        print(f"  {i:2d}. {topic}")

    print("\n=== SymPy nuclear decay equations ===")
    for k, eq in nuclear_decay_sympy_5().items():
        print(f"  {k}: {eq}")
