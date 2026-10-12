"""Cells, molecules, and Hilbert space: one master equation from ion channels to qubits.

Three objects that look unrelated are the same linear-algebra machine:

  * CELL PHYSIOLOGY -- an ion channel flickers closed <-> open. The open
    probability obeys a classical MASTER EQUATION  dp/dt = Q p, where Q is a
    rate matrix whose columns sum to 0 (probability is conserved in the L1 norm).
  * MOLECULES / QM -- an electronic excitation shared by two chromophores
    (a photosynthetic "dimer") is a vector in a 2-dim HILBERT SPACE and evolves
    unitarily, d psi/dt = -i H psi (conserved in the L2 norm). Population sloshes
    back and forth: Rabi oscillation.
  * QIS -- a warm, wet environment keeps "measuring" which molecule holds the
    excitation. In the DENSITY MATRIX this is pure dephasing (a Lindblad term)
    that kills off-diagonal coherence at rate gamma. For large gamma the quantum
    dynamics collapses onto a classical two-state master equation with hopping
    rate  k = 2 J^2 gamma / (gamma^2 + Delta^2)  -- the same equation as the ion
    channel. Decoherence is the bridge from Hilbert space to Markov chains.

Units: energies in cm^-1 (spectroscopy convention), times in femtoseconds,
hbar = 1 internally after converting cm^-1 -> rad/fs. Parameters below are
illustrative textbook magnitudes, not fits to any specific protein. Whether
long-lived coherence matters for real photosynthesis at body temperature is
an open, contested research question; this module shows the mechanism, it
does not settle that question. Education.
"""
import numpy as np
import sympy as sp
from scipy.linalg import expm

from dgs.membrane_biophysics import nernst_potential

C_CM_PER_FS = 2.99792458e10 * 1e-15        # speed of light [cm/fs]
K_B_CM = 0.6950348                          # Boltzmann constant [cm^-1 / K]
BODY_T_K = 310.0


def cm_to_rad_per_fs(nu_cm):
    """Energy E = h c nu~  ->  angular frequency omega = 2 pi c nu~ [rad/fs]."""
    return 2 * np.pi * C_CM_PER_FS * np.asarray(nu_cm, float)


def thermal_energy_cm(T_K=BODY_T_K):
    """k_B T in wavenumbers; ~215 cm^-1 at body temperature."""
    if T_K <= 0:
        raise ValueError("T_K must be positive")
    return K_B_CM * T_K


# ── 1. cell physiology: the ion channel as a classical two-state system ──

def channel_generator(k_open, k_close):
    """Rate matrix Q for closed (index 0) <-> open (index 1).
    dp/dt = Q p, columns sum to zero, off-diagonals non-negative."""
    if k_open < 0 or k_close < 0:
        raise ValueError("rates must be non-negative")
    return np.array([[-k_open, k_close], [k_open, -k_close]], float)


def master_evolve(Q, p0, times):
    """p(t) = expm(Q t) p0 for each t. Returns array (len(times), n)."""
    p0 = np.asarray(p0, float)
    return np.array([expm(Q * tt) @ p0 for tt in np.atleast_1d(times)])


def stationary_open_probability(k_open, k_close):
    return k_open / (k_open + k_close)


def telegraph_trace(k_open, k_close, t_end, rng=None, start_open=False):
    """Single-channel record by the Gillespie algorithm: exponential dwell
    times with mean 1/k. Returns (switch_times, states) as step-function data."""
    rng = np.random.default_rng(rng)
    t, s = 0.0, int(start_open)
    times, states = [0.0], [s]
    while t < t_end:
        rate = k_close if s else k_open
        t += rng.exponential(1 / rate)
        s = 1 - s
        times.append(min(t, t_end))
        states.append(s)
    return np.array(times), np.array(states)


def time_average_open(times, states):
    """Fraction of time open in a telegraph record (ergodic estimate of p_open)."""
    dwell = np.diff(times)
    return float(np.sum(dwell * states[:-1]) / (times[-1] - times[0]))


def single_channel_current_pA(V_mV, gamma_pS=20.0, C_out=5.0, C_in=140.0, z=1):
    """Open-channel current i = gamma (V - E_rev) for a K+ channel, using the
    repo's Nernst potential for E_rev. pS * mV = 1e-15 A = 1e-3 pA."""
    E_rev_mV = 1e3 * nernst_potential(C_out, C_in, z=z)
    return gamma_pS * (V_mV - E_rev_mV) * 1e-3


# ── 2. molecules in Hilbert space: the excitonic dimer ───────────────

def dimer_hamiltonian(delta_cm, J_cm):
    """H = [[+Delta/2, J], [J, -Delta/2]] in rad/fs. Site basis |1>, |2> =
    'excitation on chromophore 1 / 2'. Delta = site-energy gap, J = coupling."""
    d, J = cm_to_rad_per_fs(delta_cm), cm_to_rad_per_fs(J_cm)
    return np.array([[d / 2, J], [J, -d / 2]], complex)


def rabi_population(t_fs, delta_cm, J_cm):
    """Coherent transfer P_2(t) = (4J^2/Omega^2) sin^2(Omega t / 2),
    Omega = sqrt(Delta^2 + 4 J^2) (all in rad/fs)."""
    d, J = cm_to_rad_per_fs(delta_cm), cm_to_rad_per_fs(J_cm)
    Om = np.sqrt(d**2 + 4 * J**2)
    return 4 * J**2 / Om**2 * np.sin(Om * np.asarray(t_fs, float) / 2) ** 2


def liouvillian(H, gamma=0.0, sink=0.0, loss=0.0):
    """Superoperator L with vec(rho)' = L vec(rho) (row-major vec), for
        rho' = -i (H_eff rho - rho H_eff^dag) - gamma (rho - diag(rho))
    H_eff = H - i(sink/2)|2><2| - i(loss/2) I. gamma = pure dephasing rate
    [1/fs] of site coherences; sink = trapping from site 2 into a reaction
    center; loss = excitation recombination from every site."""
    n = H.shape[0]
    Heff = H.astype(complex).copy()
    Heff[-1, -1] -= 0.5j * sink
    Heff -= 0.5j * loss * np.eye(n)
    I = np.eye(n)
    L = -1j * (np.kron(Heff, I) - np.kron(I, Heff.conj()))
    if gamma:
        off = np.ones((n, n)) - I
        L -= gamma * np.diag(off.ravel())
    return L


def evolve_density(H, rho0, times, gamma=0.0, sink=0.0, loss=0.0):
    """rho(t) for each t, shape (len(times), n, n)."""
    L = liouvillian(H, gamma, sink, loss)
    n = H.shape[0]
    v0 = np.asarray(rho0, complex).ravel()
    return np.array([(expm(L * tt) @ v0).reshape(n, n) for tt in np.atleast_1d(times)])


def site_ket(k, n=2):
    v = np.zeros(n, complex)
    v[k] = 1
    return v


def projector(v):
    v = np.asarray(v, complex)
    return np.outer(v, v.conj())


def purity(rho):
    """Tr rho^2: 1 for a pure state, 1/n for maximally mixed."""
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy_bits(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


def bloch_vector(rho):
    """(x, y, z) = (2 Re rho12, -2 Im rho12, rho11 - rho22) for a 2x2 rho."""
    return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])


def incoherent_rate(delta_cm, J_cm, gamma):
    """Strong-dephasing hopping rate k = 2 J^2 gamma / (gamma^2 + Delta^2) [1/fs],
    from adiabatically eliminating the coherence (see derive_incoherent_rate)."""
    d, J = cm_to_rad_per_fs(delta_cm), cm_to_rad_per_fs(J_cm)
    return 2 * J**2 * gamma / (gamma**2 + d**2)


def transfer_efficiency(delta_cm, J_cm, gamma, sink=1 / 1000.0, loss=1 / 20000.0):
    """Quantum yield of trapping: eta = sink * int_0^inf rho_22(t) dt with the
    excitation starting on site 1. Exact: int e^{Lt} dt v0 = -L^{-1} v0."""
    H = dimer_hamiltonian(delta_cm, J_cm)
    L = liouvillian(H, gamma, sink, loss)
    v0 = projector(site_ket(0)).ravel()
    integral = -np.linalg.solve(L, v0).reshape(2, 2)
    return float(sink * integral[1, 1].real)


# ── 3. computer science: what Hilbert space costs ────────────────────

def hilbert_memory_bytes(n_sites, representation="density", manifold="full"):
    """Memory for a complex128 state of n two-level molecules.
    manifold='full': dim 2^n (any number of excitations, like n qubits);
    manifold='single': dim n (exactly one excitation -- the photosynthesis case).
    representation='vector' (psi) or 'density' (rho, dim^2 entries)."""
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if manifold not in ("full", "single") or representation not in ("vector", "density"):
        raise ValueError("bad manifold or representation")
    dim = 2**n_sites if manifold == "full" else n_sites
    return 16 * (dim if representation == "vector" else dim**2)


# ── SymPy derivations ────────────────────────────────────────────────

def derive_rabi():
    """Diagonalize the dimer and return P_2(t) = |<2|e^{-iHt}|1>|^2 simplified."""
    d, J, t = sp.symbols("Delta J t", positive=True)
    H = sp.Matrix([[d / 2, J], [J, -d / 2]])
    U = (-sp.I * H * t).exp()
    P2 = sp.simplify(sp.expand(sp.Abs(U[1, 0]) ** 2, complex=True))
    Om = sp.sqrt(d**2 + 4 * J**2)
    target = 4 * J**2 / Om**2 * sp.sin(Om * t / 2) ** 2
    return P2, target, sp.simplify(sp.expand_trig(P2 - target).rewrite(sp.exp))


def derive_incoherent_rate():
    """Adiabatic elimination: set d rho12/dt = 0, solve for rho12, insert into
    d rho11/dt. Returns k such that d rho11/dt = k (rho22 - rho11)."""
    d, J, g = sp.symbols("Delta J gamma", positive=True)
    r11, r22 = sp.symbols("rho11 rho22", real=True)
    r12 = sp.symbols("rho12")
    H = sp.Matrix([[d / 2, J], [J, -d / 2]])
    rho = sp.Matrix([[r11, r12], [sp.conjugate(r12), r22]])
    drho = -sp.I * (H * rho - rho * H)
    eq12 = sp.Eq(drho[0, 1] - g * r12, 0)
    sol12 = sp.solve(eq12, r12)[0]
    # rho21 = conj(rho12); build d rho11/dt with both
    d11 = drho[0, 0].subs(sp.conjugate(r12), sp.conjugate(sol12)).subs(r12, sol12)
    k = sp.simplify(sp.expand_complex(d11 / (r22 - r11)))
    return k


def derive_master_stationary():
    """Q p = 0 with p0 + p1 = 1  ->  p_open = k_o / (k_o + k_c)."""
    ko, kc, p = sp.symbols("k_o k_c p", positive=True)
    Q = sp.Matrix([[-ko, kc], [ko, -kc]])
    return sp.solve((Q * sp.Matrix([1 - p, p]))[1], p)[0]


if __name__ == "__main__":
    print("kT(310 K) =", round(thermal_energy_cm(), 1), "cm^-1")
    print("k_incoh SymPy:", derive_incoherent_rate())
    for g in (0.0, 0.01, 0.1, 1.0, 10.0):
        print(f"gamma={g:6.2f}/fs  eta={transfer_efficiency(300, 50, g):.3f}")
