"""Separation of variables, residuals, and where the Schrodinger equation earns money.

In plain words: the time-dependent Schrodinger equation
        i hbar dPsi/dt = -(hbar^2 / 2m) d^2Psi/dx^2 + V(x) Psi
looks hard, but when V does not depend on time you can GUESS Psi(x, t) = X(x) T(t).
Divide by X T and the left side depends only on t, the right side only on x; the only
way they can be equal for all x and t is if both are the same constant E:
        i hbar T'/T = E                  ->  T(t) = exp(-i E t / hbar)
        -(hbar^2/2m) X'' + V X = E X     ->  the time-independent equation (an eigenproblem).
Boundary conditions then allow only certain E: quantization. Linearity lets you add
separated solutions back up to get every other solution.

The RESIDUAL is how you check any claimed solution: plug it in and compute
        r = i hbar dPsi/dt - H Psi.
r = 0 exactly (SymPy) means it is a solution; on a grid, ||H psi - E psi|| measures
discretization error and shrinks like dx^2 for the 3-point Laplacian. A wrong energy
gives a residual that does not shrink. This is the same check used for the Gaussian in
dgs.diffraction_dispersion (free-particle Schrodinger = diffraction = dispersion).

Industry models built here (each one is a separated, 1D Schrodinger problem):
  * Gate-oxide direct tunneling (Intel, Texas Instruments, Applied Materials): a
    rectangular barrier; leakage grows ~3.5x per 0.1 nm thinner SiO2, which drove the
    switch to high-k HfO2 gate dielectrics.
  * Fowler-Nordheim tunneling in NAND flash (Micron, Western Digital): a triangular
    barrier under high field; WKB exponent 4 sqrt(2m) phi^{3/2} / (3 hbar e F).
  * Tunnel ionization behind attosecond pulses (2023 Physics Nobel): the SAME
    triangular-barrier exponent, with an atom's ionization energy and a laser's field.
  * Transmon qubits (IBM, Alphabet/Google): Schrodinger's equation in the junction phase
    phi with V = -E_J cos(phi); solved in the charge basis. Macroscopic quantum tunneling
    and energy quantization in a circuit were the 2025 Physics Nobel (Clarke, Devoret,
    Martinis).
  * Trapped-ion qubits (Honeywell via Quantinuum): the harmonic oscillator; ground-state
    size x0 = sqrt(hbar / 2 m omega) and the Lamb-Dicke parameter eta = dk x0.
  * Quantum chemistry (ExxonMobil, Pfizer, Merck as users): Born-Oppenheimer separation of
    nuclei from electrons; the H2 vibration as a Morse oscillator.
  * Quantum-well lasers in optical transceivers (Intel, Broadcom): particle in a box with
    an effective mass.

Company lists are illustrative and written from memory (Fortune 500 membership changes
yearly); they say who uses the physics, not what any company's internal models are.
NumPy + SymPy. Education.
"""

import math

import numpy as np
import sympy as sp

from dgs.tunneling import rectangular_barrier_T

HBAR = 1.054571817e-34      # J s
H_PLANCK = 6.62607015e-34   # J s
ME = 9.1093837015e-31       # kg
QE = 1.602176634e-19        # C
AMU = 1.66053906660e-27     # kg
C_LIGHT = 299792458.0       # m/s
EPS0 = 8.8541878128e-12     # F/m


def _pos(name, v):
    if not np.all(np.isfinite(v)) or np.any(np.asarray(v) <= 0):
        raise ValueError(f"{name} must be positive, got {v}")


# ---------------------------------------------------------------- SymPy: separation + residuals

def sympy_separation():
    """Separate the TDSE and return residuals of the classic separated solutions.

    Every residual here should simplify to exactly 0, except 'wrong_energy', which is the
    deliberate negative control (energy off by 1%)."""
    x, y, zc, t = sp.symbols("x y z t", real=True)
    hbar, m, L, w, k = sp.symbols("hbar m L omega k", positive=True)
    E = sp.symbols("E")
    n, n1, n2, n3 = sp.symbols("n n_1 n_2 n_3", positive=True, integer=True)
    X, T, V = sp.Function("X"), sp.Function("T"), sp.Function("V")

    def tdse_residual(Psi, Vx, coords=(x,)):
        lap = sum(sp.diff(Psi, c, 2) for c in coords)
        return sp.simplify(sp.I * hbar * sp.diff(Psi, t) - (-(hbar ** 2) / (2 * m) * lap + Vx * Psi))

    # the separation step: plug X(x)T(t) in, divide by X T
    Psi = X(x) * T(t)
    lhs = sp.I * hbar * sp.diff(Psi, t) / Psi
    rhs = (-(hbar ** 2) / (2 * m) * sp.diff(Psi, x, 2) + V(x) * Psi) / Psi
    T_sol = sp.dsolve(sp.Eq(sp.I * hbar * sp.diff(T(t), t), E * T(t)), T(t)).rhs

    out = {"lhs": sp.simplify(lhs), "rhs": sp.simplify(rhs), "T_sol": T_sol}

    # infinite well
    En = n ** 2 * sp.pi ** 2 * hbar ** 2 / (2 * m * L ** 2)
    box = sp.sqrt(2 / L) * sp.sin(n * sp.pi * x / L) * sp.exp(-sp.I * En * t / hbar)
    out["box_E"] = En
    out["box"] = tdse_residual(box, 0)
    out["box_bc"] = (box.subs(x, 0), sp.simplify(box.subs(x, L)))
    out["wrong_energy"] = tdse_residual(box.subs(t, t * sp.Rational(101, 100)), 0)

    # free particle plane wave (the Fourier mode used by the FFT propagators)
    plane = sp.exp(sp.I * (k * x - hbar * k ** 2 / (2 * m) * t))
    out["plane"] = tdse_residual(plane, 0)

    # harmonic oscillator, first five states (Hermite-Gauss)
    xi = sp.sqrt(m * w / hbar) * x
    out["qho"] = [tdse_residual(sp.hermite(j, xi) * sp.exp(-xi ** 2 / 2)
                                * sp.exp(-sp.I * (j + sp.Rational(1, 2)) * w * t),
                                m * w ** 2 * x ** 2 / 2) for j in range(5)]

    # 3D box: three separation constants, energies add
    E3 = sum(nj ** 2 for nj in (n1, n2, n3)) * sp.pi ** 2 * hbar ** 2 / (2 * m * L ** 2)
    box3 = (sp.sin(n1 * sp.pi * x / L) * sp.sin(n2 * sp.pi * y / L) * sp.sin(n3 * sp.pi * zc / L)
            * sp.exp(-sp.I * E3 * t / hbar))
    out["box3"] = tdse_residual(box3, 0, coords=(x, y, zc))
    out["box3_E"] = E3

    # superposition of two box states: still a solution (residual 0), but NOT separable
    E1, E2 = En.subs(n, 1), En.subs(n, 2)
    sup = (sp.sin(sp.pi * x / L) * sp.exp(-sp.I * E1 * t / hbar)
           + sp.sin(2 * sp.pi * x / L) * sp.exp(-sp.I * E2 * t / hbar))
    out["superposition"] = tdse_residual(sup, 0)
    out["bohr_omega"] = sp.simplify((E2 - E1) / hbar)
    out["symbols"] = dict(x=x, t=t, hbar=hbar, m=m, L=L, omega=w, k=k, n=n)
    return out


# ---------------------------------------------------------------- numeric eigen-residuals

def fd_hamiltonian(V, dx, m=1.0, hbar=1.0):
    """Dense 3-point finite-difference H = -(hbar^2/2m) D2 + diag(V), Dirichlet walls."""
    V = np.asarray(V, float)
    if V.ndim != 1 or V.size < 3:
        raise ValueError("V must be a 1D array with at least 3 samples")
    _pos("dx", dx)
    c = hbar ** 2 / (2 * m * dx ** 2)
    return np.diag(V + 2 * c) - c * np.eye(V.size, k=1) - c * np.eye(V.size, k=-1)


def eigen_residuals(H, energies, states):
    """Relative residual ||H psi - E psi|| / ||psi|| for each column of states."""
    states = np.atleast_2d(np.asarray(states))
    if states.shape[0] != H.shape[0]:
        states = states.T
    R = H @ states - states * np.asarray(energies)[None, :]
    return np.linalg.norm(R, axis=0) / np.linalg.norm(states, axis=0)


def box_residual_convergence(ns=(50, 100, 200, 400, 800), level=1, L=1.0, energy_scale=1.0):
    """Residual of the ANALYTIC box state on FD grids of increasing size.

    energy_scale=1 uses the exact E_n (residual ~ dx^2 -> 0); energy_scale=1.01 is the
    wrong-energy control, whose residual levels off at 0.01 E_n."""
    if level < 1:
        raise ValueError("level must be >= 1")
    out = []
    for n in ns:
        x = np.linspace(0, L, n + 2)[1:-1]
        dx = x[1] - x[0]
        psi = np.sin(level * np.pi * x / L)
        E = energy_scale * (level * np.pi / L) ** 2 / 2
        out.append((dx, eigen_residuals(fd_hamiltonian(np.zeros(n), dx), [E], psi[:, None])[0]))
    return np.array(out)


# ---------------------------------------------------------------- industry models

def box_energy_eV(n, L_nm, m_eff=1.0):
    """Infinite-well level E_n = n^2 h^2 / (8 m* L^2) in eV (m_eff in units of m_e)."""
    _pos("L_nm", L_nm); _pos("m_eff", m_eff)
    return np.asarray(n) ** 2 * H_PLANCK ** 2 / (8 * m_eff * ME * (L_nm * 1e-9) ** 2) / QE


def gate_oxide_transmission(t_ox_nm, barrier_eV=3.1, E_eV=0.1, m_eff=0.5):
    """Direct tunneling probability through a rectangular SiO2 gate oxide (dgs.tunneling)."""
    _pos("t_ox_nm", t_ox_nm)
    if not 0 < E_eV < barrier_eV:
        raise ValueError("need 0 < E_eV < barrier_eV")
    return rectangular_barrier_T(E_eV * QE, barrier_eV * QE, np.asarray(t_ox_nm) * 1e-9,
                                 m=m_eff * ME, hbar=HBAR)


def equivalent_oxide_thickness(t_phys_nm, k_highk, k_sio2=3.9):
    """EOT = t_phys * k_SiO2 / k_highk: the SiO2 thickness with the same capacitance."""
    _pos("t_phys_nm", t_phys_nm); _pos("k_highk", k_highk)
    return t_phys_nm * k_sio2 / k_highk


def triangular_barrier_exponent(phi_eV, F_V_per_m, m_eff=1.0):
    """WKB exponent for a triangular barrier V = phi - eFx:
        2 gamma = 4 sqrt(2 m) phi^{3/2} / (3 hbar e F).
    Fowler-Nordheim (flash) and tunnel ionization (attosecond physics) share it."""
    _pos("phi_eV", phi_eV); _pos("F_V_per_m", F_V_per_m); _pos("m_eff", m_eff)
    phi = phi_eV * QE
    return 4 * np.sqrt(2 * m_eff * ME) * phi ** 1.5 / (3 * HBAR * QE * np.asarray(F_V_per_m))


def fowler_nordheim_B(phi_eV, m_eff=1.0):
    """FN field constant B (V/m) so that J ~ F^2 exp(-B/F); ~6.83e9 sqrt(m*/m) phi^{3/2}."""
    return triangular_barrier_exponent(phi_eV, 1.0, m_eff)


def laser_peak_field(I_W_cm2):
    """Peak electric field of a linearly polarized laser: F = sqrt(2 I / (c eps0))."""
    _pos("I_W_cm2", I_W_cm2)
    return np.sqrt(2 * np.asarray(I_W_cm2) * 1e4 / (C_LIGHT * EPS0))


def ponderomotive_energy_eV(I_W_cm2, lam_m):
    """U_p = e^2 F^2 / (4 m omega^2), the electron's mean quiver energy, in eV."""
    _pos("lam_m", lam_m)
    om = 2 * np.pi * C_LIGHT / lam_m
    return QE ** 2 * laser_peak_field(I_W_cm2) ** 2 / (4 * ME * om ** 2) / QE


def keldysh_gamma(Ip_eV, I_W_cm2, lam_m):
    """gamma = sqrt(Ip / (2 Up)); gamma < 1 means tunneling dominates over multiphoton."""
    _pos("Ip_eV", Ip_eV)
    return np.sqrt(Ip_eV / (2 * ponderomotive_energy_eV(I_W_cm2, lam_m)))


def hhg_cutoff_eV(Ip_eV, I_W_cm2, lam_m):
    """High-harmonic cutoff Ip + 3.17 Up: the photon energy limit of attosecond sources."""
    return Ip_eV + 3.17 * ponderomotive_energy_eV(I_W_cm2, lam_m)


def transmon_levels(EJ, EC, ng=0.0, n_charge=30, n_levels=4):
    """Eigenvalues of H = 4 EC (n - ng)^2 - EJ cos(phi) in the charge basis |n|<=n_charge.

    cos(phi) couples n -> n+-1 with amplitude 1/2, so H is tridiagonal. Units of EJ, EC."""
    _pos("EJ", EJ); _pos("EC", EC)
    if n_charge < 1:
        raise ValueError("n_charge must be >= 1")
    nn = np.arange(-n_charge, n_charge + 1)
    H = np.diag(4 * EC * (nn - ng) ** 2) - EJ / 2 * (np.eye(nn.size, k=1) + np.eye(nn.size, k=-1))
    E, vec = np.linalg.eigh(H)
    return E[:n_levels], vec[:, :n_levels], H


def transmon_asymptotic(EJ, EC):
    """Large EJ/EC estimates: f01 ~ sqrt(8 EJ EC) - EC, anharmonicity ~ -EC (Koch et al. 2007)."""
    _pos("EJ", EJ); _pos("EC", EC)
    return np.sqrt(8 * EJ * EC) - EC, -EC


def transmon_charge_dispersion(EJ, EC, m=0):
    """Peak-to-peak charge dispersion of level m (Koch et al. 2007, eq. 2.18):
        eps_m ~ EC 2^{4m+5} / m! sqrt(2/pi) (EJ/2EC)^{m/2+3/4} exp(-sqrt(8 EJ/EC)).
    Exponentially small: tunneling between neighboring cos(phi) wells."""
    _pos("EJ", EJ); _pos("EC", EC)
    return (EC * 2 ** (4 * m + 5) / math.factorial(m) * np.sqrt(2 / np.pi)
            * (EJ / (2 * EC)) ** (m / 2 + 0.75) * np.exp(-np.sqrt(8 * EJ / EC)))


def ion_ground_state_size(mass_amu, f_trap_Hz):
    """x0 = sqrt(hbar / (2 m omega)), the zero-point spread of a trapped ion (m)."""
    _pos("mass_amu", mass_amu); _pos("f_trap_Hz", f_trap_Hz)
    return np.sqrt(HBAR / (2 * mass_amu * AMU * 2 * np.pi * f_trap_Hz))


def lamb_dicke(mass_amu, f_trap_Hz, lam_m, angle_deg=90.0):
    """eta = dk x0 for two Raman beams crossing at angle_deg: dk = 2 k sin(angle/2)."""
    _pos("lam_m", lam_m)
    dk = 2 * (2 * np.pi / lam_m) * np.sin(np.deg2rad(angle_deg) / 2)
    return dk * ion_ground_state_size(mass_amu, f_trap_Hz)


def morse_levels_cm(n, we_cm, De_eV):
    """Morse vibrational levels G(n) = we (n+1/2) - we xe (n+1/2)^2, we xe = we^2 / (4 De)."""
    _pos("we_cm", we_cm); _pos("De_eV", De_eV)
    De_cm = De_eV * QE / (H_PLANCK * C_LIGHT * 100)
    v = np.asarray(n) + 0.5
    return we_cm * v - we_cm ** 2 / (4 * De_cm) * v ** 2


def morse_fd_levels_cm(n_levels, we_cm, De_eV, mu_amu, n_grid=3000):
    """Same levels from the FD Hamiltonian with V = De (1 - e^{-a r})^2, plus their residuals."""
    De = De_eV * QE
    mu = mu_amu * AMU
    we = 2 * np.pi * C_LIGHT * 100 * we_cm
    a = we * np.sqrt(mu / (2 * De))
    r = np.linspace(-1.5 / a, 12 / a, n_grid)
    V = De * (1 - np.exp(-a * r)) ** 2
    H = fd_hamiltonian(V, r[1] - r[0], m=mu, hbar=HBAR)
    E, vec = np.linalg.eigh(H)
    res = eigen_residuals(H, E[:n_levels], vec[:, :n_levels]) / np.abs(E[:n_levels])
    return E[:n_levels] / (H_PLANCK * C_LIGHT * 100), res


INDUSTRY = [
    dict(companies="Intel, Texas Instruments, Applied Materials", physics="direct tunneling through gate oxide",
         model="rectangular barrier", function="gate_oxide_transmission"),
    dict(companies="Micron, Western Digital", physics="Fowler-Nordheim program/erase in NAND flash",
         model="triangular barrier (WKB)", function="triangular_barrier_exponent"),
    dict(companies="IBM, Alphabet (Google Quantum AI)", physics="superconducting transmon qubits",
         model="Schrodinger in phase: 4EC n^2 - EJ cos(phi)", function="transmon_levels"),
    dict(companies="Honeywell (majority owner of Quantinuum)", physics="trapped-ion qubits, motional sidebands",
         model="harmonic oscillator", function="lamb_dicke"),
    dict(companies="ExxonMobil, Pfizer, Merck (users of quantum chemistry)", physics="molecular energies, spectra",
         model="Born-Oppenheimer separation; Morse oscillator", function="morse_fd_levels_cm"),
    dict(companies="Intel, Broadcom", physics="quantum-well lasers / modulators in optical transceivers",
         model="particle in a box, effective mass", function="box_energy_eV"),
    dict(companies="Corning", physics="pulse propagation in optical fiber",
         model="(nonlinear) Schrodinger in time", function="dgs.diffraction_dispersion"),
    dict(companies="Nvidia", physics="GPU simulation of quantum circuits / wavefunctions (cuQuantum, CUDA-Q)",
         model="large linear Schrodinger evolution", function="(software, not modelled)"),
]
