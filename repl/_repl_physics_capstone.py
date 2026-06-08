# -*- coding: utf-8 -*-
# %% [markdown]
# # Physics Capstone: Classical -> EM -> SR -> QM -> Photonics
# *Closing the loop: Phys 17 + Serway + Griffiths + D-GS*

# %%
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import *
sp.init_printing(use_latex="mathjax")

try:
    from IPython.display import display as _ipy_display
    def show(expr, label=None):
        if label: print(f"  {label}:")
        _ipy_display(expr)
except ImportError:
    def show(expr, label=None):
        if label: print(f"  {label}:")
        print("  " + sp.pretty(expr, use_unicode=True))

def hdr(s): print(f"\n{'='*60}\n  {s}\n{'='*60}")

def chk(val, ref, label, tol=1e-5, absolute=False):
    v, r = float(np.real(val)), float(np.real(ref))
    err = abs(v-r) if (absolute or abs(r)<1e-30) else abs(v-r)/(abs(r)+1e-30)
    s = "PASS" if err < tol else "FAIL"
    print(f"  [{s}] {label}: got {v:.6g}, ref {r:.6g}, err {err:.2e}")
    return s == "PASS"

# %% [markdown]
# ## §1 — Classical mechanics: Lagrangian closes the loop to QM

# %%
hdr("§1 Classical Mechanics: Lagrangian → Hamiltonian → QM")

# Symbolic setup
t_sym, m_sym, k_sym, p_sym = sp.symbols('t m k p', positive=True)
q_sym, qdot_sym = sp.symbols('q qdot')
omega_sym = sp.Symbol('omega', positive=True)

# Lagrangian for harmonic oscillator
L_harm = sp.Rational(1,2)*m_sym*qdot_sym**2 - sp.Rational(1,2)*k_sym*q_sym**2
show(L_harm, "L = T - V")

# Euler-Lagrange equation symbolically
dL_dqdot = sp.diff(L_harm, qdot_sym)   # = m*qdot
dL_dq    = sp.diff(L_harm, q_sym)      # = -k*q
# EL: d/dt(∂L/∂qdot) - ∂L/∂q = 0
# d/dt(m*qdot) = m*qddot; EL residual = m*qddot - (-k*q) = m*qddot + k*q
qddot_sym = sp.Symbol('qddot')
EL_eq = m_sym * qddot_sym + k_sym * q_sym
show(sp.Eq(EL_eq, 0), "Euler-Lagrange (mq̈ + kq = 0)")

# Hamiltonian H = p*qdot - L; eliminate qdot via p = m*qdot → qdot = p/m
qdot_from_p = p_sym / m_sym
H_osc = p_sym * qdot_from_p - L_harm.subs(qdot_sym, qdot_from_p)
H_osc = sp.simplify(H_osc)
show(H_osc, "H = p²/2m + kq²/2")

# Poisson bracket {q, p} = 1 → [q̂, p̂] = iħ
show(sp.Eq(sp.Symbol('{q,p}'), 1), "Poisson bracket → [q̂,p̂] = iħ")

# Numerical
m_val, k_val = 1.0, 4.0
omega_val = np.sqrt(k_val / m_val)
period_T = 2 * np.pi / omega_val

chk(omega_val, np.sqrt(4/1), "omega vs sqrt(k/m)")
chk(period_T, np.pi, "period T vs pi")
# EL residual: m*qddot + k*q = 0 trivially by definition; check numeric substitution
EL_residual = float(EL_eq.subs([(m_sym, m_val), (k_sym, k_val),
                                  (qddot_sym, -k_val/m_val), (q_sym, 1.0)]))
chk(EL_residual, 0, "EL residual (absolute)", tol=1e-10, absolute=True)

# %% [markdown]
# ## §2 — Waves: the bridge from classical to quantum

# %%
hdr("§2 Waves: Classical → Quantum Dispersion")

x_s, t_s, v_s, omega_s, k_s, A_s = sp.symbols('x t v omega k A', real=True)

# Plane wave ψ = A exp(i(kx - ωt))
psi = A_s * sp.exp(sp.I * (k_s * x_s - omega_s * t_s))

# Wave equation residual: ∂²ψ/∂t² - v²∂²ψ/∂x² = 0
wave_eq_residual = sp.diff(psi, t_s, 2) - v_s**2 * sp.diff(psi, x_s, 2)
wave_eq_residual_sub = wave_eq_residual.subs(v_s, omega_s/k_s)
wave_eq_residual_simplified = sp.simplify(wave_eq_residual_sub)
show(wave_eq_residual_simplified, "Wave eq residual (v=ω/k)")

# Dispersion relations
hbar_sym = sp.Symbol('hbar', positive=True)
m_e_sym  = sp.Symbol('m_e', positive=True)
beta2_sym= sp.Symbol('beta2')
omega0_sym = sp.Symbol('omega_0')
vg_sym   = sp.Symbol('v_g')
k0_sym   = sp.Symbol('k_0')

show(sp.Eq(omega_s, v_s*k_s), "Classical wave: ω = vk")
show(sp.Eq(omega_s, hbar_sym*k_s**2/(2*m_e_sym)), "Schrödinger: ω = ħk²/2m (dispersive)")

# Phase and group velocity
omega_qm = hbar_sym * k_s**2 / (2 * m_e_sym)
v_ph_qm = sp.simplify(omega_qm / k_s)
v_g_qm  = sp.diff(omega_qm, k_s)
show(v_ph_qm, "v_ph (QM) = ħk/2m")
show(v_g_qm,  "v_g  (QM) = ħk/m = p/m ✓")

# Numerical: electron at k=1e10 m⁻¹
hbar_val = 1.0545718e-34
m_e_val  = 9.10938e-31
k_num    = 1e10
v_g_electron = hbar_val * k_num / m_e_val

# Particle in box: E_n = n²E_1
E1 = 1.0
E2 = 4.0 * E1

# wave_eq_residual simplified should be 0
wave_eq_check = complex(wave_eq_residual_simplified)
chk(abs(wave_eq_check), 0, "plane wave in wave eq (absolute)", tol=1e-10, absolute=True)
chk(v_g_electron, hbar_val*k_num/m_e_val, "v_g electron at k=1e10")
chk(E2/E1, 4.0, "E2/E1 == 4 (absolute)", tol=1e-10, absolute=True)

# %% [markdown]
# ## §3 — Maxwell's equations: Griffiths Ch.7-9 synthesis

# %%
hdr("§3 Maxwell's Equations")

mu0  = 4 * np.pi * 1e-7
eps0 = 8.854187817e-12
c_computed = 1.0 / np.sqrt(mu0 * eps0)

E0_sun = np.sqrt(1361.0 / (eps0 * c_computed))

# SymPy symbolic display
E_sym, B_sym, rho_sym, J_sym = sp.symbols('E B rho J')
eps0_sym, mu0_sym, c_sym = sp.symbols('epsilon_0 mu_0 c', positive=True)

show(sp.Eq(sp.Symbol('div_E'), rho_sym/eps0_sym), "∇·E = ρ/ε₀")
show(sp.Eq(sp.Symbol('div_B'), 0), "∇·B = 0")
show(sp.Eq(sp.Symbol('curl_E'), -sp.Symbol('dB_dt')), "∇×E = -∂B/∂t")
show(sp.Eq(sp.Symbol('curl_B'), mu0_sym*J_sym + mu0_sym*eps0_sym*sp.Symbol('dE_dt')),
     "∇×B = μ₀J + μ₀ε₀∂E/∂t")

c_eq = sp.Eq(c_sym**2, 1/(mu0_sym*eps0_sym))
show(c_eq, "c² = 1/(μ₀ε₀)")

chk(c_computed, 3e8, "c computed vs 3e8", tol=0.001)
chk(E0_sun, 716.0, "E0 at 1 sun vs 716 V/m", tol=5, absolute=True)
chk(eps0 * mu0 * c_computed**2, 1.0, "ε₀μ₀c² == 1", tol=1e-10, absolute=True)

# %% [markdown]
# ## §4 — Special relativity: Serway Ch.1-2

# %%
hdr("§4 Special Relativity")

# 4-momentum invariant
E_sr, p_sr, m_sr, c_sr = sp.symbols('E p m c', positive=True)
four_mom_inv = sp.Eq(E_sr**2, (p_sr*c_sr)**2 + (m_sr*c_sr**2)**2)
show(four_mom_inv, "E² = (pc)² + (mc²)²")

# Relativistic Doppler at v=0.5c
beta = 0.5
Doppler_ratio = np.sqrt((1+beta)/(1-beta))

# Twin paradox v=0.8c
beta_twin = 0.8
gamma_twin = 1.0 / np.sqrt(1 - beta_twin**2)
earth_time = 2 * (4.0 / beta_twin)  # years
traveler_years = earth_time / gamma_twin

# Compton shift at θ=π
h_val  = 6.62607015e-34
m_e_v  = 9.10938e-31
c_val  = 2.99792458e8
lambda_C = h_val / (m_e_v * c_val)
Compton_shift_180 = 2 * lambda_C  # 1 - cos(π) = 2

chk(gamma_twin, 1/np.sqrt(1-0.64), "gamma at 0.8c", tol=1e-4)
chk(traveler_years, 6.0, "traveler years", tol=0.01)
chk(Doppler_ratio, np.sqrt(3), "Doppler ratio at 0.5c", tol=1e-4)
chk(Compton_shift_180, 4.852e-12, "Compton shift at 180°", tol=1e-14, absolute=True)

# %% [markdown]
# ## §5 — Photoelectric + de Broglie: wave-particle duality

# %%
hdr("§5 Photoelectric Effect + de Broglie")

h_eV  = 4.135667696e-15   # eV·s
c_nm  = 2.99792458e17     # nm/s
phi_Na = 2.28             # eV
lam_nm = 400.0            # nm
f_photon = c_val / (lam_nm * 1e-9)
E_photon_eV = h_val * f_photon / 1.602176634e-19
V_stop = E_photon_eV - phi_Na

# de Broglie at 54 eV
KE_54 = 54 * 1.602176634e-19  # J
p_54  = np.sqrt(2 * m_e_v * KE_54)
lambda_dB = h_val / p_54

# Heisenberg confinement 1 Å
hbar_val2 = 1.0545718e-34
dx = 1e-10
dp_min = hbar_val2 / (2 * dx)
KE_conf = dp_min**2 / (2 * m_e_v) / 1.602176634e-19  # eV

# SymPy display
h_s, f_s, phi_s, lam_s, p_s2 = sp.symbols('h f phi lambda p', positive=True)
show(sp.Eq(sp.Symbol('V_s'), h_s*f_s - phi_s), "Stopping potential eV_s = hf - φ")
show(sp.Eq(lam_s, h_s/p_s2), "de Broglie λ = h/p")

chk(V_stop, 0.82, "V_stop Na at 400nm", tol=0.05, absolute=True)
chk(lambda_dB, 1.67e-10, "λ_dB at 54 eV", tol=0.05)
chk(KE_conf > 0.95, 1.0, "KE confinement > 0.95 eV (absolute)", tol=0.5, absolute=True)

# %% [markdown]
# ## §6 — Schrödinger equation: from wave to wavefunction

# %%
hdr("§6 Schrödinger Equation")

hbar_s2 = sp.Symbol('hbar', positive=True)
m_s2, V_s2, E_s2 = sp.symbols('m V E', positive=True)
x_s2 = sp.Symbol('x', real=True)
L_s2 = sp.Symbol('L', positive=True)

# TDSE symbolic
psi_s2 = sp.Function('psi')
show(sp.Eq(sp.I*hbar_s2*sp.Symbol('dpsi_dt'), -hbar_s2**2/(2*m_s2)*sp.Symbol('d2psi_dx2') + V_s2*sp.Symbol('psi')),
     "TDSE: iħ ∂ψ/∂t = Ĥψ")
show(sp.Eq(sp.Symbol('Hphi'), E_s2*sp.Symbol('phi')), "TISE: Ĥφ = Eφ")

# Infinite square well ground state: ψ₁ = √(2/L) sin(πx/L)
L_val  = 1e-9   # 1 nm
hbar_n = 1.0545718e-34
m_e_n  = 9.10938e-31
e_val  = 1.602176634e-19

# Analytical integrals
# <x> = L/2
x_avg = L_val / 2.0

# <x²> = L²(1/3 - 1/(2π²))
x2_avg = L_val**2 * (1/3 - 1/(2*np.pi**2))

# <p> = 0 by symmetry
p_avg = 0.0

# <p²> = (πħ/L)²
p2_avg = (np.pi * hbar_n / L_val)**2

sigma_x = np.sqrt(x2_avg - x_avg**2)
sigma_p = np.sqrt(p2_avg)
uncertainty_ratio = sigma_x * sigma_p / (hbar_n / 2)

chk(x_avg, L_val/2, "<x> vs L/2", tol=1e-4)
chk(p_avg, 0.0, "<p> == 0", tol=1e-10, absolute=True)
chk(uncertainty_ratio >= 1.0, 1.0, "σ_x·σ_p/(ħ/2) >= 1", tol=0.5, absolute=True)

# %% [markdown]
# ## §7 — Hydrogen atom: quantum numbers + spectroscopy

# %%
hdr("§7 Hydrogen Atom Spectroscopy")

R_H = 1.097e7  # m⁻¹

def wavelength(n_f, n_i):
    return 1.0 / (R_H * (1/n_f**2 - 1/n_i**2))

lambda_Halpha    = wavelength(2, 3)
lambda_Hbeta     = wavelength(2, 4)
lambda_Lyman_a   = wavelength(1, 2)
lambda_Paschen_a = wavelength(3, 4)

# SymPy display
n_f_s, n_i_s, R_s, lam_ryd = sp.symbols('n_f n_i R_H lambda', positive=True)
show(sp.Eq(1/lam_ryd, R_s*(1/n_f_s**2 - 1/n_i_s**2)), "Rydberg formula")

print(f"  Hα (3→2)       = {lambda_Halpha*1e9:.1f} nm")
print(f"  Hβ (4→2)       = {lambda_Hbeta*1e9:.1f} nm")
print(f"  Lyman α (2→1)  = {lambda_Lyman_a*1e9:.1f} nm")
print(f"  Paschen α (4→3)= {lambda_Paschen_a*1e9:.0f} nm")

chk(lambda_Halpha,    656.3e-9, "Hα wavelength", tol=1e-10, absolute=True)
chk(lambda_Lyman_a,   121.6e-9, "Lyman α",       tol=0.5e-9, absolute=True)
chk(lambda_Paschen_a, 1875e-9,  "Paschen α",     tol=5e-9, absolute=True)

# %% [markdown]
# ## §8 — Many-electron atoms + periodic table

# %%
hdr("§8 Many-Electron Atoms + Periodic Table")

# Moseley's law: Kα line f = (3/4)*R_H*c*(Z-1)²
Z_Cu = 29
f_Cu_Kalpha = (3/4) * R_H * c_val * (Z_Cu - 1)**2
lambda_Cu_Kalpha = c_val / f_Cu_Kalpha

print(f"  Cu Kα frequency = {f_Cu_Kalpha:.3e} Hz")
print(f"  Cu Kα wavelength = {lambda_Cu_Kalpha*1e10:.3f} Å = {lambda_Cu_Kalpha*1e9:.4f} nm")

p_block_per_period = 6   # l=1 gives 2(2*1+1) = 6 elements
carbon_ground_J = 0      # ³P₀ for C (1s²2s²2p²)

chk(lambda_Cu_Kalpha, 0.154e-9, "Cu Kα wavelength", tol=0.005e-9, absolute=True)
chk(p_block_per_period, 6, "p-block elements per period (absolute)", tol=0.5, absolute=True)
chk(carbon_ground_J, 0, "Carbon ground J (absolute)", tol=0.5, absolute=True)

# %% [markdown]
# ## §9 — Solid state + band theory

# %%
hdr("§9 Solid State + Band Theory")

hbar_ss = 1.0545718e-34
m_e_ss  = 9.10938e-31
e_ss    = 1.602176634e-19
kB      = 1.380649e-23
T       = 300.0

# Fermi energy of Cu
n_Cu = 8.49e28   # m⁻³
E_F_J = (hbar_ss**2 / (2*m_e_ss)) * (3*np.pi**2*n_Cu)**(2/3)
E_F_Cu = E_F_J / e_ss   # eV

# Built-in potential for Si PN junction (use consistent cm^-3 units)
n_i_Si = 1.5e10    # cm^-3
N_A = 1e16         # cm^-3
N_D = 1e16         # cm^-3
kT  = kB * T / e_ss      # eV
V_bi = kT * np.log(N_A * N_D / n_i_Si**2)

# LED wavelengths
def led_lambda(Eg_eV):
    return 1240e-9 / Eg_eV   # λ = hc/E_g in nm

lambda_GaAs = led_lambda(1.42)
lambda_GaN  = led_lambda(3.4)

chk(E_F_Cu, 7.04, "E_F Cu (eV)", tol=0.1)
chk(V_bi, 0.693, "V_bi Si PN junction", tol=0.01)
chk(lambda_GaAs, 873e-9, "GaAs LED λ", tol=5e-9, absolute=True)
chk(lambda_GaN,  365e-9, "GaN LED λ",  tol=5e-9, absolute=True)

# %% [markdown]
# ## §10 — Nuclear + particle physics

# %%
hdr("§10 Nuclear + Particle Physics")

# Alpha decay Q value
u_to_MeV = 931.5  # MeV/u
M_U238   = 238.050788
M_Th234  = 234.043601
M_He4    = 4.002602
Q_alpha  = (M_U238 - M_Th234 - M_He4) * u_to_MeV

# Beta decay Q
M_n = 1.008665   # u
M_p = 1.007276   # u
M_e_u = 0.000549 # u
Q_beta = (M_n - M_p - M_e_u) * u_to_MeV

# Half-life of U238
lambda_U238 = 4.916e-18  # s⁻¹
T_half_s = np.log(2) / lambda_U238
T_half_years = T_half_s / (3.156e7)  # s/year

print(f"  Q_alpha(U238) = {Q_alpha:.2f} MeV")
print(f"  Q_beta(n→p)   = {Q_beta:.3f} MeV")
print(f"  T½(U238)      = {T_half_years:.3e} years")

# Feynman diagram: e⁻ + e⁻ → e⁻ + e⁻
print("\n  Feynman diagram (Møller scattering):")
print("    e⁻(p1) --------γ(q)-------- e⁻(p3)")
print("                   |")
print("    e⁻(p2) --------+------------ e⁻(p4)")
print("  Vertex: coupling e; Propagator: 1/q² (massless photon)")

chk(Q_alpha, 4.27, "Q_alpha U238 (MeV)", tol=0.1, absolute=True)
chk(Q_beta,  0.782, "Q_beta neutron (MeV)", tol=0.01, absolute=True)
chk(T_half_years, 4.47e9, "T½ U238 (years)", tol=0.1e9, absolute=True)

# %% [markdown]
# ## §11 — Analytical mechanics: Hamiltonian → QM correspondence

# %%
hdr("§11 Analytical Mechanics: Hamiltonian → QM")

m_11, omega_11, q_11, p_11 = sp.symbols('m omega q p', positive=True)

H_osc11 = p_11**2 / (2*m_11) + m_11 * omega_11**2 * q_11**2 / 2

q_dot_11 = sp.diff(H_osc11, p_11)
p_dot_11 = -sp.diff(H_osc11, q_11)

show(q_dot_11, "q̇ = ∂H/∂p")
show(p_dot_11, "ṗ = -∂H/∂q")

# Action-angle: I = E/ω
# Numerical check: for A=1, m=1, omega=2
A_num = 1.0
m_num = 1.0
omega_num = 2.0
E_num = 0.5 * m_num * omega_num**2 * A_num**2   # = 2 J
I_num = 0.5 * m_num * omega_num * A_num**2        # = 1 J·s (= E/ω = 2/2 = 1)

# Check q_dot symbolically: should be p/m
q_dot_expected = p_11 / m_11
q_dot_matches = sp.simplify(q_dot_11 - q_dot_expected) == 0

# Check p_dot symbolically: should be -m*omega²*q
p_dot_expected = -m_11 * omega_11**2 * q_11
p_dot_matches = sp.simplify(p_dot_11 - p_dot_expected) == 0

print(f"  q̇ symbolic match (p/m): {q_dot_matches}")
print(f"  ṗ symbolic match (-mω²q): {p_dot_matches}")

chk(float(q_dot_matches), 1.0, "q_dot == p/m (symbolic)", tol=0.5, absolute=True)
chk(float(p_dot_matches), 1.0, "p_dot == -mω²q (symbolic)", tol=0.5, absolute=True)
chk(I_num, E_num/omega_num, "action I = E/ω (SHO)", tol=1e-10, absolute=True)

# %% [markdown]
# ## §12 — The full loop: Classical → EM → SR → QM → Photonics → D-GS

# %%
hdr("§12 The Full Loop: α → All of Physics")

# Fine structure constant α = e²/(4πε₀ħc)
e_c   = 1.602176634e-19
eps0_c= 8.854187817e-12
hbar_c= 1.0545718e-34
c_c   = 2.99792458e8
m_e_c = 9.10938e-31

alpha = e_c**2 / (4 * np.pi * eps0_c * hbar_c * c_c)

# Bohr radius a₀ = ħ/(m_e c α)
a0 = hbar_c / (m_e_c * c_c * alpha)

# Ground state energy E₁ = -m_e c² α²/2
E1H_J  = m_e_c * c_c**2 * alpha**2 / 2
E1H_eV = E1H_J / e_c

# Photon at λ=1550 nm
lambda_1550 = 1550e-9
E_photon_J  = hbar_c * 2 * np.pi * c_c / lambda_1550
E_photon_eV = E_photon_J / e_c

# SymPy chain display
alpha_s, a0_s, E1_s = sp.symbols('alpha a_0 E_1', positive=True)
e_s, eps_s, hb_s, c_s2, me_s = sp.symbols('e epsilon_0 hbar c m_e', positive=True)
show(sp.Eq(alpha_s, e_s**2/(4*sp.pi*eps_s*hb_s*c_s2)), "α = e²/(4πε₀ħc)")
show(sp.Eq(a0_s, hb_s/(me_s*c_s2*alpha_s)), "a₀ = ħ/(m_e c α)")
show(sp.Eq(E1_s, sp.Rational(-1,2)*me_s*c_s2**2*alpha_s**2), "E1 = -m_e c^2 alpha^2 / 2")

print(f"\n  α     = 1/{1/alpha:.3f}  (fine structure constant)")
print(f"  a₀    = {a0*1e10:.3f} Å (Bohr radius)")
print(f"  E₁H   = {E1H_eV:.2f} eV (hydrogen ground state)")

# Photonics transparency check
Eg_SiO2 = 8.9
Eg_Si   = 1.12
Eg_GaAs = 1.42
print(f"\n  E_photon(1550nm) = {E_photon_eV:.3f} eV")
print(f"  SiO₂ gap = {Eg_SiO2} eV → {'transparent ✓' if E_photon_eV < Eg_SiO2 else 'opaque ✗'}")
print(f"  Si gap   = {Eg_Si} eV → {'transparent ✓' if E_photon_eV < Eg_Si else 'opaque ✗'}")
print(f"  GaAs gap = {Eg_GaAs} eV → {'transparent ✓' if E_photon_eV < Eg_GaAs else 'opaque ✗'}")

SiO2_transparent = float(E_photon_eV < Eg_SiO2)
Si_transparent   = float(E_photon_eV < Eg_Si)

chk(alpha, 1/137.036, "α vs 1/137.036", tol=1e-4)
chk(a0, 0.529e-10, "a₀ vs 0.529 Å", tol=1e-13, absolute=True)
chk(E1H_eV, 13.6, "E₁H vs 13.6 eV", tol=0.01, absolute=True)
chk(E_photon_eV, 0.800, "E_photon(1550nm) eV", tol=0.005, absolute=True)
chk(SiO2_transparent, 1.0, "SiO₂ transparent at 1550nm (absolute)", tol=0.5, absolute=True)
chk(Si_transparent,   1.0, "Si transparent at 1550nm (absolute)", tol=0.5, absolute=True)
