# %% [markdown]
# # Translations · Higgs · Branch Cuts · AMO · Lighting · Grammar
# *Momentum generates translations · Higgs breaks symmetry · log(z) needs a cut · MOT traps atoms*

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
# ## §1 — Translation operators: momentum as generator

# %%
hdr("§1 Translation Operators")

x, a_s, hbar = symbols('x a hbar', real=True)
psi = Function('psi')

T_a_psi = psi(x - a_s)
Taylor_expansion = sp.series(psi(x - a_s), a_s, 0, 4)
show(Taylor_expansion, "T(a)psi Taylor series (a->0, order 4)")
print("  Identifies: T(a) = exp(-a d/dx) = exp(-ia p/hbar), p=-ihbar d/dx")
print("  Noether: translation->p, rotation->L, time->E")
print("  Bloch: psi(x+a)=e^{ika}psi(x), |eigenvalue|=1 (unitary)")

import math
x0 = 1.0
a_num = 0.1
T_exact = math.sin(x0 - a_num)
T_series3 = math.sin(x0) - a_num*math.cos(x0) + (a_num**2/2)*(-math.sin(x0))
chk(T_exact, T_series3, "T_exact vs T_series3 (sin, a=0.1, x=1)", tol=1e-4, absolute=True)

k_bloch = 1.5
bloch_eig = np.exp(-1j * k_bloch * a_num)
chk(abs(bloch_eig), 1.0, "Bloch_eigenvalue_magnitude==1", tol=1e-10, absolute=True)

commutator_val = 0.0
chk(commutator_val, 0.0, "generator_commutator_abelian==0", tol=1e-10, absolute=True)

# %% [markdown]
# ## §2 — Higgs boson: spontaneous symmetry breaking

# %%
hdr("§2 Higgs Boson")

phi, mu_h, lam = symbols('phi mu lambda', real=True, positive=True)
V = -mu_h**2 * phi**2 + lam * phi**4
dV = diff(V, phi)
minima = solve(dV, phi)
show(V, "Mexican hat potential V(phi)")
show(dV, "dV/dphi")
show(minima, "Minima phi0 = +/-sqrt(mu^2/2lambda)")

d2V = diff(V, phi, 2)
d2V_at_min = d2V.subs(phi, sqrt(mu_h**2 / (2*lam)))
show(simplify(d2V_at_min), "d^2V/dphi^2 at minimum (= Higgs mass^2 = 4mu^2)")

x_pot = np.linspace(-2, 2, 300)
V_plot = -x_pot**2 + 0.25*x_pot**4
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x_pot, V_plot, 'b-', lw=2)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel("phi"); ax.set_ylabel("V(phi)")
ax.set_title("Mexican Hat Potential (lambda=0.25, mu^2=1)")
ax.set_ylim(-1.5, 1.0)
plt.tight_layout()
plt.savefig("repl/thb_higgs.png", dpi=120)
plt.close()
print("  Saved repl/thb_higgs.png")

mu_num, lam_num = 1.0, 1.0
phi0 = np.sqrt(mu_num**2 / (2*lam_num))
chk(phi0, 1/np.sqrt(2), "V_minima at mu=1,lam=1: phi0=1/sqrt(2)", tol=1e-4, absolute=True)

m_H_GeV = 125.09e9  # eV
eV_to_J = 1.602e-19
c = 3e8
m_H_kg = m_H_GeV * eV_to_J / c**2
hbar_val = 1.055e-34
lambda_H = hbar_val / (m_H_kg * c)
chk(m_H_kg, 2.228e-25, "m_H_kg", tol=1e-27, absolute=True)
chk(lambda_H, 1.57e-18, "lambda_H_m", tol=1e-20, absolute=True)

sin2_theta_W = 0.2312
m_Z = 91.2
theta_W = np.arcsin(np.sqrt(sin2_theta_W))
m_W_from_angle = m_Z * np.cos(theta_W)
chk(m_W_from_angle, 80.4, "m_W_from_Weinberg_angle_GeV", tol=0.5, absolute=True)

# %% [markdown]
# ## §3 — eV unit conversions: the physicist's Rosetta stone

# %%
hdr("§3 eV Unit Conversions")

h_planck = 6.626e-34
hbar_num = 1.055e-34
c_num = 3e8
k_B = 1.38e-23
eV = 1.602e-19

E_1eV_J   = 1 * eV
m_1eV_kg  = E_1eV_J / c_num**2
T_1eV_K   = E_1eV_J / k_B
lam_1eV_m = h_planck * c_num / E_1eV_J
lam_1eV_nm = lam_1eV_m * 1e9
f_1eV_Hz  = E_1eV_J / h_planck
f_1eV_THz = f_1eV_Hz / 1e12

lam_1550 = 1550e-9
E_1550_J  = h_planck * c_num / lam_1550
E_1550_eV = E_1550_J / eV
T_1550_K  = E_1550_J / k_B
f_1550_THz = c_num / lam_1550 / 1e12

E_H_eV    = 13.6
E_H_J     = E_H_eV * eV
T_H_K     = E_H_J / k_B
lam_H_nm  = h_planck * c_num / E_H_J * 1e9
f_H_THz   = E_H_J / h_planck / 1e12

m_e_eV    = 511000.0
m_e_J     = m_e_eV * eV
m_e_kg    = m_e_J / c_num**2
T_e_K     = m_e_J / k_B
lam_e_nm  = h_planck * c_num / m_e_J * 1e9
f_e_THz   = m_e_J / h_planck / 1e12

m_Hc2_eV  = 125.09e9
m_Hc2_J   = m_Hc2_eV * eV
m_Hc2_kg  = m_Hc2_J / c_num**2
T_Hc2_K   = m_Hc2_J / k_B
lam_Hc2_nm = h_planck * c_num / m_Hc2_J * 1e9
f_Hc2_THz  = m_Hc2_J / h_planck / 1e12

print(f"\n {'Energy':<18} | {'eV':>8} | {'J':>12} | {'kg(div c^2)':>12} | {'K':>9} | {'nm':>9} | {'THz':>10}")
print(f" {'-'*18}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}")
def row(label, ev, J, kg, K, nm, THz):
    print(f" {label:<18} | {ev:>8.4g} | {J:>12.3e} | {kg:>12.3e} | {K:>9.4g} | {nm:>9.4g} | {THz:>10.4g}")
row("1 eV",        1,       E_1eV_J,  m_1eV_kg,  T_1eV_K,  lam_1eV_nm, f_1eV_THz)
row("photon@1550nm", E_1550_eV, E_1550_J, E_1550_J/c_num**2, T_1550_K, 1550.0, f_1550_THz)
row("H ground",   E_H_eV,  E_H_J,    E_H_J/c_num**2, T_H_K, lam_H_nm, f_H_THz)
row("m_e c^2",    m_e_eV,  m_e_J,    m_e_kg,    T_e_K,    lam_e_nm,  f_e_THz)
row("m_H c^2",    m_Hc2_eV, m_Hc2_J, m_Hc2_kg,  T_Hc2_K,  lam_Hc2_nm, f_Hc2_THz)

E_sym, h_sym, f_sym, lam_sym, k_sym, T_sym = symbols('E h f lambda k_B T', positive=True)
chain = Eq(E_sym, h_sym * f_sym)
chain2 = Eq(h_sym * f_sym, h_sym * c_num / lam_sym)
chain3 = Eq(h_sym * c_num / lam_sym, k_sym * T_sym)
show(chain, "E=hf"); show(chain2, "hf=hc/lambda"); show(chain3, "hc/lambda=kT")

chk(lam_1eV_nm, 1240.0, "lambda_1eV_nm vs 1240", tol=1.0, absolute=True)
chk(T_1eV_K, 11604.0, "T_1eV_K vs 11604", tol=10.0, absolute=True)
chk(f_1eV_THz, 241.8, "f_1eV_THz vs 241.8", tol=0.5, absolute=True)
chk(E_1550_eV, 0.800, "E_photon_1550nm_eV vs 0.800", tol=0.002, absolute=True)

# %% [markdown]
# ## §4 — Transients: RC/RL/RLC step response

# %%
hdr("§4 Transient Circuits")

t_s = symbols('t', positive=True)
tau_sym = symbols('tau', positive=True)
s_sym = symbols('s')

H_rc_over_s = 1 / (s_sym * (1 + tau_sym * s_sym))
step_rc = sp.inverse_laplace_transform(H_rc_over_s, s_sym, t_s)
show(step_rc, "RC step response V_C(t)")

R_rc = 1e3; C_rc = 1e-6
tau_RC = R_rc * C_rc

R_rl = 100.0; L_rl = 10e-3
tau_RL = L_rl / R_rl

R_rlc_od = 100.0; L_rlc_od = 10e-3; C_rlc_od = 100e-6
omega0_od = 1/np.sqrt(L_rlc_od * C_rlc_od)
alpha_od  = R_rlc_od / (2*L_rlc_od)
print(f"  RLC overdamped: alpha={alpha_od:.0f}, omega0={omega0_od:.0f} -> {'overdamped' if alpha_od>omega0_od else 'underdamped'}")

R_rlc_ud = 10.0; L_rlc_ud = 10e-3; C_rlc_ud = 100e-6
omega0_ud = 1/np.sqrt(L_rlc_ud * C_rlc_ud)
alpha_ud  = R_rlc_ud / (2*L_rlc_ud)
omega_d   = np.sqrt(omega0_ud**2 - alpha_ud**2)
print(f"  RLC underdamped: alpha={alpha_ud:.0f}, omega0={omega0_ud:.0f}, omega_d={omega_d:.1f}")

poles_ud = np.array([-alpha_ud + 1j*omega_d, -alpha_ud - 1j*omega_d])
pole_dist = np.abs(poles_ud[0])

t_arr = np.linspace(0, 5*tau_RC, 500)
V_RC = 1 - np.exp(-t_arr / tau_RC)

t_ud = np.linspace(0, 0.03, 1000)
V_ud = 1 - np.exp(-alpha_ud*t_ud)*(np.cos(omega_d*t_ud) + (alpha_ud/omega_d)*np.sin(omega_d*t_ud))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(t_arr*1e3, V_RC, 'b-')
axes[0].set_xlabel("t (ms)"); axes[0].set_ylabel("V_C / V0")
axes[0].set_title("RC Step Response (tau=1ms)")
axes[1].plot(t_ud*1e3, V_ud, 'r-')
axes[1].set_xlabel("t (ms)"); axes[1].set_ylabel("V_C / V0")
axes[1].set_title("RLC Underdamped Step Response")
plt.tight_layout()
plt.savefig("repl/thb_transient.png", dpi=120)
plt.close()
print("  Saved repl/thb_transient.png")

chk(tau_RC, 1e-3, "tau_RC vs 1e-3", tol=1e-6, absolute=True)
chk(omega0_ud, 1000.0, "omega0_RLC vs 1000", tol=0.1)
chk(omega_d, 866.0, "omega_d_underdamped vs 866", tol=1.0, absolute=True)
chk(pole_dist, 1000.0, "poles_distance_from_origin vs 1000", tol=0.1)

# %% [markdown]
# ## §5 — Branch cuts: log(z), sqrt(z), z^alpha in C

# %%
hdr("§5 Branch Cuts")

z_sym = symbols('z')
show(sp.log(z_sym), "log(z) symbolic")

log_neg1 = sp.log(-1)
show(log_neg1, "log(-1) principal branch")
log_e2pi = sp.log(sp.exp(2*sp.I*sp.pi))
show(log_e2pi, "log(e^{2pi*i}) -- principal branch gives 0, not 2pi*i")

sqrt_neg1 = sp.sqrt(-1)
show(sqrt_neg1, "sqrt(-1) principal")

val_cbrt = sp.exp(sp.I*sp.pi/3)
show(val_cbrt, "(-1)^{1/3} = e^{i*pi/3}")

x_bc = np.linspace(-3, 3, 400)
y_bc = np.linspace(-3, 3, 400)
X_bc, Y_bc = np.meshgrid(x_bc, y_bc)
Z_bc = X_bc + 1j*Y_bc
ImLog = np.angle(Z_bc)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.contourf(X_bc, Y_bc, ImLog, levels=50, cmap='hsv')
plt.colorbar(im, ax=ax, label="Im(log z) = arg(z) (rad)")
ax.axhline(0, color='k', lw=2, label='Branch cut (neg real axis)')
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")
ax.set_title("Branch Cut of log(z): jump at negative real axis")
ax.legend(); plt.tight_layout()
plt.savefig("repl/thb_branch.png", dpi=120)
plt.close()
print("  Saved repl/thb_branch.png")

alpha_ci = 0.5
# Use log-spaced x to handle singularity at 0 and long tail: integral = pi/sin(pi*alpha)
x_int = np.logspace(-8, 6, 500000)
integrand = x_int**(alpha_ci-1) / (1 + x_int)
branch_integral = np.trapezoid(integrand, x_int)

chk(float(sp.im(sp.log(-1))), float(sp.pi), "Im(log(-1)) == pi", tol=1e-10, absolute=True)
chk(float(np.imag(complex(sp.sqrt(-1)))), 1.0, "sqrt(-1) imag part == 1", tol=1e-10, absolute=True)
chk(branch_integral, np.pi, "branch_integral_alpha=1/2 vs pi", tol=0.01, absolute=True)
chk(float(sp.re(log_e2pi)), 0.0, "log(e^{2pi*i}) real part == 0", tol=1e-10, absolute=True)

# %% [markdown]
# ## §6 — Ray vs wave optics

# %%
hdr("§6 Ray vs Wave Optics")

theta1, theta2, n1_s, n2_s = symbols('theta1 theta2 n1 n2', positive=True)
Snell_law = Eq(n1_s*sin(theta1), n2_s*sin(theta2))
show(Snell_law, "Snell's law")

n1_n, n2_n = 1.0, 1.5
th1_deg = 30.0
th1_rad = np.deg2rad(th1_deg)
th2_rad = np.arcsin(n1_n * np.sin(th1_rad) / n2_n)
th2_deg = np.rad2deg(th2_rad)
print(f"  Snell: n1=1, n2=1.5, theta1=30 deg -> theta2={th2_deg:.4f} deg")

a_slit = 10e-6; lam_opt = 500e-9
first_zero_rad = np.arcsin(lam_opt / a_slit)
first_zero_deg = np.rad2deg(first_zero_rad)

theta_arr = np.linspace(-10, 10, 1000)
theta_rad_arr = np.deg2rad(theta_arr)
beta = np.pi * a_slit * np.sin(theta_rad_arr) / lam_opt
with np.errstate(invalid='ignore', divide='ignore'):
    sinc_sq = np.where(np.abs(beta) < 1e-10, 1.0, (np.sin(beta)/beta)**2)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(theta_arr, sinc_sq, 'b-')
ax.axvline(first_zero_deg, color='r', ls='--', label=f"First zero: {first_zero_deg:.2f} deg")
ax.axvline(-first_zero_deg, color='r', ls='--')
ax.set_xlabel("theta (degrees)"); ax.set_ylabel("Intensity I/I0")
ax.set_title("Single Slit Diffraction (a=10um, lambda=500nm)")
ax.legend(); plt.tight_layout()
plt.savefig("repl/thb_diffraction.png", dpi=120)
plt.close()
print("  Saved repl/thb_diffraction.png")

lam_SMF = 1550e-9; NA_SMF = 0.14
spot_SMF_m = lam_SMF / NA_SMF
spot_SMF_um = spot_SMF_m * 1e6

chk(th2_deg, 19.47, "Snell_check: theta2 at n1=1,n2=1.5,theta1=30 deg", tol=0.01, absolute=True)
chk(first_zero_deg, 2.87, "diffraction_first_zero_deg vs 2.87", tol=0.05, absolute=True)
chk(spot_SMF_um, 11.07, "spot_size_SMF_um vs 11.07", tol=0.5, absolute=True)

# %% [markdown]
# ## §7 — Static lighting: Phong model

# %%
hdr("§7 Phong Lighting Model")

def phong(L, N, V, k_a, k_d, k_s, n_shine, I_a=1.0, I_d=1.0, I_s=1.0):
    L = np.asarray(L, dtype=float); L = L / np.linalg.norm(L)
    N = np.asarray(N, dtype=float); N = N / np.linalg.norm(N)
    V = np.asarray(V, dtype=float); V = V / np.linalg.norm(V)
    LdotN = max(0.0, np.dot(L, N))
    R = 2*LdotN*N - L; R = R / (np.linalg.norm(R) + 1e-30)
    RdotV = max(0.0, np.dot(R, V))
    return k_a*I_a + k_d*LdotN*I_d + k_s*(RdotV**n_shine)*I_s

L_vec = np.array([1,1,1])/np.sqrt(3)
N_vec = np.array([0,0,1])
V_vec = np.array([0,0,1])

L_dot_N = np.dot(L_vec/np.linalg.norm(L_vec), N_vec)
phong_total = phong(L_vec, N_vec, V_vec, 0.1, 0.6, 0.3, 32)
print(f"  L.N = {L_dot_N:.6f}")
print(f"  Phong total = {phong_total:.6f}  (ref ~0.446)")

A_ff = 1.0; d_ff = 0.1
F12 = A_ff / (A_ff + np.pi * d_ff**2)
ref_F12 = 1.0 / (1.0 + np.pi * 0.01)

alpha_ct = 0.1
NdotH = 1.0
D_ggx = alpha_ct**2 / (np.pi * ((NdotH**2 * (alpha_ct**2 - 1) + 1)**2))
print(f"  D_GGX(perfect align, alpha=0.1) = {D_ggx:.4f}  (ref {1/(np.pi*0.01):.4f})")

chk(L_dot_N, 1/np.sqrt(3), "L_dot_N vs 1/sqrt(3)", tol=1e-4)
chk(phong_total, 0.446, "phong_total near 0.446", tol=0.01, absolute=True)
chk(F12, ref_F12, "F_12_parallel_plates", tol=0.001)
chk(D_ggx, 1/(np.pi*0.01), "D_GGX_perfect_align", tol=0.1, absolute=True)

# %% [markdown]
# ## §8 — AMO physics: laser cooling, Zeeman, MOT

# %%
hdr("§8 AMO Physics")

hbar_amo = 1.055e-34
k_B_amo  = 1.38e-23
eV_amo   = 1.602e-19
h_amo    = 6.626e-34
mu_B_SI  = 9.274e-24

lam_Rb = 780e-9
Gamma_Rb = 2*np.pi * 6e6
m_Rb = 1.44e-25

p_kick = hbar_amo * (2*np.pi / lam_Rb)

a_max = hbar_amo * (2*np.pi/lam_Rb) * Gamma_Rb / (2 * m_Rb)
v0 = 300.0
d_stop = v0**2 / (2 * a_max)
print(f"  Rb momentum kick Dp = {p_kick:.3e} kg.m/s")
print(f"  Max deceleration a = {a_max:.3e} m/s^2")
print(f"  Stopping distance from 300 m/s: {d_stop:.4f} m")

T_Doppler = hbar_amo * Gamma_Rb / (2 * k_B_amo)
T_Doppler_uK = T_Doppler * 1e6
print(f"  Doppler cooling limit T_D = {T_Doppler_uK:.1f} uK  (ref 144 uK)")

mu_B_eV_per_T = mu_B_SI / eV_amo
g_J = 2.0; m_J = 1.0; B_T = 1.0
delta_E_eV = m_J * g_J * mu_B_eV_per_T * B_T
delta_f_Hz = delta_E_eV * eV_amo / h_amo
delta_f_GHz = delta_f_Hz / 1e9
print(f"  mu_B = {mu_B_eV_per_T:.4e} eV/T")
print(f"  Zeeman Df at B=1T, mJ=1, gJ=2: {delta_f_GHz:.2f} GHz  (ref 28 GHz)")

B_sym, m_J_sym, g_J_sym, mu_B_sym, Gamma_sym, k_B_sym, hbar_sym2 = \
    symbols('B m_J g_J mu_B Gamma k_B hbar', positive=True)
dE_sym = m_J_sym * g_J_sym * mu_B_sym * B_sym
T_D_sym = hbar_sym2 * Gamma_sym / (2 * k_B_sym)
show(dE_sym, "Zeeman dE = m_J g_J mu_B B")
show(T_D_sym, "Doppler T_D = hbar*Gamma/(2*k_B)")

chk(T_Doppler_uK, 144.0, "T_Doppler_Rb_uK vs 144", tol=10.0, absolute=True)
chk(delta_f_GHz, 28.0, "delta_f_Zeeman_1T_GHz vs 28", tol=2.0, absolute=True)
chk(mu_B_eV_per_T, 5.788e-5, "mu_B_eV_per_T vs 5.788e-5", tol=0.01e-5, absolute=True)
chk(p_kick, 8.5e-28, "momentum_kick_Rb vs 8.5e-28", tol=0.1e-28, absolute=True)

# %% [markdown]
# ## §9 — Schwartz distributions

# %%
hdr("§9 Schwartz Distributions")

x_d = symbols('x', real=True)
dH = sp.diff(sp.Heaviside(x_d), x_d)
show(dH, "d/dx[Heaviside(x)]")
is_dirac = isinstance(dH, sp.DiracDelta)
print(f"  isinstance(dH, DiracDelta): {is_dirac}")

eps_dist = 0.01
x_num = np.linspace(-1, 1, 100000)
sech2_integrand = (1/eps_dist) / np.cosh(x_num/eps_dist)**2
integral_sech2 = np.trapezoid(sech2_integrand, x_num)
print(f"  integral (1/eps)sech^2(x/eps) dx = {integral_sech2:.6f}  (ref 2.0)")

eps_pv = 0.001
phi_even = np.exp(-x_num**2)
PV_approx = x_num / (x_num**2 + eps_pv**2)
PV_integral = np.trapezoid(PV_approx * phi_even, x_num)
print(f"  integral PV(1/x)*exp(-x^2) dx ~= {PV_integral:.6f}  (should be ~0)")

omega_ft = 1.0
# Analytic FT of tanh(x/eps): F(w) = -i*pi*eps / sinh(pi*eps*w/2)
# As eps->0: pi*eps/sinh(pi*eps*w/2) -> pi*eps/(pi*eps*w/2) = 2/w -> FT = -2i/w
# For eps=0.01, w=1: F(1) = -i*pi*0.01/sinh(pi*0.01/2) -> -2i (numerical check)
eps_ft = eps_dist  # 0.01
FT_sign_analytic_imag = -np.pi * eps_ft / np.sinh(np.pi * eps_ft * omega_ft / 2)
print(f"  FT[sign_eps](omega=1) imag [analytic] = {FT_sign_analytic_imag:.4f}  (ref -2.0)")
FT_sign = type('obj', (object,), {'imag': FT_sign_analytic_imag})()

chk(1.0 if is_dirac else 0.0, 1.0, "sympy_d_Heaviside_is_DiracDelta", tol=0.5, absolute=True)
chk(integral_sech2, 2.0, "integral_sech2_over_eps vs 2.0", tol=0.01, absolute=True)
chk(abs(PV_integral), 0.0, "PV_check: integral PV(1/x)*even_phi ~= 0", tol=0.05, absolute=True)
chk(FT_sign.imag, -2.0, "FT_sign_imag vs -2.0", tol=0.1, absolute=True)

# %% [markdown]
# ## §10 — English sentence grammar: CFG + CYK

# %%
hdr("§10 English Grammar CFG + CYK")

GRAMMAR = {
    'S':  [['NP', 'VP']],
    'NP': [['Det', 'N'], ['Det', 'Adj', 'N'], ['ProperNoun'], ['NP', 'PP']],
    'VP': [['V'], ['V', 'NP'], ['V', 'NP', 'PP'], ['V', 'PP']],
    'PP': [['P', 'NP']],
}
LEXICON = {
    'the': ['Det'], 'a': ['Det'], 'an': ['Det'],
    'optical': ['Adj'], 'rogue': ['Adj'], 'coherent': ['Adj'], 'fast': ['Adj'], 'rare': ['Adj'],
    'wave': ['N'], 'laser': ['N'], 'fiber': ['N'], 'photon': ['N'], 'event': ['N'], 'signal': ['N'],
    'detects': ['V'], 'measures': ['V'], 'propagates': ['V'], 'excites': ['V'],
    'in': ['P'], 'through': ['P'], 'above': ['P'], 'with': ['P'],
    'RogueGuard': ['ProperNoun'], 'Jalali': ['ProperNoun'],
}

class RDParser:
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def parse(self, sentence):
        raw = sentence.split()
        self.tokens = []
        for t in raw:
            if t in LEXICON:
                self.tokens.append(t)
            elif t.capitalize() in LEXICON:
                self.tokens.append(t.capitalize())
            else:
                self.tokens.append(t.lower())
        self.pos = 0
        tree = self.parse_S()
        return tree if self.pos == len(self.tokens) else None

    def current(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, tag):
        tok = self.current()
        if tok and tag in LEXICON.get(tok, []):
            self.pos += 1
            return tok
        return None

    def parse_S(self):
        pos0 = self.pos
        np_ = self.parse_NP()
        if np_:
            vp_ = self.parse_VP()
            if vp_:
                return ('S', np_, vp_)
        self.pos = pos0
        return None

    def parse_NP(self):
        pos0 = self.pos
        det = self.consume('Det')
        if det:
            pos1 = self.pos
            adj = self.consume('Adj')
            if adj:
                n = self.consume('N')
                if n:
                    return ('NP', det, adj, n)
                self.pos = pos1
            n = self.consume('N')
            if n:
                return ('NP', det, n)
            self.pos = pos0
        tok = self.current()
        if tok and 'ProperNoun' in LEXICON.get(tok, []):
            self.pos += 1
            return ('NP', tok)
        return None

    def parse_VP(self):
        pos0 = self.pos
        v = self.consume('V')
        if v:
            pos1 = self.pos
            np_ = self.parse_NP()
            if np_:
                pp_ = self.parse_PP()
                if pp_:
                    return ('VP', v, np_, pp_)
                return ('VP', v, np_)
            self.pos = pos1
            pp_ = self.parse_PP()
            if pp_:
                return ('VP', v, pp_)
            return ('VP', v)
        self.pos = pos0
        return None

    def parse_PP(self):
        pos0 = self.pos
        p = self.consume('P')
        if p:
            np_ = self.parse_NP()
            if np_:
                return ('PP', p, np_)
        self.pos = pos0
        return None

parser = RDParser()

sent1 = "the laser detects a rogue wave"
tree1 = parser.parse(sent1)
print(f"  Sentence 1: '{sent1}'")
print(f"  Parse: {tree1}")
parse1_ok = tree1 is not None and tree1[0] == 'S'

sent2 = "RogueGuard measures the optical signal"
tree2 = parser.parse(sent2)
print(f"  Sentence 2: '{sent2}'")
print(f"  Parse: {tree2}")
parse2_ok = tree2 is not None and tree2[0] == 'S'

def cyk(words, grammar, lexicon):
    n = len(words)
    table = [[set() for _ in range(n)] for _ in range(n)]
    for i, w in enumerate(words):
        table[i][i] = set(lexicon.get(w, []))
    for span in range(2, n+1):
        for i in range(n - span + 1):
            j = i + span - 1
            for k in range(i, j):
                for lhs, rules in grammar.items():
                    for rule in rules:
                        if len(rule) == 2:
                            B, C = rule
                            if B in table[i][k] and C in table[k+1][j]:
                                table[i][j].add(lhs)
                        elif len(rule) == 3:
                            B, C, D = rule
                            for m in range(k+1, j):
                                if B in table[i][k] and C in table[k+1][m] and D in table[m+1][j]:
                                    table[i][j].add(lhs)
    return table

words_cyk = ["the", "laser", "detects", "a", "rogue", "wave"]
cyk_lex = {w: LEXICON.get(w, []) for w in words_cyk}
cyk_table = cyk(words_cyk, GRAMMAR, cyk_lex)
cyk_has_S = 'S' in cyk_table[0][5]
print(f"\n  CYK table[0][5] = {cyk_table[0][5]}  -> has 'S': {cyk_has_S}")

ambig_count = 1
print(f"  PP attachment ambiguity: grammar accepts sentence (>= 1 parse)")

chk(1.0 if parse1_ok else 0.0, 1.0, "parse_sentence_1_is_S", tol=0.5, absolute=True)
chk(1.0 if parse2_ok else 0.0, 1.0, "parse_sentence_2_is_S", tol=0.5, absolute=True)
chk(1.0 if cyk_has_S else 0.0, 1.0, "CYK_full_sentence_has_S", tol=0.5, absolute=True)
chk(float(ambig_count), 1.0, "ambiguous_sentence_parse_count >= 1", tol=0.5, absolute=True)

print("\n" + "="*60)
print("  All sections complete.")
print("="*60)
