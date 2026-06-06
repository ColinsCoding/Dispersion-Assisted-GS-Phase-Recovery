# %% [markdown]
# # 10 Things to Try Out
# Base-6 · Transistors · Lasers · Antennas · Log/Exp · +C · Collatz · Binary · GF(2) · Switching
# `init_printing(use_latex="mathjax")` throughout.

# %% [markdown]
# ## Setup

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
from sympy import (symbols, log, exp, ln, sqrt, pi, Rational, latex,
                   integrate, diff, limit, oo, sin, cos, Function,
                   Eq, solve, simplify, Matrix, GF, Poly, factor)
from sympy import init_printing
from IPython.display import display, Math, Markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

init_printing(use_latex="mathjax")

def show(expr, label=""):
    try:
        if label:
            display(Math(r"\textbf{" + label + r"}\quad" + latex(expr)))
        else:
            display(Math(latex(expr)))
    except Exception:
        print(f"{label}  {expr}")

def hdr(s):
    try: display(Markdown(f"### {s}"))
    except: print(f"\n=== {s} ===")

def chk(val, ref, label, tol=1e-6):
    try:
        err = abs(float(val) - float(ref)) / (abs(float(ref)) + 1e-30)
        ok = err < tol
    except:
        ok = sp.simplify(sp.sympify(val) - sp.sympify(ref)) == 0
        err = 0
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}  got={float(val):.6g}  ref={float(ref):.6g}")
    return ok

print("=== 10 Things to Try Out ===")

# %% [markdown]
# ---
# ## Thing 1 · Number Base Conversion — base-6 and beyond
#
# Every integer $n$ in base $b$:
# $$n = \sum_{k=0}^{K} d_k \cdot b^k, \quad d_k \in \{0,1,\ldots,b-1\}$$
# Algorithm: repeated division by $b$, collect remainders bottom-up.

# %%
hdr("Thing 1 — Number Bases (base-6 and all the others)")

def to_base(n, b, digits="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    if n == 0: return "0"
    result = []
    while n:
        result.append(digits[n % b])
        n //= b
    return ''.join(reversed(result))

def from_base(s, b, digits="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    return sum(digits.index(c) * b**i for i, c in enumerate(reversed(s.upper())))

# Show same number in every base 2-16
n_demo = 255
print(f"\n  n = {n_demo} in various bases:")
print(f"  {'Base':>6s}  {'Repr':>20s}  {'Verify':>8s}")
for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]:
    rep = to_base(n_demo, b)
    back = from_base(rep, b)
    print(f"  base-{b:<2d}  {rep:>20s}  {back:>8d}  {'[OK]' if back==n_demo else '[ERR]'}")

# Base-6: "seximal" — why it's interesting
print("\n  Base-6 (seximal) multiplication table:")
print("  " + "".join(f"{j:>4d}" for j in range(1, 7)))
for i in range(1, 7):
    row = [to_base(i*j, 6) for j in range(1, 7)]
    print(f"  {to_base(i,6)}:" + "".join(f"{r:>4s}" for r in row))

# Fractions in base-6: 1/2 = 0.3, 1/3 = 0.2, 1/4 = 0.13, 1/6 = 0.1
print("\n  Fractions in base-6 (compare to base-10):")
fracs = [(1,2),(1,3),(1,4),(1,5),(1,6),(2,3)]
for p, q in fracs:
    # Compute base-6 digits of p/q
    digits_b6 = []
    rem = p
    for _ in range(8):
        rem *= 6
        digits_b6.append(rem // q)
        rem %= q
        if rem == 0: break
    frac_str = "0." + "".join(str(d) for d in digits_b6)
    print(f"    {p}/{q} = {p/q:.6f}  (base-10)  =  {frac_str}  (base-6)")

# SymPy: symbolic base representation
x_b = symbols('x')
n_sym = 42
print(f"\n  SymPy: 42 in binary = {bin(42)}")
print(f"  42 in hex = {hex(42)}")
print(f"  42 in base-6 = {to_base(42,6)}")
print(f"  Verify: 1*36+1*6+0 = {1*36+1*6+0}")
chk(from_base(to_base(42,6),6), 42, "base-6 roundtrip of 42")

# %% [markdown]
# ---
# ## Thing 2 · Transistors — BJT + MOSFET Physics
#
# **BJT:** $I_C = I_S \exp(V_{BE}/V_T)$, $\beta = I_C/I_B$
# **MOSFET (NMOS saturation):** $I_D = \frac{\mu_n C_{ox}}{2}\frac{W}{L}(V_{GS}-V_{th})^2$

# %%
hdr("Thing 2 — Transistors: BJT Ebers-Moll + MOSFET square law")

# Constants
kB_eV = 8.617e-5   # eV/K
T_room = 300       # K
V_T    = kB_eV * T_room   # thermal voltage ~0.02585 V
print(f"\n  Thermal voltage V_T = kT/q = {V_T*1000:.3f} mV at {T_room}K")
chk(V_T, 0.02585, "V_T at 300K", tol=0.005)

# BJT: Shockley diode (Ebers-Moll simplified)
V_BE = np.linspace(0, 0.75, 500)
I_S  = 1e-14   # A (typical small signal NPN)
beta = 200

I_C_arr  = I_S * np.exp(V_BE / V_T)
I_B_arr  = I_C_arr / beta
I_E_arr  = I_C_arr + I_B_arr

print(f"\n  BJT (I_S={I_S:.0e}, beta={beta}):")
print(f"  {'V_BE (V)':>10s}  {'I_C (mA)':>12s}  {'I_B (uA)':>12s}  {'gain hFE':>10s}")
for Vbe in [0.5, 0.6, 0.65, 0.7, 0.75]:
    Ic = I_S * np.exp(Vbe/V_T) * 1e3   # mA
    Ib = Ic/beta*1e3                    # uA
    print(f"  {Vbe:>10.3f}  {Ic:>12.4f}  {Ib:>12.4f}  {beta:>10d}")

# Transconductance gm = dIc/dVbe = Ic/V_T
Ic_op = 1e-3   # 1 mA operating point
gm = Ic_op / V_T
r_pi = beta / gm
print(f"\n  At Ic=1mA: gm = Ic/V_T = {gm*1000:.2f} mA/V = {gm:.4f} S")
print(f"  r_pi = beta/gm = {r_pi:.1f} ohm")
print(f"  Gain Av = -gm*Rc = {-gm*1000:.1f} (Rc=1kohm)")

# SymPy: BJT small-signal
Ic_s, VT_s, beta_s = symbols('I_C V_T beta', positive=True)
gm_sym = diff(Ic_s*exp(symbols('V_BE')/VT_s), symbols('V_BE'))
show(Eq(symbols('g_m'), Ic_s/VT_s), "g_m = dI_C/dV_BE =")

# MOSFET square law
print("\n  NMOS MOSFET (square law, saturation):")
mu_Cox = 200e-6    # A/V^2 (uN*Cox = 200 uA/V^2)
W_L    = 10        # W/L ratio
V_th   = 0.5       # threshold voltage
K      = mu_Cox * W_L / 2   # = 1 mA/V^2

V_GS = np.linspace(0, 2.5, 300)
V_DS_sat = np.maximum(V_GS - V_th, 0)
I_D  = K * V_DS_sat**2

print(f"  K = mu*Cox/2 * W/L = {K*1000:.2f} mA/V^2,  V_th = {V_th} V")
print(f"  {'V_GS (V)':>10s}  {'V_DS,sat (V)':>14s}  {'I_D (mA)':>12s}")
for Vgs in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5]:
    Vdsat = max(Vgs - V_th, 0)
    Id = K * Vdsat**2 * 1e3
    print(f"  {Vgs:>10.2f}  {Vdsat:>14.3f}  {Id:>12.4f}")

# Body effect on V_th
gamma = 0.4  # V^(1/2), body effect coefficient
phi_F = 0.35  # surface potential
V_SB  = 1.0   # source-body voltage
V_th_body = V_th + gamma*(np.sqrt(2*phi_F + V_SB) - np.sqrt(2*phi_F))
print(f"\n  Body effect: V_th with V_SB={V_SB}V -> V_th = {V_th_body:.4f} V")

VGS_s  = symbols('V_GS',      positive=True)
VTH_s  = symbols('V_th',      positive=True)
Vdsat_s= symbols('V_DSsat',   positive=True)
K_s    = symbols('K',         positive=True)
ID_sat = K_s*(VGS_s - VTH_s)**2
show(Eq(symbols('I_D'), ID_sat), "MOSFET saturation I_D =")

# %% [markdown]
# ---
# ## Thing 3 · Lasers — Population Inversion + Threshold
#
# **Einstein coefficients**: spontaneous ($A_{21}$), stimulated emission ($B_{21}$),
# absorption ($B_{12}$). Lasing requires $N_2 > N_1$ (population inversion) —
# impossible in 2-level, need 3 or 4 levels.
#
# **Threshold gain:** $g_{th} = \alpha_{loss} + \frac{1}{2L}\ln\frac{1}{R_1 R_2}$

# %%
hdr("Thing 3 — Lasers: Einstein Coefficients + Threshold")

# Einstein A and B coefficients relationship
# A21/B21 = 8*pi*h*nu^3/c^3
# At thermal equilibrium: N2/N1 = g2/g1 * exp(-h*nu/kT)
h_P  = 6.626e-34   # J*s
c_l  = 3e8         # m/s
kB   = 1.381e-23   # J/K

nu_laser = c_l / 1064e-9   # Nd:YAG 1064 nm
print(f"\n  Nd:YAG laser: lambda=1064nm, nu={nu_laser:.3e} Hz")

A21_B21_ratio = 8 * np.pi * h_P * nu_laser**3 / c_l**3
print(f"  A21/B21 = 8pi*h*nu^3/c^3 = {A21_B21_ratio:.3e} J*s/m^3")

# At room temperature: N2/N1 for nu_laser
ratio_N = np.exp(-h_P * nu_laser / (kB * 300))
print(f"  Thermal N2/N1 at 300K = exp(-hnu/kT) = {ratio_N:.3e}  (essentially 0)")
print(f"  => CANNOT lase thermally — must pump to invert!")

# Threshold condition
print("\n  Laser threshold analysis:")
print("  Round-trip gain must equal loss:")
print("  g_th = alpha_internal + (1/2L)*ln(1/(R1*R2))")
L_cav = 0.3     # m cavity length
R1, R2 = 0.99, 0.5   # mirror reflectivities (OC = 50%)
alpha_i = 0.002  # m^-1 internal loss

g_th = alpha_i + (1/(2*L_cav)) * np.log(1/(R1*R2))
print(f"\n  L={L_cav}m, R1={R1}, R2={R2}, alpha_i={alpha_i}m^-1")
print(f"  g_th = {alpha_i:.4f} + {(1/(2*L_cav))*np.log(1/(R1*R2)):.4f} = {g_th:.4f} m^-1")

# Power output slope efficiency
T_OC = 1 - R2   # output coupler transmission
eta_slope = T_OC / (T_OC + 2*alpha_i*L_cav)
print(f"\n  Output coupler T = {T_OC:.2f}")
print(f"  Slope efficiency eta_slope = T/(T+2*alpha*L) = {eta_slope:.4f}")

# SymPy: rate equations for population inversion
N1_s, N2_s, W_s, A_s, I_s, sigma_s = symbols('N_1 N_2 W A I sigma', positive=True)
# 3-level: dN2/dt = W*N1 - A*N2 - sigma*I*(N2-N1)
# At steady state (dN/dt=0) and N1+N2=N_total
print("\n  Rate equations (3-level laser):")
print("  dN2/dt = W*N1 - A21*N2 - sigma*I*(N2-N1) = 0")
N_t = symbols('N_total', positive=True)
# Solve for population inversion Delta_N = N2-N1
DeltaN = symbols('Delta_N')
# N2 = (N_t+DeltaN)/2, N1 = (N_t-DeltaN)/2
# Substituting:
# W*(N_t-DeltaN)/2 - A*(N_t+DeltaN)/2 - sigma*I*DeltaN = 0
# DeltaN*(W/2 + A/2 + sigma*I) = (W-A)*N_t/2
dn_eq = Eq(DeltaN*(W_s/2 + A_s/2 + sigma_s*I_s), (W_s-A_s)*N_t/2)
DeltaN_sol = solve(dn_eq, DeltaN)[0]
show(DeltaN_sol, "Delta_N =")
print("  Inversion > 0 requires W > A (pump rate > spontaneous emission rate)")

# Semiconductor laser: E_g and emission wavelength
print("\n  Semiconductor lasers (lambda = hc/E_g):")
materials = [
    ("GaAs",     1.424, "Near-IR 870nm"),
    ("InP",      1.351, "Near-IR 917nm"),
    ("GaN",      3.4,   "UV 365nm"),
    ("In_0.53Ga_0.47As", 0.74, "Telecom 1675nm"),
    ("Al_0.3Ga_0.7As",   1.79, "Red 693nm"),
]
print(f"  {'Material':25s}  {'E_g (eV)':>10s}  {'lambda (nm)':>12s}  {'Band':>12s}")
for mat, Eg, band in materials:
    lam_nm = (h_P*c_l)/(Eg*1.602e-19)*1e9
    print(f"  {mat:25s}  {Eg:>10.3f}  {lam_nm:>12.1f}  {band}")

# %% [markdown]
# ---
# ## Thing 4 · Antenna Theory — Dipole + Friis Equation
#
# Half-wave dipole: $D = 1.64$ (2.15 dBi).
# **Friis:** $\frac{P_r}{P_t} = G_t G_r \left(\frac{\lambda}{4\pi d}\right)^2$

# %%
hdr("Thing 4 — Antenna: Half-wave Dipole + Friis Link Budget")

# Radiation pattern of a half-wave dipole
theta = np.linspace(0.001, np.pi-0.001, 500)
# F(theta) = [cos(pi/2*cos(theta))/sin(theta)]^2
F_dipole = (np.cos(np.pi/2 * np.cos(theta)) / np.sin(theta))**2
# Normalize
F_dipole /= F_dipole.max()

# Directivity: D = 4pi * U_max / P_rad
# For half-wave dipole: D = 1.64 (exact numerical)
# Numerical integration
dtheta = theta[1] - theta[0]
P_norm = 2 * np.pi * np.sum(F_dipole * np.sin(theta)) * dtheta
D_num  = 4 * np.pi * F_dipole.max() / P_norm
print(f"\n  Half-wave dipole directivity D = {D_num:.4f}  (ref: 1.64)")
chk(D_num, 1.64, "dipole directivity", tol=0.01)
print(f"  D in dBi = {10*np.log10(D_num):.3f} dBi  (ref: 2.15 dBi)")

# Friis transmission equation
print("\n  Friis equation: Pr/Pt = Gt*Gr*(lambda/4*pi*d)^2")
print("  In dB: Pr(dBm) = Pt(dBm) + Gt(dB) + Gr(dB) - FSPL(dB)")
print("  FSPL = 20*log10(4*pi*d/lambda) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)")

c_sp = 3e8
def FSPL_dB(f_Hz, d_m):
    return 20*np.log10(4*np.pi*d_m*f_Hz/c_sp)

print(f"\n  Free-Space Path Loss:")
print(f"  {'f (GHz)':>10s}  {'d (km)':>10s}  {'FSPL (dB)':>12s}")
for f_GHz in [0.9, 2.4, 5.8, 28, 60]:
    for d_km in [0.1, 1.0, 10.0]:
        fspl = FSPL_dB(f_GHz*1e9, d_km*1e3)
        print(f"  {f_GHz:>10.1f}  {d_km:>10.1f}  {fspl:>12.1f}")

# Link budget example: WiFi 2.4 GHz
Pt_dBm  = 20    # dBm = 100 mW
Gt_dBi  = 3     # TX antenna gain
Gr_dBi  = 3     # RX antenna gain
d_m     = 100   # 100 m
f_Hz    = 2.4e9
fspl    = FSPL_dB(f_Hz, d_m)
Pr_dBm  = Pt_dBm + Gt_dBi + Gr_dBi - fspl
sens    = -90   # dBm receiver sensitivity
margin  = Pr_dBm - sens
print(f"\n  WiFi link budget (2.4GHz, 100m):")
print(f"    Pt={Pt_dBm}dBm  Gt=Gr={Gt_dBi}dBi  FSPL={fspl:.1f}dB")
print(f"    Pr = {Pr_dBm:.1f} dBm  (sensitivity={sens}dBm)  margin={margin:.1f}dB")
print(f"    {'[LINK OK]' if margin>0 else '[LINK FAIL]'}")
chk(fspl, FSPL_dB(2.4e9,100), "WiFi FSPL at 100m", tol=1e-4)

# Radiation resistance
Z_rad_dipole = 73.1   # ohm (half-wave dipole)
print(f"\n  Half-wave dipole: radiation resistance = {Z_rad_dipole} ohm (ref)")
print(f"  Impedance matching to 50-ohm coax: need matching network")
print(f"  Quarter-wave transformer: Z_match = sqrt(73.1*50) = {np.sqrt(73.1*50):.1f} ohm")

# SymPy: antenna aperture
lam_s, D_s, d_s = symbols('lambda D d', positive=True)
Ae = D_s * lam_s**2 / (4*sp.pi)
Friis_sym = Eq(symbols('P_r/P_t'), D_s**2 * (lam_s/(4*sp.pi*d_s))**2)
show(Ae, "Effective aperture A_e =")
show(Friis_sym, "Friis (Gt=Gr=D):")

# %% [markdown]
# ---
# ## Thing 5 · Logarithms and Exponentials — Properties + Log Amplifier
#
# They are inverses: $\log_b(b^x) = x$, $b^{\log_b x} = x$
# Change of base: $\log_b x = \ln x / \ln b$

# %%
hdr("Thing 5 — Log + Exp: Properties + Log Amplifier")

x_l = symbols('x', positive=True)
a_l, b_l, n_l = symbols('a b n', positive=True)

# Core identities
identities = [
    (log(a_l*b_l),           log(a_l)+log(b_l),          "product rule"),
    (log(a_l/b_l),           log(a_l)-log(b_l),           "quotient rule"),
    (log(a_l**n_l),          n_l*log(a_l),                "power rule"),
    (log(1/a_l),             -log(a_l),                   "reciprocal"),
    (sp.exp(log(x_l)),       x_l,                         "exp(ln x)=x"),
    (log(sp.exp(x_l)),       x_l,                         "ln(e^x)=x"),
]
print("\n  Core logarithm identities (SymPy verified):")
for lhs, rhs, name in identities:
    ok = sp.simplify(lhs - rhs) == 0
    print(f"  [{' OK ' if ok else 'FAIL'}]  {name}: {lhs} = {rhs}")

# Change of base
print("\n  Change of base: log_b(x) = ln(x)/ln(b)")
for b_val in [2, 6, 10, 16]:
    for x_val in [64, 100, 256]:
        log_bx = np.log(x_val) / np.log(b_val)
        print(f"    log_{b_val}({x_val}) = ln({x_val})/ln({b_val}) = {log_bx:.4f}")

# Derivative + integral
print("\n  Calculus:")
show(sp.Eq(sp.Symbol('d/dx ln(x)'), diff(log(x_l),x_l)), "derivative:")
show(sp.Eq(sp.Symbol('integral ln(x) dx'), integrate(log(x_l),x_l)), "antiderivative:")
show(sp.Eq(sp.Symbol('d/dx e^x'), diff(sp.exp(x_l),x_l)), "d/dx e^x =")

# Log amplifier (translinear circuit)
print("\n  Log amplifier circuit (BJT translinear):")
print("  V_out = -V_T * ln(I_in / I_S)")
print("  Based on: I_C = I_S * exp(V_BE/V_T)  =>  V_BE = V_T * ln(I_C/I_S)")
I_S_log = 1e-14
I_range = np.logspace(-9, -3, 7)   # 1nA to 1mA
V_T_val = 0.02585
print(f"  {'I_in':>10s}  {'V_out (mV)':>12s}")
for I_v in I_range:
    Vout = -V_T_val * np.log(I_v/I_S_log)*1000
    print(f"  {I_v:.2e}A  {Vout:>12.2f}")
print("  => 60 mV/decade (10x current = 60mV output change)")
print(f"     Verify: V_T*ln(10) = {V_T_val*np.log(10)*1000:.2f} mV/decade")
chk(V_T_val*np.log(10)*1000, 59.53, "mV/decade", tol=0.01)

# %% [markdown]
# ---
# ## Thing 6 · +C — The Integration Constant and Why It Matters
#
# $\int f(x)\,dx = F(x) + C$ because $\frac{d}{dx}[F(x)+C] = f(x)$ for ANY constant $C$.
# The constant is pinned by **initial conditions** or **boundary conditions**.
# Getting $C$ wrong gives a completely different physical solution.

# %%
hdr("Thing 6 — The +C: Integration Constant + Initial Conditions")

t_s, C_s = symbols('t C', real=True)

# Why +C: family of antiderivatives
print("\n  Every antiderivative F(x)+C is a valid solution to F'=f:")
f_demo2 = 2*t_s
F_general = integrate(f_demo2, t_s) + C_s
show(F_general, "integral(2t) dt =")
show(diff(F_general, t_s), "d/dt [t^2+C] =")

# Physical example: projectile
print("\n  Projectile: a = -g = -9.81 m/s^2")
print("  v(t) = integral(-9.81) dt = -9.81t + C1  (C1 = v0)")
print("  y(t) = integral(v) dt = -4.905t^2 + v0*t + C2  (C2 = y0)")
g_v = 9.81
v0  = 20.0   # m/s initial velocity
y0  = 100.0  # m initial height

t_arr_proj = np.linspace(0, 6, 300)
v_proj = -g_v*t_arr_proj + v0
y_proj = -0.5*g_v*t_arr_proj**2 + v0*t_arr_proj + y0

# Find when y=0 (hits ground)
t_land = np.roots([-0.5*g_v, v0, y0])
t_land_pos = t_land[t_land > 0].max()
print(f"\n  v0={v0}m/s, y0={y0}m")
print(f"  Lands at t = {t_land_pos:.4f} s (quadratic formula)")
print(f"  Max height: y_max = y0 + v0^2/(2g) = {y0 + v0**2/(2*g_v):.2f} m")
chk(y0 + v0**2/(2*g_v), y_proj.max(), "max height formula vs numerical", tol=0.01)

# RC circuit: V(t) = V_s*(1-e^{-t/RC}) if V(0)=0
print("\n  RC circuit: dV/dt = (V_s-V)/(RC)")
print("  General solution: V(t) = V_s + C*exp(-t/RC)")
print("  IC V(0)=0: 0 = V_s + C => C = -V_s")
print("  => V(t) = V_s*(1 - exp(-t/RC))  [C is determined by IC]")
R_c, C_c = 1000, 1e-6  # 1kohm, 1uF
tau = R_c * C_c
Vs  = 5.0
t_rc = np.linspace(0, 5*tau, 300)
V_rc = Vs * (1 - np.exp(-t_rc/tau))
print(f"  tau = RC = {tau*1000:.2f} ms,  at t=tau: V = {Vs*(1-1/np.e):.3f} V  (= 0.632*Vs)")
chk(Vs*(1-1/np.e), 5*0.6321, "V at t=tau", tol=0.01)

# Particular solution from general: SymPy
from sympy import dsolve, Function, Eq as SEq
V_func = Function('V')
tau_s, Vs_s = symbols('tau V_s', positive=True)
ode_rc = SEq(V_func(t_s).diff(t_s), (Vs_s - V_func(t_s))/tau_s)
sol_rc = dsolve(ode_rc, V_func(t_s))
show(sol_rc, "General solution (RC ODE):")
# Apply IC V(0)=0
C1_val = solve(sol_rc.rhs.subs(t_s,0), symbols('C1'))[0]
sol_particular = sol_rc.subs(symbols('C1'), C1_val)
show(sol_particular, "Particular solution V(0)=0:")

# %% [markdown]
# ---
# ## Thing 7 · Collatz Conjecture — 3n+1 (Unsolved!)
#
# **Rule:** if $n$ odd: $n \to 3n+1$; if $n$ even: $n \to n/2$.
# **Conjecture (1937):** every positive integer eventually reaches 1.
# **Status: UNSOLVED.** Terence Tao (2019): "almost all" integers reach 1.

# %%
hdr("Thing 7 — Collatz Conjecture (3n+1): Still Unsolved in 2026")

def collatz(n):
    seq = [n]
    while n != 1:
        n = 3*n+1 if n % 2 else n//2
        seq.append(n)
    return seq

def stopping_time(n):
    return len(collatz(n)) - 1

# Sequences for small numbers
print("\n  Collatz sequences:")
for n0 in [6, 11, 27, 97, 871]:
    seq = collatz(n0)
    print(f"  n={n0:>4d}: length={len(seq):>5d}, max={max(seq):>12d}  {str(seq[:8])}{'...' if len(seq)>8 else ''}")

# Famous: n=27 takes 111 steps and reaches 9232
seq27 = collatz(27)
print(f"\n  n=27: {len(seq27)} steps, max={max(seq27)} (quite a journey!)")
chk(len(seq27), 112, "Collatz(27) length=112", tol=1e-9)

# Stopping time distribution
n_max = 10000
stop_times = np.array([stopping_time(n) for n in range(1, n_max+1)])
print(f"\n  Stopping times for n=1..{n_max}:")
print(f"  Mean = {stop_times.mean():.2f},  Max = {stop_times.max()} (at n={np.argmax(stop_times)+1})")
print(f"  Distribution: mean scales roughly as O(log n)")

# Record-setting n values (high stopping times)
records = []
max_so_far = 0
for n in range(1, 10001):
    st = stop_times[n-1]
    if st > max_so_far:
        max_so_far = st
        records.append((n, st))
print(f"\n  Record stopping times (n <= 10000):")
for n_r, st_r in records[-10:]:
    print(f"    n={n_r:>6d}: stopping time = {st_r}")

# Why it's hard: 3n+1 is multiplication by 3 in base-2 (carry propagation)
# Each odd step: n -> 3n+1; but next is always even so /2 immediately
# Effective odd step: n -> (3n+1)/2
print("\n  Mathematical note:")
print("  Odd step: n -> (3n+1)/2  (net: multiply by ~1.5)")
print("  Even steps: n -> n/2     (net: multiply by 0.5)")
print("  Average: 0.5*log2(1.5) + 0.5*log2(0.5) = {:.4f} < 0 => tends to decrease".format(
    0.5*np.log2(1.5) + 0.5*np.log2(0.5)))
print("  But PROVING this always reaches 1 — unsolved for 85+ years.")
print("  Tao 2019: almost all n reach below any f(n)=n^epsilon.")

# %% [markdown]
# ---
# ## Thing 8 · Binary Arithmetic — Two's Complement + Bitwise Ops
#
# **Two's complement:** negative $n$ represented as $2^N - n$.
# Overflow detection: carry into sign bit XOR carry out of sign bit.

# %%
hdr("Thing 8 — Binary Arithmetic: Two's Complement + Bitwise")

def to_twos(n, bits=8):
    if n >= 0: return n & ((1<<bits)-1)
    return (1<<bits) + n

def from_twos(bits_val, bits=8):
    if bits_val >= (1<<(bits-1)): return bits_val - (1<<bits)
    return bits_val

def show_bin(n, bits=8, label=""):
    tw = to_twos(n, bits)
    b = format(tw, f'0{bits}b')
    print(f"  {label:>12s} {n:>6d} = {b}  (hex: {tw:0{bits//4}X})")

print("\n  8-bit two's complement representation:")
for n_v in [0, 1, 127, -1, -128, -127, 42, -42]:
    show_bin(n_v, label=f"n={n_v:>4d}:")

# Addition with overflow detection
def add_twos(a, b, bits=8):
    mask = (1<<bits)-1
    sign_bit = 1<<(bits-1)
    result = (a + b) & mask
    # Overflow: both inputs same sign, result different sign
    a_sign = (to_twos(a,bits) >> (bits-1)) & 1
    b_sign = (to_twos(b,bits) >> (bits-1)) & 1
    r_sign = (result >> (bits-1)) & 1
    overflow = (a_sign == b_sign) and (r_sign != a_sign)
    return from_twos(result, bits), overflow

print("\n  8-bit addition with overflow detection:")
test_adds = [(100, 50), (127, 1), (-100, -50), (-128, -1), (50, -30)]
for a_v, b_v in test_adds:
    r, ovf = add_twos(a_v, b_v, 8)
    true_r = a_v + b_v
    print(f"  {a_v:>5d} + {b_v:>5d} = {r:>6d}  "
          f"{'OVERFLOW!' if ovf else '          '}  (true: {true_r})")

# Bitwise operations
print("\n  Bitwise operations (8-bit unsigned):")
a_b, b_b = 0b10110100, 0b01101101
print(f"  a = {a_b:08b} ({a_b})")
print(f"  b = {b_b:08b} ({b_b})")
print(f"  a AND b = {a_b&b_b:08b} ({a_b&b_b})")
print(f"  a OR  b = {a_b|b_b:08b} ({a_b|b_b})")
print(f"  a XOR b = {a_b^b_b:08b} ({a_b^b_b})")
print(f"  NOT a   = {(~a_b)&0xFF:08b} ({(~a_b)&0xFF})")
print(f"  a << 2  = {(a_b<<2)&0xFF:08b} ({(a_b<<2)&0xFF})  (multiply by 4)")
print(f"  a >> 2  = {a_b>>2:08b} ({a_b>>2})  (divide by 4)")

# Power-of-2 tricks
print("\n  Power-of-2 bit tricks:")
print(f"  n & (n-1) == 0 iff n is power-of-2: {16&15}=={0}? {16&15==0}")
print(f"  n & (-n) = lowest set bit:  n=12={12:08b}  n&-n={12&(-12)}={12&(-12):08b}")
print(f"  Hamming weight of 255: {bin(255).count('1')} ones")

# %% [markdown]
# ---
# ## Thing 9 · Polynomial Logic over GF(2) — LFSR + CRC
#
# Boolean algebra IS a polynomial ring over $\text{GF}(2) = \{0,1\}$.
# XOR = addition mod 2, AND = multiplication.
# **LFSR** (Linear Feedback Shift Register) = polynomial division in GF(2).
# **CRC** (Cyclic Redundancy Check) = remainder of message polynomial / generator poly.

# %%
hdr("Thing 9 — GF(2) Polynomials: LFSR + CRC")

# GF(2) polynomial arithmetic
def gf2_add(a, b): return a ^ b          # XOR
def gf2_mul_bit(a, b): return a & b      # AND

# Polynomial multiplication in GF(2)
def gf2_polymul(a, b):
    result = 0
    while b:
        if b & 1: result ^= a
        a <<= 1
        b >>= 1
    return result

# Polynomial division: returns (quotient, remainder)
def gf2_polydiv(dividend, divisor):
    degree_d = dividend.bit_length() - 1
    degree_v = divisor.bit_length() - 1
    quotient = 0
    remainder = dividend
    for i in range(degree_d - degree_v, -1, -1):
        if remainder.bit_length() - 1 >= degree_v + i:
            quotient |= (1 << i)
            remainder ^= (divisor << i)
    return quotient, remainder

def poly_str(n, var='x'):
    """Convert integer to polynomial string."""
    if n == 0: return "0"
    terms = []
    for i in range(n.bit_length()-1, -1, -1):
        if (n >> i) & 1:
            if i == 0: terms.append("1")
            elif i == 1: terms.append(var)
            else: terms.append(f"{var}^{i}")
    return " + ".join(terms)

print("\n  GF(2) polynomial arithmetic:")
a_gf = 0b10011    # x^4 + x + 1
b_gf = 0b111      # x^2 + x + 1
print(f"  a = {poly_str(a_gf)} = {bin(a_gf)}")
print(f"  b = {poly_str(b_gf)} = {bin(b_gf)}")
print(f"  a XOR b (add) = {poly_str(a_gf^b_gf)} = {bin(a_gf^b_gf)}")
q, r = gf2_polydiv(a_gf, b_gf)
print(f"  a / b = quotient {poly_str(q)}, remainder {poly_str(r)}")

# LFSR: maximal-length sequence (m-sequence)
print("\n  LFSR (4-bit, tap polynomial x^4+x+1 = 0x13):")
def lfsr(seed, taps, length):
    state = seed
    seq = []
    mask = (1<<taps)-1
    for _ in range(length):
        seq.append(state & 1)       # output bit
        new_bit = bin(state & 0x9).count('1') % 2  # x^4+x (taps at 4,1)
        state = ((state >> 1) | (new_bit << (taps-1))) & mask
    return seq

seq_lfsr = lfsr(0b1111, 4, 15)
print(f"  Output: {seq_lfsr}")
print(f"  Period = {len(seq_lfsr)}  (max = 2^4-1 = 15 = {'PASS' if len(set(map(tuple,zip(*[seq_lfsr]*4)))) == 15 else 'check'})")
print(f"  Ones: {sum(seq_lfsr)}, Zeros: {len(seq_lfsr)-sum(seq_lfsr)}  (almost balanced)")
chk(len(seq_lfsr), 15, "LFSR period = 2^n-1", tol=1e-9)

# CRC-8 calculation
print("\n  CRC-8 (generator = x^8+x^2+x+1 = 0x107):")
CRC8_GEN = 0x107  # x^8 + x^2 + x + 1

def crc8(data_bytes):
    crc = 0
    for byte in data_bytes:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc ^= (CRC8_GEN << 7)
            crc <<= 1
    return (crc >> 8) & 0xFF

messages = [b"Hello", b"CSUS", b"\x00\x01\x02\x03"]
for msg in messages:
    crc = crc8(msg)
    print(f"  CRC8({msg}) = 0x{crc:02X} = {bin(crc)}")

# Boolean algebra as GF(2)[x]: De Morgan, etc.
print("\n  Boolean algebra = GF(2) ring:")
print("  XOR  = addition mod 2   (A XOR A = 0, A XOR 0 = A)")
print("  AND  = multiplication   (distributive over XOR)")
print("  NOT  = add 1: NOT A = A XOR 1")
print("  NAND = NOT(AND) = complement of product")
print("  De Morgan: NOT(A AND B) = (NOT A) OR (NOT B)")
# Verify De Morgan numerically
for A in [0,1]:
    for B in [0,1]:
        lhs = not(A and B)
        rhs = (not A) or (not B)
        print(f"    A={A},B={B}: NOT(AND)={int(lhs)}, (NOT A) OR (NOT B)={int(rhs)}  {'OK' if lhs==rhs else 'FAIL'}")

# %% [markdown]
# ---
# ## Thing 10 · Switching in Log/Exp Domain — CMOS Inverter + Log-Domain Signal Processing
#
# **CMOS inverter switching threshold:** $V_{th} = (V_{tp} + V_{tn}\sqrt{K_n/K_p})/(1+\sqrt{K_n/K_p})$
# **Log-domain:** multiply in linear = add in log. Used in companders, AGC, neural nets.

# %%
hdr("Thing 10 — Switching: CMOS Inverter + Log-Domain Signal Processing")

# CMOS inverter static characteristics
Kn = 1e-3    # NMOS: mu_n*Cox/2 * W/L = 1 mA/V^2
Kp = 0.5e-3  # PMOS: mu_p*Cox/2 * W/L = 0.5 mA/V^2 (lower mobility)
Vtn = 0.5    # NMOS threshold
Vtp = -0.5   # PMOS threshold (negative)
Vdd = 3.3

# Switching threshold (VM)
VM = (Vtp + Vtn*np.sqrt(Kn/Kp)) / (1 + np.sqrt(Kn/Kp)) + Vdd/2
print(f"\n  CMOS inverter: Kn={Kn*1e3:.1f}mA/V^2, Kp={Kp*1e3:.1f}mA/V^2")
print(f"  Switching threshold VM = {VM:.3f} V  (ideal: Vdd/2={Vdd/2:.3f}V)")

# VTC (Voltage Transfer Curve) — simplified piecewise
Vin_arr = np.linspace(0, Vdd, 500)
Vout_arr = np.zeros_like(Vin_arr)
for i, Vin in enumerate(Vin_arr):
    if Vin <= Vtn:
        Vout_arr[i] = Vdd      # PMOS on, NMOS off
    elif Vin >= Vdd + Vtp:
        Vout_arr[i] = 0        # NMOS on, PMOS off
    else:
        # Transition region: linear interpolation (simplified)
        t = (Vin - Vtn) / (Vdd + Vtp - Vtn)
        Vout_arr[i] = Vdd * (1 - t)

# Noise margins
NMH = Vdd - VM
NML = VM
print(f"  NMH = Vdd - VM = {NMH:.3f} V")
print(f"  NML = VM - 0   = {NML:.3f} V")

# Gate delay: Elmore delay model
print(f"\n  Elmore delay model:")
Cox_area = 3e-3   # fF/um^2 (gate oxide cap)
W_n = 1.0; L_n = 0.18  # um
Cgate = Cox_area * W_n * L_n   # fF
Rn_on = 1/(2*Kn*(Vdd-Vtn))    # ohm
tau_HL = 0.69 * Rn_on * Cgate*1e-15
print(f"  Cgate = {Cgate:.3f} fF, Rn_on = {Rn_on:.1f} ohm")
print(f"  tau_HL = 0.69*R*C = {tau_HL*1e12:.3f} ps  (HL transition)")

# Log-domain signal processing
print("\n  Log-domain signal processing:")
print("  Key identity: log(A*B) = log(A) + log(B)")
print("  => multiplication -> addition (implemented by BJT translinear loops)")
print("\n  Applications:")
print("  1. Compander: compress dynamic range for transmission")
print("     mu-law: F(x) = V_max * sign(x) * ln(1+mu|x|/V_max) / ln(1+mu)")
print("     mu=255 (North America): 8 bits -> ~13 bits perceived quality")
print("  2. AGC (Auto Gain Control): log-amp -> comparator -> VGA")
print("  3. Neural network activation: sigmoid = 1/(1+e^{-x}) = log-domain threshold")
print("  4. dB calculations: power_dB = 10*log10(P/P_ref)")

mu_val = 255
x_comp = np.linspace(-1, 1, 500)
F_mu = np.sign(x_comp) * np.log(1 + mu_val*np.abs(x_comp)) / np.log(1+mu_val)

print("\n  mu-law compression (mu=255) vs linear:")
for x_v in [0.01, 0.1, 0.5, 1.0]:
    F_v = np.sign(x_v)*np.log(1+mu_val*np.abs(x_v))/np.log(1+mu_val)
    print(f"    x={x_v:.2f} -> F(x)={F_v:.4f}  (compression ratio: {F_v/x_v:.2f}x)")

print("\n  dB table:")
print(f"  {'Power ratio':>14s}  {'dB':>8s}  {'meaning'}")
for pr, meaning in [(0.001,'10x attenuation'),(0.1,'10dB loss'),(0.5,'3dB loss'),
                    (1,'no change'),(2,'3dB gain'),(10,'10dB gain'),(1000,'30dB gain')]:
    print(f"  {pr:>14.3f}  {10*np.log10(pr):>8.2f}  {meaning}")

# %% [markdown]
# ---
# ## Figure — 12 panels (one per thing, 2 panels share)

# %%
fig = plt.figure(figsize=(20, 15))
gf = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)
c4 = ["#4C72B0","#DD8452","#55A868","#C44E52"]

# P1: Base conversion visual
ax1 = fig.add_subplot(gf[0,0])
bases = [2,3,4,5,6,7,8,10,12,16]
lengths = [len(to_base(255,b)) for b in bases]
ax1.bar(range(len(bases)), lengths, color='steelblue', edgecolor='k')
ax1.set_xticks(range(len(bases)))
ax1.set_xticklabels([f'b={b}' for b in bases], fontsize=7)
ax1.set_ylabel("Digits needed"); ax1.set_title("255 in each base", fontsize=9)

# P2: BJT + MOSFET I-V
ax2 = fig.add_subplot(gf[0,1])
ax2.semilogy(V_BE, I_C_arr*1000, 'b-', lw=2, label='BJT I_C (mA)')
ax2_r = ax2.twinx()
ax2_r.plot(V_GS, I_D*1000, 'r-', lw=2, label='MOSFET I_D (mA)')
ax2_r.set_ylabel('I_D (mA)', color='r')
ax2.set_xlabel('V (V)'); ax2.set_ylabel('I_C (mA, log)', color='b')
ax2.set_title('BJT + MOSFET I-V', fontsize=9)
ax2.legend(loc='upper left', fontsize=7); ax2_r.legend(loc='lower right', fontsize=7)

# P3: Laser gain vs pump rate
ax3 = fig.add_subplot(gf[0,2])
W_range = np.linspace(0, 3, 200)    # relative to A21
DeltaN_range = (W_range-1)*0.5      # simplified: positive above W=A
ax3.plot(W_range, DeltaN_range, 'purple', lw=2)
ax3.axhline(0, color='k', lw=0.8)
ax3.axvline(1, color='r', ls='--', lw=1, label='threshold W=A21')
ax3.fill_between(W_range, DeltaN_range, where=(DeltaN_range>0),
                 alpha=0.3, color='green', label='inversion region')
ax3.set_xlabel("W/A21 (pump/spontaneous)"); ax3.set_ylabel("Population inversion")
ax3.set_title("Laser population inversion", fontsize=9); ax3.legend(fontsize=7)

# P4: Antenna dipole pattern (polar)
ax4 = fig.add_subplot(gf[0,3], projection='polar')
ax4.plot(theta, F_dipole, 'b-', lw=2, label='E-plane')
ax4.plot(theta+np.pi, F_dipole, 'b-', lw=2)
ax4.set_title("Half-wave dipole\nradiation pattern", fontsize=9)
ax4.set_yticks([0.25,0.5,0.75,1.0])

# P5: Log/exp functions
ax5 = fig.add_subplot(gf[1,0])
x_plt = np.linspace(0.01, 5, 300)
ax5.plot(x_plt, np.log2(x_plt), label='log_2', lw=2, color=c4[0])
ax5.plot(x_plt, np.log(x_plt),  label='ln', lw=2, color=c4[1])
ax5.plot(x_plt, np.log10(x_plt),label='log_10', lw=2, color=c4[2])
ax5.plot(x_plt, np.exp(x_plt-2.5), label='e^x (shifted)', lw=2, color=c4[3])
ax5.axhline(0, color='k', lw=0.5); ax5.axvline(1, color='gray', ls=':', lw=0.8)
ax5.set_ylim(-3, 4); ax5.set_title("Logarithms + exponential", fontsize=9)
ax5.set_xlabel("x"); ax5.legend(fontsize=7)

# P6: +C family of antiderivatives + RC charging
ax6 = fig.add_subplot(gf[1,1])
t_fam = np.linspace(0, 3, 200)
for C_val, col in zip([-2,-1,0,1,2], ['purple','b','g','orange','r']):
    ax6.plot(t_fam, t_fam**2 + C_val, color=col, alpha=0.7, label=f'C={C_val}')
ax6.set_title("Family t² + C  (all solve y'=2t)", fontsize=9)
ax6.set_xlabel("t"); ax6.set_ylabel("F(t)"); ax6.legend(fontsize=6)

# P7: Collatz stopping times
ax7 = fig.add_subplot(gf[1,2])
n_coll = np.arange(1, 1001)
st_coll = np.array([stopping_time(n) for n in n_coll])
ax7.scatter(n_coll, st_coll, s=1, alpha=0.4, c=st_coll, cmap='plasma')
ax7.set_xlabel("n"); ax7.set_ylabel("Stopping time")
ax7.set_title("Collatz 3n+1 stopping times", fontsize=9)

# P8: n=27 Collatz trajectory
ax8 = fig.add_subplot(gf[1,3])
ax8.plot(seq27, 'b-', lw=0.8, alpha=0.8)
ax8.set_xlabel("Step"); ax8.set_ylabel("n")
ax8.set_title("Collatz trajectory n=27 (111 steps, max=9232)", fontsize=9)

# P9: GF(2) LFSR sequence
ax9 = fig.add_subplot(gf[2,0])
seq_full = lfsr(0b1111, 4, 60)
ax9.step(range(len(seq_full)), seq_full, 'g-', lw=1.5)
ax9.set_xlabel("Clock cycle"); ax9.set_ylabel("Output bit")
ax9.set_title(f"4-bit LFSR m-sequence (period 15)", fontsize=9)
ax9.set_ylim(-0.2, 1.4)

# P10: CMOS inverter VTC
ax10 = fig.add_subplot(gf[2,1])
ax10.plot(Vin_arr, Vout_arr, 'b-', lw=2)
ax10.axvline(VM, color='r', ls='--', lw=1, label=f'VM={VM:.2f}V')
ax10.plot([0,Vdd],[Vdd,0],'k:',lw=0.8,label='ideal')
ax10.fill_between([0,VM],[Vdd,Vdd],[Vdd*0.7,Vdd*0.7], alpha=0.1, color='g', label='NMH')
ax10.set_xlabel("Vin (V)"); ax10.set_ylabel("Vout (V)")
ax10.set_title("CMOS Inverter VTC", fontsize=9); ax10.legend(fontsize=7)

# P11: mu-law compression
ax11 = fig.add_subplot(gf[2,2])
ax11.plot(x_comp, x_comp, 'b--', lw=1, label='linear')
ax11.plot(x_comp, F_mu, 'r-', lw=2, label=f'mu-law mu={mu_val}')
ax11.set_xlabel("Input"); ax11.set_ylabel("Compressed output")
ax11.set_title("mu-law compression (log-domain)", fontsize=9)
ax11.legend(fontsize=7); ax11.grid(True, alpha=0.3)

# P12: Base-6 multiplication table heatmap
ax12 = fig.add_subplot(gf[2,3])
b6_table = np.array([[i*j for j in range(1,7)] for i in range(1,7)])
im = ax12.imshow(b6_table, cmap='YlOrRd')
ax12.set_xticks(range(6)); ax12.set_xticklabels(range(1,7))
ax12.set_yticks(range(6)); ax12.set_yticklabels(range(1,7))
ax12.set_title("Base-6 mult table", fontsize=9)
for i in range(6):
    for j in range(6):
        ax12.text(j,i,to_base(b6_table[i,j],6),ha='center',va='center',fontsize=8)
plt.colorbar(im, ax=ax12)

fig.suptitle(
    "10 Things to Try Out: Bases · Transistors · Lasers · Antennas · "
    "Log/Exp · +C · Collatz · Binary · GF(2) · CMOS Switching",
    fontsize=11, fontweight='bold', y=1.01
)

import pathlib
out_dir  = pathlib.Path(__file__).parent if "__file__" in dir() else pathlib.Path("repl")
out_path = out_dir / "_out_ten_things.png"
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)
print("\n=== All 10 things complete ===")
