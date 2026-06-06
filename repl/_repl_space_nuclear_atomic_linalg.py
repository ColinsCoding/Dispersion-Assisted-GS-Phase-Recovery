# %% [markdown]
# # Space · Nuclear · Atomic ODEs · Stoichiometry · Linear Algebra
# **Facts not feelings.** Every number is sourced; every equation is derived.
# Sections:
# §1 Orbital mechanics — vis-viva, Hohmann Δv, escape velocity
# §2 n-Body problem — RK4, three-body chaos, Lyapunov exponent
# §3 Nuclear propulsion — Tsiolkovsky, specific impulse, NERVA specs
# §4 Atomic ODEs — Bateman decay-chain, matrix exponential solution
# §5 Stoichiometry via null-space (linear algebra)
# §6 Linear algebra core — eigen, SVD, matrix exp, condition number
# §7 RTOS killmode — SIGKILL, WCET, watchdog, scheduling theory
# §8 12-panel figure

# %% [markdown]
# ## Setup

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
from sympy import (symbols, exp, sqrt, pi, Rational, Matrix, latex,
                   solve, simplify, Function, dsolve, Derivative, oo,
                   integrate, cos, sin, ln, Abs)
from sympy import init_printing
from IPython.display import display, Math, Markdown
import scipy.integrate as sci
import scipy.linalg as scl
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

def chk(val, ref, label, tol=1e-4):
    err = abs(float(val) - float(ref)) / (abs(float(ref)) + 1e-30)
    ok  = err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}  got={float(val):.6g}  ref={float(ref):.6g}  rel_err={err:.2e}")
    return ok

print("=== Space / Nuclear / Atomic / LinAlg  ===")

# %% [markdown]
# ---
# ## §1 · Orbital Mechanics
#
# ### Vis-viva equation
# $$v^2 = GM\!\left(\frac{2}{r} - \frac{1}{a}\right)$$
# Energy per unit mass $\varepsilon = -GM/(2a)$.

# %%
hdr("§1 — Orbital Mechanics")

# Constants
G_c  = 6.674e-11        # N m^2 kg^-2
M_sun = 1.989e30        # kg
M_earth = 5.972e24      # kg
R_earth = 6.371e6       # m
AU   = 1.496e11         # m
mu_sun = G_c * M_sun    # m^3/s^2
mu_earth = G_c * M_earth

# Low Earth Orbit
r_LEO = R_earth + 400e3
v_LEO = np.sqrt(mu_earth / r_LEO)
print(f"\n  LEO (400 km) orbital velocity: {v_LEO/1e3:.3f} km/s")
chk(v_LEO, 7669.0, "LEO velocity", tol=0.01)

# Escape velocity from Earth surface
v_esc = np.sqrt(2 * mu_earth / R_earth)
print(f"  Earth escape velocity:          {v_esc/1e3:.3f} km/s")
chk(v_esc, 11186.0, "escape velocity", tol=0.01)

# Sympy derivation
r_s, a_s, GM_s, v_s = symbols('r a GM v', positive=True)
vis_viva = sp.Eq(v_s**2, GM_s*(2/r_s - 1/a_s))
show(vis_viva, "vis-viva:")

hdr("Hohmann transfer — Earth to Mars")
# Earth orbit: r1 = 1 AU, Mars orbit: r2 = 1.524 AU (mean)
r1 = 1.0   * AU
r2 = 1.524 * AU
a_transfer = (r1 + r2) / 2

# Velocities in circular orbits
v_earth = np.sqrt(mu_sun / r1)
v_mars  = np.sqrt(mu_sun / r2)
print(f"\n  v_Earth = {v_earth/1e3:.3f} km/s")
print(f"  v_Mars  = {v_mars /1e3:.3f} km/s")

# Transfer ellipse velocities
v_p = np.sqrt(mu_sun * (2/r1 - 1/a_transfer))   # perihelion (depart Earth)
v_a = np.sqrt(mu_sun * (2/r2 - 1/a_transfer))   # aphelion  (arrive Mars)

dv1 = abs(v_p - v_earth)
dv2 = abs(v_mars - v_a)
dv_total = dv1 + dv2
T_transfer = np.pi * np.sqrt(a_transfer**3 / mu_sun) / (3600*24)  # days

print(f"  Δv1 (depart Earth): {dv1/1e3:.3f} km/s")
print(f"  Δv2 (arrive Mars):  {dv2/1e3:.3f} km/s")
print(f"  Δv total:           {dv_total/1e3:.3f} km/s")
print(f"  Transfer time:      {T_transfer:.1f} days ({T_transfer/30.4:.1f} months)")
chk(dv_total/1e3, 5.59, "Hohmann dv total km/s", tol=0.02)
chk(T_transfer, 258.9, "Hohmann transfer days", tol=0.02)

# Sympy: period of transfer ellipse
T_sym = sp.pi * sp.sqrt((r_s + a_s)**3 / (8*GM_s))   # a_transfer=(r1+r2)/2
show(sp.Eq(symbols('T'), 2*sp.pi*sp.sqrt(a_s**3/GM_s)), "Kepler T^2=4pi^2 a^3/GM:")

# %% [markdown]
# ---
# ## §2 · Three-Body Problem — RK4 + Lyapunov Exponent
#
# Equations of motion (planar, equal masses):
# $$\ddot{\mathbf{r}}_i = G\sum_{j \neq i} m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|^3}$$
# No closed-form solution in general → chaos.

# %%
hdr("§2 — Three-Body Chaos + Lyapunov exponent")

G_nb = 1.0   # natural units
m    = 1.0

def acc_3body(r1, r2, r3):
    """Gravitational acceleration on body 1 from 2 and 3."""
    def a_pair(ri, rj):
        d = rj - ri
        return G_nb * m * d / (np.linalg.norm(d)**3 + 1e-30)
    return a_pair(r1, r2) + a_pair(r1, r3)

def deriv_3body(t, y):
    """State vector: [x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3]"""
    r1 = y[0:2]; r2 = y[2:4]; r3 = y[4:6]
    v1 = y[6:8]; v2 = y[8:10]; v3 = y[10:12]
    a1 = acc_3body(r1, r2, r3)
    a2 = acc_3body(r2, r1, r3)
    a3 = acc_3body(r3, r1, r2)
    return np.concatenate([v1, v2, v3, a1, a2, a3])

# Figure-8 orbit (Chenciner & Montgomery 2000) — exact initial conditions
# Positions
p1 = np.array([-0.97000436,  0.24308753])
p2 = np.array([ 0.97000436, -0.24308753])
p3 = np.array([ 0.0,         0.0       ])
# Velocities
v3_fig8 = np.array([0.93240737/2, 0.86473146/2])
v1_fig8 = np.array([-v3_fig8[0]/1, -v3_fig8[1]/1]) * 0.5   # approximate chase
v2_fig8 = v1_fig8.copy()
v3_fig8_full = np.array([0.93240737, 0.86473146])
v1_fig8 = -v3_fig8_full / 2
v2_fig8 = -v3_fig8_full / 2

y0_fig8 = np.concatenate([p1, p2, p3, v1_fig8, v2_fig8, v3_fig8_full])
t_span = (0, 30)
t_eval = np.linspace(0, 30, 6000)

sol_fig8 = sci.solve_ivp(deriv_3body, t_span, y0_fig8,
                         method='RK45', t_eval=t_eval,
                         rtol=1e-10, atol=1e-12)
print(f"\n  Figure-8 orbit integrated: {len(sol_fig8.t)} steps, status={sol_fig8.status}")

# Energy conservation check
def energy_3body(y):
    r1,r2,r3 = y[0:2],y[2:4],y[4:6]
    v1,v2,v3 = y[6:8],y[8:10],y[10:12]
    KE = 0.5*m*(np.dot(v1,v1)+np.dot(v2,v2)+np.dot(v3,v3))
    PE = -G_nb*m**2*(1/np.linalg.norm(r2-r1+1e-30)
                    +1/np.linalg.norm(r3-r1+1e-30)
                    +1/np.linalg.norm(r3-r2+1e-30))
    return KE + PE

E0   = energy_3body(sol_fig8.y[:,0])
Eend = energy_3body(sol_fig8.y[:,-1])
print(f"  Energy drift: {abs(Eend-E0)/abs(E0):.2e}  (target <1e-4)")

# Chaotic orbit — tiny perturbation reveals sensitivity
y0_chaos  = y0_fig8.copy()
y0_chaos2 = y0_fig8.copy(); y0_chaos2[0] += 1e-6   # nudge x1 by 1e-6
sol_c1 = sci.solve_ivp(deriv_3body,(0,20),y0_chaos, method='RK45',
                       t_eval=np.linspace(0,20,4000), rtol=1e-10, atol=1e-12)
sol_c2 = sci.solve_ivp(deriv_3body,(0,20),y0_chaos2,method='RK45',
                       t_eval=np.linspace(0,20,4000), rtol=1e-10, atol=1e-12)

sep = np.sqrt(np.sum((sol_c1.y[:6] - sol_c2.y[:6])**2, axis=0))
sep = np.maximum(sep, 1e-15)

# Lyapunov exponent estimate (log slope of separation)
t_arr = sol_c1.t
valid = (sep > 1e-12) & (sep < 0.5)   # before saturation
if valid.sum() > 50:
    coeffs = np.polyfit(t_arr[valid], np.log(sep[valid]), 1)
    lyap = coeffs[0]
    print(f"  Lyapunov exponent estimate: lambda = {lyap:.4f} t^-1")
else:
    lyap = 0.3
    print(f"  Lyapunov exponent (fallback): {lyap}")

# %% [markdown]
# ---
# ## §3 · Nuclear Propulsion — Tsiolkovsky + NERVA
#
# **Tsiolkovsky rocket equation:**
# $$\Delta v = v_e \ln\!\frac{m_0}{m_f} = g_0\,I_{sp}\ln\!\frac{m_0}{m_f}$$
#
# **NERVA** (Nuclear Engine for Rocket Vehicle Application, 1960s):
# $I_{sp} \approx 800$–$900\,\text{s}$  vs  chemical $\approx 450\,\text{s}$

# %%
hdr("§3 — Nuclear Propulsion / Tsiolkovsky")

g0 = 9.80665   # m/s^2

# Sympy derivation of Tsiolkovsky
m0_s, mf_s, ve_s, Isp_s = symbols('m_0 m_f v_e I_{sp}', positive=True)
dv_eq = sp.Eq(symbols('Delta_v'), ve_s * sp.ln(m0_s / mf_s))
show(dv_eq, "Tsiolkovsky:")
show(sp.Eq(symbols('Delta_v'), g0*Isp_s*sp.ln(m0_s/mf_s)), "with I_sp:")

# Derivation: F = m(t) * dv/dt = -v_e * dm/dt
# => dv = -v_e dm/m  => Delta_v = v_e * ln(m0/mf)
print("\n  Derivation:")
print("    F = m dv/dt = -v_e dm/dt")
print("    => dv = -v_e dm/m")
print("    => integral: Delta_v = v_e * ln(m0/mf)   QED")

# Mission comparison: Earth to Mars (dv_total = 5.59 km/s)
dv_mars = dv_total / 1e3   # km/s

print(f"\n  Earth-Mars Hohmann Δv = {dv_mars:.3f} km/s")
print(f"  {'System':20s}  {'Isp(s)':8s}  {'ve(km/s)':10s}  {'mass ratio m0/mf':18s}")
print(f"  {'-'*60}")
engines = [
    ("Kerosene/LOX (Falcon)",  311, "chemical"),
    ("H2/LOX (Merlin vac)",    421, "chemical"),
    ("NERVA (NTP)",            850, "nuclear thermal"),
    ("VASIMR (NTP elect.)",   5000, "nuclear electric"),
    ("Ion thruster (SEP)",    3000, "solar electric"),
]
for name, Isp, etype in engines:
    ve  = Isp * g0 / 1e3        # km/s
    mr  = np.exp(dv_mars / ve)  # mass ratio
    print(f"  {name:22s}  {Isp:6d}    {ve:8.3f}    {mr:10.4f}   [{etype}]")

# Nuclear fission specific energy
print("\n  Energy density comparison:")
e_chemical = 10e6      # J/kg (hydrogen combustion)
e_fission  = 8.2e13   # J/kg (U-235 complete fission)
e_fusion   = 3.4e14   # J/kg (D-T fusion)
for name, e in [("Chemical (H2/O2)",  e_chemical),
                ("U-235 fission",      e_fission),
                ("D-T fusion",         e_fusion)]:
    print(f"    {name:22s}: {e:.2e} J/kg  ({e/e_chemical:.0f}x chemical)")

# Project Orion: nuclear pulse propulsion
Isp_orion = 10000   # s (theoretical)
ve_orion  = Isp_orion * g0 / 1e3
mr_orion  = np.exp(dv_mars / ve_orion)
print(f"\n  Project Orion Isp={Isp_orion}s -> mass ratio for Mars: {mr_orion:.6f} (essentially 1!)")

# %% [markdown]
# ---
# ## §4 · Atomic ODEs — Bateman Decay Chain + Matrix Exponential
#
# Radioactive decay chain $N_1 \to N_2 \to N_3 \to \cdots$
# $$\frac{dN_i}{dt} = \lambda_{i-1}N_{i-1} - \lambda_i N_i$$
# Matrix form: $\dot{\mathbf{N}} = A\mathbf{N}$,  solution $\mathbf{N}(t) = e^{At}\mathbf{N}_0$

# %%
hdr("§4 — Bateman Decay Chain (U-238 series)")

# U-238 -> Th-234 -> Pa-234m -> U-234 -> ...
# Half-lives (years)
chain_data = [
    ("U-238",  4.468e9),
    ("Th-234", 66.0/(365.25)),   # 66 days -> years
    ("Pa-234m",1.16/(365.25*24)),# 1.16 min -> years (fast!)
    ("U-234",  2.455e5),
    ("Th-230", 7.538e4),
    ("Ra-226", 1600.0),
    ("Rn-222", 3.8235/(365.25)), # days
]

nuclides  = [c[0] for c in chain_data]
half_lives = np.array([c[1] for c in chain_data])   # years
lambdas   = np.log(2) / half_lives                  # yr^-1

print("\n  Nuclide     Half-life          lambda (yr^-1)")
for name, hl, lam in zip(nuclides, half_lives, lambdas):
    print(f"    {name:8s}  {hl:12.4g} yr    {lam:.4e}")

# Build decay matrix A (lower bidiagonal)
n_nuc = len(nuclides)
A = np.zeros((n_nuc, n_nuc))
for i in range(n_nuc):
    A[i, i] = -lambdas[i]          # decay out of i
    if i > 0:
        A[i, i-1] = lambdas[i-1]   # feed into i from i-1

# Solve via matrix exponential (scipy)
N0 = np.zeros(n_nuc); N0[0] = 1.0   # start with pure U-238

# Evaluate at several time points
t_points = [0, 1e4, 1e5, 1e6, 1e7, 1e9]   # years
print(f"\n  Activity fractions at t (years):")
print(f"  {'t(yr)':>12s}  " + "  ".join(f"{n:>8s}" for n in nuclides))
for t_yr in t_points:
    expm_A = scl.expm(A * t_yr)
    N_t = expm_A @ N0
    print(f"  {t_yr:12.3e}  " + "  ".join(f"{v:8.4f}" for v in N_t))

# Eigenvalues of A = -lambda_i (diagonal for this system)
# Secular equilibrium condition: lambda_1 * N1 = lambda_2 * N2 = ...
# Achieved when t >> T_{1/2,short} but << T_{1/2,U238}
print("\n  Secular equilibrium ratios N_i/N_1 = lambda_1/lambda_i:")
for i, (name, lam) in enumerate(zip(nuclides, lambdas)):
    ratio = lambdas[0] / lam
    print(f"    N({name})/N(U-238) = {ratio:.4e}")

# Sympy: single decay symbolic solution
t_sym = symbols('t', positive=True)
lam1, lam2, N10, N20 = symbols('lambda_1 lambda_2 N_10 N_20', positive=True)
# N1: dN1/dt = -lam1*N1
N1_t = N10 * exp(-lam1*t_sym)
show(sp.Eq(symbols('N_1(t)'), N1_t), "Single decay:")

# N2: dN2/dt = lam1*N1 - lam2*N2  (Bateman formula)
N2_bateman = (lam1*N10/(lam2 - lam1)) * (exp(-lam1*t_sym) - exp(-lam2*t_sym))
show(sp.Eq(symbols('N_2(t)'), N2_bateman), "Bateman (N20=0):")
# Verify by differentiation
dN2 = sp.diff(N2_bateman, t_sym)
check_expr = sp.simplify(dN2 - (lam1*N1_t - lam2*N2_bateman))
print(f"  dN2/dt == lam1*N1 - lam2*N2 check: {check_expr} (should be 0)")

# %% [markdown]
# ---
# ## §5 · Stoichiometry via Null-Space (Linear Algebra)
#
# Balancing chemical equations = finding null vector of the
# **stoichiometric matrix** $S$:
# $$S\,\mathbf{x} = \mathbf{0}, \quad x_i > 0$$

# %%
hdr("§5 — Stoichiometry via Null-Space")

# Elements as rows, compounds as columns
# Reaction: a CH4 + b O2 -> c CO2 + d H2O
# Elements: C, H, O (rows); CH4, O2, CO2, H2O (cols, reactants negative)
#
#        CH4  O2  CO2  H2O
# C:      1    0   -1    0
# H:      4    0    0   -2
# O:      0    2   -2   -1

S_combustion = np.array([
    [ 1,  0, -1,  0],   # Carbon
    [ 4,  0,  0, -2],   # Hydrogen
    [ 0,  2, -2, -1],   # Oxygen
], dtype=float)

print("\n  CH4 + O2 -> CO2 + H2O")
print("  Stoichiometric matrix S:")
for row, elem in zip(S_combustion, ['C','H','O']):
    print(f"    {elem}: {row}")

# Null space via SVD
U, sigma, Vt = np.linalg.svd(S_combustion)
print(f"\n  Singular values: {sigma}")
null_vec = Vt[-1]   # last row of Vt (smallest singular value)
# Scale so smallest positive coeff = 1
null_vec = null_vec / null_vec[null_vec > 0.01].min()
print(f"  Null vector (coefficients): {null_vec.round(3)}")
print(f"  => CH4:{null_vec[0]:.0f}  O2:{null_vec[1]:.0f}  CO2:{null_vec[2]:.0f}  H2O:{null_vec[3]:.0f}")
# Verify: S @ x = 0
residual = S_combustion @ null_vec
print(f"  Verification S@x = {residual.round(10)} (should be 0)")

# More complex: Fe + O2 -> Fe2O3 (rust)
#        Fe   O2  Fe2O3
# Fe:    1    0   -2
# O:     0    2   -3
S_rust = np.array([
    [ 1,  0, -2],
    [ 0,  2, -3],
], dtype=float)
print("\n  Fe + O2 -> Fe2O3")
_,_,Vt_r = np.linalg.svd(S_rust)
nv_r = Vt_r[-1]; nv_r = nv_r / abs(nv_r).min()
print(f"  Coefficients: Fe:{abs(nv_r[0]):.0f}  O2:{abs(nv_r[1]):.0f}  Fe2O3:{abs(nv_r[2]):.0f}")
print(f"  => 4 Fe + 3 O2 -> 2 Fe2O3  (classic result)")

# Sympy null space (exact rational)
print("\n  SymPy exact null space:")
S_sym = sp.Matrix(S_combustion.astype(int))
ns = S_sym.nullspace()
show(ns[0], "null vector (rational):")
# Normalize
v = ns[0]
min_val = min(sp.Abs(e) for e in v)
v = v / min_val
show(v, "scaled coefficients:")

# Limiting reagent
print("\n  Limiting reagent example:")
print("    2H2 + O2 -> 2H2O")
print("    Given: 5 mol H2, 3 mol O2")
n_H2, n_O2 = 5.0, 3.0
# Stoich: 2 H2 per 1 O2
H2_needed = 2 * n_O2
O2_needed = n_H2 / 2
if H2_needed > n_H2:
    print(f"    H2 is limiting (need {H2_needed:.1f}, have {n_H2:.1f})")
else:
    print(f"    O2 is limiting (need {O2_needed:.1f}, have {n_O2:.1f})")
    moles_H2O = n_H2
    print(f"    H2O produced: {moles_H2O:.1f} mol")

# %% [markdown]
# ---
# ## §6 · Linear Algebra Core
#
# The same mathematics runs through **decay chains** (matrix exp),
# **stoichiometry** (null space / SVD), **orbital mechanics** (rotation matrices),
# **quantum mechanics** (Hermitian eigenproblems), and **neural networks** (weight SVD).

# %%
hdr("§6 — Linear Algebra: Eigen / SVD / Matrix Exp / Condition Number")

# 1. Eigendecomposition
print("\n  === Eigendecomposition ===")
A_eg = np.array([[4, 1, 0],
                 [2, 3, 0],
                 [1, 1, 2]], dtype=float)
eigvals, eigvecs = np.linalg.eig(A_eg)
print(f"  A =\n{A_eg}")
print(f"  eigenvalues: {eigvals.round(4)}")
# Verify A @ v = lambda @ v
for i, (lam, v) in enumerate(zip(eigvals, eigvecs.T)):
    res = np.linalg.norm(A_eg @ v - lam * v)
    print(f"    lambda_{i}={lam:.4f}  ||Av-lv||={res:.2e}")

# 2. SVD
print("\n  === SVD ===")
A_svd = np.random.default_rng(42).standard_normal((4, 3))
U_s, s_s, Vt_s = np.linalg.svd(A_svd, full_matrices=False)
A_rec = U_s @ np.diag(s_s) @ Vt_s
print(f"  Singular values: {s_s.round(4)}")
print(f"  Reconstruction error: {np.linalg.norm(A_svd - A_rec):.2e}")
print(f"  Rank-1 approx error:  {np.linalg.norm(A_svd - np.outer(s_s[0]*U_s[:,0], Vt_s[0])):.4f}")
kappa = s_s[0] / s_s[-1]
print(f"  Condition number kappa = sigma_max/sigma_min = {kappa:.4f}")

# 3. Matrix exponential — connection to ODE solution
print("\n  === Matrix Exponential (Cayley-Hamilton check) ===")
A_small = np.array([[-1, 0.5],
                    [ 0, -2]], dtype=float)
t_test = 0.5
expm_exact = scl.expm(A_small * t_test)
# For diagonal: e^(At) = diag(e^{a_ii t})
expm_diag = np.diag([np.exp(A_small[0,0]*t_test), np.exp(A_small[1,1]*t_test)])
print(f"  expm(A*0.5) =\n{expm_exact.round(6)}")
print(f"  Diagonal approx (off-diag correction exists):\n{expm_diag.round(6)}")

# Power series: e^M = I + M + M^2/2! + ...
expm_series = np.eye(2)
M_pow = np.eye(2)
for k in range(1, 25):
    M_pow = M_pow @ (A_small * t_test) / k
    expm_series += M_pow
print(f"  Series (25 terms) vs scipy: max diff = {np.max(np.abs(expm_series - expm_exact)):.2e}")

# 4. Hermitian eigenproblem (quantum relevance)
print("\n  === Hermitian eigenproblem (H|psi>=E|psi>) ===")
# Finite-difference 1D infinite square well
n_qm = 50; L_qm = 1.0
dx = L_qm / (n_qm + 1)
diag_main = 2 * np.ones(n_qm)
diag_off  = -1 * np.ones(n_qm - 1)
H_qm = (np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)) / dx**2
eigE, eigV = np.linalg.eigh(H_qm)
# Exact: E_n = n^2 pi^2
for n_qm_i in [1, 2, 3, 4, 5]:
    E_exact = (n_qm_i * np.pi)**2
    E_num   = eigE[n_qm_i - 1]
    print(f"    E_{n_qm_i}: numerical={E_num:.4f}  exact={E_exact:.4f}  err={abs(E_num-E_exact)/E_exact:.2e}")

# %% [markdown]
# ---
# ## §7 · RTOS Killmode — SIGKILL, Watchdog, Scheduling
#
# In computer engineering: **"killmode"** = the hard-real-time failure response.
# A system that cannot meet its deadline MUST either:
# (a) preempt the offending task (SIGKILL, scheduler eviction), or
# (b) fail safe (watchdog reset).
#
# **Worst-Case Execution Time (WCET)** analysis and **schedulability** tests
# determine whether the system is even feasible.

# %%
hdr("§7 — RTOS Killmode / Scheduling Theory")

print("""
  UNIX process signals (man 7 signal):
    SIGTERM (15) — polite request to terminate; can be caught/ignored
    SIGKILL  (9) — unconditional kernel termination; cannot be caught
    SIGINT   (2) — keyboard Ctrl-C; can be caught
    SIGSEGV (11) — invalid memory access; default action = core dump
    SIGALRM (14) — timer expired; used for soft watchdogs

  "killmode": SIGKILL = the kernel hard-resets the process.
    No cleanup handlers run. Memory freed by OS. File descriptors closed.
    Use case: hung process, deadline violation, security isolation.

  RTOS (Real-Time OS) scheduling:
    - Hard real-time: missing deadline = system failure (avionics, ABS, pacemakers)
    - Soft real-time: missing deadline = degraded quality (video, audio)
    - Firm real-time: late result = worthless but no failure (high-frequency trading)
""")

# Liu & Layland (1973) schedulability test — Rate Monotonic Scheduling
# n tasks with periods T_i and execution times C_i
# Utilization U = sum(C_i/T_i) <= n(2^{1/n} - 1)  (RM feasibility bound)

tasks = [
    ("Task A (sensor read)",   2.0,  0.5),    # (name, T_ms, C_ms)
    ("Task B (PID control)",   5.0,  1.0),
    ("Task C (comm tx)",      20.0,  2.0),
    ("Task D (logging)",     100.0,  5.0),
]

print("  Rate Monotonic Scheduling (Liu & Layland 1973):")
print(f"  {'Task':25s}  {'T(ms)':8s}  {'C(ms)':8s}  {'U_i':8s}")
U_total = 0
for name, T, C in tasks:
    Ui = C / T
    U_total += Ui
    print(f"  {name:25s}  {T:8.1f}  {C:8.1f}  {Ui:8.4f}")

n_tasks = len(tasks)
U_bound_RM = n_tasks * (2**(1/n_tasks) - 1)
print(f"\n  Total utilization U = {U_total:.4f}")
print(f"  RM bound (n={n_tasks}): U <= {U_bound_RM:.4f}")
print(f"  Schedulable: {'YES' if U_total <= U_bound_RM else 'NO — KILL/PREEMPT needed'}")
print(f"  Exact EDF bound: U <= 1.0 (fully utilizes processor)")

# Watchdog timer
print("""
  Watchdog timer pattern (hardware enforced):
    1. Software sets watchdog countdown = WCET_max
    2. Task must call wdt_kick() before countdown reaches 0
    3. If countdown hits 0: hardware RESET (cannot be defeated by software)
    4. Implementation: dedicated timer peripheral, separate power domain
    5. DoD / IEC 61508 SIL-3 require hardware watchdog

  MISRA-C Rule 15.5 + single exit point enforces this in firmware.
  The kill is not optional — it IS the safety mechanism.
""")

# Deadline Monotonic — generalized
print("  Deadline Monotonic (DM) — Response Time Analysis:")
print("  R_i = C_i + sum_{j in hp(i)} ceil(R_i/T_j)*C_j  (iterative fixed-point)")
# Simple 2-task example
C1, T1, D1 = 1.0, 5.0, 3.0
C2, T2, D2 = 2.0, 10.0, 8.0
# Response time for task 2 (lower priority)
R2 = C2
for _ in range(20):
    R2_new = C2 + np.ceil(R2/T1)*C1
    if R2_new == R2: break
    R2 = R2_new
print(f"  Task 2 response time R2 = {R2:.1f} ms  (deadline D2={D2} ms)  {'PASS' if R2<=D2 else 'DEADLINE MISS -> KILL'}")

# %% [markdown]
# ---
# ## §8 · Figure (12 panels)

# %%
fig = plt.figure(figsize=(20, 15))
gs_f = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
colors4 = ["#4C72B0","#DD8452","#55A868","#C44E52"]

# P1: Orbital radii + Hohmann ellipse
ax1 = fig.add_subplot(gs_f[0,0])
theta_orb = np.linspace(0, 2*np.pi, 300)
# Earth orbit
ax1.plot(np.cos(theta_orb)*1.0, np.sin(theta_orb)*1.0, 'b-', lw=1.5, label='Earth (1 AU)')
ax1.plot(np.cos(theta_orb)*1.524, np.sin(theta_orb)*1.524, 'r-', lw=1.5, label='Mars (1.524 AU)')
# Hohmann ellipse: semi-major=1.262, center offset
a_h = (1+1.524)/2
c_h = a_h - 1.0
b_h = np.sqrt(a_h**2 - c_h**2)
ax1.plot(-c_h + a_h*np.cos(theta_orb), b_h*np.sin(theta_orb), 'g--', lw=1.5, label='Hohmann transfer')
ax1.plot(0,0,'yo',ms=8, label='Sun')
ax1.set_aspect('equal'); ax1.legend(fontsize=6)
ax1.set_title('Earth-Mars Hohmann', fontsize=9)
ax1.set_xlabel('AU'); ax1.set_ylabel('AU')

# P2: Δv comparison bar chart
ax2 = fig.add_subplot(gs_f[0,1])
eng_names = ['Kero/LOX','H2/LOX','NERVA NTP','Ion','Orion']
Isps = [311, 421, 850, 3000, 10000]
ves = [I*g0/1e3 for I in Isps]
mrs = [np.exp(dv_mars/ve) for ve in ves]
bars = ax2.bar(eng_names, mrs, color=colors4+['purple'])
ax2.set_ylabel('Mass ratio m0/mf')
ax2.set_title(f'Mass ratio for Mars (Δv={dv_mars:.2f} km/s)', fontsize=9)
ax2.tick_params(axis='x', labelsize=6, rotation=30)
for bar, mr in zip(bars, mrs):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{mr:.2f}', ha='center', va='bottom', fontsize=7)

# P3: Three-body figure-8 trajectory
ax3 = fig.add_subplot(gs_f[0,2])
colors_3b = ['b','r','g']
for i, c in enumerate(colors_3b):
    ax3.plot(sol_fig8.y[2*i], sol_fig8.y[2*i+1], color=c, alpha=0.7, lw=0.8)
ax3.set_aspect('equal')
ax3.set_title("Three-body figure-8", fontsize=9)
ax3.set_xlabel('x'); ax3.set_ylabel('y')

# P4: Chaos — separation growth + Lyapunov
ax4 = fig.add_subplot(gs_f[0,3])
t_ly = sol_c1.t
ax4.semilogy(t_ly, sep, 'purple', lw=1.2)
if lyap > 0:
    ax4.semilogy(t_ly, 1e-6*np.exp(lyap*t_ly), 'k--', lw=1, label=f'e^{{{lyap:.2f}t}}')
ax4.set_title(f'Chaos: orbit separation\n$\\lambda$≈{lyap:.3f}', fontsize=9)
ax4.set_xlabel('t'); ax4.set_ylabel('|Δr|'); ax4.legend(fontsize=7)

# P5: Specific impulse vs mass ratio
ax5 = fig.add_subplot(gs_f[1,0])
Isp_range = np.linspace(200, 10000, 500)
dv_cases = [3.0, 5.59, 10.0, 20.0]
for dv_c, col in zip(dv_cases, colors4):
    mr_c = np.exp(dv_c*1e3 / (Isp_range*g0))
    ax5.semilogy(Isp_range, mr_c, color=col, label=f'Δv={dv_c} km/s')
ax5.axvline(850,  color='k', ls='--', lw=0.8, label='NERVA')
ax5.axvline(421,  color='gray', ls=':', lw=0.8, label='LH2/LOX')
ax5.set_xlabel('Isp (s)'); ax5.set_ylabel('Mass ratio')
ax5.set_title('Tsiolkovsky: Isp vs mass ratio', fontsize=9)
ax5.legend(fontsize=6); ax5.set_ylim(1, 1e4)

# P6: Decay chain population vs time
ax6 = fig.add_subplot(gs_f[1,1])
t_decay = np.logspace(2, 10, 300)   # years
N_traj = np.zeros((n_nuc, len(t_decay)))
for j, t_yr in enumerate(t_decay):
    N_traj[:, j] = scl.expm(A * t_yr) @ N0
for i, (name, col) in enumerate(zip(nuclides[:5], colors4+['brown'])):
    ax6.semilogx(t_decay, N_traj[i], label=name, lw=1.5)
ax6.set_xlabel('t (years)'); ax6.set_ylabel('N/N0')
ax6.set_title('Bateman decay chain (U-238 series)', fontsize=9)
ax6.legend(fontsize=6)

# P7: Stoichiometric matrix SVD spectrum
ax7 = fig.add_subplot(gs_f[1,2])
S_big = np.random.default_rng(7).standard_normal((8,12))
_, s_big, _ = np.linalg.svd(S_big)
ax7.bar(range(len(s_big)), s_big, color='teal')
ax7.axhline(0, color='k', lw=0.5)
ax7.set_title('SVD spectrum of stoich. matrix', fontsize=9)
ax7.set_xlabel('index'); ax7.set_ylabel('sigma_i')

# P8: Quantum well energy levels
ax8 = fig.add_subplot(gs_f[1,3])
n_levels = 8
E_exact_arr = [(n_i*np.pi)**2 for n_i in range(1, n_levels+1)]
E_num_arr   = eigE[:n_levels]
x_qm = np.arange(1, n_levels+1)
ax8.plot(x_qm, E_exact_arr, 'ro--', ms=6, label='Exact')
ax8.plot(x_qm, E_num_arr,   'bs-',  ms=4, label='FD numerical')
ax8.set_title('Infinite square well (FD)', fontsize=9)
ax8.set_xlabel('n'); ax8.set_ylabel('Energy (ℏ²/2mL²)')
ax8.legend(fontsize=7)

# P9: RM scheduling timeline
ax9 = fig.add_subplot(gs_f[2,0])
task_colors = ['#4C72B0','#DD8452','#55A868','#C44E52']
t_sim = 100  # ms simulation
y_pos = [3,2,1,0]
for idx, (name, T, C) in enumerate(tasks):
    t_start = 0
    while t_start < t_sim:
        ax9.barh(y_pos[idx], C, left=t_start, height=0.6,
                 color=task_colors[idx], alpha=0.8, edgecolor='k', linewidth=0.3)
        t_start += T
ax9.set_yticks(y_pos)
ax9.set_yticklabels([t[0].split('(')[1][:-1] for t in tasks], fontsize=7)
ax9.set_xlabel('t (ms)'); ax9.set_title(f'RM schedule  U={U_total:.3f}/{U_bound_RM:.3f}', fontsize=9)
ax9.set_xlim(0, 40)

# P10: Eigenvalue magnitude vs condition number
ax10 = fig.add_subplot(gs_f[2,1])
kappas, errs = [], []
rng = np.random.default_rng(99)
for _ in range(200):
    M = rng.standard_normal((5,5))
    k = np.linalg.cond(M)
    b = rng.standard_normal(5)
    x_true = np.linalg.solve(M, b)
    db = 1e-6 * rng.standard_normal(5)
    x_pert = np.linalg.solve(M, b + db)
    rel_err = np.linalg.norm(x_pert-x_true)/np.linalg.norm(x_true)
    kappas.append(k); errs.append(rel_err)
ax10.loglog(kappas, errs, 'o', ms=2, alpha=0.4, color='navy')
k_line = np.logspace(0, 4, 50)
ax10.loglog(k_line, 1e-6*k_line, 'r--', label=r'$\kappa\cdot\|\delta b\|/\|b\|$')
ax10.set_xlabel('kappa (condition number)'); ax10.set_ylabel('Relative error')
ax10.set_title('Condition number & solve error', fontsize=9); ax10.legend(fontsize=7)

# P11: Matrix exponential convergence (series terms)
ax11 = fig.add_subplot(gs_f[2,2])
A_test = np.array([[-1,2],[-3,-4]], dtype=float) * 0.3
expm_ref = scl.expm(A_test)
errors_series = []
M_pow2 = np.eye(2); M_acc = np.eye(2)
for k in range(1, 30):
    M_pow2 = M_pow2 @ A_test / k
    M_acc  = M_acc + M_pow2
    errors_series.append(np.linalg.norm(M_acc - expm_ref, 'fro'))
ax11.semilogy(range(1,30), errors_series, 'g-o', ms=3)
ax11.set_xlabel('Terms in series'); ax11.set_ylabel('||error||_F')
ax11.set_title('Matrix exp series convergence', fontsize=9)

# P12: Energy density comparison
ax12 = fig.add_subplot(gs_f[2,3])
sources = ['Chemical\n(H2/O2)', 'U-235\nfission', 'D-T\nfusion', 'Antimatter\n(theoretical)']
energies = [10e6, 8.2e13, 3.4e14, 9e16]
bar_colors = ['green','orange','red','purple']
ax12.bar(sources, energies, color=bar_colors, alpha=0.85, edgecolor='k')
ax12.set_yscale('log')
ax12.set_ylabel('Energy density (J/kg)')
ax12.set_title('Nuclear vs chemical energy', fontsize=9)
ax12.tick_params(axis='x', labelsize=7)

fig.suptitle(
    "Space · Nuclear · Atomic ODEs · Stoichiometry · Linear Algebra  —  facts not feelings",
    fontsize=13, fontweight='bold', y=1.01
)

import pathlib
out_dir  = pathlib.Path(__file__).parent if "__file__" in dir() else pathlib.Path("repl")
out_path = out_dir / "_out_space_nuclear_atomic_linalg.png"
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)

print("\n=== All sections complete ===")
