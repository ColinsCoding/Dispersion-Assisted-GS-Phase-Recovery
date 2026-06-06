# %% [markdown]
# # Linear Algebra & Differential Equations for Chemistry + Genetics
# `init_printing(use_latex="mathjax")` throughout.
#
# §1  Stoichiometric matrix — null space, rank, conservation laws
# §2  Chemical kinetics ODEs — first/second order, half-life, integrated rate laws
# §3  Coupled reaction networks — matrix ODE, eigenvalue stability (Jacobian)
# §4  Michaelis-Menten enzyme kinetics — quasi-steady-state derivation
# §5  Equilibrium and Le Chatelier — K_eq from thermodynamics, ICE table as linear system
# §6  Mendelian genetics — dominant/recessive, Punnett square as matrix, Hardy-Weinberg ODE
# §7  Figure (12 panels)

# %% [markdown]
# ## Setup

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
from sympy import (symbols, Matrix, Rational, latex, simplify, exp, sqrt,
                   integrate, diff, solve, ln, pi, oo, Function, dsolve,
                   Derivative, Eq, limit)
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
    try:
        err = abs(float(val) - float(ref)) / (abs(float(ref)) + 1e-30)
        ok  = err < tol
    except:
        ok = bool(sp.simplify(val - ref) == 0)
        err = 0
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}  got={float(val):.6g}  ref={float(ref):.6g}")
    return ok

print("=== LinAlg + ODEs for Chemistry + Genetics ===")

# %% [markdown]
# ---
# ## §1 · Stoichiometric Matrix — The Linear Algebra of Chemistry
#
# Every balanced reaction is a **linear algebra problem**.
# Reactions → columns, species → rows.
# $$S \cdot \mathbf{v} = \frac{d\mathbf{c}}{dt}$$
# - **Null space** of $S^T$ → conservation laws (mass, charge, atoms)
# - **Null space** of $S$   → flux modes at steady state ($S\mathbf{v}=0$)
# - **Rank** of $S$         → degrees of freedom in the system

# %%
hdr("§1 — Stoichiometric Matrix")

print("""
  Reaction network (glycolysis fragment):
    R1: Glucose + ATP -> G6P + ADP            (hexokinase)
    R2: G6P -> F6P                            (isomerase)
    R3: F6P + ATP -> F1,6BP + ADP            (PFK)
    R4: F1,6BP -> DHAP + G3P                  (aldolase)

  Species: Glucose, G6P, F6P, F1,6BP, ATP, ADP, DHAP, G3P
  Rows = species (8), Cols = reactions (4)
""")

# Stoichiometric matrix (rows=species, cols=reactions)
# Order: Glc, G6P, F6P, F1,6BP, ATP, ADP, DHAP, G3P
S_met = np.array([
#  R1   R2   R3   R4
  [-1,   0,   0,   0],   # Glucose
  [ 1,  -1,   0,   0],   # G6P
  [ 0,   1,  -1,   0],   # F6P
  [ 0,   0,   1,  -1],   # F1,6BP
  [-1,   0,  -1,   0],   # ATP
  [ 1,   0,   1,   0],   # ADP
  [ 0,   0,   0,   1],   # DHAP
  [ 0,   0,   0,   1],   # G3P
], dtype=float)

species = ["Glucose","G6P","F6P","F1,6BP","ATP","ADP","DHAP","G3P"]
reactions = ["R1(hexokinase)","R2(isomerase)","R3(PFK)","R4(aldolase)"]

print("  Stoichiometric matrix S (rows=species, cols=reactions):")
print(f"  {'':>12s}" + "".join(f"{r:>16s}" for r in reactions))
for i, sp_name in enumerate(species):
    print(f"  {sp_name:>12s}" + "".join(f"{int(S_met[i,j]):>16d}" for j in range(4)))

# Rank
rank = np.linalg.matrix_rank(S_met)
print(f"\n  Rank(S) = {rank}  (out of min(8,4)=4)")
print(f"  Degrees of freedom = reactions - rank = 4 - {rank} = {4-rank}")

# SVD analysis
U_s, sv, Vt_s = np.linalg.svd(S_met, full_matrices=False)
print(f"  Singular values: {sv.round(4)}")

# Null space of S^T -> conservation laws
# S^T has shape (4,8); null space gives conserved quantities
S_sym = Matrix(S_met.astype(int))
print("\n  Null space of S^T (conservation laws):")
ns_left = S_sym.T.nullspace()
for i, ns_v in enumerate(ns_left):
    conserved = " + ".join(f"{int(ns_v[j])}*{species[j]}"
                           for j in range(8) if ns_v[j] != 0)
    print(f"    CL{i+1}: {conserved} = const")

# At steady state: S*v = 0 — find flux modes
print("\n  Steady-state flux modes (null space of S):")
ns_right = S_sym.nullspace()
if ns_right:
    for i, ns_v in enumerate(ns_right):
        print(f"    FM{i+1}: v = {[str(ns_v[j]) for j in range(4)]}")
else:
    print("    Full column rank — unique steady-state flux direction")

# Simple atom-balance example: CH4 + 2O2 -> CO2 + 2H2O
print("\n  Atom balance as null space (combustion):")
print("  S*x=0 where S is atom matrix, x are stoich coefficients")
# Already done in previous notebook — show concept
S_comb = Matrix([
    [ 1,  0, -1,  0],   # C
    [ 4,  0,  0, -2],   # H
    [ 0,  2, -2, -1],   # O
])
ns_comb = S_comb.nullspace()
show(ns_comb[0].T, "null vector [CH4, O2, CO2, H2O]:")
v_norm = ns_comb[0] / min(abs(e) for e in ns_comb[0] if e != 0)
show(v_norm.T, "balanced coefficients:")
print(f"  => CH4 + {v_norm[1]}*O2 -> {v_norm[2]}*CO2 + {v_norm[3]}*H2O")

# %% [markdown]
# ---
# ## §2 · Chemical Kinetics ODEs — Integrated Rate Laws
#
# | Order | Rate law | Integrated | Half-life |
# |---|---|---|---|
# | 0 | $r = k$ | $[A] = [A]_0 - kt$ | $t_{1/2}=[A]_0/2k$ |
# | 1 | $r = k[A]$ | $[A] = [A]_0 e^{-kt}$ | $t_{1/2}=\ln2/k$ |
# | 2 | $r = k[A]^2$ | $1/[A] = 1/[A]_0 + kt$ | $t_{1/2}=1/(k[A]_0)$ |

# %%
hdr("§2 — Integrated Rate Laws (SymPy derivation)")

t, k_s, A0, B0 = symbols('t k A_0 B_0', positive=True)
A_func = Function('A')

# 0th order: dA/dt = -k
ode0 = Eq(A_func(t).diff(t), -k_s)
sol0 = dsolve(ode0, A_func(t), ics={A_func(0): A0})
show(sol0, "0th order [A](t):")
t_half0 = solve(sol0.rhs - A0/2, t)[0]
show(t_half0, "t_{1/2} (0th order):")
chk(t_half0.subs([(A0,1),(k_s,0.5)]), 1.0, "0th order half-life")

# 1st order: dA/dt = -k*A
ode1 = Eq(A_func(t).diff(t), -k_s*A_func(t))
sol1 = dsolve(ode1, A_func(t), ics={A_func(0): A0})
show(sol1, "1st order [A](t):")
t_half1 = solve(sol1.rhs - A0/2, t)[0]
show(t_half1, "t_{1/2} (1st order) =")
chk(float(t_half1.evalf(subs={k_s:sp.log(2)})), 1.0, "1st order half-life = ln2/k")

# 2nd order: dA/dt = -k*A^2
ode2 = Eq(A_func(t).diff(t), -k_s*A_func(t)**2)
sol2 = dsolve(ode2, A_func(t), ics={A_func(0): A0})
show(sol2, "2nd order [A](t):")
t_half2 = solve(sol2.rhs - A0/2, t)[0]
show(t_half2, "t_{1/2} (2nd order):")
chk(float(t_half2.subs([(A0,2),(k_s,1)])), 0.5, "2nd order half-life = 1/(k*A0)")

# Numerical comparison
print("\n  Numerical: [A](t) for each order (k=0.1, A0=1.0)")
t_arr = np.linspace(0, 30, 300)
k_v = 0.1; A0_v = 1.0
A_0th = np.maximum(A0_v - k_v*t_arr, 0)
A_1st = A0_v * np.exp(-k_v * t_arr)
A_2nd = A0_v / (1 + k_v * A0_v * t_arr)

# Half-life checks
hl0 = A0_v / (2*k_v)
hl1 = np.log(2) / k_v
hl2 = 1 / (k_v * A0_v)
print(f"  0th: t_1/2 = A0/(2k) = {hl0:.2f} s")
print(f"  1st: t_1/2 = ln2/k   = {hl1:.3f} s")
print(f"  2nd: t_1/2 = 1/(kA0) = {hl2:.2f} s")

chk(A_1st[np.argmin(abs(t_arr-hl1))], 0.5, "1st order: [A](t_1/2)=0.5", tol=0.01)
chk(A_2nd[np.argmin(abs(t_arr-hl2))], 0.5, "2nd order: [A](t_1/2)=0.5", tol=0.01)

# Arrhenius k(T)
print("\n  Arrhenius k(T) = A*exp(-Ea/RT):")
Ea = 50e3   # J/mol typical activation energy
R_gas = 8.314
A_freq = 1e13   # pre-exponential (s^-1)
T_vals = np.array([273, 298, 310, 373, 500, 1000])
for T_v in T_vals:
    k_v2 = A_freq * np.exp(-Ea/(R_gas*T_v))
    print(f"    T={T_v}K: k = {k_v2:.3e} s^-1")

# %% [markdown]
# ---
# ## §3 · Coupled Reaction Networks — Matrix ODE + Eigenvalue Stability
#
# Linear reaction system:
# $$\frac{d\mathbf{c}}{dt} = K\,\mathbf{c}, \qquad
#   \mathbf{c}(t) = e^{Kt}\mathbf{c}_0 = \sum_i \alpha_i \mathbf{v}_i e^{\lambda_i t}$$
# Stability: $\text{Re}(\lambda_i) < 0$ for all $i$ → concentrations decay to equilibrium.

# %%
hdr("§3 — Coupled Reactions: Matrix ODE + Eigenvalues")

# Consecutive reactions: A -> B -> C  (k1, k2)
# d[A]/dt = -k1*[A]
# d[B]/dt = +k1*[A] - k2*[B]
# d[C]/dt = +k2*[B]

k1, k2 = 0.3, 0.1  # s^-1

K_mat = np.array([
    [-k1,  0,   0],
    [ k1, -k2,  0],
    [  0,  k2,  0],
])
print(f"\n  A -> B -> C  (k1={k1}, k2={k2})")
print(f"  Rate matrix K:\n{K_mat}")

eigvals, eigvecs = np.linalg.eig(K_mat)
print(f"\n  Eigenvalues: {eigvals.round(4)}")
print("  All Re(lambda) <= 0: stable (concentrations bounded)")

# Analytical solution via matrix exponential
c0 = np.array([1.0, 0.0, 0.0])
t_rxn = np.linspace(0, 40, 500)
c_traj = np.array([scl.expm(K_mat * ti) @ c0 for ti in t_rxn])

# Check mass conservation: sum [A]+[B]+[C] = 1 always
mass = c_traj.sum(axis=1)
print(f"  Mass conservation check: max deviation = {abs(mass-1).max():.2e}  [{'PASS' if abs(mass-1).max()<1e-10 else 'FAIL'}]")

# Sympy closed-form: Bateman-like
t_s = symbols('t', positive=True)
k1_s, k2_s = symbols('k_1 k_2', positive=True)
A_t = sp.exp(-k1_s*t_s)
B_t = (k1_s/(k2_s-k1_s)) * (sp.exp(-k1_s*t_s) - sp.exp(-k2_s*t_s))
C_t = 1 - A_t - B_t
show(sp.Eq(sp.Symbol('A(t)'), A_t), "A(t) =")
show(sp.Eq(sp.Symbol('B(t)'), B_t), "B(t) =")
check_mass = sp.simplify(A_t + B_t + C_t - 1)
print(f"  A(t)+B(t)+C(t)-1 = {check_mass}  [PASS if 0]")

# Jacobian stability for nonlinear: autocatalytic A + B -> 2B (Lotka)
print("\n  Autocatalytic / Lotka-Volterra oscillator:")
print("  dA/dt = a*A - b*A*B")
print("  dB/dt = c*A*B - d*B")
a_lv, b_lv, c_lv, d_lv = 1.0, 0.1, 0.1, 0.5
# Fixed point (non-trivial): A* = d/c, B* = a/b
A_star = d_lv/c_lv; B_star = a_lv/b_lv
print(f"  Fixed point: A*={A_star}, B*={B_star}")
# Jacobian at fixed point
J_lv = np.array([
    [a_lv - b_lv*B_star,  -b_lv*A_star],
    [ c_lv*B_star,         c_lv*A_star - d_lv],
])
eig_lv = np.linalg.eigvals(J_lv)
print(f"  Jacobian eigenvalues: {eig_lv}")
print(f"  Re(lambda) = {eig_lv.real}  => {'center (neutral stability)' if abs(eig_lv.real).max()<1e-10 else 'stable/unstable'}")

# Simulate Lotka-Volterra
def lotka(t, y):
    A_v, B_v = y
    return [a_lv*A_v - b_lv*A_v*B_v,
            c_lv*A_v*B_v - d_lv*B_v]

sol_lv = sci.solve_ivp(lotka, [0, 60], [10.0, 5.0],
                       t_eval=np.linspace(0,60,2000), rtol=1e-9)
print(f"  LV simulation: {len(sol_lv.t)} steps, status={sol_lv.status}")

# %% [markdown]
# ---
# ## §4 · Michaelis-Menten Enzyme Kinetics
#
# Mechanism: $E + S \underset{k_{-1}}{\stackrel{k_1}{\rightleftharpoons}} ES \xrightarrow{k_{cat}} E + P$
#
# **Quasi-steady-state approximation** (Briggs-Haldane):
# $$\frac{d[ES]}{dt} \approx 0 \implies [ES] = \frac{[E]_T[S]}{K_M + [S]}$$
# $$v = k_{cat}[ES] = \frac{v_{max}[S]}{K_M + [S]}, \quad K_M = \frac{k_{-1}+k_{cat}}{k_1}$$

# %%
hdr("§4 — Michaelis-Menten Kinetics (QSSA derivation)")

S_s, E_s, ES_s, P_s = symbols('[S] [E] [ES] [P]', positive=True)
k1_mm, km1_mm, kcat_mm = symbols('k_1 k_{-1} k_{cat}', positive=True)
ET_s, KM_s, vmax_s = symbols('[E]_T K_M v_{max}', positive=True)

# Full ODE system
dS  = -k1_mm*E_s*S_s + km1_mm*ES_s
dES = k1_mm*E_s*S_s - km1_mm*ES_s - kcat_mm*ES_s
dE  = -k1_mm*E_s*S_s + km1_mm*ES_s + kcat_mm*ES_s
dP  = kcat_mm*ES_s

show(Eq(sp.Symbol('d[S]/dt'), dS), "Full ODE:")
show(Eq(sp.Symbol('d[ES]/dt'), dES))

# QSSA: d[ES]/dt = 0, [E] = [E]_T - [ES]
# k1*(ET-ES)*S - k-1*ES - kcat*ES = 0
# k1*ET*S = ES*(k1*S + k-1 + kcat)
# => ES = ET*S / (S + (k-1+kcat)/k1) = ET*S/(S+KM)
ES_qssa = ET_s*S_s / (S_s + KM_s)
v_MM    = kcat_mm * ES_qssa
show(ES_qssa, "[ES]_{QSSA} =")
show(v_MM, "v = k_{cat}[ES] =")

# Rewrite in terms of vmax
v_MM_vmax = vmax_s*S_s / (S_s + KM_s)
show(v_MM_vmax, "Michaelis-Menten:")

# Linearization: Lineweaver-Burk (double reciprocal)
# 1/v = KM/(vmax*S) + 1/vmax
LB_lhs = sp.Symbol('1/v')
LB_rhs = KM_s/(vmax_s*S_s) + 1/vmax_s
show(Eq(LB_lhs, LB_rhs), "Lineweaver-Burk (1/v vs 1/[S]):")

# Numerical example
vmax_v = 10.0  # mM/s
KM_v   = 2.0   # mM
S_arr  = np.linspace(0.01, 20, 300)
v_arr  = vmax_v * S_arr / (KM_v + S_arr)

print(f"\n  Michaelis-Menten: vmax={vmax_v} mM/s, KM={KM_v} mM")
print(f"  {'[S] (mM)':>10s}  {'v (mM/s)':>10s}  {'v/vmax':>8s}")
for S_v, v_v in zip([0.5,1,2,5,10,20], vmax_v*np.array([0.5,1,2,5,10,20])/(KM_v+np.array([0.5,1,2,5,10,20]))):
    print(f"  {S_v:>10.1f}  {v_v:>10.4f}  {v_v/vmax_v:>8.4f}")

# At S=KM: v = vmax/2 (definition of KM)
v_at_KM = vmax_v * KM_v / (KM_v + KM_v)
chk(v_at_KM, vmax_v/2, "v([S]=KM) = vmax/2")

# Full ODE simulation (not just QSSA)
k1_v = 10.0; km1_v = 2.0; kcat_v = 5.0
KM_full = (km1_v + kcat_v)/k1_v
vmax_full = kcat_v  # per unit [E]T
ET_v = 0.01  # mM (enzyme much less than substrate)
S0_v = 1.0

def mm_full(t, y):
    S_v2, E_v2, ES_v2, P_v2 = y
    dS2  = -k1_v*E_v2*S_v2 + km1_v*ES_v2
    dE2  = -k1_v*E_v2*S_v2 + (km1_v+kcat_v)*ES_v2
    dES2 =  k1_v*E_v2*S_v2 - (km1_v+kcat_v)*ES_v2
    dP2  =  kcat_v*ES_v2
    return [dS2, dE2, dES2, dP2]

sol_mm = sci.solve_ivp(mm_full, [0, 5], [S0_v, ET_v, 0, 0],
                       t_eval=np.linspace(0,5,1000), rtol=1e-9)

# QSSA prediction at same conditions
v_qssa_pred = vmax_full * ET_v * sol_mm.y[0] / (KM_full + sol_mm.y[0])
dP_num = np.gradient(sol_mm.y[3], sol_mm.t)
print(f"\n  Full ODE vs QSSA: max v discrepancy = {np.max(abs(dP_num[10:] - v_qssa_pred[10:])):.4e} mM/s")

# %% [markdown]
# ---
# ## §5 · Chemical Equilibrium — K_eq as Linear Algebra
#
# At equilibrium: $\Delta G = 0 \implies K_{eq} = e^{-\Delta G^0/RT}$
# ICE (Initial-Change-Equilibrium) table = solving a linear (or quadratic) system.

# %%
hdr("§5 — Equilibrium: K_eq + ICE table as linear system")

# SymPy: Gibbs free energy -> K_eq
DG0_s, R_s, T_eq_s = symbols('Delta_G^0 R T', real=True)
K_eq_s = sp.exp(-DG0_s/(R_s*T_eq_s))
show(Eq(sp.Symbol('K_{eq}'), K_eq_s), "K_eq from thermodynamics:")

# Example: N2 + 3H2 <-> 2NH3 (Haber process)
print("\n  Haber process: N2 + 3H2 <-> 2NH3")
print("  DG0 = -32.9 kJ/mol at 298K")
DG0_haber = -32.9e3   # J/mol
R_gas = 8.314; T_298 = 298
K_eq_haber = np.exp(-DG0_haber/(R_gas*T_298))
print(f"  K_eq(298K) = exp({-DG0_haber/1000:.1f}kJ / {R_gas*T_298/1000:.3f}kJ) = {K_eq_haber:.2e}")

# ICE table as system of equations
# Start: [N2]=1M, [H2]=3M, [NH3]=0
# Change: -x, -3x, +2x
# Equil: 1-x, 3-3x, 2x
# K = (2x)^2 / ((1-x)(3-3x)^3)
x = symbols('x', positive=True)
Keq_expr = (2*x)**2 / ((1-x)*(3-3*x)**3)
print("\n  ICE: [N2]=1-x, [H2]=3-3x, [NH3]=2x")
show(Eq(sp.Symbol('K_{eq}'), Keq_expr), "K_eq =")

# For large K_eq, reaction goes nearly to completion.
# Solve numerically:
from scipy.optimize import brentq
def K_residual(xv):
    return (2*xv)**2 / ((1-xv+1e-15)*(3-3*xv+1e-15)**3) - K_eq_haber
try:
    x_eq = brentq(K_residual, 1e-6, 0.9999)
    print(f"  x_eq = {x_eq:.6f}")
    print(f"  [N2]  = {1-x_eq:.6f} M")
    print(f"  [H2]  = {3-3*x_eq:.6f} M")
    print(f"  [NH3] = {2*x_eq:.6f} M")
    print(f"  K check = {(2*x_eq)**2/((1-x_eq)*(3-3*x_eq)**3):.4e}  (target {K_eq_haber:.4e})")
except Exception as e:
    print(f"  (brentq: {e} — K very large, x->1)")
    x_eq = 0.999

# Le Chatelier: K_eq vs temperature (van't Hoff)
print("\n  van't Hoff: d(ln K)/dT = DH0/RT^2")
DH0_haber = -92e3   # J/mol (exothermic)
T_range = np.linspace(300, 800, 200)
lnK_range = -DG0_haber/(R_gas*298) + DH0_haber/R_gas * (1/298 - 1/T_range)
K_range = np.exp(lnK_range)
print(f"  {'T(K)':>8s}  {'K_eq':>12s}  {'ln K':>8s}")
for T_v, K_v in zip([298,400,500,600,700,800], np.interp([298,400,500,600,700,800], T_range, K_range)):
    print(f"  {T_v:>8.0f}  {K_v:>12.4e}  {np.log(K_v):>8.3f}")
print("  (Lower T -> larger K for exothermic; higher T -> faster kinetics)")

# %% [markdown]
# ---
# ## §6 · Genetics — Dominant/Recessive + Hardy-Weinberg ODE
#
# **Mendelian inheritance**: allele combinations as **outer product** (matrix).
# **Hardy-Weinberg**: allele frequencies $p + q = 1$ are conserved under
# random mating — eigenvalue of the mating operator equals 1.

# %%
hdr("§6 — Mendelian Genetics + Hardy-Weinberg ODE")

# Punnett square as outer product
# Monohybrid Aa x Aa
print("\n  Monohybrid cross: Aa x Aa")
parent1 = np.array(['A','a'])
parent2 = np.array(['A','a'])

# Build Punnett square
print(f"\n       {parent1[0]}    {parent1[1]}")
genotypes = {}
for j, a2 in enumerate(parent2):
    row = f"  {a2}  "
    for a1 in parent1:
        combo = ''.join(sorted([a1,a2], key=lambda x: (x.lower(), x=='A')))
        row += f"  {a1+a2}  "
        genotypes[a1+a2] = genotypes.get(a1+a2, 0) + 1
    print(row)

print(f"\n  Genotype frequencies:")
total = sum(genotypes.values())
phenos = {}
for g, cnt in sorted(genotypes.items()):
    dom = 'A' in g   # dominant if any A
    pheno = "Dominant" if dom else "Recessive"
    phenos[pheno] = phenos.get(pheno,0) + cnt
    print(f"    {g}: {cnt}/{total} = {cnt/total:.4f}  ({pheno})")

print(f"\n  Phenotype ratio:")
for p, cnt in phenos.items():
    print(f"    {p}: {cnt}/{total} = {cnt/total:.4f}  ({cnt}:{total-cnt} ratio)")
chk(phenos["Dominant"]/total, 0.75, "dominant phenotype 3/4")
chk(phenos["Recessive"]/total, 0.25, "recessive phenotype 1/4")

# Dihybrid: AaBb x AaBb
print("\n  Dihybrid cross: AaBb x AaBb")
gametes = ['AB','Ab','aB','ab']
dihybrid = np.zeros((4,4), dtype=object)
pheno_counts = {'dom_dom':0,'dom_rec':0,'rec_dom':0,'rec_rec':0}
for i, g1 in enumerate(gametes):
    for j, g2 in enumerate(gametes):
        combo = g1+g2
        # Count A/a and B/b
        A_count = combo.count('A')
        B_count = combo.count('B')
        if A_count>0 and B_count>0: pheno_counts['dom_dom'] += 1
        elif A_count>0:             pheno_counts['dom_rec'] += 1
        elif B_count>0:             pheno_counts['rec_dom'] += 1
        else:                       pheno_counts['rec_rec'] += 1

total_di = sum(pheno_counts.values())
print(f"  Phenotype ratios (16 total):")
for ph, cnt in pheno_counts.items():
    print(f"    {ph}: {cnt}/16 = {cnt/total_di:.4f}")
chk(pheno_counts['dom_dom']/total_di, 9/16, "dihybrid 9/16 dom-dom")
chk(pheno_counts['rec_rec']/total_di, 1/16, "dihybrid 1/16 rec-rec")

# Hardy-Weinberg equilibrium
print("\n  Hardy-Weinberg Equilibrium:")
print("  p + q = 1  (p = freq(A), q = freq(a))")
print("  After random mating: freq(AA)=p^2, freq(Aa)=2pq, freq(aa)=q^2")
print("  Verify: p^2 + 2pq + q^2 = (p+q)^2 = 1")

p_sym, q_sym = symbols('p q', positive=True)
HW = (p_sym + q_sym)**2
HW_expanded = sp.expand(HW)
show(HW_expanded, "(p+q)^2 =")
print(f"  (p+q)^2 = 1 check: {sp.simplify(HW_expanded.subs(q_sym, 1-p_sym) - 1) == 0}")

# HW as eigenvalue problem
# Mating matrix M where M_ij = probability of genotype i producing allele j
# Allele frequency vector [p, q] is an eigenvector with eigenvalue 1
print("\n  HW as eigenvalue: mating preserves allele frequencies")
# AA: produces A with prob 1
# Aa: produces A with prob 0.5, a with prob 0.5
# aa: produces a with prob 1
# If genotype freqs are [p^2, 2pq, q^2]:
# p_next = 1*p^2 + 0.5*(2pq) + 0*q^2 = p^2+pq = p(p+q) = p  (fixed!)
p_v = 0.6
q_v = 1 - p_v
freqs = np.array([p_v**2, 2*p_v*q_v, q_v**2])
p_next = 1*freqs[0] + 0.5*freqs[1] + 0*freqs[2]
chk(p_next, p_v, "HW: p preserved after one generation")

# Simulation: approach to HW from non-equilibrium
print("\n  Convergence to HW equilibrium (starting from non-HW freqs):")
f_AA, f_Aa, f_aa = 0.5, 0.1, 0.4   # not at HW
print(f"  Start: f(AA)={f_AA} f(Aa)={f_Aa} f(aa)={f_aa}")
print(f"  {'Gen':>5s}  {'f(AA)':>8s}  {'f(Aa)':>8s}  {'f(aa)':>8s}  {'p':>6s}  {'HW dist':>10s}")
for gen in range(6):
    p_g = f_AA + 0.5*f_Aa
    q_g = 1 - p_g
    hw_dist = abs(f_AA - p_g**2) + abs(f_Aa - 2*p_g*q_g) + abs(f_aa - q_g**2)
    print(f"  {gen:>5d}  {f_AA:>8.4f}  {f_Aa:>8.4f}  {f_aa:>8.4f}  {p_g:>6.4f}  {hw_dist:>10.6f}")
    # After random mating: go to HW in ONE generation
    f_AA = p_g**2
    f_Aa = 2*p_g*q_g
    f_aa = q_g**2

# Selection: frequency change with fitness
print("\n  Selection dynamics: wAA=1.0, wAa=0.9, waa=0.5 (recessive lethal-ish)")
w_AA, w_Aa, w_aa = 1.0, 0.9, 0.5
p_sel = 0.1   # start with rare dominant
p_traj = [p_sel]
for _ in range(100):
    q_sel = 1 - p_sel
    w_bar = w_AA*p_sel**2 + 2*w_Aa*p_sel*q_sel + w_aa*q_sel**2
    p_sel_new = (w_AA*p_sel**2 + w_Aa*p_sel*q_sel) / w_bar
    p_traj.append(p_sel_new)
    p_sel = p_sel_new
print(f"  p(A) after 100 generations: {p_traj[-1]:.6f}  (started at {p_traj[0]:.2f})")

# %% [markdown]
# ---
# ## §7 · Figure

# %%
fig = plt.figure(figsize=(20, 15))
gf = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)
c4 = ["#4C72B0","#DD8452","#55A868","#C44E52"]

# P1: SVD spectrum of stoichiometric matrix
ax1 = fig.add_subplot(gf[0,0])
_, sv_met, _ = np.linalg.svd(S_met, full_matrices=False)
ax1.bar(range(len(sv_met)), sv_met, color='steelblue', edgecolor='k')
ax1.set_title("Stoich matrix SVD spectrum", fontsize=9)
ax1.set_xlabel("index"); ax1.set_ylabel("singular value")

# P2: Integrated rate laws
ax2 = fig.add_subplot(gf[0,1])
ax2.plot(t_arr, A_0th, label="0th order", lw=2, color=c4[0])
ax2.plot(t_arr, A_1st, label="1st order", lw=2, color=c4[1])
ax2.plot(t_arr, A_2nd, label="2nd order", lw=2, color=c4[2])
ax2.axhline(0.5, color='k', ls=':', lw=1, label='[A]_0/2')
ax2.set_title("Integrated rate laws (k=0.1,A0=1)", fontsize=9)
ax2.set_xlabel("t (s)"); ax2.set_ylabel("[A]"); ax2.legend(fontsize=7)

# P3: Consecutive A->B->C
ax3 = fig.add_subplot(gf[0,2])
for i, (label, col) in enumerate(zip(['[A]','[B]','[C]'], c4)):
    ax3.plot(t_rxn, c_traj[:,i], label=label, lw=2, color=col)
ax3.set_title(f"A->B->C (k1={k1}, k2={k2})", fontsize=9)
ax3.set_xlabel("t (s)"); ax3.set_ylabel("concentration"); ax3.legend(fontsize=8)

# P4: Lotka-Volterra
ax4 = fig.add_subplot(gf[0,3])
ax4.plot(sol_lv.t, sol_lv.y[0], label='Prey A', color=c4[0], lw=2)
ax4.plot(sol_lv.t, sol_lv.y[1], label='Pred B', color=c4[1], lw=2)
ax4.axhline(A_star, ls='--', color=c4[0], lw=0.8)
ax4.axhline(B_star, ls='--', color=c4[1], lw=0.8)
ax4.set_title("Lotka-Volterra oscillator", fontsize=9)
ax4.set_xlabel("t"); ax4.legend(fontsize=7)

# P5: Michaelis-Menten curve
ax5 = fig.add_subplot(gf[1,0])
ax5.plot(S_arr, v_arr, 'b-', lw=2, label='MM velocity')
ax5.axhline(vmax_v, color='r', ls='--', lw=1, label=f'vmax={vmax_v}')
ax5.axvline(KM_v,  color='g', ls='--', lw=1, label=f'KM={KM_v}')
ax5.axhline(vmax_v/2, color='gray', ls=':', lw=1)
ax5.set_title("Michaelis-Menten", fontsize=9)
ax5.set_xlabel("[S] (mM)"); ax5.set_ylabel("v (mM/s)"); ax5.legend(fontsize=7)

# P6: Lineweaver-Burk
ax6 = fig.add_subplot(gf[1,1])
S_LB = np.linspace(0.5, 20, 100)
v_LB = vmax_v*S_LB/(KM_v+S_LB)
ax6.plot(1/S_LB, 1/v_LB, 'purple', lw=2)
ax6.set_xlabel("1/[S]"); ax6.set_ylabel("1/v")
ax6.set_title("Lineweaver-Burk (double reciprocal)", fontsize=9)
slope_LB = KM_v/vmax_v
ax6.text(0.02, 0.85, f"slope=KM/vmax={slope_LB:.2f}\ny-int=1/vmax={1/vmax_v:.2f}",
         transform=ax6.transAxes, fontsize=8)

# P7: K_eq vs temperature (van't Hoff)
ax7 = fig.add_subplot(gf[1,2])
ax7.semilogy(T_range, K_range, 'r-', lw=2)
ax7.axhline(1, color='k', ls=':', lw=1, label='K=1 (DG=0)')
ax7.set_title("Haber: K_eq vs T (van't Hoff)", fontsize=9)
ax7.set_xlabel("T (K)"); ax7.set_ylabel("K_eq"); ax7.legend(fontsize=7)

# P8: Hardy-Weinberg genotype frequencies
ax8 = fig.add_subplot(gf[1,3])
p_range = np.linspace(0, 1, 200)
q_range = 1 - p_range
f_AA_r = p_range**2
f_Aa_r = 2*p_range*q_range
f_aa_r = q_range**2
ax8.plot(p_range, f_AA_r, label='f(AA)=p²', color=c4[0], lw=2)
ax8.plot(p_range, f_Aa_r, label='f(Aa)=2pq', color=c4[1], lw=2)
ax8.plot(p_range, f_aa_r, label='f(aa)=q²', color=c4[2], lw=2)
ax8.set_xlabel("p = freq(A)"); ax8.set_ylabel("genotype frequency")
ax8.set_title("Hardy-Weinberg genotype freqs", fontsize=9)
ax8.legend(fontsize=7)

# P9: Selection dynamics
ax9 = fig.add_subplot(gf[2,0])
ax9.plot(range(len(p_traj)), p_traj, 'b-', lw=2)
ax9.set_xlabel("Generation"); ax9.set_ylabel("p(A)")
ax9.set_title(f"Selection: wAA={w_AA}, wAa={w_Aa}, waa={w_aa}", fontsize=9)

# P10: Punnett dihybrid phenotype pie chart
ax10 = fig.add_subplot(gf[2,1])
labels_di = [f'{k}\n({v}/16)' for k,v in pheno_counts.items()]
sizes_di  = list(pheno_counts.values())
ax10.pie(sizes_di, labels=labels_di, colors=c4, autopct='%1.0f%%', startangle=90)
ax10.set_title("Dihybrid AaBb x AaBb\nphenotype ratios", fontsize=9)

# P11: Full MM ODE vs QSSA
ax11 = fig.add_subplot(gf[2,2])
ax11.plot(sol_mm.t, sol_mm.y[0], label='[S]', lw=2, color=c4[0])
ax11.plot(sol_mm.t, sol_mm.y[3], label='[P]', lw=2, color=c4[1])
ax11.plot(sol_mm.t, sol_mm.y[2]*100, label='[ES]x100', lw=2, color=c4[2])
ax11.set_title("MM full ODE: E+S<->ES->E+P", fontsize=9)
ax11.set_xlabel("t (s)"); ax11.legend(fontsize=7)

# P12: Arrhenius k(T)
ax12 = fig.add_subplot(gf[2,3])
T_arr_k = np.linspace(250, 1200, 300)
k_arr2 = A_freq * np.exp(-Ea/(R_gas*T_arr_k))
ax12.semilogy(T_arr_k, k_arr2, 'orange', lw=2)
ax12.set_xlabel("T (K)"); ax12.set_ylabel("k (s^-1)")
ax12.set_title(f"Arrhenius (Ea={Ea/1e3:.0f} kJ/mol)", fontsize=9)

fig.suptitle(
    "Linear Algebra + ODEs for Chemistry & Genetics\n"
    "Stoichiometry · Kinetics · Enzyme · Equilibrium · Mendelian · Hardy-Weinberg",
    fontsize=12, fontweight='bold', y=1.01
)

import pathlib
out_dir  = pathlib.Path(__file__).parent if "__file__" in dir() else pathlib.Path("repl")
out_path = out_dir / "_out_linalg_odes_chemistry_genetics.png"
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)

print("\n=== All §1-§7 complete ===")
