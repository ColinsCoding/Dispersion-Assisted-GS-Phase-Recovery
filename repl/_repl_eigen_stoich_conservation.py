# %% [markdown]
# # Eigenvectors · Stoichiometry · Conservation Laws
# `init_printing(use_latex="mathjax")` — all math renders as LaTeX in Jupyter.
#
# **Paranoid Android axis**: every "paranoid" system (unstable, oscillating,
# drifting) has eigenvalues in the RIGHT half-plane or outside the unit circle.
# Every conservation law is a LEFT null vector of the stoichiometric matrix.
# Same linear algebra, different physical skin.
#
# **Structure:**
# §1  init_printing — how it works, what it does
# §2  Eigenvectors from scratch — what they ARE geometrically
# §3  Eigenvalues → stability → paranoid vs calm systems
# §4  Stoichiometry as linear algebra — balancing via null space
# §5  Conservation laws — atom balance = left null vector
# §6  Markov chains — eigenvector = steady-state distribution
# §7  Quantum mechanics teaser — Schrödinger = eigenvalue problem

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import (Matrix, symbols, sqrt, Rational, latex, simplify,
                   eye, zeros, diag, exp, I, pi, cos, sin,
                   GramSchmidt, det, trace, factor, Eq, Symbol)
from sympy import init_printing

# ── THE MAGIC LINE ─────────────────────────────────────────────────────────────
init_printing(use_latex='mathjax')
# In Jupyter: every sp.Matrix, sp.Eq, sp.Expr displayed with display()
# renders via MathJax → real LaTeX, not ASCII art.
# In terminal: falls back gracefully to unicode pretty-print.
# ──────────────────────────────────────────────────────────────────────────────

try:
    from IPython.display import display as _D
    IN_JUPYTER = True
except ImportError:
    IN_JUPYTER = False

def show(expr, label=None):
    if label: print(f"\n  {label}")
    if IN_JUPYTER:
        _D(expr)
    else:
        print("  " + sp.pretty(expr, use_unicode=True))

def hdr(s):
    bar = '─' * 64
    print(f'\n{bar}\n  {s}\n{bar}')

def chk(val, ref, label, tol=1e-8, absolute=False):
    try: v, r = float(val), float(ref)
    except: print(f'  [FAIL]  {label}  (not float)'); return
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    print(f"  [{'PASS' if err<tol else 'FAIL'}]  {label}  got={v:.8g}  ref={r:.8g}")

print("=== Eigenvectors · Stoichiometry · Conservation Laws ===")
print("    init_printing(use_latex='mathjax') active")

# %% [markdown]
# ---
# ## §1 · How `init_printing` Works
#
# Three rendering layers:
#
# | Context | How math displays | Trigger |
# |---------|-------------------|---------|
# | Jupyter + `display(expr)` | MathJax **LaTeX** | `init_printing(use_latex='mathjax')` |
# | Terminal | unicode art `√2` | `use_unicode=True` (default) |
# | Plain ASCII | `sqrt(2)` | `use_unicode=False` |
#
# **Rule**: always call `display(expr)`, never `print(expr)` or `sp.pretty(expr)`.
# `init_printing` hooks into `display()`, not `print()`.

# %%
hdr("§1 — init_printing demo")

x, lam = symbols('x lambda')

expr1 = sp.sqrt(x**2 + 1) / (x - sp.Rational(1,3))
expr2 = sp.Matrix([[1, sp.sqrt(2)], [0, sp.Rational(3,4)]])
expr3 = sp.Eq(sp.Symbol('lambda'), (sp.sqrt(5) - 1) / 2)

print("  These render as LaTeX in Jupyter, unicode in terminal:\n")
show(expr1, "A fraction with sqrt:")
show(expr2, "A 2×2 matrix:")
show(expr3, "Golden ratio eigenvalue:")

# %% [markdown]
# ---
# ## §2 · Eigenvectors — What They ARE
#
# **Geometric definition**: v is an eigenvector of A if A just STRETCHES v,
# never rotates it:
#
#   A v = λ v
#
# λ (eigenvalue) = the stretch factor.
# - λ > 1  → v grows (unstable mode)
# - 0 < λ < 1 → v shrinks (stable mode)
# - λ < 0  → v flips and stretches (oscillatory decay)
# - |λ| = 1 → v conserved in norm (rotation, Markov steady-state)
#
# **How to find λ**: det(A - λI) = 0 (characteristic equation)
# **How to find v**: null space of (A - λI)

# %%
hdr("§2 — Eigenvectors: geometric + algebraic")

# 2×2 example: A = [[3,1],[0,2]]
A = Matrix([[3, 1], [0, 2]])
show(A, "A =")

# Characteristic polynomial
lam_sym = symbols('lambda')
char_poly = det(A - lam_sym * eye(2))
show(sp.Eq(Symbol('det(A-λI)'), char_poly), "Characteristic polynomial:")
eigenvals_sym = sp.solve(char_poly, lam_sym)
print(f"  Eigenvalues: {eigenvals_sym}")

# SymPy eigenvects: returns [(eigenvalue, multiplicity, [eigenvectors])]
evects = A.eigenvects()
for lv, mult, vecs in evects:
    show(sp.Eq(Symbol('λ'), lv), f"  λ={lv}, multiplicity={mult}")
    for v in vecs:
        show(v.T, "  v =")
    # Verify: A*v = λ*v
    v0 = vecs[0]
    residual = A * v0 - lv * v0
    chk(float(residual.norm()), 0, f"Av=λv residual for λ={lv}", tol=1e-10, absolute=True)

# Symmetric matrix → orthogonal eigenvectors
print("\n  Symmetric matrix → real orthogonal eigenvectors:")
S = Matrix([[4, 2], [2, 1]])
show(S, "S =")
S_evects = S.eigenvects()
v1 = S_evects[0][2][0].normalized()
v2 = S_evects[1][2][0].normalized()
dot_12 = v1.dot(v2)
show(sp.Eq(Symbol('v₁·v₂'), simplify(dot_12)), "Orthogonality check:")
chk(float(simplify(dot_12)), 0, "symmetric eigvecs orthogonal", tol=1e-10, absolute=True)

# Diagonalisation: A = P D P^{-1}
P_mat, D_mat = A.diagonalize()
show(P_mat, "P (columns = eigenvectors):")
show(D_mat, "D (diagonal = eigenvalues):")
A_reconstructed = P_mat * D_mat * P_mat.inv()
diff_mat = simplify(A - A_reconstructed)
chk(float(diff_mat.norm()), 0, "A = PDP⁻¹ reconstruction", tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## §3 · Eigenvalues → Stability → Paranoid vs Calm
#
# "Paranoid Android" system: eigenvalues in the RIGHT half-plane
# (continuous) or OUTSIDE the unit circle (discrete).
# The system state grows without bound — paranoid, amplifying, unstable.
#
# "Calm" system: all eigenvalues in LHP / inside unit circle.
# Any perturbation decays back to equilibrium.
#
# The same test works for: circuits, mechanical systems, population models,
# chemical reactors, GS convergence (§8 of remote_control notebook).

# %%
hdr("§3 — Stability via eigenvalues: paranoid vs calm")

t_num = np.linspace(0, 5, 500)

systems = {
    "Calm (stable oscillator)":    np.array([[-0.5, 2.0], [-2.0, -0.5]]),
    "Paranoid (unstable saddle)":  np.array([[ 1.0, 0.5], [ 0.0, -0.2]]),
    "Marginal (pure imaginary)":   np.array([[ 0.0, 1.0], [-1.0,  0.0]]),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (label, A_num) in zip(axes, systems.items()):
    eigs = np.linalg.eigvals(A_num)
    # Simulate trajectory from x0=[1,0] via matrix exponential (Euler)
    dt = t_num[1] - t_num[0]
    x = np.array([1.0, 0.0])
    traj = [x.copy()]
    for _ in t_num[1:]:
        x = x + dt * (A_num @ x)
        if np.linalg.norm(x) > 1e4: x = x / np.linalg.norm(x) * 1e4
        traj.append(x.copy())
    traj = np.array(traj)
    ax.plot(t_num, traj[:,0], label='x₁', linewidth=2)
    ax.plot(t_num, traj[:,1], label='x₂', linewidth=2, linestyle='--')
    re_parts = [f'{e.real:.2f}' for e in eigs]
    stable = all(e.real < 0 for e in eigs)
    ax.set_title(f'{label}\nRe(λ)={re_parts}  {"✓calm" if stable else "✗paranoid"}',
                 fontsize=9)
    ax.set_xlabel('t')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('repl/_fig_eigen_stability.png', dpi=110, bbox_inches='tight')
plt.close()
print("  Saved: repl/_fig_eigen_stability.png")

# Numerical checks
for label, A_num in systems.items():
    eigs = np.linalg.eigvals(A_num)
    max_re = max(e.real for e in eigs)
    if "Calm" in label:
        chk(max_re < 0, 1, f"{label}: max Re(λ)<0", tol=1e-9, absolute=True)
    elif "Paranoid" in label:
        chk(max_re > 0, 1, f"{label}: max Re(λ)>0 (unstable)", tol=1e-9, absolute=True)
    else:
        chk(abs(max_re) < 1e-10, 1, f"{label}: Re(λ)≈0 (marginal)", tol=1e-8, absolute=True)

# %% [markdown]
# ---
# ## §4 · Stoichiometry as Linear Algebra
#
# Balancing a chemical equation = finding the NULL SPACE of the
# stoichiometric matrix S.
#
# Example: combustion of methane
#
#   a CH₄  +  b O₂  →  c CO₂  +  d H₂O
#
# Each column = one molecule; each row = one element.
# Balance: S · [a, b, c, d]ᵀ = 0  (atom conservation)
#
# The null vector gives the integer coefficients.

# %%
hdr("§4 — Stoichiometry via null space")

print("""
  Reaction: a CH₄ + b O₂ → c CO₂ + d H₂O

  Atom       CH₄   O₂   CO₂   H₂O
  Carbon      1     0    -1     0
  Hydrogen    4     0     0    -2
  Oxygen      0     2    -2    -1
""")

# Stoichiometric matrix: reactants positive, products negative
S_stoich = Matrix([
    [ 1,  0, -1,  0],   # Carbon
    [ 4,  0,  0, -2],   # Hydrogen
    [ 0,  2, -2, -1],   # Oxygen
])
show(S_stoich, "S (atom × molecule):")

# Null space = coefficients that balance the equation
null_vecs = S_stoich.nullspace()
show(null_vecs[0].T, "Null vector (proportional to balanced coefficients):")

# Scale to integers
null_raw = null_vecs[0]
# Multiply by LCM of denominators
from sympy import lcm
denoms = [sp.denom(c) for c in null_raw]
scale = 1
for d in denoms:
    scale = sp.lcm(scale, d)
coeffs = (null_raw * scale)
print(f"\n  Balanced: {coeffs[0]}CH₄ + {coeffs[1]}O₂ → {coeffs[2]}CO₂ + {coeffs[3]}H₂O")

# Verify: S · coeffs = 0
residual = S_stoich * coeffs
chk(float(residual.norm()), 0, "Atom balance residual = 0", tol=1e-10, absolute=True)

# Second example: glucose combustion C₆H₁₂O₆ + O₂ → CO₂ + H₂O
print("\n  Glucose combustion: a C₆H₁₂O₆ + b O₂ → c CO₂ + d H₂O")
S_glucose = Matrix([
    [ 6,  0, -1,  0],   # Carbon
    [12,  0,  0, -2],   # Hydrogen
    [ 6,  2, -2, -1],   # Oxygen
])
null_g = S_glucose.nullspace()[0]
denoms_g = [sp.denom(c) for c in null_g]
scale_g = 1
for d in denoms_g:
    scale_g = sp.lcm(scale_g, d)
cg = null_g * scale_g
print(f"  Balanced: {cg[0]}C₆H₁₂O₆ + {cg[1]}O₂ → {cg[2]}CO₂ + {cg[3]}H₂O")
chk(float((S_glucose * cg).norm()), 0, "Glucose atom balance = 0", tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## §5 · Conservation Laws = Left Null Vectors
#
# A LEFT null vector w of S means:  wᵀ S = 0ᵀ
#
# Physically: w is a conserved quantity — a linear combination of
# species concentrations that NEVER CHANGES during reaction.
#
# In stoichiometry: atom counts are left null vectors.
# In mechanics: energy, momentum, charge.
# In a Markov chain: the stationary distribution is the LEFT eigenvector
# for eigenvalue 1.
#
# Same math, three physics skins.

# %%
hdr("§5 — Conservation laws = left null vectors")

# For the methane combustion matrix S_stoich above:
# Left null space = rows of S_stoich itself span left null space of S·v=0?
# No — LEFT null space = null space of S^T
print("  Left null space of S (= conservation laws beyond atom balance):")
S_T = S_stoich.T
left_null = S_T.nullspace()
if left_null:
    for v in left_null:
        show(v.T, "  Left null vector w (wᵀS=0):")
else:
    print("  Left null space is trivial (rank S = n_atoms, fully determined)")
    print("  → atom balance uniquely determines all conservation laws")

# Generic 2-species chemical system to show the principle
print("\n  Generic A ⇌ B system:")
# dA/dt = -k1*A + k2*B,   dB/dt = k1*A - k2*B
k1, k2 = 2.0, 1.0
# Stoichiometric matrix for A→B (column 1) and B→A (column 2)
# Reaction 1: -1 A, +1 B   Reaction 2: +1 A, -1 B
S_AB = Matrix([[-1, 1], [1, -1]])
show(S_AB, "S for A⇌B:")
left_null_AB = S_AB.T.nullspace()
show(left_null_AB[0].T, "Left null vector w (conservation):")
print("  → wᵀ·[A,B] = A + B = constant  (total concentration conserved)")

# Verify numerically
w_cons = np.array([1.0, 1.0])
A_ode = np.array([[-k1, k2], [k1, -k2]])
eigs_ode = np.linalg.eigvals(A_ode)
print(f"  Eigenvalues of rate matrix: {eigs_ode}")
chk(min(abs(eigs_ode)), 0, "Rate matrix has λ=0 (conserved mode)", tol=1e-10, absolute=True)
# Eigenvector for λ=0 is the conserved direction [1,1]
# Conservation check via LEFT null vector: w^T A = 0 → w=[1,1] means A+B const
w_left = np.array([1.0, 1.0])
left_resid = w_left @ A_ode
chk(np.max(np.abs(left_resid)), 0, "left null [1,1]: A+B conserved",
    tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## §6 · Markov Chains — Steady State = Dominant Eigenvector
#
# A Markov transition matrix T has columns summing to 1.
# The steady-state distribution π satisfies: T π = π
# → π is the eigenvector for eigenvalue λ = 1.
#
# This IS the same eigenvector equation.
# Population genetics, PageRank, queueing theory —
# all "find the λ=1 eigenvector of a stochastic matrix."

# %%
hdr("§6 — Markov chain: steady state = λ=1 eigenvector")

# 3-state weather model: Sunny, Cloudy, Rainy
T_markov = Matrix([
    [Rational(7,10), Rational(2,10), Rational(1,10)],   # Sunny row
    [Rational(2,10), Rational(6,10), Rational(2,10)],   # Cloudy row
    [Rational(1,10), Rational(2,10), Rational(7,10)],   # Rainy row
])
show(T_markov, "Transition matrix T (columns sum to 1):")

# Column sums = 1
col_sums = [sum(T_markov.col(j)) for j in range(3)]
print(f"  Column sums: {col_sums}")
chk(float(col_sums[0]), 1, "col 0 sums to 1", tol=1e-10, absolute=True)

# Find stationary distribution: T π = π  →  (T-I)π = 0
evects_m = T_markov.eigenvects()
pi_vec = None
for ev, mult, vecs in evects_m:
    if abs(float(ev) - 1.0) < 1e-9:
        pi_vec = vecs[0]
        break

# Normalize: sum = 1
pi_vec_norm = pi_vec / sum(pi_vec)
show(pi_vec_norm.T, "Steady-state distribution π:")

# Verify T π = π
residual_pi = T_markov * pi_vec_norm - pi_vec_norm
chk(float(residual_pi.norm()), 0, "Tπ = π (stationary)", tol=1e-8, absolute=True)

# Also verify by power iteration
T_num = np.array(T_markov.tolist(), dtype=float)
pi_power = np.array([1/3, 1/3, 1/3])
for _ in range(200):
    pi_power = T_num @ pi_power
pi_sympy = np.array([float(c) for c in pi_vec_norm])
chk(np.max(np.abs(pi_power - pi_sympy)), 0,
    "Power iteration matches eigenvector", tol=1e-8, absolute=True)
print(f"  π = Sunny:{pi_sympy[0]:.3f}  Cloudy:{pi_sympy[1]:.3f}  Rainy:{pi_sympy[2]:.3f}")

# %% [markdown]
# ---
# ## §7 · Quantum Mechanics Teaser — Schrödinger = Eigenvalue Problem
#
# Time-independent Schrödinger equation:
#
#   H |ψ⟩ = E |ψ⟩
#
# H = Hamiltonian operator (Hermitian matrix in finite basis)
# |ψ⟩ = eigenstate (eigenvector)
# E = energy level (eigenvalue — must be real, H is Hermitian)
#
# Particle in a box (finite difference):
# H_ij = -ℏ²/(2m) * (ψ_{i+1} - 2ψ_i + ψ_{i-1})/Δx²  (kinetic energy)
# Eigenvectors = standing wave modes  sin(nπx/L)
# Eigenvalues  = energy levels  E_n = n²π²ℏ²/(2mL²)

# %%
hdr("§7 — Schrödinger as eigenvalue problem: particle in a box")

N_box = 100          # grid points
L_box = 1.0          # box length [Å normalized]
hbar_m = 1.0         # ℏ²/(2m) = 1 in natural units

dx = L_box / (N_box + 1)
x_grid = np.linspace(dx, L_box - dx, N_box)

# Kinetic energy matrix (second-derivative finite difference)
diag_main = np.full(N_box,  2.0 * hbar_m / dx**2)
diag_off  = np.full(N_box-1, -1.0 * hbar_m / dx**2)
H_box = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)

# Solve eigenvalue problem
E_numeric, psi_numeric = np.linalg.eigh(H_box)

# Analytic energy levels: E_n = n² π² / (2L²)  (ℏ=2m=1 units → ℏ²/2m=1)
E_analytic = [(n**2 * np.pi**2 * hbar_m) / (L_box**2) for n in range(1, 6)]

print("  Energy levels (natural units, ℏ²/2m = 1):")
print(f"  {'n':>3}  {'E_numeric':>12}  {'E_analytic':>12}  {'error':>10}")
for n in range(1, 6):
    err = abs(E_numeric[n-1] - E_analytic[n-1]) / E_analytic[n-1]
    print(f"  {n:>3}  {E_numeric[n-1]:>12.4f}  {E_analytic[n-1]:>12.4f}  {err:>10.2e}")
    chk(E_numeric[n-1], E_analytic[n-1], f"E_{n} matches analytic", tol=0.005)

# Eigenvectors = standing waves
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for i, ax in enumerate(axes):
    n = i + 1
    psi = psi_numeric[:, i]
    psi_norm = psi / np.sqrt(np.trapezoid(psi**2, x_grid))
    psi_analytic = np.sqrt(2/L_box) * np.sin(n * np.pi * x_grid / L_box)
    # Align sign
    if np.dot(psi_norm, psi_analytic) < 0:
        psi_norm = -psi_norm
    ax.plot(x_grid, psi_norm, 'b-', linewidth=2, label=f'numeric n={n}')
    ax.plot(x_grid, psi_analytic, 'r--', linewidth=1.5, label='analytic', alpha=0.7)
    ax.set_title(f'ψ_{n}(x),  E={E_numeric[i]:.2f}', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
plt.suptitle('Particle-in-a-box eigenstates (Schrödinger = eigenvalue problem)',
             fontsize=10)
plt.tight_layout()
plt.savefig('repl/_fig_schrodinger_box.png', dpi=110, bbox_inches='tight')
plt.close()
print("  Saved: repl/_fig_schrodinger_box.png")

# Hermitian guarantee: all eigenvalues real
imag_parts = np.imag(E_numeric[:5])
chk(np.max(np.abs(imag_parts)), 0,
    "Hermitian H → all eigenvalues real", tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## Summary — One Table to Rule Them All
#
# | Physical system | Matrix | Eigenvectors | Eigenvalues |
# |----------------|--------|--------------|-------------|
# | Paranoid Android (unstable ODE) | Rate matrix A | modes of motion | Re(λ)>0 = grows |
# | Stoichiometry | S (atom×molecule) | balanced equation | null space: λ=0 |
# | Conservation law | Sᵀ | conserved quantity | left null: λ=0 |
# | Markov chain | T (stochastic) | steady-state π | dominant λ=1 |
# | Quantum mechanics | H (Hamiltonian) | wavefunctions ψ_n | energy levels E_n |
# | PCA / ML | Covariance Σ | principal components | variance explained |
# | D-GS phase recovery | H(ω) dispersion | phase modes | diversity = spread |
#
# **The thread**: eigenvector = the direction a linear system "wants to go".
# Eigenvalue = how fast/slow, growing/decaying, real/oscillatory.

# %%
hdr("Done — all checks")
print("  §1 init_printing  §2 eigenvectors  §3 stability")
print("  §4 stoichiometry  §5 conservation  §6 Markov  §7 Schrödinger")
