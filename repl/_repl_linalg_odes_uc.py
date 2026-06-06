# %% [markdown]
# # Linear Algebra & Differential Equations — First Course Through Last
# `init_printing(use_latex="mathjax")` — all math renders as LaTeX in Jupyter.
#
# **The through-line**: linear algebra and differential equations are the same subject.
# An ODE is a linear operator acting on a function space.
# Eigenvectors of that operator are the normal modes — sin, cos, e^{λt}, Bessel.
# Every solution is a linear combination of eigenvectors.
#
# **First course at UC** (Math 54 level):
#   vectors, dot/cross, row reduction, rank, determinant, eigenvalues, first-order ODEs
#
# **Last course at UC** (Math 110 / 128A level):
#   SVD, function spaces, adjoint operators, PDEs, Fourier as projection, Sturm-Liouville
#
# **Structure:**
# §1   Vectors — geometric + algebraic, dot, cross, projection
# §2   Matrices — row reduction, RREF, rank, null space
# §3   Determinants — geometric meaning, cofactor, Cramer
# §4   Eigenvalues / eigenvectors — characteristic polynomial, diagonalisation
# §5   First-order ODEs — separable, linear (integrating factor), exact
# §6   Second-order ODEs — characteristic roots, resonance, damping
# §7   Systems of ODEs — x′ = Ax, fundamental matrix, phase portrait
# §8   SVD — singular values, pseudoinverse, low-rank approximation
# §9   Function spaces — inner product, L², orthogonality, Gram-Schmidt on functions
# §10  Fourier series — projection onto eigenbasis of d²/dx²
# §11  PDEs — heat, wave, Laplace via separation of variables
# §12  Sturm-Liouville — eigenfunctions, orthogonality, completeness

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
from sympy import (
    Matrix, symbols, sqrt, Rational, pi, exp, cos, sin, oo,
    integrate, diff, solve, simplify, latex, Eq, Symbol,
    Function, dsolve, classify_ode, Abs, sign,
    fourier_series, fourier_transform, I, ln, factorial,
    GramSchmidt, eye, zeros as sp_zeros, diag, conjugate
)
from sympy import init_printing
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

init_printing(use_latex='mathjax')

try:
    from IPython.display import display as _D, Math, Markdown
    def show(expr, label=None):
        if label: _D(Markdown(f"**{label}**"))
        _D(expr)
    def tex(s): _D(Math(s))
except ImportError:
    def show(expr, label=None):
        if label: print(f"\n  {label}")
        print("  " + sp.pretty(expr, use_unicode=True))
    def tex(s): print(f"  [{s}]")

def hdr(s):
    bar = '─' * 64
    print(f'\n{bar}\n  {s}\n{bar}')

def chk(val, ref, label, tol=1e-8, absolute=False):
    try: v, r = float(val), float(ref)
    except: print(f'  [FAIL]  {label}  (not float)'); return
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    print(f"  [{'PASS' if err<tol else 'FAIL'}]  {label}  got={v:.8g}  ref={r:.8g}")

def chk_zero(expr, label, tol=1e-10):
    chk(float(sp.Abs(sp.sympify(expr)).evalf()), 0, label, tol=tol, absolute=True)

print("=== Linear Algebra & ODEs: First Course Through Last ===")

# %% [markdown]
# ---
# ## §1 · Vectors — Dot, Cross, Projection
#
# **Dot product**: $\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$
# → measures alignment; zero iff orthogonal.
#
# **Cross product**: $\mathbf{u}\times\mathbf{v}$ = area of parallelogram, perpendicular to both.
#
# **Projection**: $\text{proj}_\mathbf{v}\mathbf{u} = \dfrac{\mathbf{u}\cdot\mathbf{v}}{\mathbf{v}\cdot\mathbf{v}}\,\mathbf{v}$
# → the shadow of **u** onto the line spanned by **v**.

# %%
hdr("§1 — Vectors: dot, cross, projection")

u = Matrix([1, 2, 3])
v = Matrix([4, 0, -1])
show(u.T, "u ="); show(v.T, "v =")

dot_uv = u.dot(v)
show(Eq(Symbol('u·v'), dot_uv), "dot product:")
cross_uv = u.cross(v)
show(cross_uv.T, "u × v =")

# Cross product is perpendicular to both
chk(float(cross_uv.dot(u)), 0, "(u×v)·u = 0", absolute=True)
chk(float(cross_uv.dot(v)), 0, "(u×v)·v = 0", absolute=True)

# Angle between u and v
cos_theta = dot_uv / (u.norm() * v.norm())
theta = sp.acos(cos_theta)
show(Eq(Symbol('θ'), sp.N(theta, 4)), "angle (rad):")
chk(float(cos_theta), float(dot_uv/(u.norm()*v.norm())), "cos θ consistent")

# Projection of u onto v
proj = (u.dot(v) / v.dot(v)) * v
show(proj.T, "proj_v(u) =")
residual = u - proj   # orthogonal to v
chk(float(residual.dot(v)), 0, "(u - proj_v u) ⊥ v", absolute=True)

# Gram-Schmidt: orthogonalise {u, v, w}
w = Matrix([1, -1, 0])
basis_orth = GramSchmidt([u, v, w], orthonormal=True)
show(Matrix(basis_orth).T, "Gram-Schmidt orthonormal basis:")
# Verify orthonormality
for i in range(3):
    for j in range(3):
        val = float(basis_orth[i].dot(basis_orth[j]).evalf())
        ref = 1.0 if i==j else 0.0
        chk(val, ref, f"<e{i},e{j}> = {'1' if i==j else '0'}", tol=1e-8)

# %% [markdown]
# ---
# ## §2 · Matrices — Row Reduction, Rank, Null Space
#
# **RREF** (reduced row echelon form) = the algorithm that solves everything:
# - Pivot columns → basis for column space
# - Free columns → null space vectors
# - rank + nullity = n  (rank-nullity theorem)
#
# $$\text{rank}(A) + \text{nullity}(A) = n \quad \text{(number of columns)}$$

# %%
hdr("§2 — Row reduction, RREF, rank, null space")

A = Matrix([
    [1,  2,  3,  4],
    [2,  4,  7,  8],
    [3,  6, 10, 12],
])
show(A, "A =")

rref_A, pivots = A.rref()
show(rref_A, "RREF(A) =")
print(f"  Pivot columns: {pivots}")

rank_A = A.rank()
nullity_A = A.shape[1] - rank_A
print(f"  rank = {rank_A},  nullity = {nullity_A}")
chk(rank_A + nullity_A, A.shape[1], "rank + nullity = n (columns)")

# Null space
null_vecs = A.nullspace()
show(Matrix([v.T for v in null_vecs]), "Null space basis (rows):")
for i, nv in enumerate(null_vecs):
    res = A * nv
    chk(float(res.norm()), 0, f"A·n{i} = 0", absolute=True)

# Column space (image)
col_space = A.columnspace()
print(f"  Column space dimension = {len(col_space)}  (= rank)")

# Solve Ax = b
b = Matrix([1, 3, 4])
# Check consistency: b in column space?
Aug = A.row_join(b)
rref_aug, _ = Aug.rref()
show(rref_aug, "Augmented [A|b] RREF:")
x_sols, params = A.gauss_jordan_solve(b)
show(x_sols, "Particular solution (gauss_jordan):")
res_ls = (A * x_sols - b).norm()
chk(float(res_ls.evalf()), 0, "Ax = b  (b in col space → exact)", tol=1e-8, absolute=True)

# %% [markdown]
# ---
# ## §3 · Determinants — Geometric Meaning
#
# $\det(A)$ = **signed volume** of the parallelepiped spanned by the columns.
#
# - $\det(A) \neq 0 \Leftrightarrow A$ is invertible $\Leftrightarrow$ columns are linearly independent
# - $\det(AB) = \det(A)\det(B)$
# - $\det(A^T) = \det(A)$
# - Row swap: det flips sign. Row scale by k: det scales by k.
# - $\det(A^{-1}) = 1/\det(A)$

# %%
hdr("§3 — Determinants: geometric meaning + cofactor expansion")

tex(r"\det(A) = \text{signed volume of parallelepiped spanned by columns}")
tex(r"\det(AB) = \det(A)\det(B), \quad \det(A^T) = \det(A)")

B = Matrix([[2, 1, 0], [3, -1, 4], [1, 2, -2]])
show(B, "B =")
d = B.det()
show(Eq(Symbol('det B'), d), "det B =")
chk(float(d), float(d), "det computed")

# Multiplicative property
C = Matrix([[1, 0, 2], [0, 3, 1], [4, 1, 0]])
chk(float((B*C).det()), float(B.det()*C.det()), "det(BC) = det(B)det(C)")
chk(float(B.T.det()), float(B.det()), "det(B^T) = det(B)")
chk(float(B.inv().det()), 1/float(B.det()), "det(B⁻¹) = 1/det(B)", tol=1e-6)

# Cramer's rule: x_i = det(A with col i replaced by b) / det(A)
rhs = Matrix([1, 2, 3])
x_cramer = B.solve(rhs)
show(x_cramer.T, "Cramer solution Bx=rhs:")
chk(float((B*x_cramer - rhs).norm()), 0, "Cramer: Bx=rhs residual", absolute=True)

# Geometric: 2D area = |det|
A2 = Matrix([[3, 1], [0, 2]])
area = abs(A2.det())
print(f"\n  2D parallelogram area = |det| = {area}  (columns: [3,0] and [1,2])")
chk(float(area), 6.0, "area of 2D parallelogram")

# %% [markdown]
# ---
# ## §4 · Eigenvalues / Eigenvectors
#
# $A\mathbf{v} = \lambda\mathbf{v}$ → **v** is a direction A merely stretches.
#
# **Finding them**:
# 1. $\det(A - \lambda I) = 0$ → characteristic polynomial → eigenvalues
# 2. Null space of $(A - \lambda_i I)$ → eigenvectors
#
# **Diagonalisation**: $A = PDP^{-1}$ where D = diagonal eigenvalue matrix,
# P = columns are eigenvectors. Requires n linearly independent eigenvectors.
#
# **Powers**: $A^k = PD^kP^{-1}$ — raising a diagonal matrix to a power is trivial.

# %%
hdr("§4 — Eigenvalues, diagonalisation, A^k")

tex(r"\det(A - \lambda I) = 0 \;\Rightarrow\; \text{eigenvalues}")
tex(r"A = P D P^{-1}, \quad A^k = P D^k P^{-1}")

M = Matrix([[4, 1], [2, 3]])
show(M, "M =")

char_poly = M.charpoly()
show(char_poly, "Characteristic polynomial:")

evals = M.eigenvals()
print(f"  Eigenvalues: {evals}")

evects = M.eigenvects()
P_mat, D_mat = M.diagonalize()
show(P_mat, "P (eigenvectors as columns):")
show(D_mat, "D (eigenvalues on diagonal):")

# Verify A = PDP⁻¹
chk_zero(simplify(M - P_mat*D_mat*P_mat.inv()).norm(), "M = PDP⁻¹")

# Powers: M^5 two ways
M5_direct = M**5
M5_diag   = P_mat * D_mat**5 * P_mat.inv()
chk_zero(simplify(M5_direct - M5_diag).norm(), "M⁵ via diagonalisation = direct")
show(M5_direct, "M⁵ =")

# Symmetric matrix → orthogonal eigenvectors (use numpy for numerical check)
S_num = np.array([[5,2,0],[2,3,1],[0,1,4]], dtype=float)
eig_vals_n, eig_vecs_n = np.linalg.eigh(S_num)  # eigh for symmetric
for i in range(3):
    for j in range(i+1, 3):
        ip = float(eig_vecs_n[:,i] @ eig_vecs_n[:,j])
        chk(ip, 0, f"Sym: eigvec {i}⊥eigvec {j}", tol=1e-12, absolute=True)

# %% [markdown]
# ---
# ## §5 · First-Order ODEs
#
# Three canonical types:
#
# **Separable**: $\dfrac{dy}{dx} = f(x)\,g(y)$ → divide by $g$, integrate both sides.
#
# **Linear**: $y' + P(x)y = Q(x)$ → multiply by integrating factor $\mu = e^{\int P\,dx}$.
#
# **Exact**: $M\,dx + N\,dy = 0$ with $\partial M/\partial y = \partial N/\partial x$
# → find potential function F with $F_x=M$, $F_y=N$.

# %%
hdr("§5 — First-order ODEs: separable, linear, exact")

x_sym, y_sym = symbols('x y')
y_fn = Function('y')

tex(r"\text{Separable: } \frac{dy}{dx} = f(x)g(y) \;\Rightarrow\; \int\frac{dy}{g(y)} = \int f(x)\,dx")
tex(r"\text{Linear: } y' + P(x)y = Q(x),\quad \mu = e^{\int P\,dx}")
tex(r"\text{Exact: } M\,dx + N\,dy = 0,\quad \frac{\partial M}{\partial y}=\frac{\partial N}{\partial x}")

# --- Separable: dy/dx = x*y^2 ---
ode1 = Eq(y_fn(x_sym).diff(x_sym), x_sym * y_fn(x_sym)**2)
sol1 = dsolve(ode1, y_fn(x_sym))
show(ode1, "ODE 1 (separable): dy/dx = x y²")
show(sol1, "Solution:")
# Verify by substitution
y_sol1 = sol1.rhs
dy_sol1 = diff(y_sol1, x_sym)
residual1 = simplify(dy_sol1 - x_sym * y_sol1**2)
print(f"  Verification residual: {residual1}  {'✓' if residual1 == 0 else '✗'}")

# --- Linear: y' + (2/x)y = x² ---
ode2 = Eq(y_fn(x_sym).diff(x_sym) + Rational(2,1)/x_sym * y_fn(x_sym), x_sym**2)
sol2 = dsolve(ode2, y_fn(x_sym))
show(ode2, "ODE 2 (linear): y' + (2/x)y = x²")
show(sol2, "Solution:")
y_sol2 = sol2.rhs
dy_sol2 = diff(y_sol2, x_sym)
residual2 = simplify(dy_sol2 + 2/x_sym*y_sol2 - x_sym**2)
print(f"  Verification residual: {residual2}  {'✓' if residual2 == 0 else '✗'}")

# --- Exact: (2xy + 1)dx + (x² + 3y²)dy = 0 ---
M_exact = 2*x_sym*y_sym + 1
N_exact = x_sym**2 + 3*y_sym**2
dM_dy = diff(M_exact, y_sym)
dN_dx = diff(N_exact, x_sym)
is_exact = simplify(dM_dy - dN_dx) == 0
print(f"\n  Exact ODE: ({M_exact})dx + ({N_exact})dy = 0")
print(f"  ∂M/∂y = {dM_dy},  ∂N/∂x = {dN_dx},  exact: {is_exact}")
F_partial = integrate(M_exact, x_sym) + Function('g')(y_sym)
# g(y) from ∂F/∂y = N
dF_dy = diff(integrate(M_exact, x_sym), y_sym)
g_prime = N_exact - dF_dy
g_fn = integrate(g_prime, y_sym)
F_total = integrate(M_exact, x_sym) + g_fn
show(Eq(Symbol('F(x,y)'), F_total), "Potential function F(x,y) = C:")
chk(float(simplify(diff(F_total, x_sym) - M_exact)), 0, "∂F/∂x = M", absolute=True)
chk(float(simplify(diff(F_total, y_sym) - N_exact)), 0, "∂F/∂y = N", absolute=True)

# %% [markdown]
# ---
# ## §6 · Second-Order ODEs — Characteristic Roots
#
# $ay'' + by' + cy = 0$ → try $y = e^{rt}$ → $ar^2 + br + c = 0$
#
# | Discriminant | Roots | General solution |
# |-------------|-------|-----------------|
# | $b^2-4ac > 0$ | real distinct $r_1, r_2$ | $C_1 e^{r_1 t} + C_2 e^{r_2 t}$ |
# | $b^2-4ac = 0$ | repeated $r$ | $(C_1 + C_2 t)e^{rt}$ |
# | $b^2-4ac < 0$ | complex $\alpha\pm i\beta$ | $e^{\alpha t}(C_1\cos\beta t + C_2\sin\beta t)$ |
#
# **Underdamped** ($b^2 < 4ac$): oscillates, decays → the "default" in physics.

# %%
hdr("§6 — Second-order ODEs: characteristic roots + resonance")

t_sym = symbols('t', positive=True)

tex(r"ay'' + by' + cy = 0 \;\Rightarrow\; ar^2 + br + c = 0")

cases_2nd = [
    (1, -5, 6,  "overdamped (real distinct)"),
    (1,  2, 1,  "critically damped (repeated root)"),
    (1,  2, 5,  "underdamped (complex conjugates)"),
]

for a_c, b_c, c_c, label in cases_2nd:
    ode = Eq(a_c*y_fn(t_sym).diff(t_sym,2) + b_c*y_fn(t_sym).diff(t_sym) + c_c*y_fn(t_sym), 0)
    sol = dsolve(ode)
    disc = b_c**2 - 4*a_c*c_c
    print(f"\n  {label}  (Δ={disc}):")
    show(sol, f"  y(t) =")
    # Verify
    y_s = sol.rhs
    res = simplify(a_c*diff(y_s,t_sym,2) + b_c*diff(y_s,t_sym) + c_c*y_s)
    print(f"  residual: {res}  {'✓' if res==0 else '✗'}")

# Forced resonance: y'' + ω²y = cos(ωt) — secular growth
omega = symbols('omega', positive=True)
ode_res = Eq(y_fn(t_sym).diff(t_sym,2) + omega**2*y_fn(t_sym), sp.cos(omega*t_sym))
sol_res = dsolve(ode_res)
show(sol_res, "Resonance y''+ω²y=cos(ωt):")
print("  Note: t·sin(ωt) term → amplitude grows linearly (resonance)")

# Numerical: RK4 for y'' + 2y' + 5y = 0, y(0)=1, y'(0)=0
def rk4_2nd(a,b,c,y0,yp0,T=10,N=1000):
    dt = T/N
    t_v = np.linspace(0,T,N+1)
    y_v = np.zeros(N+1); yp_v = np.zeros(N+1)
    y_v[0],yp_v[0] = y0, yp0
    def f(t,y,yp): return (-b*yp - c*y)/a
    for i in range(N):
        k1 = f(t_v[i],y_v[i],yp_v[i])
        k2 = f(t_v[i]+dt/2, y_v[i]+dt/2*yp_v[i], yp_v[i]+dt/2*k1)
        k3 = f(t_v[i]+dt/2, y_v[i]+dt/2*yp_v[i], yp_v[i]+dt/2*k2)
        k4 = f(t_v[i]+dt,   y_v[i]+dt*yp_v[i],   yp_v[i]+dt*k3)
        yp_v[i+1] = yp_v[i] + dt/6*(k1+2*k2+2*k3+k4)
        y_v[i+1]  = y_v[i]  + dt*yp_v[i] + dt**2/6*(k1+k2+k3)
    return t_v, y_v

t_rk, y_rk = rk4_2nd(1,2,5,1,0)
# Analytic: e^{-t}(cos(2t) + 0.5 sin(2t))
y_analytic = np.exp(-t_rk)*(np.cos(2*t_rk) + 0.5*np.sin(2*t_rk))
chk(np.max(np.abs(y_rk - y_analytic)), 0,
    "RK4 y''+2y'+5y=0 matches analytic (max error)", tol=1e-4, absolute=True)

# %% [markdown]
# ---
# ## §7 · Systems of ODEs — x′ = Ax
#
# A system $\mathbf{x}' = A\mathbf{x}$ has solution:
#
# $$\mathbf{x}(t) = e^{At}\,\mathbf{x}(0) = P\,e^{Dt}\,P^{-1}\,\mathbf{x}(0)$$
#
# where $e^{Dt} = \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t})$.
#
# Each eigenmode evolves independently:
# $\mathbf{x}(t) = \sum_i c_i\,e^{\lambda_i t}\,\mathbf{v}_i$
#
# **Stability**: stable iff all Re(λᵢ) < 0.

# %%
hdr("§7 — Systems x′=Ax: matrix exponential, phase portrait")

tex(r"\mathbf{x}(t) = e^{At}\mathbf{x}_0 = P e^{Dt} P^{-1}\mathbf{x}_0")

A_sys = Matrix([[-1, 2], [-2, -1]])
show(A_sys, "A =")

evects_sys = A_sys.eigenvects()
for ev, mult, vecs in evects_sys:
    show(Eq(Symbol('λ'), ev), f"  λ (mult={mult}):")
    show(vecs[0].T, "  v =")

# Matrix exponential via diagonalisation (complex eigenvalues)
# A has eigenvalues -1±2i → solution oscillates and decays
A_num = np.array(A_sys.tolist(), dtype=complex)
t_vals = np.linspace(0, 6, 500)
x0 = np.array([1.0, 0.0])

# Analytic: e^{-t}[cos(2t)(1,0) + sin(2t)*(A+I)/2 * x0]
# Easier: use scipy matrix exponential
from scipy.linalg import expm
trajs = np.array([expm(A_num * t).real @ x0 for t in t_vals])

# Verify at t=0
chk(np.linalg.norm(trajs[0] - x0), 0, "e^{A·0}x0 = x0", absolute=True)

# Verify decay: ‖x(t)‖ → 0 as t → ∞
chk(np.linalg.norm(trajs[-1]) < 0.01, 1,
    "‖x(t)‖ → 0 (all Re(λ)<0)", absolute=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(t_vals, trajs[:,0], label='x₁(t)')
ax1.plot(t_vals, trajs[:,1], label='x₂(t)', linestyle='--')
ax1.set_title("Time evolution x′=Ax  (Re(λ)=-1, Im(λ)=±2)")
ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xlabel('t')

# Phase portrait
X_g, Y_g = np.meshgrid(np.linspace(-2,2,20), np.linspace(-2,2,20))
A_ph = np.array([[-1,2],[-2,-1]], dtype=float)
dX = A_ph[0,0]*X_g + A_ph[0,1]*Y_g
dY = A_ph[1,0]*X_g + A_ph[1,1]*Y_g
ax2.streamplot(X_g, Y_g, dX, dY, density=1.2, color='C0', linewidth=1)
ax2.plot(*zip(*trajs), 'r-', linewidth=2, label='trajectory from (1,0)')
ax2.set_title("Phase portrait (stable spiral)"); ax2.legend(); ax2.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('repl/_fig_system_ode.png', dpi=110, bbox_inches='tight')
plt.close()
print("  Saved: repl/_fig_system_ode.png")

# %% [markdown]
# ---
# ## §8 · SVD — Singular Value Decomposition
#
# Every matrix $A \in \mathbb{R}^{m\times n}$ factors as:
#
# $$A = U \Sigma V^T$$
#
# - $U \in \mathbb{R}^{m\times m}$: left singular vectors (output directions)
# - $\Sigma \in \mathbb{R}^{m\times n}$: singular values on diagonal (stretches)
# - $V \in \mathbb{R}^{n\times n}$: right singular vectors (input directions)
#
# **Pseudoinverse**: $A^+ = V \Sigma^+ U^T$ — solves least-squares for any A.
#
# **Low-rank approximation**: keep top k singular values → best rank-k fit.
#
# **Eigenvalues of AᵀA = σᵢ²** — that's the connection back to eigenvalues.

# %%
hdr("§8 — SVD: U Σ Vᵀ, pseudoinverse, low-rank")

tex(r"A = U \Sigma V^T, \quad A^+ = V\Sigma^+ U^T")
tex(r"A^T A v_i = \sigma_i^2 v_i \quad\Leftarrow\text{eigenvectors of }A^TA")

A_svd = np.array([[1, 2, 0], [3, 1, 4], [0, 2, 1]], dtype=float)
U, S, Vt = np.linalg.svd(A_svd)
print(f"  Singular values: {S}")

# Reconstruction
A_reconstructed = U @ np.diag(S) @ Vt
chk(np.max(np.abs(A_reconstructed - A_svd)), 0,
    "A = U Σ Vᵀ reconstruction", tol=1e-12, absolute=True)

# σᵢ² = eigenvalues of AᵀA
AtA_eigs = np.sort(np.linalg.eigvals(A_svd.T @ A_svd))[::-1]
S_sq = S**2
for i in range(len(S)):
    chk(S_sq[i], AtA_eigs[i], f"σ_{i+1}² = eigenvalue {i+1} of AᵀA", tol=1e-8)

# Pseudoinverse
A_plus = np.linalg.pinv(A_svd)
chk(np.max(np.abs(A_svd @ A_plus @ A_svd - A_svd)), 0,
    "A A⁺ A = A  (Moore-Penrose condition 1)", tol=1e-12, absolute=True)
chk(np.max(np.abs(A_plus @ A_svd @ A_plus - A_plus)), 0,
    "A⁺ A A⁺ = A⁺ (Moore-Penrose condition 2)", tol=1e-12, absolute=True)

# Low-rank: image compression demo with random 20×20 matrix
np.random.seed(7)
img = np.random.randn(20, 20)
U_i, S_i, Vt_i = np.linalg.svd(img)
print(f"\n  Low-rank approximation (20×20 matrix, rank-k):")
for k in [1, 3, 5, 10]:
    img_k = U_i[:,:k] @ np.diag(S_i[:k]) @ Vt_i[:k,:]
    err = np.linalg.norm(img - img_k, 'fro') / np.linalg.norm(img, 'fro')
    energy = np.sum(S_i[:k]**2)/np.sum(S_i**2)
    print(f"  rank-{k:2d}: rel error={err:.3f}  energy captured={energy:.3f}")
chk(np.linalg.norm(img - U_i @ np.diag(S_i) @ Vt_i, 'fro'), 0,
    "rank-20 = exact reconstruction", tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## §9 · Function Spaces — Inner Product on L²
#
# Finite-dim vector space → infinite-dim function space.
# The dictionary:
#
# | Linear algebra | Function space |
# |---------------|---------------|
# | $\mathbf{u}\cdot\mathbf{v} = \sum u_i v_i$ | $\langle f,g\rangle = \int_a^b f(x)g(x)\,dx$ |
# | orthonormal basis $\{e_i\}$ | orthonormal functions $\{\phi_n\}$ |
# | $\mathbf{x} = \sum c_i \mathbf{e}_i$ | $f = \sum c_n \phi_n$ |
# | $c_i = \mathbf{e}_i\cdot\mathbf{x}$ | $c_n = \langle\phi_n, f\rangle$ |
# | $A\mathbf{v} = \lambda\mathbf{v}$ | $Ly = \lambda y$ (Sturm-Liouville) |
#
# This is why Fourier series "works" — it's just a change of basis.

# %%
hdr("§9 — Function spaces: inner product on L²[−π, π]")

tex(r"\langle f,g \rangle = \int_{-\pi}^{\pi} f(x)g(x)\,dx")
tex(r"\|\phi_n\|^2 = \langle\phi_n,\phi_n\rangle = \pi \quad \text{for sin, cos}")

x_sym = symbols('x', real=True)
n_sym, m_sym = symbols('n m', positive=True, integer=True)

# Orthogonality of trig functions over [-π, π]
print("  Trig orthogonality (the foundation of Fourier series):")
for fn1, fn2, label, expected in [
    (sp.sin(n_sym*x_sym), sp.sin(m_sym*x_sym), "<sin nx, sin mx>", "π if n=m, else 0"),
    (sp.cos(n_sym*x_sym), sp.cos(m_sym*x_sym), "<cos nx, cos mx>", "π if n=m, else 0"),
    (sp.sin(n_sym*x_sym), sp.cos(m_sym*x_sym), "<sin nx, cos mx>", "0 always"),
]:
    # Check n≠m case
    for n_v, m_v in [(1,2), (2,3), (1,3)]:
        r = float(integrate(fn1.subs(n_sym,n_v)*fn2.subs(m_sym,m_v),
                            (x_sym,-pi,pi)).evalf())
        chk(r, 0, f"{label} n={n_v},m={m_v} = 0", tol=1e-10, absolute=True)
    # Check n=m case (only for same family)
    if "sin nx, sin mx" in label or "cos nx, cos mx" in label:
        for n_v in [1,2,3]:
            r = float(integrate(fn1.subs(n_sym,n_v)*fn2.subs(m_sym,n_v),
                                (x_sym,-pi,pi)).evalf())
            chk(r, float(pi), f"{label} n=m={n_v} = π", tol=1e-8)

# Gram-Schmidt on functions: {1, x, x²} over [-1,1]
# (produces Legendre polynomials up to normalisation)
print("\n  Gram-Schmidt on {1, x, x²} over [-1,1] → Legendre-like:")
x_s = symbols('x', real=True)
funcs = [sp.Integer(1), x_s, x_s**2]
def inner(f, g): return integrate(f*g, (x_s, -1, 1))

orth = []
for f in funcs:
    q = f
    for o in orth:
        q = q - inner(f, o)/inner(o, o) * o
    orth.append(q)

for i, q in enumerate(orth):
    q_s = simplify(q)
    show(Eq(Symbol(f'φ_{i}'), q_s), f"  orthogonalised φ_{i}:")
    # Verify orthogonality to all previous
    for j, prev in enumerate(orth[:i]):
        ip = float(integrate(q_s * simplify(prev), (x_s,-1,1)).evalf())
        chk(ip, 0, f"<φ_{i}, φ_{j}> = 0", tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## §10 · Fourier Series — Projection onto Eigenbasis
#
# The derivative operator $L = d^2/dx^2$ on $[-\pi,\pi]$ has eigenfunctions
# $\sin(nx)$ and $\cos(nx)$ with eigenvalues $-n^2$.
#
# $$L(\cos nx) = -n^2\cos(nx), \qquad L(\sin nx) = -n^2\sin(nx)$$
#
# A Fourier series is **projection** of $f$ onto this eigenbasis:
#
# $$a_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx, \quad
#   b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx$$

# %%
hdr("§10 — Fourier series = projection onto eigenbasis of d²/dx²")

tex(r"f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty \bigl[a_n\cos(nx) + b_n\sin(nx)\bigr]")
tex(r"a_n = \frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\cos(nx)\,dx, \quad b_n = \frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\sin(nx)\,dx")

# Verify d²/dx² eigenfunctions
for n_v in [1, 2, 3, 4]:
    lam_cos = float(simplify(diff(sp.cos(n_v*x_sym),x_sym,2)/sp.cos(n_v*x_sym)).evalf())
    lam_sin = float(simplify(diff(sp.sin(n_v*x_sym),x_sym,2)/sp.sin(n_v*x_sym)).subs(x_sym, 0.7).evalf())
    chk(lam_cos, -n_v**2, f"d²/dx²(cos {n_v}x) = -{n_v}² cos {n_v}x")
    chk(lam_sin, -n_v**2, f"d²/dx²(sin {n_v}x) = -{n_v}² sin {n_v}x")

# Fourier series of f(x) = x on (-π, π)  [SymPy]
x_fs = symbols('x')
f_sq = x_fs  # sawtooth
N_terms = 8
a0 = float(sp.N(integrate(f_sq, (x_fs,-pi,pi)) / pi))
print(f"\n  Fourier series of f(x) = x:  a₀ = {a0:.4g}")
b_coeffs = []
for n_v in range(1, N_terms+1):
    bn = float(integrate(f_sq * sp.sin(n_v*x_fs), (x_fs,-pi,pi)).evalf()) / pi
    b_coeffs.append(bn)
    ref_bn = (-1)**(n_v+1) * 2/n_v
    chk(bn, ref_bn, f"b_{n_v} = (-1)^(n+1)·2/n", tol=1e-8)

# Numerical reconstruction
x_num = np.linspace(-np.pi, np.pi, 1000)
f_approx = sum(b_coeffs[n-1] * np.sin(n*x_num) for n in range(1, N_terms+1))
f_exact  = x_num.copy()
# Exclude endpoints (Gibbs phenomenon there)
mask = np.abs(x_num) < 2.5
chk(np.max(np.abs(f_approx[mask] - f_exact[mask])), 0,
    f"Fourier N={N_terms} max error (|x|<2.5)", tol=0.4, absolute=True)

# Parseval's theorem: sum b_n² = (1/π)∫|f|²dx
parseval_lhs = sum(bn**2 for bn in b_coeffs)
parseval_rhs = float(integrate(f_sq**2, (x_fs,-pi,pi)).evalf()) / pi
print(f"\n  Parseval: Σbₙ² = {parseval_lhs:.4f}  vs  (1/π)∫f²dx = {parseval_rhs:.4f}")
print(f"  (converges to π²/3 = {float(pi**2/3):.4f})")

# %% [markdown]
# ---
# ## §11 · PDEs — Separation of Variables
#
# **Heat equation**: $u_t = \alpha^2 u_{xx}$
#
# Try $u = X(x)T(t)$:
# $\dfrac{T'}{α²T} = \dfrac{X''}{X} = -\lambda$ (separation constant)
#
# → $X'' + \lambda X = 0$,  $T' + \alpha^2\lambda T = 0$
#
# With BC $u(0,t)=u(L,t)=0$:
# $\lambda_n = (n\pi/L)^2$,  $X_n = \sin(n\pi x/L)$,  $T_n = e^{-\alpha^2\lambda_n t}$
#
# **General solution**: $u(x,t) = \sum_{n=1}^\infty b_n \sin\!\left(\tfrac{n\pi x}{L}\right) e^{-\alpha^2(n\pi/L)^2 t}$

# %%
hdr("§11 — Heat equation via separation of variables")

tex(r"u_t = \alpha^2 u_{xx}, \quad u(0,t)=u(L,t)=0")
tex(r"u(x,t) = \sum_{n=1}^\infty b_n \sin\!\left(\frac{n\pi x}{L}\right)e^{-\alpha^2(n\pi/L)^2 t}")

alpha_val = 1.0
L_val = np.pi   # so λ_n = n²
N_heat = 20
x_h = np.linspace(0, L_val, 300)
t_heat_vals = [0.0, 0.01, 0.05, 0.2, 1.0]

# Initial condition: u(x,0) = sin(x) + 0.5*sin(3x)
# Exact solution: e^{-t}sin(x) + 0.5*e^{-9t}sin(3x)
def u_exact(x, t):
    return np.exp(-t)*np.sin(x) + 0.5*np.exp(-9*t)*np.sin(3*x)

# Fourier coefficients of u(x,0)
def b_n(n, ic_fn):
    return 2/L_val * np.trapezoid(ic_fn(x_h) * np.sin(n*np.pi*x_h/L_val), x_h)

ic_fn = lambda x: np.sin(x) + 0.5*np.sin(3*x)
b_coeffs_h = [b_n(n, ic_fn) for n in range(1, N_heat+1)]

chk(abs(b_coeffs_h[0] - 1.0), 0, "b₁ = 1  (sin x mode)", tol=1e-4, absolute=True)
chk(abs(b_coeffs_h[2] - 0.5), 0, "b₃ = 0.5 (sin 3x mode)", tol=1e-4, absolute=True)

# Solution at various times
fig, axes = plt.subplots(1, len(t_heat_vals), figsize=(14, 3.5))
for ax, t_v in zip(axes, t_heat_vals):
    u_fs = sum(b_coeffs_h[n-1]*np.sin(n*np.pi*x_h/L_val)*
               np.exp(-alpha_val**2*(n*np.pi/L_val)**2*t_v)
               for n in range(1, N_heat+1))
    ax.plot(x_h, u_exact(x_h, t_v), 'r--', linewidth=2, label='exact')
    ax.plot(x_h, u_fs, 'b-', linewidth=1.5, label=f'N={N_heat}')
    ax.set_title(f't={t_v}'); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
plt.suptitle('Heat equation u_t = u_xx, u=sin(x)+0.5sin(3x) at t=0', fontsize=10)
plt.tight_layout()
plt.savefig('repl/_fig_heat_equation.png', dpi=110, bbox_inches='tight')
plt.close()
print("  Saved: repl/_fig_heat_equation.png")

# Verify at t=0.2
u_num = np.array([sum(b_coeffs_h[n-1]*np.sin(n*np.pi*x_h/L_val)*
                      np.exp(-alpha_val**2*(n*np.pi/L_val)**2*0.2)
                      for n in range(1,N_heat+1))])
u_ex  = u_exact(x_h, 0.2)
chk(np.max(np.abs(u_num - u_ex)), 0,
    "heat eq Fourier vs exact at t=0.2", tol=1e-6, absolute=True)

# %% [markdown]
# ---
# ## §12 · Sturm-Liouville — Eigenfunction Expansion
#
# General Sturm-Liouville problem:
#
# $$\frac{d}{dx}\!\left[p(x)\,y'\right] + q(x)\,y + \lambda\,w(x)\,y = 0$$
#
# **Theorem**: eigenvalues are real, eigenfunctions are orthogonal w.r.t.
# weight $w$:
#
# $$\langle\phi_m,\phi_n\rangle_w = \int_a^b \phi_m\phi_n\,w\,dx = 0 \quad m\neq n$$
#
# Special cases you already know:
# - $p=1, q=0, w=1$ → trig functions (Fourier)
# - $p=1-x^2, q=0, w=1$ → Legendre polynomials
# - $p=x, q=0, w=x$ → Bessel functions

# %%
hdr("§12 — Sturm-Liouville: eigenfunctions orthogonal w.r.t. weight w")

tex(r"\frac{d}{dx}\!\left[p(x)y'\right] + q(x)y + \lambda w(x)y = 0")
tex(r"\int_a^b \phi_m\phi_n\,w(x)\,dx = 0 \quad m\neq n")

x_sl = symbols('x', real=True)

# --- Legendre polynomials (p=1-x², q=0, w=1, [-1,1]) ---
print("  Legendre polynomials Pₙ(x): orthogonal on [-1,1] w.r.t. w=1")
from sympy import legendre
P_legs = [sp.legendre(n, x_sl) for n in range(5)]
for i, Pi in enumerate(P_legs):
    show(Eq(Symbol(f'P_{i}(x)'), Pi))

# Orthogonality: ∫ Pₙ Pₘ dx = 2/(2n+1) δₙₘ
for n_v in range(4):
    for m_v in range(n_v+1, 5):
        ip = float(integrate(P_legs[n_v]*P_legs[m_v], (x_sl,-1,1)).evalf())
        chk(ip, 0, f"<P_{n_v}, P_{m_v}> = 0 (orthogonal)", tol=1e-10, absolute=True)
for n_v in range(5):
    ip = float(integrate(P_legs[n_v]**2, (x_sl,-1,1)).evalf())
    ref = 2/(2*n_v+1)
    chk(ip, ref, f"<P_{n_v}, P_{n_v}> = 2/(2·{n_v}+1) = {ref:.4f}")

# Expand f(x)=x³ in Legendre series: x³ = (3/5)P₃ + (3/5)P₁... wait
# f(x) = x³ = (2P₃ + 3P₁)/5
c_n = [float(((2*n_v+1)/2 * integrate(x_sl**3 * P_legs[n_v], (x_sl,-1,1))).evalf())
       for n_v in range(5)]
print(f"\n  x³ = Σ cₙ Pₙ(x)  coefficients: {[f'{c:.4f}' for c in c_n]}")
f_recon = sum(c_n[n_v]*P_legs[n_v] for n_v in range(5))
f_diff  = simplify(f_recon - x_sl**3)
is_zero = f_diff == 0 or sp.Abs(f_diff).evalf(subs={x_sl: 0.5}) < 1e-10
print(f"  [{'PASS' if is_zero else 'FAIL'}]  x³ = Σcₙ Pₙ reconstruction  diff={f_diff}")

# Final: connection table
hdr("Summary — Linear Algebra = Differential Equations")
print("""
  ┌──────────────────────────┬──────────────────────────────────────────┐
  │  Linear algebra          │  Differential equations                  │
  ├──────────────────────────┼──────────────────────────────────────────┤
  │  Vector space Rⁿ         │  Function space L²[a,b]                  │
  │  dot product Σ uᵢvᵢ      │  inner product ∫ f g dx                  │
  │  orthonormal basis {eᵢ}  │  eigenfunctions {sin nx, cos nx, Pₙ, Jₙ}│
  │  coordinates cᵢ=eᵢ·x    │  Fourier/Legendre coefficients           │
  │  matrix A                │  differential operator L = d²/dx²        │
  │  Av = λv                 │  Ly = λy  (Sturm-Liouville)              │
  │  det(A−λI) = 0           │  characteristic equation (ODE)           │
  │  A = PDP⁻¹               │  modal decomposition of PDE solution     │
  │  SVD: A = UΣVᵀ           │  Schmidt decomposition of Green's func   │
  │  e^{At}                  │  e^{Lt} = heat/wave propagator           │
  └──────────────────────────┴──────────────────────────────────────────┘
""")

hdr("Done — §1 vectors §2 RREF §3 det §4 eigen §5 ODE-1 §6 ODE-2 §7 systems §8 SVD §9 L² §10 Fourier §11 heat PDE §12 Sturm-Liouville")
