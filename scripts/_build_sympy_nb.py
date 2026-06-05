import nbformat, textwrap

def code(src): return nbformat.v4.new_code_cell(textwrap.dedent(src).strip())
def md(src):   return nbformat.v4.new_markdown_cell(textwrap.dedent(src).strip())

nb = nbformat.v4.new_notebook()
nb.cells = [

md("""# SymPy Reference Notebook
Symbolic mathematics in Python. Every cell renders LaTeX automatically.
Covers: algebra, calculus, linear algebra, ODEs, Bessel, dispersion, GS."""),

# ── Setup ─────────────────────────────────────────────────────────────────
md("## 0. Setup: `init_printing`"),

code("""
import sympy as sp

# init_printing: renders all SymPy output as LaTeX in Jupyter
# use_unicode=True  -> pretty Unicode in terminal
# use_latex=True    -> MathJax in Jupyter (automatic when use_latex not set)
sp.init_printing(use_unicode=True, use_latex='mathjax')

# Declare symbols with assumptions (helps simplify)
x, y, z   = sp.symbols('x y z', real=True)
t, s      = sp.symbols('t s', real=True, positive=True)
n, m, k   = sp.symbols('n m k', integer=True, positive=True)
a, b, c   = sp.symbols('a b c', real=True)
omega, nu = sp.symbols('omega nu', real=True)
hbar, m_s = sp.symbols('hbar m', positive=True)
lam       = sp.Symbol('lambda', positive=True)

print("Symbols declared. SymPy version:", sp.__version__)
"""),

code("""
# Verify init_printing works: this should render as LaTeX
expr = sp.sqrt(sp.pi) * sp.exp(-x**2)
print("Expression (terminal fallback):", expr)
expr   # <- in Jupyter this renders as LaTeX automatically
"""),

# ── Algebra ───────────────────────────────────────────────────────────────
md("""## 1. Algebra: expand, factor, simplify, apart, together

| Function | Does |
|----------|------|
| `sp.expand(e)` | Distribute all products |
| `sp.factor(e)` | Factor into irreducibles |
| `sp.simplify(e)` | Heuristic simplification |
| `sp.cancel(e)` | Cancel common factors in rational |
| `sp.apart(e, x)` | Partial fraction decomposition |
| `sp.trigsimp(e)` | Simplify trig identities |
| `sp.radsimp(e)` | Rationalize denominators |
"""),

code("""
# Expand
poly = (x + 2)**4
sp.expand(poly)
"""),

code("""
# Factor
sp.factor(x**4 - 16)
"""),

code("""
# Simplify trig: sin^2 + cos^2 = 1
expr = sp.sin(x)**2 + sp.cos(x)**2
sp.simplify(expr)
"""),

code("""
# Partial fractions
f = (3*x + 5) / (x**2 - x - 6)
sp.apart(f, x)
"""),

code("""
# Solve algebraic equations
sol = sp.solve(x**3 - 3*x + 2, x)
print("Roots:", sol)
sp.factor(x**3 - 3*x + 2)
"""),

# ── Calculus ──────────────────────────────────────────────────────────────
md("""## 2. Calculus: diff, integrate, limit, series, summation"""),

code("""
# Derivatives
f = sp.sin(x) * sp.exp(-a*x**2)
df  = sp.diff(f, x)
d2f = sp.diff(f, x, 2)
print("f   =", f)
print("f'  =", df)
print("f'' =", sp.simplify(d2f))
"""),

code("""
# Definite and indefinite integrals
# Gaussian integral: int_{-inf}^{inf} exp(-x^2) dx = sqrt(pi)
gauss = sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))
gauss
"""),

code("""
# Fourier transform: int exp(-a*x^2) * exp(-i*k*x) dx
k_sym = sp.Symbol('k', real=True)
a_pos = sp.Symbol('a', positive=True)
ft = sp.integrate(
    sp.exp(-a_pos*x**2) * sp.exp(-sp.I*k_sym*x),
    (x, -sp.oo, sp.oo)
)
sp.simplify(ft)
"""),

code("""
# Limits
print("lim x->0 sin(x)/x =", sp.limit(sp.sin(x)/x, x, 0))
print("lim x->inf (1+1/x)^x =", sp.limit((1 + 1/x)**x, x, sp.oo))
print("lim x->0+ x*log(x) =", sp.limit(x*sp.log(x), x, 0, '+'))
"""),

code("""
# Taylor series (around x=0, order 6)
series_sin  = sp.series(sp.sin(x),  x, 0, 8)
series_cos  = sp.series(sp.cos(x),  x, 0, 8)
series_exp  = sp.series(sp.exp(x),  x, 0, 6)
print("sin(x) =", series_sin)
print("cos(x) =", series_cos)
print("exp(x) =", series_exp)
"""),

code("""
# Sum: geometric series
n_sym = sp.Symbol('n', integer=True, nonneg=True)
r     = sp.Symbol('r')
geo_finite   = sp.summation(r**n_sym, (n_sym, 0, 10))
geo_infinite = sp.summation(r**n_sym, (n_sym, 0, sp.oo))
print("Sum r^n, n=0..10  =", sp.factor(geo_finite))
print("Sum r^n, n=0..inf =", geo_infinite, " (|r|<1)")
"""),

# ── Linear Algebra ────────────────────────────────────────────────────────
md("""## 3. Linear Algebra: Matrix, eigenvalues, SVD, null space"""),

code("""
# Define a matrix symbolically
A = sp.Matrix([
    [2,  1,  0],
    [1,  3,  1],
    [0,  1,  2],
])
print("det(A) =", A.det())
print("trace  =", A.trace())
A
"""),

code("""
# Eigenvalues and eigenvectors
evals = A.eigenvals()    # {eigenvalue: multiplicity}
evecs = A.eigenvects()   # [(eigenvalue, mult, [vecs]), ...]
print("Eigenvalues:", evals)
for lam_v, mult, vecs in evecs:
    print(f"  lambda={lam_v}  (mult={mult})  eigvec={vecs[0].T}")
"""),

code("""
# Symbolic matrix: covariance-like
C = sp.Matrix([
    [a,   b],
    [b,   a+c],
])
sp.pprint(C.eigenvals())
"""),

code("""
# Null space (same as stoichiometry balancing from _repl_stoichiometry.py)
# CH4 + O2 -> CO2 + H2O
#       CH4  O2  CO2  H2O
M_chem = sp.Matrix([
    [ 1,   0,  -1,   0],   # C
    [ 4,   0,   0,  -2],   # H
    [ 0,   2,  -2,  -1],   # O
])
ns = M_chem.nullspace()
print("Null space vector (balanced stoichiometry):")
sp.pprint(ns[0].T)
# Ratio: CH4 : O2 : CO2 : H2O = 1:2:1:2
"""),

code("""
# Singular value decomposition (symbolic small matrix)
B = sp.Matrix([[1, 2], [3, 4], [5, 6]])
U, S, V = B.singular_value_decomposition()
print("Singular values:", [sp.simplify(s) for s in S.diagonal()])
# verify: B = U * S * V.T
residual = sp.simplify(B - U*S*V.T)
print("Residual (should be zero):", residual)
"""),

# ── ODEs ──────────────────────────────────────────────────────────────────
md("""## 4. ODEs: dsolve, initial conditions, systems"""),

code("""
# First order: RC circuit  dV/dt + V/RC = 0
RC = sp.Symbol('RC', positive=True)
V  = sp.Function('V')
ode1 = sp.Eq(V(t).diff(t) + V(t)/RC, 0)
sol1 = sp.dsolve(ode1, V(t))
print("RC discharge:"); sp.pprint(sol1)
"""),

code("""
# Second order: harmonic oscillator  x'' + omega^2 x = 0
x_fn   = sp.Function('x')
omega0 = sp.Symbol('omega0', positive=True)
ode2   = sp.Eq(x_fn(t).diff(t, 2) + omega0**2 * x_fn(t), 0)
sol2   = sp.dsolve(ode2, x_fn(t))
sp.pprint(sol2)
"""),

code("""
# Apply initial conditions: x(0)=1, x'(0)=0
C1, C2 = sp.symbols('C1 C2')
sol_gen = sol2.rhs
ic1 = sp.Eq(sol_gen.subs(t, 0), 1)        # x(0)=1
ic2 = sp.Eq(sol_gen.diff(t).subs(t, 0), 0) # x'(0)=0
consts = sp.solve([ic1, ic2], [C1, C2])
x_particular = sol_gen.subs(consts)
print("x(t) with x(0)=1, x'(0)=0:")
sp.pprint(sp.simplify(x_particular))
"""),

code("""
# Schrodinger equation (1D particle in box): -hbar^2/2m * psi'' = E*psi
# psi'' + k^2*psi = 0,  k = sqrt(2mE)/hbar
psi = sp.Function('psi')
k   = sp.Symbol('k', positive=True)
ode_se = sp.Eq(psi(x).diff(x, 2) + k**2 * psi(x), 0)
sol_se = sp.dsolve(ode_se, psi(x))
sp.pprint(sol_se)
"""),

# ── Bessel + Special Functions ────────────────────────────────────────────
md("""## 5. Bessel Functions + Special Functions"""),

code("""
# Bessel equation: r^2 R'' + r R' + (k^2 r^2 - n^2) R = 0
# Solution: R(r) = C1 * J_n(k*r) + C2 * Y_n(k*r)
r   = sp.Symbol('r', positive=True)
k_r = sp.Symbol('k_r', positive=True)
n_b = sp.Integer(0)

# SymPy Bessel functions
J0 = sp.besselj(0, k_r*r)
J1 = sp.besselj(1, k_r*r)
Y0 = sp.bessely(0, k_r*r)
K0 = sp.besselk(0, k_r*r)  # modified, decaying (evanescent)
I0 = sp.besseli(0, k_r*r)  # modified, growing

print("J_0(k*r) ="); sp.pprint(J0)
print()
print("Derivative d/dr J_0(k*r) = -k * J_1(k*r):")
dJ0 = sp.diff(J0, r)
sp.pprint(sp.simplify(dJ0))
"""),

code("""
# Recurrence relation: J_{n-1}(x) + J_{n+1}(x) = (2n/x)*J_n(x)
x_b  = sp.Symbol('x', positive=True)
n_b2 = sp.Integer(1)
lhs  = sp.besselj(n_b2-1, x_b) + sp.besselj(n_b2+1, x_b)
rhs  = 2*n_b2/x_b * sp.besselj(n_b2, x_b)
diff = sp.simplify(lhs - rhs)
print("J_0(x) + J_2(x) - 2/x * J_1(x) =", diff)
"""),

code("""
# Gamma function and factorial connection
n_int = sp.Symbol('n', integer=True, positive=True)
print("Gamma(n+1) = n! verified symbolically:")
for val in [1, 2, 3, 4, 5]:
    print(f"  Gamma({val+1}) = {sp.gamma(val+1)} = {sp.factorial(val)}")

# Reflection formula: Gamma(x)*Gamma(1-x) = pi/sin(pi*x)
x_g = sp.Symbol('x')
refl = sp.simplify(sp.gamma(x_g) * sp.gamma(1-x_g) - sp.pi/sp.sin(sp.pi*x_g))
print("Reflection formula residual:", refl)
"""),

# ── Dispersion + GS ───────────────────────────────────────────────────────
md("""## 6. Dispersion + GS Phase Recovery (the project)"""),

code("""
# Dispersion relation for fiber: beta(omega)
# Taylor expand around omega_0
omega_0 = sp.Symbol('omega_0', positive=True)
beta_0  = sp.Symbol('beta_0',  real=True)
beta_1  = sp.Symbol('beta_1',  real=True)   # 1/v_g
beta_2  = sp.Symbol('beta_2',  real=True)   # GVD
beta_3  = sp.Symbol('beta_3',  real=True)   # TOD
d_omega = omega - omega_0  # detuning

# Dispersion relation (Taylor series)
beta = beta_0 + beta_1*d_omega + sp.Rational(1,2)*beta_2*d_omega**2 + \
       sp.Rational(1,6)*beta_3*d_omega**3

print("Dispersion relation beta(omega):")
sp.pprint(beta)
"""),

code("""
# Transfer function H(nu) = exp(i*pi*D*nu^2)
# Verify it is all-pass: |H|^2 = 1
D_sym = sp.Symbol('D', real=True)
nu_sym = sp.Symbol('nu', real=True)
H = sp.exp(sp.I * sp.pi * D_sym * nu_sym**2)
H_conj = sp.conjugate(H)
mag_sq = sp.simplify(sp.expand_complex(H * H_conj))
print("|H(nu)|^2 =", mag_sq, "  (all-pass confirmed)")
"""),

code("""
# GS constraint sets as algebraic equations
# C1: |E|^2 = I1  ->  projection: E_new = sqrt(I1) * E/|E|
# C2: |F{H*E}|^2 = I2  ->  project in Fourier domain

# Phase ambiguity: global phase U(1)
# E and E*exp(i*phi_0) give SAME intensities I1, I2 for any phi_0

phi0    = sp.Symbol('phi_0', real=True)
E_sym   = sp.Symbol('E')               # complex (no assumption)
E_shift = E_sym * sp.exp(sp.I * phi0)

# |E_shift|^2 = |E|^2 * |exp(i*phi0)|^2 = |E|^2 (since |exp(i*phi0)|=1)
mag_orig  = sp.Abs(E_sym)**2
mag_shift = sp.Abs(E_shift)**2
print("U(1) phase ambiguity:")
print("|E|^2 =", mag_orig)
print("|E*exp(i*phi0)|^2 = E_sym*conj(E_sym) (same)")
print("-> GS result is defined only up to global phase offset phi_0")
"""),

code("""
# Even symmetry of H(nu) = exp(i*pi*D*nu^2)
# nu^2 is even: H(-nu) = H(nu)
H_pos = sp.exp(sp.I * sp.pi * D_sym * nu_sym**2)
H_neg = H_pos.subs(nu_sym, -nu_sym)
diff_H = sp.simplify(H_pos - H_neg)
print("H(nu) - H(-nu) =", diff_H, "  (even function confirmed)")
print()
print("Physical meaning:")
print("  Even H -> real input E stays real after dispersing")
print("  Hermitian FFT symmetry preserved")
print("  GS can use rfft (half the FFT) -> 2x speedup")
"""),

code("""
# Normalization: particle in box psi_n
# Verify int_0^L |psi_n|^2 dx = 1
L   = sp.Symbol('L', positive=True)
n_q = sp.Symbol('n', integer=True, positive=True)
psi_n = sp.sqrt(2/L) * sp.sin(n_q * sp.pi * x / L)
norm  = sp.integrate(sp.conjugate(psi_n) * psi_n, (x, 0, L))
print("Normalization integral =", sp.simplify(norm))

# Orthogonality
psi_1 = sp.sqrt(2/L) * sp.sin(1 * sp.pi * x / L)
psi_2 = sp.sqrt(2/L) * sp.sin(2 * sp.pi * x / L)
overlap = sp.simplify(sp.integrate(psi_1 * psi_2, (x, 0, L)))
print("<psi_1|psi_2> =", overlap, "  (orthogonal)")
"""),

# ── Printing Modes ────────────────────────────────────────────────────────
md("""## 7. Printing Modes Reference

```python
# In Jupyter: LaTeX rendering (default after init_printing)
expr

# Force LaTeX string (for reports, papers)
sp.latex(expr)

# Pretty-print to terminal (Unicode)
sp.pprint(expr, use_unicode=True)

# ASCII only (for Windows terminal without Unicode)
sp.pprint(expr, use_unicode=False)

# Print to string
str(expr)

# Full precision float evaluation
expr.evalf(50)  # 50 significant digits

# Substitute values
expr.subs(x, sp.pi/4)
expr.subs([(x, 1), (a, 2)])  # multiple substitutions

# Lambdify: convert SymPy expr to fast numpy function
import numpy as np
f_numpy = sp.lambdify(x, sp.sin(x)**2 + sp.cos(x)**2, 'numpy')
print(f_numpy(np.linspace(0, np.pi, 5)))  # should be all 1.0
```
"""),

code("""
# Quick demo of all print modes
expr = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo))

print("str(expr)    :", str(expr))
print("sp.latex()   :", sp.latex(expr))
print()
print("sp.pprint (unicode):")
sp.pprint(expr, use_unicode=True)
print()
print("Evaluated    :", expr.doit())
"""),

code("""
# Lambdify demo: SymPy -> NumPy (fast numerical evaluation)
import numpy as np

# Define symbolically
f_sym = sp.sin(x)**2 / (1 + sp.cos(x)**2)
print("Symbolic:", f_sym)

# Convert to numpy function
f_np = sp.lambdify(x, f_sym, 'numpy')
x_vals = np.linspace(0, 2*np.pi, 6)
print("Numerical:", np.round(f_np(x_vals), 4))

# This is the bridge: symbolic -> verify -> lambdify -> GS inner loop
"""),

]

nbformat.write(nb, 'notebooks/sympy_reference.ipynb')
print('Written: notebooks/sympy_reference.ipynb')
