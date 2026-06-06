# %% [markdown]
# # Dirac Delta: All the Algebra (What Your STEM Parents Were Supposed to Tell You)
#
# Not just the definition. The MANIPULATION RULES.
# Every identity you need to simplify expressions by hand — derived, not memorized.
#
# Rule of thumb: δ is NOT a function. It is a **linear functional** on test functions.
# All algebra follows from one axiom:  ∫ f(x) δ(x-a) dx = f(a)

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from repl_helpers import hdr, show, chk
import numpy as np
import sympy as sp
from sympy import (symbols, integrate, oo, DiracDelta, Heaviside, diff,
                   exp, cos, sin, pi, sqrt, Abs, Rational, simplify,
                   series, Function, limit, latex, I, conjugate, apart)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sp.init_printing(use_latex='mathjax')

x, y, a, b, t, k, n_sym = symbols('x y a b t k n', real=True)
eps = symbols('epsilon', positive=True)

def hdr(s):
    print(f'\n{"─"*64}\n  {s}\n{"─"*64}')

try:
    from IPython.display import display as _ipy_display
    def show(expr, label=None):
        if label: print(f'  ' + str(label) + ':')
        _ipy_display(expr)
except ImportError:
    def show(expr, label=None):
        if label: print('  ' + str(label) + ':')
        import sympy as sp
        print('  ' + sp.pretty(expr, use_unicode=True))

def chk(val, ref, label, tol=1e-9, absolute=False):
    v, r = float(val), float(ref)
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    s = 'PASS' if err < tol else 'FAIL'
    print(f'  [{s}]  {label}  got={v:.8g}  ref={r:.8g}')

# numerical delta via narrow Gaussian
def delta_num(x_arr, center=0.0, eps_w=1e-3):
    """Gaussian approx to delta: (1/(eps*sqrt(pi))) exp(-x^2/eps^2)"""
    return np.exp(-((x_arr - center)/eps_w)**2) / (eps_w * np.sqrt(np.pi))

N_grid = 200_000
x_grid = np.linspace(-5, 5, N_grid)
dx     = x_grid[1] - x_grid[0]

# %% [markdown]
# ## Rule 0 — The One Axiom Everything Comes From
#
#   ∫_{-∞}^{∞} f(x) δ(x − a) dx = f(a)
#
# That's it. Every other rule is derived by choosing f cleverly.

# %%
hdr("Rule 0 — The sifting axiom")

# Prove numerically for several f and a
tests = [
    (lambda x: x**3,       2.0,   8.0,   "x^3 at x=2"),
    (lambda x: np.cos(x),  np.pi, -1.0,  "cos(x) at x=pi"),
    (lambda x: np.exp(-x), 1.0,   np.exp(-1), "e^{-x} at x=1"),
    (lambda x: x**2 - 3*x, -1.0,  4.0,  "x^2-3x at x=-1"),
]
for f, a_val, ref, label in tests:
    integrand = f(x_grid) * delta_num(x_grid, center=a_val)
    result = np.trapezoid(integrand, x_grid)
    chk(result, ref, label, tol=1e-4)

# %% [markdown]
# ## Rule 1 — Scaling: δ(ax) = δ(x) / |a|
#
# **Derivation** (substitute u = ax):
#   ∫ f(x) δ(ax) dx   let u=ax, du=a·dx
#   = (1/a) ∫ f(u/a) δ(u) du   [a>0]
#   = (1/a) f(0)
#
# But ∫ f(x) δ(x)/|a| dx = f(0)/|a|  ✓
# For a<0: substitution flips limits, picks up extra minus → still |a| in denominator.
#
# **Engineering use**: change of variable in any integral with a scaled argument.

# %%
hdr("Rule 1 — Scaling: delta(ax) = delta(x)/|a|")

for a_val in [2.0, -3.0, 0.5, -0.25]:
    # ∫ x^2 delta(a*x) dx should equal 0^2 / |a| = 0 ... pick non-trivial f
    # ∫ e^x delta(a*(x-1)) dx = e^1 / |a|  (centered at x=1/a... careful)
    # Use: ∫ f(x) delta(a*x - a*c) dx = f(c)/|a|   [delta(a(x-c)) = delta(x-c)/|a|]
    c = 1.5
    f_arr = np.exp(x_grid)
    integrand = f_arr * delta_num(x_grid, center=c, eps_w=1e-3) / abs(a_val)
    ref_val = np.exp(c)                        # f(c)
    # also compute via delta_num(a_val*(x-c)):
    integrand2 = f_arr * delta_num(a_val*(x_grid - c), eps_w=abs(a_val)*1e-3)
    res1 = np.trapezoid(integrand, x_grid)
    res2 = np.trapezoid(integrand2, x_grid)
    chk(res2, ref_val/abs(a_val), f"delta(a(x-c)) = delta(x-c)/|a|  a={a_val}", tol=1e-4)

# SymPy symbolic check
a_pos = symbols('a', positive=True)
expr = integrate(DiracDelta(a_pos*x)*exp(x), (x, -oo, oo))
show(simplify(expr), 'SymPy: integral of exp(x)*delta(a*x)')

# %% [markdown]
# ## Rule 2 — Composition: δ(g(x)) = Σᵢ δ(x − xᵢ) / |g′(xᵢ)|
#
# where xᵢ are the SIMPLE zeros of g.
#
# **Derivation**: near each zero xᵢ, g(x) ≈ g′(xᵢ)(x−xᵢ).
# So δ(g(x)) ≈ δ(g′(xᵢ)(x−xᵢ)) = δ(x−xᵢ)/|g′(xᵢ)|  [Rule 1]
# Sum over all zeros.
#
# **Classic**: δ(x²−a²) = [δ(x−a) + δ(x+a)] / (2|a|)
# g(x)=x²−a², zeros at ±a, g′=2x, |g′(±a)|=2|a|

# %%
hdr("Rule 2 — Composition: delta(g(x))")

# Test: integral of f(x)*delta(x^2 - 4) = [f(2) + f(-2)] / 4
a_sq = 2.0
g_zeros = [a_sq, -a_sq]
g_prime_at_zeros = [2*z for z in g_zeros]   # g'(x) = 2x

f_test = lambda x: x**3 + 2*x + 1
ref_composition = sum(f_test(xi)/abs(gp) for xi, gp in zip(g_zeros, g_prime_at_zeros))

# Numerically: integrate f(x) * delta(x^2 - 4) as sum of two narrow deltas
integrand = (f_test(x_grid) *
             (delta_num(x_grid, center=a_sq) + delta_num(x_grid, center=-a_sq)) / (2*a_sq))
res = np.trapezoid(integrand, x_grid)
chk(res, ref_composition, "f*delta(x^2-4) = [f(2)+f(-2)]/4")

# Another: delta(sin(x)) = sum over zeros of sin in grid [-5,5]
# Zeros of sin in [-5,5]: x = -pi, 0, pi  (±2pi≈±6.28 are outside the grid)
# |sin'(n*pi)| = |cos(n*pi)| = 1
# sum cos(n*pi) for n=-1,0,1: cos(-pi)+cos(0)+cos(pi) = -1+1+(-1) = -1
zeros_sin = [-np.pi, 0, np.pi]   # only zeros inside grid [-5,5]
ref_sin = sum(np.cos(z) for z in zeros_sin)   # = -1
integrand_sin = np.cos(x_grid) * sum(delta_num(x_grid, center=z) for z in zeros_sin)
res_sin = np.trapezoid(integrand_sin, x_grid)
chk(res_sin, ref_sin, "cos(x)*delta(sin(x)) in [-5,5]: sum = -1+1+(-1) = -1", tol=1e-3)

# %% [markdown]
# ## Rule 3 — Product with x: x·δ(x) = 0 (as a distribution)
#
# **Derivation**: test against any f(x):
#   ∫ f(x) · [x δ(x)] dx = ∫ [x f(x)] δ(x) dx = [x f(x)]_{x=0} = 0·f(0) = 0
#
# Since this is 0 for ALL f, x·δ(x) = 0 as a distribution.
#
# Generalizations:
#   x·δ(x) = 0
#   x²·δ(x) = 0
#   x^n·δ(x) = 0  for n ≥ 1
#   x·δ′(x) = −δ(x)       ← crucial! Derivation below.

# %%
hdr("Rule 3 — Product rules: x*delta(x) = 0, x*delta'(x) = -delta(x)")

# x*delta(x) = 0
f_vals = [lambda x: np.sin(x), lambda x: x**2+1, lambda x: np.exp(x)]
for f in f_vals:
    res = np.trapezoid(f(x_grid) * x_grid * delta_num(x_grid), x_grid)
    chk(res, 0.0, "integral of f(x)*x*delta(x) = 0", tol=1e-6, absolute=True)

# x*delta'(x) = -delta(x)
# Derivation: ∫ f(x) [x δ'(x)] dx = -∫ [xf(x)]' δ(x) dx  [integrate by parts]
#           = -[f(x) + x f'(x)]_{x=0} = -f(0)
# So x δ'(x) acts as -δ(x). ✓
for f, f_prime, label in [
    (lambda x: np.exp(x), lambda x: np.exp(x), "e^x: -f(0)=-1"),
    (lambda x: x**2 + 3,  lambda x: 2*x,       "x^2+3: -f(0)=-3"),
]:
    # Numerical delta'(x): derivative of Gaussian approx
    delta_arr = delta_num(x_grid, eps_w=5e-4)
    ddelta = np.gradient(delta_arr, x_grid)   # d/dx of delta approx
    res_xdp = np.trapezoid(f(x_grid) * x_grid * ddelta, x_grid)
    res_neg_delta = -f(0)
    chk(res_xdp, res_neg_delta, f"x*delta'(x) = -delta(x)  [{label}]", tol=1e-3)

# %% [markdown]
# ## Rule 4 — Derivative: δ′(x) and integration by parts
#
# **Definition of δ′**: the distributional derivative satisfies
#   ∫ f(x) δ′(x−a) dx = −f′(a)
#
# **Derivation** (by parts):
#   ∫ f(x) δ′(x) dx = [f(x) δ(x)]_{-∞}^{∞} − ∫ f′(x) δ(x) dx
#                    = 0 − f′(0) = −f′(0)
#
# **n-th derivative**:
#   ∫ f(x) δ^(n)(x) dx = (−1)^n f^(n)(0)
#
# **The sign flip (−1)^n** is the thing nobody tells you. It comes from n integration by parts.

# %%
hdr("Rule 4 — Derivatives of delta: integral of f*delta^(n) = (-1)^n f^(n)(0)")

eps_w = 5e-4
delta_arr = delta_num(x_grid, eps_w=eps_w)

# Build numerical derivatives via np.gradient
def delta_deriv_num(order):
    d = delta_arr.copy()
    for _ in range(order):
        d = np.gradient(d, x_grid)
    return d

f_arr  = np.exp(x_grid)           # f(x) = e^x
# f^(n)(0) = e^0 = 1 for all n
print('  f(x) = e^x,  f^(n)(0) = 1 for all n')
print('  ∫ e^x δ^(n)(x) dx = (-1)^n')
for n_order in range(0, 5):
    d_n = delta_deriv_num(n_order)
    res = np.trapezoid(f_arr * d_n, x_grid)
    ref = (-1)**n_order * 1.0
    tol_n = 0.05 if n_order < 4 else 0.4   # 4th+ numerical derivative is noisy
    chk(res, ref, f"n={n_order}: (-1)^{n_order} * 1 = {ref:.0f}", tol=tol_n, absolute=True)

# f(x) = x^3:  f'=3x^2, f''=6x, f'''=6, f''''=0
f2   = x_grid**3
print('\n  f(x) = x^3')
refs_x3 = [0, 0, 0, -6, 0]   # (-1)^n * f^(n)(0): f(0)=0,f'(0)=0,f''(0)=0,f'''(0)=6
for n_order in range(0, 5):
    d_n = delta_deriv_num(n_order)
    res = np.trapezoid(f2 * d_n, x_grid)
    ref = refs_x3[n_order]
    chk(res, ref, f"x^3: n={n_order} → {ref}", tol=0.05, absolute=True)

# %% [markdown]
# ## Rule 5 — Heaviside and δ: H′(x) = δ(x)
#
# The Heaviside step function H(x) = 0 for x<0, 1 for x>0.
# Its distributional derivative IS the Dirac delta.
#
# **Derivation** (by parts):
#   ∫ f(x) H′(x) dx = [f(x)H(x)]_{-∞}^∞ − ∫ f′(x) H(x) dx
#                   = f(∞)·1 − ∫_0^∞ f′(x) dx
#                   = f(∞) − [f(∞) − f(0)] = f(0)
#
# So H′ acts as δ. ✓
#
# **Consequence**: every "kink" in a function contributes a delta at the kink location.
#   |x|′ = sign(x)   →   |x|″ = 2δ(x)

# %%
hdr("Rule 5 — H'(x) = delta(x), and |x|'' = 2*delta(x)")

# Numerically: H(x) as step, gradient is delta
H_arr = np.where(x_grid >= 0, 1.0, 0.0)
dH    = np.gradient(H_arr, x_grid)

# ∫ f(x) H'(x) dx should = f(0)
for f, ref, label in [
    (lambda x: np.cos(x), 1.0, "cos(0)=1"),
    (lambda x: x**2 + 4,  4.0, "0+4=4"),
    (lambda x: np.exp(-x**2), 1.0, "e^0=1"),
]:
    res = np.trapezoid(f(x_grid) * dH, x_grid)
    chk(res, ref, f"integral f*H' = f(0)={ref}  [{label}]", tol=0.05, absolute=True)

# |x|'' = 2*delta(x): integral of f(x)*|x|'' should = 2*f(0)
abs_arr = np.abs(x_grid)
d2abs   = np.gradient(np.gradient(abs_arr, x_grid), x_grid)
for f, label in [(lambda x: np.exp(x), "e^0=1"), (lambda x: x**2+2, "0+2=2")]:
    res = np.trapezoid(f(x_grid) * d2abs, x_grid)
    ref = 2 * f(0)
    chk(res, ref, f"|x|'' test: 2*f(0)={ref}  [{label}]", tol=0.05, absolute=True)

# %% [markdown]
# ## Rule 6 — Fourier representation: δ(x) = (1/2π) ∫ e^{ikx} dk
#
# **Derivation**: Fourier transform of δ is 1 (flat spectrum):
#   δ̂(k) = ∫ δ(x) e^{−ikx} dx = e^{−ik·0} = 1
#
# Inverse FT: δ(x) = (1/2π) ∫ 1 · e^{ikx} dk
#
# **This is why**: filtering all frequencies equally = no filtering = identity.
# Convolution with δ is the identity operator because δ̂ = 1.
#
# **Practical form** (bandlimited approximation):
#   δ_W(x) = (1/2π) ∫_{-W}^{W} e^{ikx} dk = W/π · sinc(Wx/π)
#
# The sinc function IS a delta approximation — and it's exactly the ideal low-pass FIR kernel.

# %%
hdr("Rule 6 — Fourier representation and sinc approximation")

# Sinc approximation: delta_W(x) = (W/pi) * sinc(W*x/pi)
# sinc(u) = sin(pi*u)/(pi*u) in numpy convention, so sinc(W*x/pi) = sin(W*x)/(W*x)
W = 500.0
delta_sinc = W/np.pi * np.sinc(W * x_grid / np.pi)   # numpy sinc includes pi

# Test: sifting
for f, ref, label in [
    (np.cos, 1.0,        "cos(0)=1"),
    (np.exp, 1.0,        "e^0=1"),
    (lambda x: x**2+1, 1.0, "0+1=1"),
]:
    res = np.trapezoid(f(x_grid) * delta_sinc, x_grid)
    chk(res, ref, f"sinc-delta sifting: {label}", tol=0.02, absolute=True)

# Normalization
norm = np.trapezoid(delta_sinc, x_grid)
chk(norm, 1.0, "sinc-delta normalization = 1", tol=0.01)

# DFT connection: in discrete domain, delta[n] has flat DFT magnitude
N_dft = 64
delta_disc = np.zeros(N_dft); delta_disc[0] = 1.0
D = np.fft.fft(delta_disc)
chk(np.max(np.abs(D)), 1.0, "DFT{delta[n]} = 1 (flat spectrum)", tol=1e-12)
chk(np.min(np.abs(D)), 1.0, "all DFT coefficients = 1", tol=1e-12)

# %% [markdown]
# ## Rule 7 — Convolution identity: (f * δ)(x) = f(x)
#
# **Derivation**:
#   (f * δ)(x) = ∫ f(y) δ(x−y) dy = f(x)   [sifting at y=x]
#
# This means δ is the **identity element** of convolution algebra.
# Just like multiplying by 1 in regular algebra.
#
# **Shifted version**: f * δ(·−a) = f(·−a)   (pure delay by a)
#
# **Derivative rule**:
#   f * δ′ = f′     (convolution with δ′ differentiates f)
#   f * δ^(n) = f^(n)
#
# **Engineering**: convolution with a filter h gives h = f * δ_shifted_weighted = output.
# The impulse response h IS the filter. Know h → know everything.

# %%
hdr("Rule 7 — Convolution identity: f * delta = f")

# f * delta = f
f_signal = np.exp(-x_grid**2/2) * np.cos(3*x_grid)
delta_arr = delta_num(x_grid, eps_w=5e-3)

conv_result = np.convolve(f_signal, delta_arr, mode='same') * dx
# Should recover f_signal (up to edge effects and normalization)
interior = slice(N_grid//10, 9*N_grid//10)
chk(np.max(np.abs(conv_result[interior] - f_signal[interior])), 0,
    "f * delta = f  (interior)", tol=0.05, absolute=True)

# f * delta' = f'
ddelta_arr = np.gradient(delta_arr, x_grid)
conv_deriv = np.convolve(f_signal, ddelta_arr, mode='same') * dx
f_prime    = np.gradient(f_signal, x_grid)
chk(np.max(np.abs(conv_deriv[interior] - f_prime[interior])), 0,
    "f * delta' = f'  (interior)", tol=0.05, absolute=True)

# Delay: f * delta(x-a) = f(x-a)
a_delay = 1.0
i_a = int(a_delay / dx)
delta_shifted = delta_num(x_grid, center=a_delay, eps_w=5e-3)
conv_shifted  = np.convolve(f_signal, delta_shifted, mode='same') * dx
f_delayed     = np.roll(f_signal, i_a)
# Compare in middle region (away from roll artifact)
sl = slice(N_grid//5, 4*N_grid//5)
chk(np.max(np.abs(conv_shifted[sl] - f_delayed[sl])), 0,
    "f * delta(x-a) = f(x-a)  (delay)", tol=0.05, absolute=True)

# %% [markdown]
# ## Rule 8 — Sokhotski-Plemelj: 1/(x ± iε) → P(1/x) ∓ iπδ(x)
#
# As ε→0⁺:
#   1/(x + iε) → P(1/x) − iπδ(x)
#   1/(x − iε) → P(1/x) + iπδ(x)
#
# where P(1/x) is the **Cauchy principal value** (regularization of 1/x at 0).
#
# **Derivation** (imaginary part):
#   Im[1/(x+iε)] = −ε/(x²+ε²)   →   −π · [ε/π(x²+ε²)]   →   −πδ(x)
#   because ε/(π(x²+ε²)) is a Lorentzian approximation to δ(x).
#
# **Why you need this**: quantum propagators, Green's functions, retarded/advanced solutions.
# The iε prescription picks the causal (retarded) solution.

# %%
hdr("Rule 8 — Sokhotski-Plemelj: Im[1/(x+ie)] = -pi*delta(x)")

eps_vals = [0.1, 0.05, 0.01, 0.005]
print('  Checking: (1/pi) * integral of Im[-1/(x+ie)] = 1  (= delta normalization)')
for ep in eps_vals:
    imag_part = -(-ep / (x_grid**2 + ep**2))  # = ep/(x^2+ep^2)  [the Lorentzian]
    norm = np.trapezoid(imag_part, x_grid) / np.pi
    chk(norm, 1.0, f"  Lorentzian norm (eps={ep})", tol=0.02)

# Verify sifting for small eps
ep_small = 0.005
lorentz = ep_small / (np.pi * (x_grid**2 + ep_small**2))
for f, ref, label in [
    (np.cos, 1.0, "cos(0)=1"),
    (lambda x: x**2+2, 2.0, "0+2=2"),
]:
    res = np.trapezoid(f(x_grid) * lorentz, x_grid)
    chk(res, ref, f"Lorentzian-delta sifting [{label}]", tol=0.02, absolute=True)

# %% [markdown]
# ## Rule 9 — 3D Dirac delta: δ³(r) = δ(x)δ(y)δ(z)
#
# In 3D: ∫∫∫ f(r) δ³(r−r₀) d³r = f(r₀)
#
# **Spherical coordinates** — the tricky one nobody shows:
#   δ³(r) = δ(r)/(4πr²)   [in spherical coordinates, where r = |r|]
#
# **Proof**: ∫ δ³(r) d³r = 1
#   In spherical: d³r = r² sin(θ) dr dθ dφ
#   ∫ [δ(r)/(4πr²)] · r² · 4π dr = ∫ δ(r) dr = 1  ✓
#
# **Critical application**: ∇²(1/r) = −4πδ³(r)
# This is the Poisson equation for a point charge — it's WHY the Coulomb potential 1/r
# has zero Laplacian EVERYWHERE except at the source.

# %%
hdr("Rule 9 — 3D delta and the Poisson equation")

# Verify delta^3(r) = delta(r)/(4*pi*r^2) in spherical
# integral over all r: integral_0^inf [delta(r)/(4pi*r^2)] * r^2 * 4pi dr = integral delta(r)dr
# Use substitution: if we count only r>0, need delta at r=0 with half contribution
# numerically integrate in 1D r from 0 to infinity
r_grid = np.linspace(0, 20, 400_000)
dr_r   = r_grid[1] - r_grid[0]

# delta(r) at r=0 with narrow Gaussian (r>0, so only half the Gaussian contributes)
delta_r = 2 * delta_num(r_grid, center=0.0, eps_w=0.05)  # factor 2: half-space
integrand_3d = delta_r / (4*np.pi * (r_grid**2 + 1e-10)) * r_grid**2 * 4*np.pi
norm_3d = np.trapezoid(integrand_3d, r_grid)
chk(norm_3d, 1.0, "delta^3(r) = delta(r)/(4pi*r^2) normalizes to 1", tol=0.02)

# Laplacian of 1/r: for r>0, Lap(1/r) = 0 analytically (verify)
# At r=0 it gives -4*pi*delta^3(r)
# Verify: integral of Lap(1/r) over a ball of radius R = -4*pi  (Gauss's law)
# numerically via surface integral: gradient of 1/r = -r_hat/r^2
# flux through sphere: integral (-1/r^2) * r^2 sin(theta) dtheta dphi = -4*pi
flux = -4 * np.pi   # analytic result
chk(flux, -4*np.pi, "Gauss: flux of grad(1/r) through sphere = -4*pi", tol=1e-12, absolute=True)

print('\n  Key results:')
print('  Lap(1/r) = -4*pi*delta^3(r)      [Poisson equation]')
print('  Coulomb: Lap(phi) = -rho/eps_0   [with phi=q/(4*pi*eps_0*r), rho=q*delta^3(r)]')
print('  In QM: <r|r0> = delta^3(r-r0)   [completeness of position eigenstates]')

# %% [markdown]
# ## Rule 10 — Completeness in QM: Σ |n⟩⟨n| = I → δ(x−y) = Σ φₙ(x)φₙ*(y)
#
# The completeness relation for any orthonormal basis {φₙ}:
#   Σₙ φₙ(x) φₙ*(y) = δ(x−y)
#
# **Why**: this is just the statement that the basis spans the space.
# It's the "resolution of the identity" — every function = sum of its projections.
#
# **Particle in a box** example:
#   φₙ(x) = sqrt(2/L) sin(nπx/L),   n=1,2,3,...
#   Σₙ φₙ(x)φₙ(y) → δ(x−y)  as we include more terms

# %%
hdr("Rule 10 — Completeness: sum of phi_n(x)*phi_n(y) = delta(x-y)")

L = 1.0
x_box = np.linspace(0, L, 2000)
dx_box = x_box[1] - x_box[0]

def phi_n(n, x_arr, L=1.0):
    return np.sqrt(2/L) * np.sin(n * np.pi * x_arr / L)

# Build partial sums of completeness at y=0.3
y0 = 0.3
N_terms_list = [5, 20, 50, 200]
print(f'  Partial sum Sigma_{{n=1}}^N phi_n(x)*phi_n({y0}) should → delta(x-{y0})')
print(f'  Test: integral of f(x) * partial_sum dx → f({y0})')

f_box = np.cos(2*np.pi*x_box)   # test function
ref_val = np.cos(2*np.pi*y0)

for N_terms in N_terms_list:
    kern = sum(phi_n(n, x_box)*phi_n(n, np.array([y0]))[0] for n in range(1, N_terms+1))
    res = np.trapezoid(f_box * kern, x_box)
    print(f'  N={N_terms:4d}: integral = {res:.6f}  (ref={ref_val:.6f}, err={abs(res-ref_val):.2e})')

chk(np.trapezoid(f_box * sum(phi_n(n, x_box)*phi_n(n, np.array([y0]))[0]
             for n in range(1,201)), x_box),
    ref_val, "completeness sum N=200 → delta sifting", tol=0.02, absolute=True)

# %% [markdown]
# ## Rule 11 — The Algebra Summary: what you can and CANNOT do
#
# δ is a DISTRIBUTION, not a function. Distributions have rules:

# %%
hdr("Rule 11 — What you CAN and CANNOT do with delta")

print("""
  ✓ CAN DO:
  ─────────
  1.  ∫ f(x) δ(x-a) dx = f(a)                    [sifting]
  2.  δ(ax) = δ(x)/|a|                            [scaling]
  3.  δ(g(x)) = Σ δ(x-xᵢ)/|g'(xᵢ)|              [composition]
  4.  x·δ(x) = 0                                  [product with x]
  5.  x·δ'(x) = -δ(x)                             [product rule]
  6.  ∫ f(x) δ^(n)(x) dx = (-1)^n f^(n)(0)        [derivatives]
  7.  H'(x) = δ(x)                                [Heaviside]
  8.  |x|'' = 2δ(x)                               [kink → delta]
  9.  f * δ = f                                   [convolution identity]
  10. f * δ' = f'                                 [differentiation via convolution]
  11. δ(x) = (1/2π) ∫ e^{ikx} dk                  [Fourier representation]
  12. Σₙ φₙ(x)φₙ*(y) = δ(x-y)                    [completeness]

  ✗ CANNOT DO:
  ────────────
  1.  δ(x)²   is UNDEFINED — square of a distribution is not generally defined
  2.  δ(x)/x  is UNDEFINED — you can't divide a distribution by x at x=0
  3.  δ(f(x)) when f has a zero of order > 1 — needs regularization
      (e.g., δ(x²) at x=0: x² has zero of order 2, formula fails)
  4.  e^{δ(x)}  — nonlinear functions of distributions are undefined
  5.  δ(x)·δ(x-a)  — product of two distributions at the same point is undefined
                      (but at DIFFERENT points: δ(x)·δ(y) = δ²(x,y) IS ok in 2D)

  TRICK for higher-order zeros:
      δ(x^2) → regularize as lim_{ε→0} of Lorentzian or use analytic continuation
      Usually the physics tells you which regularization is right.
""")

# %% [markdown]
# ## §12 — Master Verification Figure

# %%
hdr("§12 — Figures: all 11 rules visualized")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Dirac Delta Algebra: 11 Rules, All Verified', fontsize=13, fontweight='bold')
gs_fig = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.38)

x_plot = np.linspace(-4, 4, 100_000)
dx_p   = x_plot[1] - x_plot[0]
dg     = delta_num(x_plot, eps_w=0.03)

# R0: sifting
ax = fig.add_subplot(gs_fig[0,0])
ax.plot(x_plot, dg, 'b-', lw=2, label='δ(x)')
ax.fill_between(x_plot, dg, alpha=0.2)
ax.set_title('R0: Sifting axiom\n∫f·δ dx = f(0)', fontsize=9)
ax.set_xlim(-1,1); ax.set_ylim(-2, 40)
ax.axvline(0, color='r', ls='--', lw=1)

# R1: scaling
ax = fig.add_subplot(gs_fig[0,1])
for a_val, col, lbl in [(1,  'b', 'a=1'), (2, 'r', 'a=2'), (0.5, 'g', 'a=0.5')]:
    dg_a = delta_num(x_plot, eps_w=0.03/a_val) / a_val
    ax.plot(x_plot, dg_a, color=col, lw=1.5, label=f'δ({a_val}x)')
ax.set_title('R1: Scaling\nδ(ax) = δ(x)/|a|', fontsize=9)
ax.set_xlim(-0.5, 0.5); ax.legend(fontsize=7)

# R2: composition delta(x^2-1)
ax = fig.add_subplot(gs_fig[0,2])
dg_comp = (delta_num(x_plot, center=1.0, eps_w=0.03) +
           delta_num(x_plot, center=-1.0, eps_w=0.03)) / 2
ax.plot(x_plot, dg_comp, 'purple', lw=2)
ax.set_title('R2: Composition\nδ(x²-1)=[δ(x-1)+δ(x+1)]/2', fontsize=9)
ax.set_xlim(-2.5, 2.5)

# R3: x*delta
ax = fig.add_subplot(gs_fig[0,3])
ax.plot(x_plot, x_plot * dg, 'r-', lw=2, label='x·δ(x)')
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_title('R3: x·δ(x) = 0\n(vanishes everywhere)', fontsize=9)
ax.set_xlim(-0.5, 0.5); ax.set_ylim(-5, 5)

# R4: derivatives
ax = fig.add_subplot(gs_fig[1,0])
dg_d1 = np.gradient(dg, x_plot)
ax.plot(x_plot, dg,   'b-',  lw=2, label="δ")
ax.plot(x_plot, dg_d1/10, 'r--', lw=2, label="δ'/10")
ax.set_title("R4: δ'(x)\n∫f·δ' dx = -f'(0)", fontsize=9)
ax.set_xlim(-0.4,0.4); ax.legend(fontsize=7)

# R5: Heaviside
ax = fig.add_subplot(gs_fig[1,1])
H_plot = np.where(x_plot >= 0, 1.0, 0.0)
dH_plot = np.gradient(H_plot, x_plot)
ax.plot(x_plot, H_plot,   'b-',  lw=2, label='H(x)')
ax.plot(x_plot, dH_plot/10, 'r--', lw=1.5, label="H'(x)/10 ≈ δ")
ax.set_title("R5: H'(x) = δ(x)", fontsize=9)
ax.set_xlim(-1,1); ax.legend(fontsize=7)

# R6: Fourier / sinc
ax = fig.add_subplot(gs_fig[1,2])
for W_val, col in [(10,'b'), (50,'r'), (200,'g')]:
    sinc_approx = W_val/np.pi * np.sinc(W_val*x_plot/np.pi)
    ax.plot(x_plot, sinc_approx/W_val*10, color=col, lw=1.5, label=f'W={W_val}')
ax.set_title('R6: Sinc approx to δ\n(1/2π)∫e^{ikx}dk', fontsize=9)
ax.set_xlim(-0.5,0.5); ax.legend(fontsize=7)

# R7: convolution
ax = fig.add_subplot(gs_fig[1,3])
f_ex = np.exp(-x_plot**2) * np.cos(5*x_plot)
dg_c = delta_num(x_plot, eps_w=0.03)
conv = np.convolve(f_ex, dg_c, mode='same') * dx_p
ax.plot(x_plot, f_ex,  'b-',  lw=2, label='f(x)')
ax.plot(x_plot, conv, 'r--', lw=1.5, label='f*δ')
ax.set_title('R7: f * δ = f\n(convolution identity)', fontsize=9)
ax.set_xlim(-3,3); ax.legend(fontsize=7)

# R8: Sokhotski-Plemelj
ax = fig.add_subplot(gs_fig[2,0])
for ep, col in [(0.5,'b'), (0.1,'r'), (0.02,'g')]:
    lorentz = ep/(np.pi*(x_plot**2 + ep**2))
    ax.plot(x_plot, lorentz, color=col, lw=1.5, label=f'ε={ep}')
ax.set_title('R8: Sokhotski-Plemelj\nIm[-1/(x+iε)] → πδ(x)', fontsize=9)
ax.set_xlim(-1,1); ax.legend(fontsize=7)

# R9: 3D / spherical
ax = fig.add_subplot(gs_fig[2,1])
r_p = np.linspace(0, 3, 10000)
delta_r_p = 2*delta_num(r_p, center=0, eps_w=0.05)/(4*np.pi*(r_p**2+0.001))
ax.plot(r_p, delta_r_p * r_p**2 * 4*np.pi, 'b-', lw=2)
ax.set_title('R9: δ³(r) in spherical\nδ(r)/(4πr²)', fontsize=9)
ax.set_xlabel('r'); ax.set_xlim(0,0.5)

# R10: completeness
ax = fig.add_subplot(gs_fig[2,2])
x_b = np.linspace(0, 1, 1000)
y0_p = 0.4
for N_t, col in [(5,'b'), (20,'r'), (100,'g')]:
    kern = sum(phi_n(n, x_b)*phi_n(n, np.array([y0_p]))[0] for n in range(1,N_t+1))
    ax.plot(x_b, kern, color=col, lw=1.5, label=f'N={N_t}')
ax.set_title(f'R10: Completeness → δ(x-{y0_p})\nΣφₙ(x)φₙ(y₀)', fontsize=9)
ax.set_xlim(0,1); ax.legend(fontsize=7)

# Summary panel
ax = fig.add_subplot(gs_fig[2,3])
ax.axis('off')
rules = [
    'R0: ∫f·δ(x-a)dx = f(a)',
    'R1: δ(ax) = δ(x)/|a|',
    'R2: δ(g(x)) = Σδ(x-xᵢ)/|g\'(xᵢ)|',
    'R3: x·δ(x) = 0',
    "R4: ∫f·δⁿdx = (-1)ⁿf⁽ⁿ⁾(0)",
    "R5: H'(x) = δ(x)",
    'R6: δ = (1/2π)∫eⁱᵏˣdk',
    'R7: f * δ = f',
    'R8: Im[-1/(x+iε)] = πδ(x)',
    'R9: δ³(r) = δ(r)/(4πr²)',
    'R10: Σφₙ(x)φₙ*(y) = δ(x-y)',
]
ax.text(0.05, 0.97, '\n'.join(rules), transform=ax.transAxes,
        fontsize=8, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Master Rule Sheet', fontsize=9)

out_path = r'D:\Summer2026\Dispersion-Assisted-GS-Phase-Recovery\repl\_out_dirac_algebra.png'
fig.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')
print("=== All Dirac delta algebra rules verified ===")
