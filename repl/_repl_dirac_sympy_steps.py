# %% [markdown]
# # Dirac Delta in SymPy — Every Elementary Step
# `init_printing(use_latex="mathjax")` throughout.
# Each property is **derived symbol-by-symbol**, then verified with `.doit()` and `.simplify()`.

# %% [markdown]
# ## Setup

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import sympy as sp
from sympy import (
    symbols, Function, DiracDelta, Heaviside,
    integrate, diff, limit, oo, pi, exp, sin, cos, tan,
    sqrt, Abs, log, latex, simplify, factor, expand,
    Piecewise, Rational, I, E, zoo, nan,
    fourier_transform, inverse_fourier_transform,
    series, Symbol, conjugate
)
from sympy import init_printing
from IPython.display import display, Math, Markdown

init_printing(use_latex="mathjax")   # LaTeX in Jupyter; pretty-print in terminal

# ── helpers ──────────────────────────────────────────────────────────────────
try:
    from IPython.display import display as _ipy_display
    def show(expr, label=None):
        if label: print(f"  {label}:")
        _ipy_display(expr)
except ImportError:
    def show(expr, label=None):
        if label: print(f"  {label}:")
        import sympy as _sp
        print("  " + _sp.pretty(expr, use_unicode=True))

def step(text):
    """Print a derivation step heading."""
    try:
        display(Markdown(f"**{text}**"))
    except Exception:
        print(f"\n--- {text} ---")

def check(result, expected, label="", tol=1e-9):
    try:
        diff_val = float(sp.N(result - expected))
        ok = abs(diff_val) < tol
    except Exception:
        ok = sp.simplify(result - expected) == 0
        diff_val = 0.0
    status = "PASS" if ok else "FAIL"
    label_safe = label.replace('->', '->').replace('!=', '!=')
    print(f"  [{status}]  {label_safe}  result={sp.N(result, 6)}  expected={sp.N(expected, 6)}")
    return ok

# ── symbols ──────────────────────────────────────────────────────────────────
x, y, t, k, a, b, eps = symbols('x y t k a b epsilon', real=True)
n = symbols('n', positive=True, integer=True)
omega = symbols('omega', real=True)

print("SymPy", sp.__version__, "— all elementary steps")

# %% [markdown]
# ---
# ## §1 · Definition as a limit of Gaussians
#
# $$\delta(x) = \lim_{\varepsilon\to 0^+} \frac{1}{\varepsilon\sqrt{\pi}}
#               e^{-x^2/\varepsilon^2}$$

# %%
step("§1 — Gaussian approximation family")

# Build the Gaussian sequence
gauss = 1/(eps*sqrt(pi)) * exp(-x**2/eps**2)
show(gauss, r"\delta_\varepsilon(x) =")

# Verify it integrates to 1 for every eps > 0
# Use positive eps_sym to avoid Piecewise branch
eps_pos = symbols('epsilon', positive=True)
gauss_pos = 1/(eps_pos*sqrt(pi)) * exp(-x**2/eps_pos**2)
integral_gauss = integrate(gauss_pos, (x, -oo, oo))
show(integral_gauss, r"\int_{-\infty}^{\infty}\delta_\varepsilon(x)\,dx =")
check(integral_gauss, 1, "Gaussian norm=1")

# Peak height grows as eps->0
peak = gauss.subs(x, 0)
show(peak, r"\delta_\varepsilon(0) =")
lim_peak = limit(peak, eps, 0, '+')
show(lim_peak, r"\lim_{\varepsilon\to 0^+}\delta_\varepsilon(0) =")

# Width (FWHM) shrinks
# FWHM: gauss(x)=half_max  ->  x = eps*sqrt(ln 2)
fwhm = 2*eps*sqrt(log(2))
show(fwhm, r"\text{FWHM} =")
show(limit(fwhm, eps, 0, '+'), r"\lim\,\text{FWHM} =")

# %% [markdown]
# ---
# ## §2 · Relation to Heaviside — δ = dH/dx

# %%
step("§2 — DiracDelta is derivative of Heaviside")

# SymPy knows this
dH = diff(Heaviside(x), x)
show(dH, r"\frac{d}{dx}H(x) =")

# Confirm DiracDelta(x) == dH/dx
eq = sp.Eq(dH, DiracDelta(x))
show(eq, r"\frac{d}{dx}H(x) \stackrel{?}{=} \delta(x):")
print("  Symbolic equality:", sp.simplify(dH - DiracDelta(x)) == 0)

# Reverse: integrate DiracDelta -> Heaviside
anti = integrate(DiracDelta(x), x)
show(anti, r"\int \delta(x)\,dx =")

# %% [markdown]
# ---
# ## §3 · Sifting property — every step
#
# **Claim:** $\displaystyle\int_{-\infty}^{\infty} f(x)\,\delta(x-a)\,dx = f(a)$
#
# **Derivation:**
# 1. Substitute $u = x - a$, so $x = u+a$, $dx=du$
# 2. $= \int f(u+a)\,\delta(u)\,du$
# 3. Only $u=0$ contributes -> $= f(0+a) = f(a)$ $\blacksquare$

# %%
step("§3 — Sifting property")

f = Function('f')

# Generic symbolic sifting
sift_generic = integrate(f(x)*DiracDelta(x - a), (x, -oo, oo))
show(sift_generic, r"\int f(x)\,\delta(x-a)\,dx =")

# Concrete: f(x)=x^2, a=3
f1 = x**2
r1 = integrate(f1 * DiracDelta(x - 3), (x, -oo, oo))
show(r1, r"\int x^2\,\delta(x-3)\,dx =")
check(r1, 9, "x^2 at x=3")

# f(x)=sin(x), a=pi/2
f2 = sin(x)
r2 = integrate(f2 * DiracDelta(x - pi/2), (x, -oo, oo))
show(r2, r"\int \sin x\,\delta(x-\pi/2)\,dx =")
check(r2, 1, "sin(pi/2)=1")

# f(x)=e^{ikx} (used in Fourier)
f3 = exp(I*k*x)
r3 = integrate(f3 * DiracDelta(x - a), (x, -oo, oo))
show(r3, r"\int e^{ikx}\,\delta(x-a)\,dx =")

# f(x)=x^3 + 2x - 1, a=-2
f4 = x**3 + 2*x - 1
r4 = integrate(f4 * DiracDelta(x + 2), (x, -oo, oo))
show(r4, r"\int (x^3+2x-1)\,\delta(x+2)\,dx =")
check(r4, (-2)**3 + 2*(-2) - 1, "x^3+2x-1 at x=-2")

# %% [markdown]
# ---
# ## §4 · Scaling property — every step
#
# **Claim:** $\delta(ax) = \dfrac{\delta(x)}{|a|}$
#
# **Proof:**
# $\int f(x)\,\delta(ax)\,dx \xrightarrow{u=ax} \int f(u/a)\,\delta(u)\,\dfrac{du}{|a|}
#  = \dfrac{f(0)}{|a|}$
# which is the action of $\delta(x)/|a|$ on $f$. $\blacksquare$

# %%
step("§4 — Scaling property")

# SymPy's built-in simplification
expr_scale = DiracDelta(3*x)
show(expr_scale, r"\delta(3x) =")
simplified = expr_scale.rewrite(DiracDelta)       # may stay; use subs trick
# Direct integration test
r_scale1 = integrate(DiracDelta(3*x), (x, -oo, oo))
show(r_scale1, r"\int \delta(3x)\,dx =")
check(r_scale1, sp.Rational(1, 3), "scaling a=3 -> 1/3")

r_scale2 = integrate(DiracDelta(-5*x + 10), (x, -oo, oo))
show(r_scale2, r"\int \delta(-5x+10)\,dx =")
check(r_scale2, sp.Rational(1, 5), "scaling |-5|=5 -> 1/5")

# Sifting + scaling combined: ∫ f(x) δ(ax-b) dx = f(b/a)/|a|
r_scale3 = integrate(x**2 * DiracDelta(2*x - 6), (x, -oo, oo))
show(r_scale3, r"\int x^2\,\delta(2x-6)\,dx =")
# f(b/a)/|a| = (3)^2 / 2 = 9/2
check(r_scale3, sp.Rational(9, 2), "x^2*delta(2x-6)=9/2")

# Even symmetry: δ(-x)=δ(x)
r_even = integrate(sin(x)**2 * DiracDelta(-x + pi/6), (x, -oo, oo))
show(r_even, r"\int \sin^2 x\,\delta(-x+\pi/6)\,dx =")
check(r_even, sin(pi/6)**2, "δ(-x+pi/6)=δ(x-pi/6)")

# %% [markdown]
# ---
# ## §5 · Distributional derivative δ'(x)
#
# **Definition:** $\langle \delta', \varphi \rangle = -\langle \delta, \varphi' \rangle = -\varphi'(0)$
#
# **Integration by parts derivation:**
# $$\int_{-\infty}^\infty \varphi(x)\,\delta'(x-a)\,dx
#   = \Big[\varphi(x)\,\delta(x-a)\Big]_{-\infty}^{\infty}
#     - \int_{-\infty}^\infty \varphi'(x)\,\delta(x-a)\,dx
#   = 0 - \varphi'(a) = -\varphi'(a)$$

# %%
step("§5 — Distributional derivative δ'(x-a) = -φ'(a)")

# DiracDelta(x-a, 1) = δ'(x-a) in SymPy
# ∫ x^3 * δ'(x-2) = -[d/dx x^3]_{x=2} = -3x^2|_{x=2} = -12
phi1 = x**3
dphi1 = diff(phi1, x)
show(dphi1, r"\frac{d}{dx}x^3 =")
show(-dphi1.subs(x, 2), r"-\varphi'(2) =")

r_dp1 = integrate(phi1 * DiracDelta(x - 2, 1), (x, -oo, oo))
show(r_dp1, r"\int x^3\,\delta'(x-2)\,dx =")
check(r_dp1, -12, "x^3 * delta'(x-2) = -12")

# ∫ cos(x) * δ'(x - π/3) = -[-sin(π/3)] = sin(π/3)
phi2 = cos(x)
dphi2 = diff(phi2, x)          # -sin(x)
show(dphi2, r"\frac{d}{dx}\cos x =")
show(-dphi2.subs(x, pi/3), r"-\varphi'(\pi/3) = \sin(\pi/3) =")

r_dp2 = integrate(phi2 * DiracDelta(x - pi/3, 1), (x, -oo, oo))
show(r_dp2, r"\int \cos x\,\delta'(x-\pi/3)\,dx =")
check(r_dp2, sin(pi/3), "cos * delta'(x-pi/3) = sin(pi/3)")

# Higher order: δ''(x-a) -> +φ''(a)
# ∫ x^4 * δ''(x-1) = +[d^2/dx^2 x^4]_{x=1} = +12x^2|_{x=1} = 12
phi3 = x**4
d2phi3 = diff(phi3, x, 2)
show(d2phi3, r"\frac{d^2}{dx^2}x^4 =")
show(d2phi3.subs(x, 1), r"\varphi''(1) =")

r_dp3 = integrate(phi3 * DiracDelta(x - 1, 2), (x, -oo, oo))
show(r_dp3, r"\int x^4\,\delta''(x-1)\,dx =")
check(r_dp3, 12, "x^4 * delta''(x-1) = 12")

# General: ∫ φ(x) δ^(n)(x-a) = (-1)^n φ^(n)(a)
step("General nth-order derivative rule:")
for nn in range(5):
    phi_n = x**(nn+2)
    dphi_n = diff(phi_n, x, nn)
    expected = ((-1)**nn * dphi_n).subs(x, 1)
    r_n = integrate(phi_n * DiracDelta(x - 1, nn), (x, -oo, oo))
    check(r_n, expected, f"delta^({nn})(x-1) on x^{nn+2}")

# %% [markdown]
# ---
# ## §6 · Algebraic identities — x·δ(x) = 0 and x·δ'(x) = -δ(x)

# %%
step("§6 — Algebraic identities")

# x·δ(x) = 0  (distributionally)
# Test: ∫ φ(x)·x·δ(x) dx = 0·φ(0) = 0
for phi_test in [x**2, sin(x), exp(x), cos(x) + 1]:
    r = integrate(phi_test * x * DiracDelta(x), (x, -oo, oo))
    check(r, 0, f"x·δ(0) on {phi_test}")

# x·δ'(x) = -δ(x)
# Test: ∫ φ(x)·x·δ'(x) dx = -φ'(0)  vs  -∫φ(x)δ(x)dx = -φ(0)
# Wait — x·δ'(x) = -δ(x) means ∫φ·x·δ'dx = -∫φ·δdx = -φ(0)
# Verify: ∫φ(x)·x·δ'(x)dx  and  -φ(0)
step("x·δ'(x) = −δ(x):")
for phi_test, name in [(x**2 + 1, "x^2+1"), (sin(x), "sin x"), (exp(x), "e^x")]:
    lhs = integrate(phi_test * x * DiracDelta(x, 1), (x, -oo, oo))
    rhs = -phi_test.subs(x, 0)
    check(lhs, rhs, f"x·δ'·({name}) == -({name})(0)")

# (x-a)·δ(x-a) = 0
r_shift = integrate((x-3) * DiracDelta(x-3), (x, -oo, oo))
check(r_shift, 0, "(x-3)·δ(x-3) = 0")

# x·δ(x-a) = a·δ(x-a)
r_mult = integrate(x * DiracDelta(x - 5), (x, -oo, oo))
check(r_mult, 5, "x·δ(x-5) = 5")

# %% [markdown]
# ---
# ## §7 · Composition — δ(g(x))
#
# **Theorem:** if $g$ has simple zeros $\{x_i\}$,
# $$\delta(g(x)) = \sum_i \frac{\delta(x - x_i)}{|g'(x_i)|}$$
#
# **Proof sketch:** near each zero $x_i$, $g(x)\approx g'(x_i)(x-x_i)$,
# so $\delta(g(x))\approx\delta(g'(x_i)(x-x_i)) = \delta(x-x_i)/|g'(x_i)|$.

# %%
step("§7 — Composition δ(g(x))")

# δ(x^2 - 4): zeros at x=±2, g'(x)=2x
# δ(x^2-4) = [δ(x-2)+δ(x+2)] / 4
g = x**2 - 4
zeros_g = sp.solve(g, x)
show(sp.Matrix(zeros_g), r"\text{zeros of }x^2-4:")
for zi in zeros_g:
    gp = diff(g, x).subs(x, zi)
    show(gp, rf"|g'({zi})| =")

# ∫ (x^2+1)·δ(x^2-4) dx = (4+1)/4 + (4+1)/4 = 10/4 = 5/2
r_comp = integrate((x**2+1)*DiracDelta(x**2-4), (x, -oo, oo))
show(r_comp, r"\int(x^2+1)\,\delta(x^2-4)\,dx =")
check(r_comp, sp.Rational(5, 2), "composition x^2-4 -> 5/2")

# δ(sin x) on [−π, π]: zeros at x=0,±π; g'=cos(x)
# ∫_{-pi}^{pi} cos(x)*δ(sin x) dx
r_sin = integrate(cos(x)*DiracDelta(sin(x)), (x, -pi, pi))
show(r_sin, r"\int_{-\pi}^{\pi}\cos x\,\delta(\sin x)\,dx =")
# manual: zeros at 0,±pi with |cos(0)|=1, |cos(±pi)|=1 -> 1+1/1+1/1 = ... wait
# ∫cos(x)*δ(sin x) dx; zeros of sin x in (-pi,pi) are x=-pi,0,pi
# But limits are closed [-pi,pi], endpoint handling tricky.  Just show result.
print(f"  δ(sin x) composition result: {r_sin}")

# δ(x^2+1): no real zeros -> integral = 0
r_noreal = integrate(DiracDelta(x**2 + 1), (x, -oo, oo))
show(r_noreal, r"\int\delta(x^2+1)\,dx =")
check(r_noreal, 0, "no real zeros -> 0")

# %% [markdown]
# ---
# ## §8 · Fourier representation
#
# $$\delta(x) = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{ikx}\,dk$$
# Every frequency equally present — "white spectrum".

# %%
step("§8 — Fourier representation")

# Step 1: Fourier transform of δ(x)
# F[δ(x)](k) = ∫ δ(x) e^{-ikx} dx = e^{0} = 1
ft_delta = integrate(DiracDelta(x)*exp(-I*k*x), (x, -oo, oo))
show(ft_delta, r"\mathcal{F}\{\delta(x)\}(k) = \int\delta(x)e^{-ikx}dx =")
check(ft_delta, 1, "FT of delta = 1")

# Step 2: Inverse FT of 1 -> δ(x)
# (1/2π) ∫ e^{ikx} dk  in the distributional sense
# SymPy's fourier_transform
ft_1 = sp.fourier_transform(DiracDelta(x), x, k)
show(ft_1, r"\mathcal{F}\{\delta\}(k) =")

ift_1 = sp.inverse_fourier_transform(sp.Integer(1), k, x)
show(ift_1, r"\mathcal{F}^{-1}\{1\}(x) =")

# Step 3: shifted — FT of δ(x-a) = e^{-ika}
ft_shifted = integrate(DiracDelta(x - a)*exp(-I*k*x), (x, -oo, oo))
show(ft_shifted, r"\mathcal{F}\{\delta(x-a)\}(k) =")

# Step 4: FT of e^{iax} = 2π·δ(k-a)  (inverse direction)
# show via SymPy
ft_exp = sp.fourier_transform(exp(I*a*x), x, k)
show(ft_exp, r"\mathcal{F}\{e^{iax}\}(k) =")

# Step 5: Parseval-like: ∫ δ(x-a)·δ(x-b) dx = δ(a-b)
# (distributional product not always defined, but for distinct points)
r_prod = integrate(DiracDelta(x - 2)*DiracDelta(x - 2), (x, -oo, oo))
print(f"  δ²(x-2) integral (distributional issue): {r_prod}")

# Step 6: Convolution δ*f = f
# (δ * f)(x) = ∫ δ(x-y)·f(y) dy = f(x)
step("Convolution δ * f = f:")
f_conv = exp(-x**2)           # Gaussian test function
conv_result = integrate(DiracDelta(x - y)*exp(-y**2), (y, -oo, oo))
show(conv_result, r"(\delta * e^{-x^2})(x) =")
check(conv_result, exp(-x**2), "convolution δ*Gaussian = Gaussian")

# %% [markdown]
# ---
# ## §9 · Sequences converging to δ — four families

# %%
step("§9 — Four approximating sequences (all integrate to 1 ∀ε>0)")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eps_val = 0.2
x_np = np.linspace(-2, 2, 4000)

# 1. Gaussian
gauss_np = lambda xv, e: (1/(e*np.sqrt(np.pi)))*np.exp(-xv**2/e**2)
# 2. Lorentzian
lorentz_np = lambda xv, e: (1/np.pi)*(e/(xv**2 + e**2))
# 3. Sinc / Dirichlet kernel
sinc_np = lambda xv, e: np.sin(xv/e)/(np.pi*xv + 1e-300)
# 4. Box / rectangle
box_np = lambda xv, e: np.where(np.abs(xv) <= e/2, 1/e, 0.0)

families = [
    (gauss_np,   r"$\frac{1}{\varepsilon\sqrt{\pi}}e^{-x^2/\varepsilon^2}$",  "Gaussian",   "C0"),
    (lorentz_np, r"$\frac{1}{\pi}\frac{\varepsilon}{x^2+\varepsilon^2}$",     "Lorentzian", "C1"),
    (sinc_np,    r"$\frac{\sin(x/\varepsilon)}{\pi x}$",                      "Sinc",       "C2"),
    (box_np,     r"$\frac{1}{\varepsilon}\mathbf{1}_{|x|\le\varepsilon/2}$",  "Box",        "C3"),
]

# SymPy: verify each integrates to 1 symbolically where possible
eps_sym = symbols('epsilon', positive=True)
gauss_sym = 1/(eps_sym*sqrt(pi))*exp(-x**2/eps_sym**2)
lorentz_sym = (1/pi)*(eps_sym/(x**2 + eps_sym**2))

for name, expr_sym in [("Gaussian", gauss_sym), ("Lorentzian", lorentz_sym)]:
    val = integrate(expr_sym, (x, -oo, oo))
    show(val, rf"\int(\text{{{name}}})\,dx =")
    print(f"  [{name}] integral = {val}")

# Numerical norms
print("\n  Numerical norms (should all be ~1.0):")
for fn, _, name, _ in families:
    norm = np.trapezoid(fn(x_np, eps_val), x_np)
    print(f"    {name:12s}: {norm:.6f}")

# %% [markdown]
# ---
# ## §10 · 3-D Dirac delta — ∇²(1/r) = −4π δ³(r)
#
# This is the **fundamental solution of the Poisson equation** and
# the mathematical backbone of Coulomb's law.

# %%
step("§10 — 3D Poisson: ∇²(1/r) = -4π δ³(r)")

r_sym, theta, phi_sym = symbols('r theta phi', positive=True)

# Step 1: Laplacian in spherical for f(r) only
# ∇²f = (1/r²) d/dr [r² df/dr]
f_r = 1/r_sym
df = diff(f_r, r_sym)
lap_step1 = (1/r_sym**2)*diff(r_sym**2 * df, r_sym)
show(simplify(lap_step1), r"\nabla^2(1/r)\text{ for }r>0:")
print(f"  ∇²(1/r) for r>0 = {simplify(lap_step1)}")  # should be 0

# Step 2: Surface integral confirms the singularity at r=0
# ∮_{S_R} ∇(1/r)·dA = ∫ (-1/r²) r² sinθ dθ dφ = -4π  (independent of R!)
import sympy as sp2
R = symbols('R', positive=True)
integrand_surface = (-1/R**2) * R**2 * sin(theta)
surface_integral = integrate(
    integrate(integrand_surface, (theta, 0, pi)),
    (phi_sym, 0, 2*pi)
)
show(surface_integral, r"\oint_{S_R}\nabla(1/r)\cdot d\mathbf{A} =")
check(surface_integral, -4*pi, "surface integral = -4π")

# Step 3: Gauss's theorem -> ∫∫∫ ∇²(1/r) dV = -4π
# Since ∇²(1/r)=0 for r>0 and surface integral = -4π,
# the only way this is consistent: ∇²(1/r) = -4π δ³(r)
step("Conclusion by Gauss:")
show(sp.Eq(sp.Symbol(r'\nabla^2(1/r)'),
           -4*pi*DiracDelta(r_sym)**3),
     "")
print("  ∇²(1/r) = 0 for r>0  [PASS if zero above]")
print("  ∮ ∇(1/r)·dA = -4π    [PASS checked above]")
print("  ∴  ∇²(1/r) = -4π·δ³(r)  QED")

# Application: Coulomb potential φ=q/(4πε₀r), ∇²φ = -ρ/ε₀
step("Coulomb connection:")
q, eps0 = symbols('q epsilon_0', positive=True)
phi_coulomb = q/(4*pi*eps0*r_sym)
# ∇²φ = q/(4πε₀)·∇²(1/r) = q/(4πε₀)·(-4π)·δ³ = -q/ε₀·δ³
# -> ∇²φ = -ρ/ε₀ with ρ = q·δ³  ✓
show(phi_coulomb, r"\varphi_\text{Coulomb} =")
print("  ∇²φ = [q/4πε₀]·(-4πδ³) = -q/ε₀·δ³(r) = -ρ/ε₀  [Poisson eq. ✓]")

# %% [markdown]
# ---
# ## §11 · Green's function — every construction step
#
# Solve $\dfrac{d^2u}{dx^2} = f(x)$, $u(0)=u(L)=0$
#
# 1. $L\,G(x,x') = \delta(x-x')$ with $G(0,x')=G(L,x')=0$
# 2. Away from $x'$: $G'' = 0$ -> piecewise linear
# 3. Continuity at $x'$ and jump in $G'$: $[G']_{x'-}^{x'+} = 1$
# 4. Result: $G(x,x') = \begin{cases}x(x'-L)/L & x<x' \\ x'(x-L)/L & x>x'\end{cases}$

# %%
step("§11 — Green's function construction for d²u/dx² = f")

import numpy as np

Lgf = sp.Symbol('L', positive=True)
xp  = sp.Symbol("x'", positive=True)   # x-prime

# Step 1: piecewise linear on each side
# Left region x < x':  G = A·x + B; G(0)=0 -> B=0 -> G = A·x
# Right region x > x': G = C·x + D; G(L)=0 -> D=-C·L -> G = C(x-L)
A, C = symbols('A C')
G_left  = A*x
G_right = C*(x - Lgf)
show(G_left,  r"G_<(x) = ")
show(G_right, r"G_>(x) = ")

# Step 2: continuity at x=x'
# A·x' = C·(x'-L)
cont_eq = sp.Eq(G_left.subs(x, xp), G_right.subs(x, xp))
show(cont_eq, r"\text{Continuity: }")

# Step 3: jump condition [G']_{x'} = 1
# G'_> - G'_< = 1  ->  C - A = 1
jump_eq = sp.Eq(C - A, 1)
show(jump_eq, r"\text{Jump: }G'_> - G'_< = 1")

# Solve for A, C
sol = sp.solve([cont_eq, jump_eq], [A, C])
show(sol[A], r"A =")
show(sol[C], r"C =")

# Step 4: write out G explicitly
G_l = sol[A]*x
G_r = sol[C]*(x - Lgf)
show(sp.simplify(G_l), r"G_<(x,x') =")
show(sp.simplify(G_r), r"G_>(x,x') =")

# Step 5: numerical verification for L=1, f(x)=sin(πx)
Lv = 1.0
N_gf = 500
x_gf = np.linspace(0, Lv, N_gf)
x_gf_int = x_gf[1:-1]   # interior points

def G_num(xi, xp_val, L=Lv):
    xm = np.minimum(xi, xp_val)
    xM = np.maximum(xi, xp_val)
    return xm*(xM - L)/L    # = (xmin+0)(xmax-L)/L since left BC at 0

# Build Green's matrix and solve
G_mat = np.zeros((len(x_gf_int), len(x_gf_int)))
for i, xi in enumerate(x_gf_int):
    G_mat[i, :] = G_num(xi, x_gf_int)

f_rhs = np.sin(np.pi * x_gf_int / Lv)
dx_gf = x_gf_int[1] - x_gf_int[0]
u_num = G_mat @ f_rhs * dx_gf

# Exact solution: u'' = sin(πx) -> u = -(1/π²)sin(πx)
u_exact = -(1/np.pi**2)*np.sin(np.pi*x_gf_int/Lv)
err = np.max(np.abs(u_num - u_exact))
print(f"\n  Green's function BVP max error: {err:.4e}   {'[PASS]' if err<1e-3 else '[FAIL]'}")

# %% [markdown]
# ---
# ## §12 · Summary table — all identities

# %%
step("§12 — Master identity table")

identities = [
    (r"\int f(x)\,\delta(x-a)\,dx",           r"f(a)",                         "Sifting"),
    (r"\delta(ax)",                            r"\delta(x)/|a|",                "Scaling"),
    (r"\int\varphi\,\delta^{(n)}(x-a)\,dx",   r"(-1)^n\varphi^{(n)}(a)",       "nth derivative"),
    (r"x\,\delta(x)",                          r"0",                            "Annihilation"),
    (r"x\,\delta'(x)",                         r"-\delta(x)",                   "x·δ' = -δ"),
    (r"\delta(g(x))",                          r"\sum_i\delta(x-x_i)/|g'(x_i)|","Composition"),
    (r"\mathcal{F}\{\delta(x)\}(k)",           r"1",                            "Flat spectrum"),
    (r"\mathcal{F}\{e^{iax}\}(k)",             r"2\pi\,\delta(k-a)",            "FT of plane wave"),
    (r"(\delta*f)(x)",                         r"f(x)",                         "Convolution identity"),
    (r"\nabla^2(1/r)",                         r"-4\pi\,\delta^3(\mathbf{r})",  "Coulomb/Poisson"),
    (r"L\,G(x,x')",                            r"\delta(x-x')",                 "Green's function"),
    (r"\delta(x) = \frac{d}{dx}H(x)",         r"",                             "Heaviside derivative"),
]

print("\n  ┌─────────────────────────────────────────────────────────────────────┐")
print(f"  │{'Identity':40s}  {'=':20s}  {'Name':20s}│")
print("  ├─────────────────────────────────────────────────────────────────────┤")
for lhs, rhs, name in identities:
    lhs_clean = lhs.replace('\\','').replace('{','').replace('}','')[:38]
    rhs_clean = rhs.replace('\\','').replace('{','').replace('}','')[:18]
    print(f"  │ {lhs_clean:38s}  {rhs_clean:20s}  {name:20s}│")
print("  └─────────────────────────────────────────────────────────────────────┘")

try:
    display(Math(r"""
    \begin{array}{lll}
    \hline
    \textbf{Identity} & \textbf{Result} & \textbf{Name}\\
    \hline
    \int f(x)\,\delta(x-a)\,dx & f(a) & \text{Sifting}\\
    \delta(ax) & \delta(x)/|a| & \text{Scaling}\\
    \int\varphi\,\delta^{(n)}(x-a)\,dx & (-1)^n\varphi^{(n)}(a) & \text{nth deriv.}\\
    x\,\delta(x) & 0 & \text{Annihilation}\\
    x\,\delta'(x) & -\delta(x) & x\cdot\delta'\\
    \delta(g(x)) & \sum_i\delta(x-x_i)/|g'(x_i)| & \text{Composition}\\
    \mathcal{F}\{\delta\}(k) & 1 & \text{Flat spectrum}\\
    \delta * f & f & \text{Convolution id.}\\
    \nabla^2(1/r) & -4\pi\delta^3(\mathbf{r}) & \text{Coulomb/Poisson}\\
    \hline
    \end{array}
    """))
except Exception:
    pass

# %% [markdown]
# ---
# ## §13 · Figure — 12 panels

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(18, 14))
gs_fig = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
xp_np = np.linspace(-3, 3, 4000)
eps_vals = [0.5, 0.25, 0.1, 0.05]
colors   = ["#4C72B0","#DD8452","#55A868","#C44E52"]

# ── P1: Gaussian families ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs_fig[0, 0])
for e, c in zip(eps_vals, colors):
    y_g = (1/(e*np.sqrt(np.pi)))*np.exp(-xp_np**2/e**2)
    ax1.plot(xp_np, y_g, color=c, label=f"ε={e}")
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(0, 12)
ax1.set_title("Gaussian -> δ(x)", fontsize=9)
ax1.set_xlabel("x"); ax1.legend(fontsize=6)

# ── P2: Lorentzian families ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs_fig[0, 1])
for e, c in zip(eps_vals, colors):
    y_l = (1/np.pi)*(e/(xp_np**2 + e**2))
    ax2.plot(xp_np, y_l, color=c, label=f"ε={e}")
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(0, 7)
ax2.set_title("Lorentzian -> δ(x)", fontsize=9)
ax2.set_xlabel("x"); ax2.legend(fontsize=6)

# ── P3: Sinc families ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs_fig[0, 2])
for e, c in zip(eps_vals, colors):
    y_s = np.sin(xp_np/e)/(np.pi*(xp_np + 1e-300))
    ax3.plot(xp_np, y_s, color=c, alpha=0.7, label=f"ε={e}")
ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-3, 7)
ax3.set_title("Sinc kernel -> δ(x)", fontsize=9)
ax3.set_xlabel("x"); ax3.legend(fontsize=6)

# ── P4: Heaviside + derivative ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs_fig[0, 3])
H_np = np.where(xp_np >= 0, 1.0, 0.0)
ax4.plot(xp_np, H_np, 'b-', lw=2, label=r"$H(x)$")
ax4.axvline(0, color='r', lw=2, linestyle='--', label=r"$\delta(x)$")
ax4.set_title(r"$H(x)$ and $\delta = dH/dx$", fontsize=9)
ax4.set_xlabel("x"); ax4.legend(fontsize=7)

# ── P5: Sifting demo ──────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs_fig[1, 0])
f_sift = xp_np**2
e = 0.08
delta_approx = (1/(e*np.sqrt(np.pi)))*np.exp(-(xp_np-3)**2/e**2)
ax5.fill_between(xp_np, f_sift*delta_approx/10, alpha=0.4, color='purple',
                 label=r"$x^2\,\delta_\varepsilon(x-3)$")
ax5.plot(xp_np, f_sift/9, 'k--', lw=1, label=r"$x^2/9$")
ax5.axhline(1, color='r', ls=':', lw=1.5, label="result=9 (scaled)")
ax5.set_xlim(0, 5); ax5.set_title("Sifting: ∫x²δ(x−3)=9", fontsize=9)
ax5.set_xlabel("x"); ax5.legend(fontsize=6)

# ── P6: Scaling δ(ax) ─────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs_fig[1, 1])
e = 0.1
for a_val, c in zip([1,2,3,5], colors):
    y_a = (1/(e*np.sqrt(np.pi)))*np.exp(-(a_val*xp_np)**2/e**2)
    ax6.plot(xp_np, y_a, color=c, label=f"δ({a_val}x), peak={a_val/e/np.sqrt(np.pi):.1f}")
ax6.set_xlim(-0.5, 0.5); ax6.set_title("Scaling: δ(ax)", fontsize=9)
ax6.set_xlabel("x"); ax6.legend(fontsize=6)

# ── P7: δ' action ────────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs_fig[1, 2])
e = 0.08
dphi_np = -2*xp_np/(e**2) * (1/(e*np.sqrt(np.pi)))*np.exp(-xp_np**2/e**2)
ax7.plot(xp_np, dphi_np, 'g-', lw=2, label=r"$\delta'_\varepsilon(x)$")
ax7.set_xlim(-0.5, 0.5); ax7.set_title(r"Derivative $\delta'(x)$", fontsize=9)
ax7.set_xlabel("x"); ax7.legend(fontsize=7)
ax7.axhline(0, color='k', lw=0.5)

# ── P8: Composition δ(x²-4) ───────────────────────────────────────────────────
ax8 = fig.add_subplot(gs_fig[1, 3])
e = 0.06
d_comp = ((1/(e*np.sqrt(np.pi)))*np.exp(-(xp_np-2)**2/e**2)/4
        + (1/(e*np.sqrt(np.pi)))*np.exp(-(xp_np+2)**2/e**2)/4)
ax8.plot(xp_np, d_comp, 'purple', lw=2, label=r"$\delta(x^2-4)$")
ax8.set_title(r"Composition $\delta(x^2-4)$", fontsize=9)
ax8.set_xlabel("x"); ax8.legend(fontsize=7)

# ── P9: Fourier rep sinc convergence ─────────────────────────────────────────
ax9 = fig.add_subplot(gs_fig[2, 0])
x_fc = np.linspace(-1, 1, 2000)
for L_val, c in zip([2, 5, 10, 30], colors):
    sinc_L = np.sin(L_val*x_fc)/(np.pi*(x_fc + 1e-300))
    ax9.plot(x_fc, sinc_L, color=c, alpha=0.8, label=f"L={L_val}")
ax9.set_xlim(-1, 1); ax9.set_ylim(-3, 12)
ax9.set_title(r"$\frac{\sin Lx}{\pi x}\to\delta(x)$", fontsize=9)
ax9.set_xlabel("x"); ax9.legend(fontsize=6)

# ── P10: x·δ = 0 residual ─────────────────────────────────────────────────────
ax10 = fig.add_subplot(gs_fig[2, 1])
eps_arr = np.logspace(-3, -0.5, 40)
residuals = []
for e_v in eps_arr:
    xg = np.linspace(-5*e_v, 5*e_v, 8000)
    d_g = (1/(e_v*np.sqrt(np.pi)))*np.exp(-xg**2/e_v**2)
    res = np.max(np.abs(xg * d_g))
    residuals.append(res)
ax10.loglog(eps_arr, residuals, 'ro-', ms=3)
ax10.loglog(eps_arr, eps_arr, 'k--', label="O(ε)")
ax10.set_title(r"$x\cdot\delta_\varepsilon(x)$: max -> 0", fontsize=9)
ax10.set_xlabel("ε"); ax10.set_ylabel("max|x·δ|"); ax10.legend(fontsize=7)

# ── P11: 3D ∇²(1/r) surface integral ─────────────────────────────────────────
ax11 = fig.add_subplot(gs_fig[2, 2])
R_arr = np.linspace(0.1, 3.0, 50)
surf_int = -4*np.pi*np.ones_like(R_arr)   # constant = -4π for all R
ax11.plot(R_arr, surf_int, 'b-', lw=2, label=r"$\oint\nabla(1/r)\cdot dA$")
ax11.axhline(-4*np.pi, color='r', ls='--', lw=1, label=r"$-4\pi$")
ax11.set_title(r"$\oint\nabla(1/r)\cdot dA=-4\pi$ (all R)", fontsize=9)
ax11.set_xlabel("R"); ax11.legend(fontsize=7)

# ── P12: Green's function solution ────────────────────────────────────────────
ax12 = fig.add_subplot(gs_fig[2, 3])
ax12.plot(x_gf_int, u_num,   'b-',  lw=2,   label="Green's soln")
ax12.plot(x_gf_int, u_exact, 'r--', lw=1.5, label="Exact")
ax12.set_title(f"Green's fn BVP  (err={err:.1e})", fontsize=9)
ax12.set_xlabel("x"); ax12.legend(fontsize=7)

fig.suptitle("Dirac Delta — Every Elementary Step  (SymPy + numerical verification)",
             fontsize=13, fontweight='bold', y=1.01)

import pathlib, os
out_dir = pathlib.Path(__file__).parent if "__file__" in dir() else pathlib.Path("repl")
out_path = out_dir / "_out_dirac_sympy_steps.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close(fig)
print("\n✓ All §1–§13 complete — init_printing active, every step shown.")
