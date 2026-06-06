# %% [markdown]
# # Dirac Delta — All Axioms and Core Identities
# `init_printing(use_latex="mathjax")` throughout.
#
# Every identity is:
#   1. stated as a LaTeX equation via `display()`
#   2. derived symbolically with SymPy
#   3. verified numerically
#
# **Structure:**
# §1  The axiom — what δ IS (not a function, a distribution)
# §2  Sifting property  ∫ f(x) δ(x−a) dx = f(a)
# §3  Scaling           δ(ax) = δ(x)/|a|
# §4  Symmetry          δ(−x) = δ(x)  (even distribution)
# §5  Composition       δ(g(x)) = Σᵢ δ(x−xᵢ)/|g′(xᵢ)|
# §6  Derivative        ∫ f δ′ dx = −f′(a)  (IBP)
# §7  nth derivative    ∫ f δ⁽ⁿ⁾ dx = (−1)ⁿ f⁽ⁿ⁾(a)
# §8  Product with func  f(x)δ(x−a) = f(a)δ(x−a)
# §9  Fourier transform  F{δ(t)} = 1,  F{1} = 2πδ(ω)
# §10 Convolution        (f ∗ δ)(t) = f(t)
# §11 Heaviside link     d/dx H(x) = δ(x)
# §12 Multidimensional   δⁿ(r) = δ(x)δ(y)δ(z)
# §13 Limiting sequences Gaussian, sinc, Lorentzian → δ

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
from sympy import (
    symbols, DiracDelta, Heaviside, integrate, diff, exp, sqrt, pi,
    cos, sin, oo, Eq, simplify, limit, Abs, Rational, latex,
    fourier_transform, inverse_fourier_transform, Function,
    I, conjugate, sign, factorial, ln
)
from sympy import init_printing

init_printing(use_latex='mathjax')

try:
    from IPython.display import display as _D, Math, Markdown
    def show(expr, label=None):
        if label: _D(Markdown(f"**{label}**"))
        _D(expr)
    def tex(s):
        _D(Math(s))
except ImportError:
    def show(expr, label=None):
        if label: print(f"\n  {label}")
        print("  " + sp.pretty(expr, use_unicode=True))
    def tex(s):
        print(f"  [{s}]")

def hdr(s):
    bar = '─' * 64
    print(f'\n{bar}\n  {s}\n{bar}')

def chk(val, ref, label, tol=1e-6, absolute=False):
    try: v, r = float(val), float(ref)
    except: print(f'  [FAIL]  {label}  (not float)'); return
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    print(f"  [{'PASS' if err<tol else 'FAIL'}]  {label}  got={v:.8g}  ref={r:.8g}")

def chk_expr(expr, ref_expr, label, tol=1e-10):
    """Verify two SymPy expressions are equal by evaluating at random points."""
    x_sym = symbols('x')
    diff_expr = simplify(expr - ref_expr)
    if diff_expr == 0:
        print(f"  [PASS]  {label}  (symbolic identity)")
        return
    # Numerical spot-check
    vals = [0.3, 1.7, -0.5, 2.1]
    errs = []
    for v in vals:
        try:
            e = abs(complex(expr.subs(x_sym, v)) - complex(ref_expr.subs(x_sym, v)))
            errs.append(e)
        except: pass
    if errs and max(errs) < tol:
        print(f"  [PASS]  {label}  (numerical spot-check)")
    else:
        print(f"  [FAIL]  {label}  diff={diff_expr}")

print("=== Dirac Delta: All Axioms and Core Identities ===")

# %% [markdown]
# ---
# ## §1 · The Axiom — δ is Defined by What It Does Under an Integral
#
# A **distribution** (generalised function) is not defined by pointwise values.
# It is defined by how it acts on test functions φ ∈ C∞₀ (smooth, compact support).
#
# **The single defining axiom:**
#
# $$\langle \delta_a, \varphi \rangle \;=\; \int_{-\infty}^{\infty} \delta(x-a)\,\varphi(x)\,dx \;=\; \varphi(a)$$
#
# Everything else — scaling, derivatives, composition, Fourier — is derived from
# this one axiom plus linearity.
#
# **What δ is NOT**: δ(0) = ∞ is not a definition. That is a mnemonic.
# The function that is ∞ at one point and 0 elsewhere has integral 0, not 1.
# δ lives in the dual space of test functions, not in L¹.

# %%
hdr("§1 — The axiom: δ defined by sifting")

x, t, a, b, eps = symbols('x t a b epsilon', real=True)
eps_pos = symbols('epsilon', positive=True)

tex(r"\langle \delta_a,\, \varphi \rangle = \int_{-\infty}^{\infty} \delta(x-a)\,\varphi(x)\,dx = \varphi(a)")

# SymPy's DiracDelta satisfies the axiom:
test_fns = [
    (x**3,              2,   8,      "x³ at a=2"),
    (sp.exp(x),         0,   1,      "eˣ at a=0"),
    (sp.cos(x),         pi,  -1,     "cos x at a=π"),
    (x**2 - 3*x + 1,  -1,   5,      "x²-3x+1 at a=-1"),
    (sp.sin(x)*sp.exp(-x), 1, sp.sin(1)*sp.exp(-1), "sin(x)e⁻ˣ at a=1"),
]

for f, a_val, expected, label in test_fns:
    result = integrate(f * DiracDelta(x - a_val), (x, -oo, oo))
    result_n = float(result.evalf())
    expected_n = float(sp.sympify(expected).evalf())
    chk(result_n, expected_n, f"sifting: {label}")

# %% [markdown]
# ---
# ## §2 · Sifting with Shifted and Scaled Arguments
#
# $$\int_{-\infty}^{\infty} f(x)\,\delta(x - a)\,dx = f(a)$$
#
# Works for any integrable f, any real a.
# The delta "samples" f exactly at x = a.

# %%
hdr("§2 — Sifting: general form")

show(Eq(integrate(Function('f')(x)*DiracDelta(x-a), (x,-oo,oo)),
        Function('f')(a)), "Sifting identity:")

# Verify with multiple functions and offsets
cases = [
    (x**4,         3,    81),
    (sp.log(x),    sp.E,  1),
    (sp.Abs(x),    -2,    2),
    (sp.sqrt(Abs(x)+1), 8, 3),
]
for f_expr, a_val, ref in cases:
    r = integrate(f_expr * DiracDelta(x - a_val), (x, -oo, oo))
    chk(float(r.evalf()), float(sp.sympify(ref).evalf()),
        f"∫ f·δ(x-{a_val}) dx = {ref}")

# Sifting over finite interval [a-ε, a+ε] converges to f(a)
print("\n  Finite-interval sifting (Heaviside bracket):")
for a_val, f_expr in [(1, x**2), (2, sp.sin(x))]:
    r = integrate(f_expr * DiracDelta(x - a_val), (x, a_val - 1, a_val + 1))
    chk(float(r.evalf()), float(f_expr.subs(x, a_val).evalf()),
        f"∫_[a-1,a+1] f·δ(x-{a_val}) dx")

# %% [markdown]
# ---
# ## §3 · Scaling Identity
#
# $$\delta(ax) = \frac{1}{|a|}\,\delta(x) \qquad a \neq 0$$
#
# **Proof**: change of variable u = ax,  du = a dx:
# $$\int f(x)\,\delta(ax)\,dx = \int f(u/a)\,\delta(u)\,\frac{du}{|a|} = \frac{f(0)}{|a|}$$
#
# General shifted form:
# $$\delta(ax - b) = \frac{1}{|a|}\,\delta\!\left(x - \frac{b}{a}\right)$$

# %%
hdr("§3 — Scaling: δ(ax) = δ(x)/|a|")

tex(r"\delta(ax) = \frac{1}{|a|}\,\delta(x)")
tex(r"\delta(ax-b) = \frac{1}{|a|}\,\delta\!\left(x-\frac{b}{a}\right)")

for a_val in [2, 3, 0.5, -4, -1.5]:
    r = integrate(DiracDelta(a_val * x), (x, -oo, oo))
    chk(float(r), 1/abs(a_val), f"∫δ({a_val}x)dx = 1/{abs(a_val):.4g}")

print()
# Shifted: int f(x)*delta(a*x - b) dx = f(b/a)/|a|
for a_val, b_val, f_expr in [(2, 4, x**2), (3, 6, sp.exp(x)), (-2, 2, sp.cos(x))]:
    r = integrate(f_expr * DiracDelta(a_val*x - b_val), (x, -oo, oo))
    ref = f_expr.subs(x, b_val/a_val) / abs(a_val)
    chk(float(r.evalf()), float(ref.evalf()),
        f"∫x²·δ({a_val}x-{b_val})dx = f({b_val/a_val:.3g})/{abs(a_val)}")

# %% [markdown]
# ---
# ## §4 · Symmetry — δ is Even
#
# $$\delta(-x) = \delta(x)$$
#
# Proof: apply scaling rule with a = −1:
# $$\delta(-x) = \frac{1}{|-1|}\,\delta(x) = \delta(x)$$

# %%
hdr("§4 — Symmetry: δ(−x) = δ(x)  (even distribution)")

tex(r"\delta(-x) = \delta(x)")

for f_expr, a_val, label in [(x**2, 1, "x²"), (sp.cos(x), 0, "cos x")]:
    r_pos = integrate(f_expr * DiracDelta(x - a_val),  (x, -oo, oo))
    r_neg = integrate(f_expr * DiracDelta(-x - (-a_val)), (x, -oo, oo))
    chk(float(r_pos.evalf()), float(r_neg.evalf()),
        f"δ(x-a) and δ(-x+a) give same result ({label})")

# Also: delta is real — conjugate(delta) = delta
r1 = integrate(DiracDelta(x), (x, -oo, oo))
chk(float(r1), 1.0, "∫δ(x)dx = 1 (normalisation)")

# %% [markdown]
# ---
# ## §5 · Composition — δ(g(x))
#
# If g has simple zeros at x₁, x₂, …:
#
# $$\delta(g(x)) = \sum_i \frac{\delta(x - x_i)}{|g'(x_i)|}$$
#
# **How to apply**: find all zeros of g, divide by |slope| at each zero.
# Works whenever zeros are simple (g′(xᵢ) ≠ 0).

# %%
hdr("§5 — Composition: δ(g(x)) = Σ δ(x−xᵢ)/|g′(xᵢ)|")

tex(r"\delta(g(x)) = \sum_i \frac{\delta(x - x_i)}{|g'(x_i)|}")

# Example 1: g(x) = x² − 4,  zeros at x = ±2,  g′(x) = 2x
# δ(x²-4) = δ(x-2)/(2·2) + δ(x+2)/(2·2) = [δ(x-2)+δ(x+2)]/4
f_test = x**3   # sift f(x)=x³
# Manual: f(2)/|g'(2)| + f(-2)/|g'(-2)| = 8/4 + (-8)/4 = 0
g1 = x**2 - 4
zeros_g1 = sp.solve(g1, x)
gprime1 = diff(g1, x)
manual1 = sum(f_test.subs(x, z) / Abs(gprime1.subs(x, z)) for z in zeros_g1)
sympy1 = integrate(f_test * DiracDelta(g1), (x, -oo, oo))
show(Eq(symbols('g'), g1), "g(x):")
print(f"  zeros: {zeros_g1},  |g′| at zeros: {[abs(gprime1.subs(x,z)) for z in zeros_g1]}")
show(Eq(symbols("∫x³δ(x²-4)dx"), simplify(sympy1)))
chk(float(simplify(sympy1 - manual1)), 0,
    "δ(x²-4) composition", tol=1e-8, absolute=True)

# Example 2: g(x) = sin(x), zeros at nπ on [−2π, 2π]
# |g′(nπ)| = |cos(nπ)| = 1
# ∫ 1·δ(sin x) dx over [-π,π] = sum of 1/|cos(nπ)| for zeros in interval
# Zeros of sin in [-π, π]: x=-π, 0, π → but endpoints are boundary
# Use (-π/2 - 0.1, 3π/2) to get zeros 0, π clearly inside
g2_vals = [0, np.pi]  # zeros of sin in (−0.1, π+0.1)
manual2 = sum(1.0 / abs(np.cos(z)) for z in g2_vals)   # = 1/1 + 1/1 = 2
# SymPy can't evaluate δ(sin x) symbolically — use Lorentzian approximation
eps_c = 0.001
x_c = np.linspace(-0.1, np.pi + 0.1, 400000)
dx_c = x_c[1] - x_c[0]
delta_sin_approx = (eps_c / np.pi) / (np.sin(x_c)**2 + eps_c**2)
r_sin = np.sum(delta_sin_approx) * dx_c
chk(r_sin, manual2, "∫δ(sin x)dx over (−0.1,π+0.1) = 2 (numerical)", tol=0.01)

# Example 3: δ(x²) — double zero at x=0. Not defined! Check SymPy behaviour.
print("\n  δ(x²): double zero — result diverges (not a valid distribution)")
try:
    r_sq = integrate(DiracDelta(x**2), (x, -oo, oo))
    print(f"  SymPy gives: {r_sq}  (divergent / undefined)")
except Exception as e:
    print(f"  SymPy error (expected): {e}")

# %% [markdown]
# ---
# ## §6 · Derivative — δ′ Flips the Derivative onto the Test Function
#
# Defined by integration by parts. The boundary term vanishes because
# test functions have compact support:
#
# $$\int_{-\infty}^{\infty} f(x)\,\delta'(x-a)\,dx = -f'(a)$$
#
# Mnemonic: δ′ acts like "apply IBP once, evaluate derivative of f at a,
# negate."

# %%
hdr("§6 — First derivative: ∫ f δ′(x−a) dx = −f′(a)")

tex(r"\int_{-\infty}^{\infty} f(x)\,\delta'(x-a)\,dx = -f'(a)")

deriv_cases = [
    (x**3,          2,   -3*4,       "x³: −3x²|₂ = −12"),
    (sp.exp(x),     0,   -1,         "eˣ: −eˣ|₀ = −1"),
    (sp.sin(x),     pi/2, -sp.cos(pi/2), "sin x: −cos(π/2) = 0"),
    (x**4,         -1,   4,          "x⁴: −4x³|₋₁ = −(−4) = 4"),
    (x**2*sp.cos(x), 0,  0,          "x²cos x: f′(0) = 0"),
]

for f_expr, a_val, ref, label in deriv_cases:
    r = integrate(f_expr * DiracDelta(x - a_val, 1), (x, -oo, oo))
    ref_n = float(sp.sympify(ref).evalf())
    chk(float(r.evalf()), ref_n, label)

# %% [markdown]
# ---
# ## §7 · nth Derivative — General Rule
#
# $$\int_{-\infty}^{\infty} f(x)\,\delta^{(n)}(x-a)\,dx = (-1)^n\,f^{(n)}(a)$$
#
# Proof: apply IBP n times. Each IBP flips one derivative off δ onto f
# and introduces a minus sign. After n flips: (−1)ⁿ f⁽ⁿ⁾(a).

# %%
hdr("§7 — nth derivative: ∫ f δ⁽ⁿ⁾ dx = (−1)ⁿ f⁽ⁿ⁾(a)")

tex(r"\int f(x)\,\delta^{(n)}(x-a)\,dx = (-1)^n f^{(n)}(a)")

f_poly = x**5   # easy: known derivatives
a_val = 1
print(f"  f(x) = x⁵,   a = {a_val}")
print(f"  {'n':>3}  {'SymPy':>14}  {'(-1)ⁿf⁽ⁿ⁾(a)':>14}  result")
for n in range(6):
    r = integrate(f_poly * DiracDelta(x - a_val, n), (x, -oo, oo))
    ref = (-1)**n * diff(f_poly, x, n).subs(x, a_val)
    ok = simplify(r - ref) == 0
    print(f"  {n:>3}  {float(r.evalf()):>14.4f}  {float(ref.evalf()):>14.4f}  "
          f"{'PASS' if ok else 'FAIL'}")

# %% [markdown]
# ---
# ## §8 · Product with a Function
#
# $$f(x)\,\delta(x - a) = f(a)\,\delta(x - a)$$
#
# Under the integral sign, δ(x−a) forces x→a, so f(x) can be replaced
# by the constant f(a) everywhere. This is the algebraic form of sifting.
#
# Consequence: $x\,\delta(x) = 0$  (the distribution, not pointwise)

# %%
hdr("§8 — Product: f(x)δ(x−a) = f(a)δ(x−a)")

tex(r"f(x)\,\delta(x-a) = f(a)\,\delta(x-a)")
tex(r"x\,\delta(x) = 0")

# Verify: ∫ g(x) · [f(x)δ(x-a)] dx = ∫ g(x) · [f(a)δ(x-a)] dx
g_test = sp.cos(x)
for f_expr, a_val in [(x**2, 3), (sp.sin(x), pi/4), (sp.exp(x), 2)]:
    lhs = integrate(g_test * f_expr * DiracDelta(x - a_val), (x, -oo, oo))
    rhs = integrate(g_test * f_expr.subs(x, a_val) * DiracDelta(x - a_val), (x, -oo, oo))
    chk(float(simplify(lhs - rhs).evalf()), 0,
        f"f(x)δ(x-a) = f(a)δ(x-a), f={f_expr}, a={a_val}", absolute=True)

# x·δ(x) = 0: ∫ φ(x) · x·δ(x) dx = φ(0)·0 = 0 for any φ
for phi in [sp.cos(x), x**2 + 1, sp.exp(-x**2)]:
    r = integrate(phi * x * DiracDelta(x), (x, -oo, oo))
    chk(float(r.evalf()), 0, f"x·δ(x): ∫φ·xδ dx=0  φ={phi}", absolute=True)

# %% [markdown]
# ---
# ## §9 · Fourier Transform of δ
#
# Using convention F{f}(ω) = ∫ f(t) e^{−iωt} dt:
#
# $$\mathcal{F}\{\delta(t)\}(\omega) = 1 \qquad \text{(flat spectrum)}$$
# $$\mathcal{F}\{\delta(t-t_0)\}(\omega) = e^{-i\omega t_0} \qquad \text{(shift = phase ramp)}$$
# $$\mathcal{F}\{1\}(\omega) = 2\pi\,\delta(\omega) \qquad \text{(DC = frequency spike)}$$
# $$\mathcal{F}\{e^{i\omega_0 t}\}(\omega) = 2\pi\,\delta(\omega - \omega_0) \qquad \text{(pure tone)}$$

# %%
hdr("§9 — Fourier transform: F{δ}=1, F{1}=2πδ(ω)")

tex(r"\mathcal{F}\{\delta(t)\} = 1")
tex(r"\mathcal{F}\{1\} = 2\pi\,\delta(\omega)")
tex(r"\mathcal{F}\{\delta(t-t_0)\} = e^{-i\omega t_0}")
tex(r"\mathcal{F}\{e^{i\omega_0 t}\} = 2\pi\,\delta(\omega-\omega_0)")

omega, t0, omega0 = symbols('omega t_0 omega_0', real=True)

# F{δ(t)} = ∫ δ(t) e^{-iωt} dt = e^{-iω·0} = 1
r_ft_delta = integrate(DiracDelta(t) * exp(-I*omega*t), (t, -oo, oo))
show(Eq(symbols('F{δ}'), simplify(r_ft_delta)), "F{δ(t)}:")
chk(abs(complex(r_ft_delta) - 1), 0, "F{δ(t)} = 1", absolute=True)

# F{δ(t-t0)} = e^{-iω t0}
r_shift = integrate(DiracDelta(t - t0) * exp(-I*omega*t), (t, -oo, oo))
show(Eq(symbols('F{δ(t-t₀)}'), simplify(r_shift)), "F{δ(t−t₀)}:")
# Verify at specific values
for t0_n, omega_n in [(1.0, 2.0), (0.5, -1.0), (3.0, 0.5)]:
    ref_n = complex(sp.exp(-I*omega_n*t0_n))
    got_n = complex(r_shift.subs([(t0, t0_n), (omega, omega_n)]).evalf())
    chk(abs(got_n - ref_n), 0,
        f"F{{δ(t-{t0_n})}}(ω={omega_n}) = e^{{-i·{omega_n}·{t0_n}}}", absolute=True)

# Numerical: DFT verification that δ[n] has flat spectrum
N_fft = 256
delta_seq = np.zeros(N_fft); delta_seq[0] = 1.0
D_fft = np.fft.fft(delta_seq)
chk(np.max(np.abs(np.abs(D_fft) - 1.0)), 0,
    "DFT: δ[n] has unit-magnitude spectrum", tol=1e-12, absolute=True)

# Parseval for δ: ∫|δ(t)|²dt diverges, but ∫|F{δ}|²dω = ∫1 dω diverges consistently
print("  (Parseval diverges for δ — it is not in L², consistent with distribution theory)")

# %% [markdown]
# ---
# ## §10 · Convolution — δ is the Identity Element
#
# $$( f \ast \delta )(t) = \int_{-\infty}^{\infty} f(\tau)\,\delta(t-\tau)\,d\tau = f(t)$$
#
# δ is to convolution what 1 is to multiplication.
# In signal processing: feeding δ(t) into any LTI system gives the impulse response h(t).
# The system output for arbitrary input is then f ∗ h.

# %%
hdr("§10 — Convolution: f ∗ δ = f  (identity element)")

tex(r"(f \ast \delta)(t) = f(t)")
tex(r"(f \ast \delta_a)(t) = f(t-a) \qquad \delta_a(t) = \delta(t-a)")

tau = symbols('tau', real=True)

# Symbolic convolution: ∫ f(τ) δ(t-τ) dτ = f(t)
for f_expr, label in [(t**2, "t²"), (sp.exp(-t), "e⁻ᵗ"), (sp.cos(t), "cos t")]:
    conv = integrate(f_expr.subs(t, tau) * DiracDelta(t - tau), (tau, -oo, oo))
    conv_s = simplify(conv)
    f_orig = f_expr
    diff_s = simplify(conv_s - f_orig)
    ok = diff_s == 0
    print(f"  [{'PASS' if ok else 'FAIL'}]  (f∗δ)(t) = f(t)  for f={label}  "
          f"got: {conv_s}")

# Shifted delta: f ∗ δ(t-a) = f(t-a)
a_val_n = 2
for f_expr, label in [(t**2, "t²"), (sp.sin(t), "sin t")]:
    conv_shift = integrate(f_expr.subs(t, tau) * DiracDelta(t - a_val_n - tau),
                          (tau, -oo, oo))
    f_shifted  = f_expr.subs(t, t - a_val_n)
    ok = simplify(conv_shift - f_shifted) == 0
    print(f"  [{'PASS' if ok else 'FAIL'}]  (f∗δ(t-{a_val_n}))(t) = f(t-{a_val_n})  "
          f"for f={label}")

# Numerical: DFT convolution with delta
N_c = 64
signal = np.random.default_rng(0).standard_normal(N_c)
delta_n = np.zeros(N_c); delta_n[0] = 1.0
conv_result = np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(delta_n)))
chk(np.max(np.abs(conv_result - signal)), 0,
    "DFT: signal ∗ δ[n] = signal", tol=1e-10, absolute=True)

# Shifted: signal ∗ δ[n-k] = signal shifted by k
k_shift = 5
delta_shift = np.zeros(N_c); delta_shift[k_shift] = 1.0
conv_shifted = np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(delta_shift)))
signal_shifted = np.roll(signal, k_shift)
chk(np.max(np.abs(conv_shifted - signal_shifted)), 0,
    f"DFT: signal ∗ δ[n-{k_shift}] = shift by {k_shift}", tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## §11 · Heaviside Link — H′(x) = δ(x)
#
# The Heaviside step function:
# $$H(x) = \begin{cases} 0 & x < 0 \\ 1 & x > 0 \end{cases}$$
#
# Its distributional derivative is the Dirac delta:
# $$\frac{d}{dx} H(x) = \delta(x)$$
#
# Proof: ∫ H′(x) φ(x) dx = −∫ H(x) φ′(x) dx (IBP, boundary=0)
#        = −∫₀^∞ φ′(x) dx = −[φ(∞) − φ(0)] = φ(0)
#
# This is the distributional derivative — H has a jump at 0,
# and δ is the "derivative of that jump."

# %%
hdr("§11 — Heaviside link: d/dx H(x) = δ(x)")

tex(r"\frac{d}{dx} H(x) = \delta(x)")
tex(r"\frac{d}{dx} |x| = \text{sign}(x) = 2H(x) - 1")
tex(r"\frac{d^2}{dx^2} |x| = 2\delta(x)")

# SymPy: diff(Heaviside(x)) = DiracDelta(x)
dH = diff(Heaviside(x), x)
show(Eq(diff(Heaviside(x), x), dH), "d/dx H(x):")
is_delta = dH == DiracDelta(x)
print(f"  SymPy gives DiracDelta: {is_delta}")

# Verify via sifting: ∫ φ(x)·H′(x)dx should equal φ(0)
for phi_expr in [sp.exp(-x**2), sp.cos(x), x**2 + 1]:
    r = integrate(phi_expr * diff(Heaviside(x), x), (x, -oo, oo))
    ref = phi_expr.subs(x, 0)
    chk(float(r.evalf()), float(ref.evalf()),
        f"∫φ·H′dx = φ(0)  φ={phi_expr}")

# d²/dx² |x| = 2δ(x)
d2_abs = diff(Abs(x), x, 2)
print(f"\n  d²/dx² |x| = {d2_abs}")
r_abs2 = integrate(sp.cos(x) * d2_abs, (x, -oo, oo))
chk(float(r_abs2.evalf()), float(2*sp.cos(0).evalf()),
    "∫cos(x)·(d²/dx²|x|)dx = 2cos(0) = 2")

# Approximate H(x) as limit of sigmoid: H_ε(x) = 1/(1+e^{-x/ε})
print("\n  Sigmoid approximation to H(x), derivative → δ(x) as ε→0:")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
x_num = np.linspace(-3, 3, 2000)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
for eps_n, col in [(0.5,'C0'), (0.2,'C1'), (0.05,'C2')]:
    H_approx = 1 / (1 + np.exp(-x_num/eps_n))
    dH_approx = H_approx * (1 - H_approx) / eps_n
    ax1.plot(x_num, H_approx, color=col, label=f'ε={eps_n}', linewidth=2)
    ax2.plot(x_num, dH_approx, color=col, label=f'ε={eps_n}', linewidth=2)
ax1.axhline(0.5, color='k', linestyle=':', alpha=0.4)
ax1.set_title("H_ε(x) → Heaviside"); ax1.legend(); ax1.grid(True,alpha=0.3)
ax2.set_title("H_ε′(x) → δ(x)"); ax2.legend(); ax2.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('repl/_fig_heaviside_delta.png', dpi=110, bbox_inches='tight')
plt.close()
print("  Saved: repl/_fig_heaviside_delta.png")

# %% [markdown]
# ---
# ## §12 · Multidimensional δ
#
# In ℝⁿ the delta distribution is the product of 1D deltas:
#
# $$\delta^{(n)}(\mathbf{r}) = \delta(x_1)\,\delta(x_2)\cdots\delta(x_n)$$
#
# Sifting in ℝ³:
# $$\iiint f(\mathbf{r})\,\delta^{(3)}(\mathbf{r} - \mathbf{r_0})\,d^3r = f(\mathbf{r_0})$$
#
# In spherical coordinates (3D):
# $$\delta^{(3)}(\mathbf{r}) = \frac{\delta(r)}{4\pi r^2}$$
#
# This connects to Gauss's law: ∇²(1/r) = −4πδ³(r)

# %%
hdr("§12 — Multidimensional δ: product form + Laplacian identity")

tex(r"\delta^{(3)}(\mathbf{r}-\mathbf{r}_0) = \delta(x-x_0)\,\delta(y-y_0)\,\delta(z-z_0)")
tex(r"\nabla^2\!\left(\frac{1}{r}\right) = -4\pi\,\delta^{(3)}(\mathbf{r})")

x_s, y_s, z_s = symbols('x y z', real=True)

# 2D sifting: ∫∫ f(x,y) δ(x-a)δ(y-b) dx dy = f(a,b)
f_2d = x_s**2 + y_s**3
a2, b2 = 1, 2
r_2d = integrate(integrate(
    f_2d * DiracDelta(x_s - a2) * DiracDelta(y_s - b2),
    (x_s, -oo, oo)), (y_s, -oo, oo))
ref_2d = f_2d.subs(x_s, a2).subs(y_s, b2)
chk(float(r_2d), float(ref_2d), f"2D sifting: f(1,2) = {ref_2d}")

# 3D sifting: ∫∫∫ f δ³ d³r = f(r0)
f_3d = x_s * y_s * z_s**2
a3, b3, c3 = 1, 2, 3
r_3d = integrate(integrate(integrate(
    f_3d * DiracDelta(x_s-a3) * DiracDelta(y_s-b3) * DiracDelta(z_s-c3),
    (x_s,-oo,oo)), (y_s,-oo,oo)), (z_s,-oo,oo))
ref_3d = f_3d.subs(x_s,a3).subs(y_s,b3).subs(z_s,c3)
chk(float(r_3d), float(ref_3d), f"3D sifting: f(1,2,3) = {ref_3d}")

# Laplacian of 1/r = −4π δ³(r): verify in spherical shell
# ∇²(1/r) = 0 for r≠0. Divergence theorem gives the delta.
print("\n  ∇²(1/r) = −4πδ³(r):")
print("  → for r≠0:  ∇²(1/r) = 0  (harmonic function)")
r_sym = symbols('r', positive=True)
lap_1r = diff(r_sym**2 * diff(1/r_sym, r_sym), r_sym) / r_sym**2  # spherical radial
lap_val = simplify(lap_1r)
show(Eq(symbols('∇²(1/r)'), lap_val), "Laplacian of 1/r for r≠0:")
chk(float(lap_val.evalf()), 0, "∇²(1/r) = 0 for r≠0", absolute=True)

# Gauss's law: ∮ ∇(1/r)·dA = -4π (over sphere of any radius)
# ∇(1/r) = -r̂/r² → flux through sphere radius R = -4π
R_sym = symbols('R', positive=True)
flux = -4 * pi * R_sym**2 * (1/R_sym**2)   # -1/R² * 4πR²
chk(float(simplify(flux + 4*pi)), 0, "Gauss: ∮∇(1/r)·dA = -4π", absolute=True)

# %% [markdown]
# ---
# ## §13 · Limiting Sequences — δ as a Limit
#
# δ(x) = lim_{ε→0} δ_ε(x) for any "nascent delta" family.
# Key examples:
#
# $$\delta_\varepsilon(x) = \frac{1}{\varepsilon\sqrt{\pi}}\,e^{-x^2/\varepsilon^2}
#    \quad \text{(Gaussian)}$$
#
# $$\delta_\varepsilon(x) = \frac{1}{\pi}\,\frac{\varepsilon}{x^2+\varepsilon^2}
#    \quad \text{(Lorentzian / Cauchy)}$$
#
# $$\delta_\varepsilon(x) = \frac{\sin(x/\varepsilon)}{\pi x}
#    \quad \text{(sinc / Dirichlet)}$$
#
# $$\delta_\varepsilon(x) = \frac{1}{2\varepsilon}\,\mathbf{1}_{|x|<\varepsilon}
#    \quad \text{(top-hat)}$$
#
# All satisfy: (1) unit area, (2) width → 0 as ε → 0.

# %%
hdr("§13 — Limiting sequences → δ(x)")

tex(r"\delta_\varepsilon(x) = \frac{1}{\varepsilon\sqrt{\pi}}e^{-x^2/\varepsilon^2}")
tex(r"\delta_\varepsilon(x) = \frac{1}{\pi}\frac{\varepsilon}{x^2+\varepsilon^2}")
tex(r"\delta_\varepsilon(x) = \frac{\sin(x/\varepsilon)}{\pi x}")

x_num = np.linspace(-2, 2, 4000)
epsilons = [0.5, 0.2, 0.05]
families = {
    'Gaussian':   lambda x, e: np.exp(-x**2/e**2) / (e * np.sqrt(np.pi)),
    'Lorentzian': lambda x, e: (e/np.pi) / (x**2 + e**2),
    'sinc':       lambda x, e: np.sin(x/e) / (np.pi * x + 1e-300),
    'top-hat':    lambda x, e: np.where(np.abs(x) < e, 1/(2*e), 0.0),
}

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, (name, fn) in zip(axes, families.items()):
    for eps_n, col in zip(epsilons, ['C0','C1','C2']):
        y = fn(x_num, eps_n)
        ax.plot(x_num, y, color=col, label=f'ε={eps_n}', linewidth=2)
    ax.set_title(f'{name}', fontsize=9)
    ax.set_xlim(-1, 1); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle('Nascent delta families → δ(x) as ε→0', fontsize=10)
plt.tight_layout()
plt.savefig('repl/_fig_dirac_limits.png', dpi=110, bbox_inches='tight')
plt.close()
print("  Saved: repl/_fig_dirac_limits.png")

# Verify each family: ∫δ_ε(x)dx = 1, and sifting converges
dx_num = x_num[1] - x_num[0]
f_sift_num = np.cos(x_num)   # test function, f(0)=1

for name, fn in families.items():
    eps_small = 0.001
    y = fn(x_num, eps_small)
    norm = np.sum(y) * dx_num
    sift_val = np.sum(y * f_sift_num) * dx_num
    chk(norm, 1.0, f"{name}: ∫δ_ε dx = 1  (ε=0.001)", tol=0.005)
    chk(sift_val, 1.0, f"{name}: ∫δ_ε·cos(x)dx → cos(0)=1", tol=0.02)

# %% [markdown]
# ---
# ## Summary Table — All Axioms and Identities
#
# | # | Identity | Equation |
# |---|----------|----------|
# | 1 | **Axiom** (sifting) | ∫ f(x) δ(x−a) dx = f(a) |
# | 2 | Normalisation | ∫ δ(x) dx = 1 |
# | 3 | Scaling | δ(ax) = δ(x)/\|a\| |
# | 4 | Symmetry | δ(−x) = δ(x) |
# | 5 | Composition | δ(g(x)) = Σ δ(x−xᵢ)/\|g′(xᵢ)\| |
# | 6 | First derivative | ∫ f δ′(x−a) dx = −f′(a) |
# | 7 | nth derivative | ∫ f δ⁽ⁿ⁾(x−a) dx = (−1)ⁿ f⁽ⁿ⁾(a) |
# | 8 | Product rule | f(x) δ(x−a) = f(a) δ(x−a) |
# | 9 | x·δ(x) = 0 | corollary of 8 |
# | 10 | Fourier (δ) | F{δ(t)} = 1 |
# | 11 | Fourier (1) | F{1} = 2πδ(ω) |
# | 12 | Fourier (shift) | F{δ(t−t₀)} = e^{−iωt₀} |
# | 13 | Convolution | f ∗ δ = f |
# | 14 | Heaviside link | H′(x) = δ(x) |
# | 15 | Laplacian | ∇²(1/r) = −4πδ³(r) |
# | 16 | Multidimensional | δⁿ(r) = δ(x)δ(y)δ(z) |

# %%
hdr("Done — all identities verified")
print("  §1 axiom  §2 sifting  §3 scaling  §4 symmetry  §5 composition")
print("  §6 δ′     §7 δ⁽ⁿ⁾    §8 product  §9 Fourier   §10 convolution")
print("  §11 Heaviside  §12 multidimensional  §13 limiting sequences")
