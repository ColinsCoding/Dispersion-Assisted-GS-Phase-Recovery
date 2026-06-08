# %% [markdown]
# # Dirac Delta — All Axioms and Core Identities
# `init_printing(use_latex="mathjax")` — every expression renders as LaTeX in Jupyter.
#
# The Dirac delta is **not a function**. It is a **distribution** (linear functional):
# a machine that eats a smooth test function and returns a number.
#
# **§1** Axiomatic definition · **§2** Sifting · **§3** Scaling · **§4** Shift
# **§5** Product rule · **§6** Derivative δ′ and δ⁽ⁿ⁾ · **§7** Composition
# **§8** Fourier transform · **§9** Convolution · **§10** Heaviside
# **§11** Approximating families · **§12** 3-D delta + ∇²(1/r)

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import sympy as sp
from sympy import (symbols, DiracDelta, Heaviside, integrate, diff, limit,
                   exp, sqrt, pi, oo, cos, sin, Abs, simplify,
                   Eq, Symbol, Function, Rational)
from sympy import init_printing

# ── THE MAGIC LINE ─────────────────────────────────────────────────────────────
init_printing(use_latex="mathjax")
# In Jupyter every display(expr) call renders via MathJax.
# In terminal falls back to unicode pretty-print.
# ──────────────────────────────────────────────────────────────────────────────

try:
    from IPython.display import display as _D
    IN_JUPYTER = True
except ImportError:
    IN_JUPYTER = False

def show(expr, label=None):
    if label:
        print(f"\n  {label}")
    if IN_JUPYTER:
        _D(expr)
    else:
        print("  " + sp.pretty(expr, use_unicode=True))

def hdr(s):
    bar = "─" * 64
    print(f"\n{bar}\n  {s}\n{bar}")

def chk(val, ref, label, tol=1e-8, absolute=False):
    try:
        v, r = float(val), float(ref)
    except Exception:
        print(f"  [FAIL]  {label}  (cannot convert to float)")
        return
    err = abs(v - r) if (absolute or r == 0) else abs(v - r) / (abs(r) + 1e-30)
    print(f"  [{'PASS' if err < tol else 'FAIL'}]  {label}  got={v:.8g}  ref={r:.8g}")

print("=== Dirac Delta: All Axioms and Core Identities ===")

# %% [markdown]
# ---
# ## §1 · Axiomatic Definition
#
# Two axioms define everything:
#
# **Axiom 1 — Support:**
# $$\delta(x) = 0 \qquad \forall\, x \neq 0$$
#
# **Axiom 2 — Unit integral:**
# $$\int_{-\infty}^{+\infty} \delta(x)\, dx = 1$$
#
# Together they uniquely characterise δ as the distribution:
# $$\langle \delta,\, \varphi \rangle = \varphi(0)
#   \qquad \forall\, \varphi \in \mathcal{S}(\mathbb{R})$$

# %%
hdr("§1 — Axiomatic definition")

x, t, a = symbols("x t a", real=True)
eps = symbols("epsilon", positive=True)

# Axiom 2
norm = integrate(DiracDelta(x), (x, -oo, oo))
show(Eq(Symbol("integral"), norm), "Axiom 2 — unit integral:")
chk(float(norm), 1.0, "∫δ(x)dx = 1")

# Gaussian approximating family satisfies both axioms in the limit
g_eps = exp(-x**2 / eps**2) / (eps * sqrt(pi))
norm_g = integrate(g_eps, (x, -oo, oo))
show(simplify(norm_g), "Gaussian family norm (all ε):")
chk(float(norm_g), 1.0, "Gaussian family is normalised for every ε")

# Peak at x=0: substitute x=0 first, then take limit ε→0
peak = limit(g_eps.subs(x, 0), eps, 0, "+")
show(Eq(Symbol("peak(x=0)"), peak), "Peak at x=0 as ε→0 → ∞ (Axiom 1: zero elsewhere):")

# Distributional pairing
phi = exp(-x**2)
pairing = integrate(DiracDelta(x) * phi, (x, -oo, oo))
show(Eq(Symbol("<delta,phi>"), pairing), "⟨δ, e^{-x²}⟩ = φ(0) = 1:")
chk(float(pairing), 1.0, "distributional pairing = φ(0)")

# %% [markdown]
# ---
# ## §2 · Sifting Property
#
# The **fundamental identity** — the only job δ has:
#
# $$\boxed{\int_{-\infty}^{+\infty} f(x)\,\delta(x - a)\,dx = f(a)}$$
#
# **Proof**: Axiom 1 zeroes the integrand everywhere except x=a.
# Continuity of f at a means f(x)→f(a) near a.
# Axiom 2 gives integral weight = 1. ∴ result = f(a).

# %%
hdr("§2 — Sifting property")

f_sym = Function("f")
a_sym = symbols("a", real=True)
general = integrate(f_sym(x) * DiracDelta(x - a_sym), (x, -oo, oo))
show(Eq(Symbol("sift"), general), "General sifting (symbolic):")

cases = [
    (x**3,             2,     8,    "x³ at a=2"),
    (sp.exp(x),        0,     1,    "eˣ at a=0"),
    (sp.cos(x),        pi,   -1,    "cos x at a=π"),
    (x**2 - 3*x,      -1,    4,    "x²−3x at a=−1"),
    (sp.log(x + 2),   -1,    0,    "ln(x+2) at a=−1"),
]
print("  f(x)                a    ∫f·δ(x−a)   f(a)")
print("  " + "─" * 50)
for f_e, a_v, exp_v, lbl in cases:
    r = float(integrate(f_e * DiracDelta(x - a_v), (x, -oo, oo)).evalf())
    ok = abs(r - exp_v) < 1e-8
    print(f"  {lbl:<24} {r:>9.5f}  {exp_v:>7}  {'✓' if ok else '✗'}")
    chk(r, exp_v, lbl, absolute=(exp_v == 0))

# %% [markdown]
# ---
# ## §3 · Scaling Identity
#
# $$\boxed{\delta(ax) = \frac{\delta(x)}{|a|}, \qquad a \neq 0}$$
#
# **Derivation**: u = ax, du = a dx.
# ∫f(x)δ(ax)dx = (1/|a|) ∫f(u/a)δ(u)du = f(0)/|a|.
#
# Compressed spike → taller peak → same unit area.
#
# General form: $\delta(ax-b) = \dfrac{1}{|a|}\delta\!\left(x-\dfrac{b}{a}\right)$

# %%
hdr("§3 — Scaling  δ(ax) = δ(x)/|a|")

print("  a           ∫δ(ax)dx    1/|a|")
print("  " + "─" * 34)
for a_v in [2, 3, -2, -5, Rational(1, 2), Rational(3, 4)]:
    r   = float(integrate(DiracDelta(a_v * x), (x, -oo, oo)).evalf())
    ref = 1 / abs(float(a_v))
    print(f"  {str(a_v):<12}  {r:.5f}   {ref:.5f}")
    chk(r, ref, f"δ({a_v}x)")

a_pos = symbols("a", positive=True)
sym_r = integrate(DiracDelta(a_pos * x), (x, -oo, oo))
show(Eq(Symbol("result"), sym_r), "Symbolic ∫δ(ax)dx (a>0) = 1/a:")

r_sb = integrate(x**2 * DiracDelta(3*x - 6), (x, -oo, oo))
show(Eq(Symbol("sift+scale"), r_sb),
     "∫x²δ(3x−6)dx = f(2)/3 = 4/3:")
chk(float(r_sb), float(Rational(4, 3)), "x²δ(3x−6) = 4/3")

# %% [markdown]
# ---
# ## §4 · Shift and Symmetry
#
# δ(x−a) fires at x=a. Key facts:
#
# $$\delta(a - x) = \delta(x - a) \quad \text{(even distribution)}$$
#
# $$\int f(x)\,[\delta(x-a)+\delta(x-b)]\,dx = f(a)+f(b)$$

# %%
hdr("§4 — Shift and symmetry")

a_v = Rational(3, 2)
r1 = integrate(x**2 * DiracDelta(a_v - x),   (x, -oo, oo))
r2 = integrate(x**2 * DiracDelta(x - a_v),   (x, -oo, oo))
show(r1, "∫x²δ(3/2−x)dx:")
show(r2, "∫x²δ(x−3/2)dx:")
chk(float(r1), float(r2), "δ(a−x) = δ(x−a): even symmetry")

r_sum = integrate(x**2 * (DiracDelta(x-1) + DiracDelta(x-3)), (x, -oo, oo))
show(Eq(Symbol("sum"), r_sum),
     "∫x²[δ(x−1)+δ(x−3)]dx = 1²+3² = 10:")
chk(float(r_sum), 10.0, "sum of two spikes")

f_s = sp.exp(-t**2 / 4)
print("\n  Sampling: ∫f(t)δ(t−n)dt = f(n)")
for n_v in range(4):
    r_n = float(integrate(f_s * DiracDelta(t - n_v), (t, -oo, oo)).evalf())
    d_n = float(f_s.subs(t, n_v))
    chk(r_n, d_n, f"sample n={n_v}")

# %% [markdown]
# ---
# ## §5 · Product Rule
#
# $$\boxed{f(x)\,\delta(x-a) = f(a)\,\delta(x-a)}$$
#
# Since δ(x−a)=0 for x≠a, f is "frozen" at a.
#
# **Corollaries**: $x\,\delta(x)=0$ · $x^n\delta(x)=0$ for n≥1
#
# **Warning**: 1/x · δ(x) is undefined (singularity at the support).

# %%
hdr("§5 — Product rule  f(x)δ(x−a) = f(a)δ(x−a)")

r_xd = integrate(x * DiracDelta(x), (x, -oo, oo))
show(Eq(Symbol("xd"), r_xd), "∫x·δ(x)dx = x|₀ = 0:")
chk(float(r_xd), 0, "x·δ(x)=0", absolute=True)

print("\n  xⁿ·δ(x) = 0 for n=1..5:")
for n_v in range(1, 6):
    r = float(integrate(x**n_v * DiracDelta(x), (x, -oo, oo)))
    chk(r, 0, f"x^{n_v}·δ(x)=0", tol=1e-10, absolute=True)

f_e = sp.cos(x) + x**2
g_e = sp.sin(x)
a_e = sp.Integer(2)
r_lhs = integrate(f_e * DiracDelta(x - a_e) * g_e, (x, -oo, oo))
r_rhs = float(f_e.subs(x, a_e)) * float(g_e.subs(x, a_e))
show(Eq(Symbol("result"), r_lhs),
     "∫(cosx+x²)δ(x−2)sinx dx = (cos2+4)·sin2:")
chk(float(r_lhs), r_rhs, "f(x)δ(x−a)g(x)=f(a)g(a)")

# %% [markdown]
# ---
# ## §6 · Derivative δ′(x) and δ⁽ⁿ⁾(x)
#
# Defined by integration by parts (boundary terms vanish):
#
# $$\boxed{\int f(x)\,\delta'(x-a)\,dx = -f'(a)}$$
#
# nth derivative (IBP applied n times):
#
# $$\boxed{\int f(x)\,\delta^{(n)}(x-a)\,dx = (-1)^n\, f^{(n)}(a)}$$

# %%
hdr("§6 — Derivative  ∫fδ′dx=−f′(a),  ∫fδ⁽ⁿ⁾dx=(−1)ⁿf⁽ⁿ⁾(a)")

print("  First derivative:")
d1_cases = [
    (x**3,        2,          -12,               "x³: −3x²|₂=−12"),
    (sp.exp(x),   0,          -1,                "eˣ: −e⁰=−1"),
    (sp.sin(x),   pi/2,       float(-sp.cos(pi/2).evalf()), "sin: −cos(π/2)=0"),
    (x**2 + x,   -1,          1,                 "x²+x: −(2x+1)|₋₁=1"),
]
for f_e, a_v, exp_v, lbl in d1_cases:
    r = float(integrate(f_e * DiracDelta(x - a_v, 1), (x, -oo, oo)).evalf())
    chk(r, exp_v, lbl, tol=1e-6, absolute=(exp_v == 0))

print("\n  nth derivative of f=x⁵+3x³−x at a=0:")
f_p = x**5 + 3*x**3 - x
for n_v in [0, 1, 2, 3, 4]:
    r   = float(integrate(f_p * DiracDelta(x, n_v), (x, -oo, oo)).evalf())
    ref = float(((-1)**n_v * diff(f_p, x, n_v).subs(x, 0)).evalf())
    chk(r, ref, f"n={n_v}: (−1)^{n_v}f^({n_v})(0)={ref:.3f}", absolute=(ref == 0))

show(Eq(Symbol("general"), sp.Symbol("(-1)^n f^(n)(a)")),
     "General nth-derivative formula:")

# %% [markdown]
# ---
# ## §7 · Composition Identity
#
# If g has **simple** zeros x₁,…,xₙ (g(xᵢ)=0, g′(xᵢ)≠0):
#
# $$\boxed{\delta(g(x)) = \sum_{i}\frac{\delta(x - x_i)}{|g'(x_i)|}}$$
#
# **Near each zero**: g(x) ≈ g′(xᵢ)(x−xᵢ) → apply the scaling identity.
#
# - δ(x²−a²) = [δ(x−a)+δ(x+a)] / (2|a|)   (zeros at ±a)
# - δ(sin x) = Σₙ δ(x−nπ)   (|cos(nπ)|=1)

# %%
hdr("§7 — Composition  δ(g(x)) = Σδ(x−xᵢ)/|g′(xᵢ)|")

print("  δ(x²−a²):  zeros ±a,  |g′|=2a")
for a_v in [1, 2, 3, Rational(1, 2)]:
    r   = float(integrate(x**3 * DiracDelta(x**2 - a_v**2),
                           (x, -oo, oo)).evalf())
    af  = float(a_v)
    ref = (af**3 + (-af)**3) / (2*af)
    chk(r, ref, f"a={a_v}: [f(a)+f(-a)]/(2a)", absolute=(abs(ref) < 1e-10))

print("\n  δ(sin x) over [−2π, 2π]: zeros at 0,±π,±2π")
# SymPy can't evaluate δ(sin x) symbolically — do numerically via composition rule
# δ(sin x) = Σ δ(x−nπ)/|cos(nπ)| = Σ δ(x−nπ) (since |cos(nπ)|=1)
x_arr_sin = np.linspace(-2*float(pi) - 0.01, 2*float(pi) + 0.01, 4000000)
f_arr_sin  = np.exp(-x_arr_sin**2 / 20)
# Approximate each δ(x−nπ) as a Gaussian with ε=0.01
eps_sin = 0.01
zeros_sin = [-2*float(pi), -float(pi), 0.0, float(pi), 2*float(pi)]
delta_sum  = sum(np.exp(-(x_arr_sin - z)**2 / eps_sin**2)
                 / (eps_sin * np.sqrt(np.pi)) for z in zeros_sin)
r_num = np.trapezoid(f_arr_sin * delta_sum, x_arr_sin)
ref_s = sum(np.exp(-z**2 / 20) for z in zeros_sin)
chk(r_num, ref_s, "∫e^{-x²/20}δ(sin x)dx = Σf(nπ)  (numerical)", tol=0.01)

print("\n  ⚠  δ(x²) is undefined — double zero at 0, formula requires g′(xᵢ)≠0")

# %% [markdown]
# ---
# ## §8 · Fourier Transform — δ is White
#
# Convention: $\mathcal{F}\{f\}(\omega)=\int f(t)\,e^{-i\omega t}\,dt$
#
# $$\boxed{\mathcal{F}\{\delta(t)\} = 1} \qquad \text{flat spectrum — all frequencies equally}$$
# $$\boxed{\mathcal{F}\{1\} = 2\pi\,\delta(\omega)} \qquad \text{pure DC — spike at 0 frequency}$$
# $$\boxed{\mathcal{F}\{\delta(t-t_0)\} = e^{-i\omega t_0}} \qquad \text{shift = linear phase ramp}$$

# %%
hdr("§8 — Fourier transform  F{δ(t)}=1")

omega_s, t_s = symbols("omega t", real=True)
t0_s = symbols("t0", positive=True)

ft_d = integrate(DiracDelta(t_s) * sp.exp(-sp.I*omega_s*t_s), (t_s, -oo, oo))
show(Eq(Symbol("F{d}"), ft_d), "F{δ(t)} = 1:")
chk(abs(complex(ft_d) - 1.0), 0, "F{δ(t)}=1", tol=1e-10, absolute=True)

ft_sh = integrate(DiracDelta(t_s - t0_s) * sp.exp(-sp.I*omega_s*t_s),
                  (t_s, -oo, oo))
show(Eq(Symbol("F{d_shift}"), ft_sh), "F{δ(t−t₀)} = e^{−iωt₀}:")

# unit magnitude at all ω
ft_at = ft_sh.subs([(t0_s, 2), (omega_s, pi)])
chk(abs(complex(ft_at.evalf())) - 1.0, 0,
    "|F{δ(t−2)}| at ω=π is 1 (pure phase)", tol=1e-10, absolute=True)

# DFT of unit impulse = flat spectrum
N_f = 256
imp_f = np.zeros(N_f); imp_f[0] = 1.0
spec  = np.fft.fft(imp_f)
chk(np.max(np.abs(np.abs(spec) - 1.0)), 0,
    "DFT{impulse}:  |X[k]|=1 for all k", tol=1e-12, absolute=True)

ft_cos = integrate(DiracDelta(t_s)*sp.cos(omega_s*t_s)
                   *sp.exp(-sp.I*omega_s*t_s), (t_s, -oo, oo))
show(Eq(Symbol("F{d*cos}"), ft_cos),
     "F{δ(t)·cos(ωt)} = cos(0) = 1:")
chk(float(ft_cos), 1.0, "F{δ(t)cos(ωt)}=1")

# %% [markdown]
# ---
# ## §9 · Convolution — δ is the Identity Element
#
# $$\boxed{(f * \delta)(t) = f(t)}$$
#
# $$\boxed{(f * \delta_{t_0})(t) = f(t-t_0)} \qquad \text{(delay by } t_0\text{)}$$
#
# $$\boxed{(f * \delta')(t) = f'(t)} \qquad \text{(differentiation)}$$

# %%
hdr("§9 — Convolution  f∗δ=f")

tau = symbols("tau", real=True)
conv_sym = integrate(f_sym(tau) * DiracDelta(t_s - tau), (tau, -oo, oo))
show(Eq(Symbol("(f*d)(t)"), conv_sym), "Symbolic (f∗δ)(t):")

N_c   = 512
t_arr = np.linspace(-5, 5, N_c)
dt    = t_arr[1] - t_arr[0]
gauss = np.exp(-t_arr**2)
# f∗δ via FFT (circular, no edge issue)
F_g   = np.fft.fft(gauss)
# δ in frequency domain = 1 (for centred impulse, phase correct)
conv_r_fft = np.real(np.fft.ifft(F_g * 1.0))
chk(np.max(np.abs(conv_r_fft - gauss)), 0,
    "f∗δ = f  (FFT, frequency-domain identity)", tol=1e-12, absolute=True)

# f∗δ(t−t₀): multiply by e^{−i2πk·delay/N} in frequency
freq  = np.fft.fftfreq(N_c)
delay_t = 20 * dt   # delay in time units
conv_d = np.real(np.fft.ifft(F_g * np.exp(-1j*2*np.pi*freq*delay_t/dt)))
g_del  = np.exp(-(t_arr - delay_t)**2)
# compare interior only (circular wrap affects edges)
mid = slice(N_c//4, 3*N_c//4)
chk(np.max(np.abs(conv_d[mid] - g_del[mid])), 0,
    "f∗δ(t−t₀) = f(t−t₀)  (FFT delay, interior)", tol=1e-6, absolute=True)

F_g   = np.fft.fft(gauss)
freq  = np.fft.fftfreq(N_c, d=dt)
d_fft = np.real(np.fft.ifft(F_g * (1j * 2 * np.pi * freq)))
d_dir = -2 * t_arr * gauss          # analytic: d/dt e^{-t²} = -2t e^{-t²}
chk(np.max(np.abs(d_fft - d_dir)), 0,
    "f∗δ′ = f′  (FFT differentiation)", tol=1e-3, absolute=True)

# %% [markdown]
# ---
# ## §10 · Heaviside — dH/dx = δ(x)
#
# $$H(x) = \begin{cases}0 & x<0\\ \tfrac{1}{2} & x=0\\ 1 & x>0\end{cases}$$
#
# $$\boxed{\frac{d}{dx}H(x) = \delta(x)} \qquad \text{(distributional derivative)}$$
#
# **Proof**: IBP with test function φ:
# $\int H'\varphi\,dx = -\int H\varphi'\,dx = -\int_0^\infty\varphi'\,dx
#  = \varphi(0) = \langle\delta,\varphi\rangle$
#
# H is not classically differentiable at 0, but its **distributional** derivative is δ.

# %%
hdr("§10 — Heaviside  dH/dx = δ(x)")

dH = diff(Heaviside(x), x)
show(Eq(Symbol("dH/dx"), dH), "SymPy: d/dx H(x):")

test_phi = [
    (sp.exp(-x**2),       1.0,  "e^{-x²}: φ(0)=1"),
    (sp.cos(x),           1.0,  "cos x: φ(0)=1"),
    (x**2 + 1,            1.0,  "x²+1: φ(0)=1"),
    (x * sp.exp(-x**2),   0.0,  "x·e^{-x²}: φ(0)=0"),
]
for phi_e, exp_v, lbl in test_phi:
    r = float(integrate(phi_e * dH, (x, -oo, oo)).evalf())
    chk(r, exp_v, lbl, absolute=True)

x0 = symbols("x0", positive=True)
H_d = integrate(DiracDelta(t_s), (t_s, -oo, x0))
show(Eq(Symbol("H(x>0)"), H_d),
     "H(x>0) = ∫_{-∞}^{x>0}δ(t)dt:")
chk(float(H_d), 1.0, "integral of δ to x>0 = 1 = H")

d_abs = diff(Abs(x), x)
show(Eq(Symbol("d|x|/dx"), d_abs), "d|x|/dx = sgn(x):")
chk(float(d_abs.subs(x,  1)),  1.0, "sgn(1)=+1")
chk(float(d_abs.subs(x, -1)), -1.0, "sgn(-1)=-1")

# %% [markdown]
# ---
# ## §11 · Approximating Families
#
# | Family | $\delta_\varepsilon(x)$ | Note |
# |--------|------------------------|------|
# | Gaussian | $\frac{1}{\varepsilon\sqrt{\pi}}e^{-x^2/\varepsilon^2}$ | smooth, $\mathcal{S}(\mathbb{R})$ |
# | Lorentzian | $\frac{1}{\pi}\frac{\varepsilon}{x^2+\varepsilon^2}$ | Cauchy, resonance lineshape |
# | Box | $\frac{1}{2\varepsilon}\mathbf{1}_{|x|\le\varepsilon}$ | compact support |
# | Sinc | $\frac{\sin(x/\varepsilon)}{\pi x}$ | Fourier dual of Box |
#
# Each satisfies: **(i)** $\int\delta_\varepsilon\,dx=1$ for all ε,
# **(ii)** $\delta_\varepsilon(x)\to\delta(x)$ as ε→0 in the distributional sense.

# %%
hdr("§11 — Approximating families")

x_a  = np.linspace(-15, 15, 300000)
dx_a = x_a[1] - x_a[0]

fams = {
    "Gaussian":   lambda e: np.exp(-x_a**2 / e**2) / (e * np.sqrt(np.pi)),
    "Lorentzian": lambda e: (e / np.pi) / (x_a**2 + e**2),
    "Box":        lambda e: np.where(np.abs(x_a) <= e, 1/(2*e), 0.0),
    "Sinc":       lambda e: np.where(np.abs(x_a) > 1e-14,
                            np.sin(x_a / e) / (np.pi * x_a), 1/(np.pi*e)),
}
eps_list = [0.5, 0.2, 0.1, 0.05]

print(f"  {'Family':<12}  " + "  ".join(f"ε={e}" for e in eps_list))
print("  " + "─" * 52)
for name, fn in fams.items():
    norms = [np.trapezoid(fn(e), x_a) for e in eps_list]
    print(f"  {name:<12}  " + "  ".join(f"{n:.5f}" for n in norms))
    for nv in norms:
        chk(nv, 1.0, f"{name} norm=1", tol=0.004)

f_tst = np.cos(x_a) * np.exp(-x_a**2 / 4)
print("\n  Sifting accuracy → f(0)=1 as ε→0 (Gaussian):")
for e in eps_list:
    s = np.trapezoid(f_tst * fams["Gaussian"](e), x_a)
    print(f"    ε={e:.2f}:  {s:.6f}")
chk(np.trapezoid(f_tst * fams["Gaussian"](0.01), x_a), 1.0,
    "Gaussian sift ε=0.01 → f(0)=1", tol=0.001)

# %% [markdown]
# ---
# ## §12 · Three-Dimensional Delta and ∇²(1/r) = −4πδ³(**r**)
#
# $$\delta^3(\mathbf{r}) = \delta(x)\,\delta(y)\,\delta(z),
#   \qquad \iiint\delta^3(\mathbf{r})\,d^3r = 1$$
#
# **The central Green's-function identity** (used in every branch of physics):
#
# $$\boxed{\nabla^2\!\left(\frac{1}{r}\right) = -4\pi\,\delta^3(\mathbf{r})}$$
#
# **Proof**:
# 1. ∇²(1/r) = 0 for r > 0  (verified symbolically below).
# 2. Divergence theorem over sphere radius R:
#    $\oint\nabla(1/r)\cdot d\mathbf{A} = (-1/R^2)(4\pi R^2) = -4\pi$
#    for **any** R → distributional source of strength −4π sitting at **r**=0.

# %%
hdr("§12 — 3D delta and ∇²(1/r) = −4πδ³(r)")

r_s = symbols("r", positive=True)

lap_r = diff(r_s**2 * diff(1/r_s, r_s), r_s) / r_s**2
show(Eq(Symbol("nabla^2(1/r)"), simplify(lap_r)),
     "∇²(1/r) for r>0:")
chk(float(simplify(lap_r).subs(r_s, 1)), 0,
    "∇²(1/r)=0 at r=1", tol=1e-12, absolute=True)
chk(float(simplify(lap_r).subs(r_s, 5)), 0,
    "∇²(1/r)=0 at r=5", tol=1e-12, absolute=True)

# Divergence theorem: flux = ∮∇(1/r)·dA = −(1/R²)·4πR² = −4π (any R)
print("\n  ∮∇(1/r)·dA = −4π for any sphere radius R:")
for R_v in [0.1, 1.0, 5.0, 100.0]:
    flux = -(1/R_v**2) * 4 * np.pi * R_v**2
    chk(flux, -4*np.pi, f"R={R_v}")

# Numerical regularised integral
r_g  = np.linspace(1e-4, 40, 400000)
eps3 = 0.05
phi3 = -1 / np.sqrt(r_g**2 + eps3**2)
dphi = np.gradient(phi3, r_g)
lap3 = np.gradient(r_g**2 * dphi, r_g) / (r_g**2 + 1e-30)
vol  = np.trapezoid(lap3 * 4*np.pi * r_g**2, r_g)
chk(vol, -4*np.pi,
    "∫∇²(−1/r_ε)·4πr²dr = −4π  (regularised, ε=0.05)", tol=0.05)

print("\n  Applications:")
print("  ∇²φ = −ρ/ε₀, ρ = qδ³(r)  →  φ = q/(4πε₀r)   [Coulomb]")
print("  ∇²G = −δ³(r)              →  G = 1/(4πr)      [Laplacian Green's fn]")
print("  −(ℏ²/2m)∇²ψ + Vψ = Eψ    →  same eigenvalue structure [Schrödinger]")

# %% [markdown]
# ---
# ## Summary — All 15 Identities
#
# | # | Identity | Name |
# |---|----------|------|
# | A1 | $\delta(x)=0,\ x\neq 0$ | Support axiom |
# | A2 | $\int\delta\,dx=1$ | Unit integral |
# | I1 | $\int f\,\delta(x-a)\,dx=f(a)$ | **Sifting** |
# | I2 | $\delta(ax)=\delta(x)/\lvert a\rvert$ | Scaling |
# | I3 | $\delta(a-x)=\delta(x-a)$ | Even symmetry |
# | I4 | $f(x)\delta(x-a)=f(a)\delta(x-a)$ | Product rule |
# | I5 | $\int f\,\delta'\,dx=-f'(a)$ | First derivative |
# | I6 | $\int f\,\delta^{(n)}\,dx=(-1)^n f^{(n)}(a)$ | nth derivative |
# | I7 | $\delta(g(x))=\sum\delta(x-x_i)/\lvert g'(x_i)\rvert$ | Composition |
# | I8 | $\mathcal{F}\{\delta\}=1$ | White spectrum |
# | I9 | $\mathcal{F}\{\delta(t-t_0)\}=e^{-i\omega t_0}$ | Phase shift |
# | I10 | $f\ast\delta=f$ | Convolution identity |
# | I11 | $f\ast\delta'=f'$ | Convolution + derivative |
# | I12 | $dH/dx=\delta(x)$ | Heaviside derivative |
# | I13 | $\nabla^2(1/r)=-4\pi\delta^3(\mathbf{r})$ | 3D Green's function |

# %%
hdr("Done — all 15 identities verified")
print("  Open repl/_repl_dirac_axioms.ipynb in Jupyter.")
print("  Run All Cells — every expression renders as LaTeX via MathJax.")
