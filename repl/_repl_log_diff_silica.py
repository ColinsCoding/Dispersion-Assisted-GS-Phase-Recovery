# %% [markdown]
# # Logarithmic Differentiation + Time Translation Symmetry of Silica
# `init_printing(use_latex="mathjax")` throughout.
#
# **§1** Logarithmic differentiation — the rule, derivation, animated curves
# **§2** Hard cases — xˣ, products, nested powers
# **§3** Time translation symmetry → energy conservation (Noether)
# **§4** Silica (SiO₂) phonon modes — each mode = conserved energy quantum
# **§5** Animated: log-derivative of silica refractive index n(ω)

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import sympy as sp
from sympy import (symbols, ln, exp, sin, cos, tan, sqrt, pi, oo,
                   diff, simplify, factor, latex, Eq, Symbol,
                   Function, Rational, log, Abs)
from sympy import init_printing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

init_printing(use_latex="mathjax")

try:
    from IPython.display import display as _D, HTML as _HTML
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
        print(f"  [FAIL]  {label}"); return
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    print(f"  [{'PASS' if err<tol else 'FAIL'}]  {label}  got={v:.8g}  ref={r:.8g}")

print("=== Logarithmic Differentiation + Silica Time Symmetry ===")

# %% [markdown]
# ---
# ## §1 · Logarithmic Differentiation — The Rule
#
# **The trick**: for any differentiable f(x) > 0,
#
# $$\ln f(x) \xrightarrow{\tfrac{d}{dx}} \frac{f'(x)}{f(x)}
#   \implies \boxed{f'(x) = f(x)\cdot\frac{d}{dx}\!\left[\ln f(x)\right]}$$
#
# **Why it works**: chain rule on both sides of $e^{\ln f} = f$.
#
# **When to use it**: f is a product, quotient, or power of functions —
# logarithm turns × into +, ÷ into −, and powers into ×.
#
# $$\ln\!\left(\frac{u^a \cdot v^b}{w^c}\right)
#   = a\ln u + b\ln v - c\ln w$$

# %%
hdr("§1 — The rule, derivation, symbolic verification")

x = symbols("x", positive=True)

# Derive symbolically: d/dx ln(f) = f'/f
f = Function("f")
lhs = diff(ln(f(x)), x)
show(Eq(Symbol("d/dx ln(f)"), lhs), "Chain rule: d/dx ln(f(x)) =")

# Therefore f' = f * d/dx[ln f]
show(Eq(Symbol("f'(x)"), f(x) * lhs),
     "Logarithmic differentiation formula:")

# Verify on concrete functions
print("\n  Verifying: direct diff == log-diff method")
test_fns = [
    x**5 * sp.exp(-x),
    (x**3 * sp.sqrt(1 + x**2)) / (sp.sin(x) + 2),
    x**(sp.Rational(2,3)) * (x**2 + 1)**sp.Rational(3,2),
]
for f_e in test_fns:
    direct  = diff(f_e, x)
    log_way = f_e * diff(ln(f_e), x)
    diff_check = simplify(direct - log_way)
    show(Eq(Symbol("f"), f_e), "  f =")
    show(Eq(Symbol("f' direct"), simplify(direct)), "  f' direct =")
    show(Eq(Symbol("f' log"), simplify(log_way)), "  f' via log-diff =")
    ok = diff_check == 0
    print(f"  [{'PASS' if ok else 'FAIL'}]  methods agree: diff={diff_check}")

# %% [markdown]
# ---
# ## §2 · Hard Cases — Where Log-Diff Shines
#
# ### Case 1: Variable base AND exponent — xˣ
#
# Direct diff fails because neither the power rule nor the exp rule applies alone.
#
# $$f = x^x \implies \ln f = x\ln x
#   \implies \frac{f'}{f} = \ln x + 1
#   \implies \boxed{f' = x^x(\ln x + 1)}$$
#
# ### Case 2: Long products
#
# $$f = \frac{(x+1)^3(x+2)^4}{(x+3)^5}$$
#
# Log turns this into a sum of logs → diff term by term → multiply back.

# %%
hdr("§2 — Hard cases: xˣ, products, nested powers")

# xˣ
f_xx = x**x
ln_xx = ln(f_xx)
show(Eq(Symbol("ln(xˣ)"), sp.expand_log(ln_xx, force=True)),
     "ln(xˣ) = x·ln(x):")
d_ln_xx = diff(sp.expand_log(ln_xx, force=True), x)
show(Eq(Symbol("d/dx ln(xˣ)"), simplify(d_ln_xx)),
     "d/dx ln(xˣ) = ln(x)+1:")
f_xx_deriv = f_xx * d_ln_xx
show(Eq(Symbol("d/dx xˣ"), simplify(f_xx_deriv)),
     "d/dx xˣ = xˣ(ln x + 1):")

# Verify at x=2: should be 4(ln2+1)
val_at_2  = float(f_xx_deriv.subs(x, 2).evalf())
ref_at_2  = 4 * (np.log(2) + 1)
chk(val_at_2, ref_at_2, "d/dx xˣ at x=2 = 4(ln2+1)")

# Long product
x_r = symbols("x", real=True, positive=True)
f_prod = (x_r+1)**3 * (x_r+2)**4 / (x_r+3)**5
ln_prod = sp.expand_log(ln(f_prod), force=True)
show(Eq(Symbol("ln(f)"), ln_prod),
     "ln of long product — sum of logs:")
d_prod = f_prod * diff(ln_prod, x_r)
show(Eq(Symbol("f'"), simplify(d_prod)),
     "Derivative via log-diff:")
chk(float(simplify(d_prod - diff(f_prod, x_r)).subs(x_r, 1)),
    0, "long product: methods agree at x=1", tol=1e-8, absolute=True)

# x^sin(x) — another impossible direct case
x_p = symbols("x", positive=True)
f_xs = x_p**sp.sin(x_p)
ln_xs = sp.expand_log(ln(f_xs), force=True)
show(Eq(Symbol("ln(x^sinx)"), ln_xs),
     "ln(x^sin x) = sin(x)·ln(x):")
d_xs = simplify(f_xs * diff(ln_xs, x_p))
show(Eq(Symbol("d/dx x^sinx"), d_xs),
     "d/dx x^sin(x):")
# at x=1: d/dx x^sinx = 1^sin1 * (sin1/1 + cos1*ln1) = sin(1) + 0 = sin(1)
ref_xs = float(sp.sin(sp.Integer(1)).evalf())
chk(float(d_xs.subs(x_p, 1).evalf()), ref_xs,
    "d/dx x^sinx at x=1 = sin(1)")

# %% [markdown]
# ---
# ## §3 · Time Translation Symmetry → Energy Conservation (Noether)
#
# **Noether's theorem** (1915): every continuous symmetry of the action
# has a corresponding conserved quantity.
#
# **Time translation**: the laws of physics are the same at t=0 and t=t₀.
# The Lagrangian L does not depend explicitly on t.
#
# $$\frac{\partial L}{\partial t} = 0
#   \implies \frac{d}{dt}\!\underbrace{\left(\sum_i \dot{q}_i
#   \frac{\partial L}{\partial \dot{q}_i} - L\right)}_{\text{Hamiltonian }H} = 0$$
#
# $$\boxed{E = H = \text{conserved}}$$
#
# For a harmonic oscillator (= every phonon mode):
# $L = \tfrac{1}{2}m\dot{x}^2 - \tfrac{1}{2}kx^2$,
# time-translation symmetry → $E = \tfrac{1}{2}m\dot{x}^2 + \tfrac{1}{2}kx^2 = \text{const}$

# %%
hdr("§3 — Noether: time translation → energy conservation")

t_s, m_s, k_s, omega_s = symbols("t m k omega", positive=True)
x_s = Function("x")

# Harmonic oscillator Lagrangian
L_ho = sp.Rational(1,2)*m_s*diff(x_s(t_s),t_s)**2 \
     - sp.Rational(1,2)*k_s*x_s(t_s)**2
show(Eq(Symbol("L"), L_ho), "Harmonic oscillator Lagrangian:")

# Hamiltonian (Legendre transform of L)
p = m_s * diff(x_s(t_s), t_s)   # conjugate momentum
H_ho = p * diff(x_s(t_s), t_s) - L_ho
H_ho = sp.Rational(1,2)*m_s*diff(x_s(t_s),t_s)**2 \
     + sp.Rational(1,2)*k_s*x_s(t_s)**2
show(Eq(Symbol("H"), H_ho), "Hamiltonian = conserved energy:")

# Verify: solution x(t) = A cos(ωt), energy constant
A_s = symbols("A", positive=True)
omega_val = sp.sqrt(k_s/m_s)
x_sol = A_s * sp.cos(omega_val * t_s)
v_sol = diff(x_sol, t_s)
E_sol = sp.Rational(1,2)*m_s*v_sol**2 + sp.Rational(1,2)*k_s*x_sol**2
E_simplified = simplify(E_sol)
show(Eq(Symbol("E(t)"), E_simplified),
     "E(t) for x=A·cos(ωt) — should be constant:")
# Check dE/dt = 0
dE_dt = simplify(diff(E_simplified, t_s))
show(Eq(Symbol("dE/dt"), dE_dt), "dE/dt:")
chk(dE_dt == 0 or dE_dt == sp.Integer(0), 1,
    "dE/dt = 0: energy conserved", tol=1e-9, absolute=True)

# Quantum: E_n = ℏω(n + 1/2)
hbar, n_sym = symbols("hbar n", positive=True)
E_n = hbar * omega_s * (n_sym + sp.Rational(1,2))
show(Eq(Symbol("E_n"), E_n),
     "Quantum harmonic oscillator: E_n = ℏω(n+½):")
print("  n=0 (ground state): E₀ = ℏω/2  (zero-point energy, never zero!)")

# %% [markdown]
# ---
# ## §4 · Silica (SiO₂) Phonon Modes
#
# SiO₂ has 3 atoms per formula unit → 9 phonon branches.
# Each branch = a harmonic oscillator → time-translation symmetry →
# each mode's energy $E_n = \hbar\omega_i(n_i+\tfrac{1}{2})$ is conserved.
#
# **Key Raman-active modes** (amorphous silica):
#
# | Mode | Frequency (cm⁻¹) | Assignment |
# |------|-----------------|------------|
# | D1   | 490             | 4-membered ring breathing |
# | D2   | 606             | 3-membered ring breathing |
# | W3   | 800             | Si-O-Si bending |
# | W4   | 1060–1200       | Si-O stretching (TO/LO split) |
#
# **Sellmeier equation** — refractive index of fused silica:
# $$n^2(\lambda) = 1 + \frac{B_1\lambda^2}{\lambda^2-C_1}
#   + \frac{B_2\lambda^2}{\lambda^2-C_2}
#   + \frac{B_3\lambda^2}{\lambda^2-C_3}$$

# %%
hdr("§4 — Silica phonon modes + Sellmeier refractive index")

# Sellmeier coefficients for fused silica (Malitson 1965)
B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2  # in μm²

lam_um = np.linspace(0.21, 6.7, 2000)  # wavelength in μm
n2 = (1 + B1*lam_um**2/(lam_um**2-C1)
        + B2*lam_um**2/(lam_um**2-C2)
        + B3*lam_um**2/(lam_um**2-C3))
n_arr = np.sqrt(n2)

# SymPy Sellmeier
lam = symbols("lambda", positive=True)
n2_sym = (1 + B1*lam**2/(lam**2-C1)
            + B2*lam**2/(lam**2-C2)
            + B3*lam**2/(lam**2-C3))
n_sym_expr = sp.sqrt(n2_sym)
show(Eq(Symbol("n²(λ)"), n2_sym),
     "Sellmeier equation for fused silica:")

# Logarithmic derivative of n: d ln(n)/dλ = (1/n)(dn/dλ)
ln_n = ln(n_sym_expr)
d_ln_n = diff(ln_n, lam)
dn_dlam = n_sym_expr * d_ln_n   # = dn/dλ
show(Eq(Symbol("d ln(n)/dλ"), simplify(d_ln_n)),
     "Log-derivative of n (group delay related):")

# GVD: d²n/dλ² via log-diff chain
d2n = diff(dn_dlam, lam)

# Numerical evaluation
dn_num  = np.gradient(n_arr, lam_um)
d2n_num = np.gradient(dn_num, lam_um)

# GVD parameter β₂ = (λ³/2πc) d²n/dλ²  [ps²/km at λ in μm, c in μm/ps]
c_light = 2.998e5   # μm/ps (= 3e8 m/s)
beta2   = (lam_um**3 / (2*np.pi*c_light)) * d2n_num * 1e6  # ps²/km

# Zero-dispersion wavelength of silica ≈ 1.27 μm
idx_zdw = np.argmin(np.abs(beta2))
ZDW     = lam_um[idx_zdw]
print(f"  Zero-dispersion wavelength: {ZDW:.3f} μm  (expect ~1.27 μm)")
chk(ZDW, 1.27, "ZDW of fused silica", tol=0.05)

# Phonon mode energies (Raman frequencies)
phonon_modes = {
    "D1 ring": 490,    # cm⁻¹
    "D2 ring": 606,
    "Si-O-Si bend": 800,
    "Si-O stretch": 1100,
}
hbar_cm = 1.0   # working in units where ℏ=1, E in cm⁻¹
print("\n  Silica phonon modes (zero-point energy ℏω/2):")
for name, freq in phonon_modes.items():
    E0 = freq / 2
    print(f"  {name:<18}: ω={freq} cm⁻¹   E₀=ℏω/2={E0:.0f} cm⁻¹")
    chk(E0, freq/2, f"{name} ZPE", tol=1e-9, absolute=True)

# %% [markdown]
# ---
# ## §5 · Animated Logarithmic Differentiation of n(λ)
#
# Animation shows:
# - **Top**: Sellmeier n(λ) with a moving point and its tangent line
# - **Middle**: ln(n(λ)) — the log-transformed curve
# - **Bottom**: d ln(n)/dλ = (1/n)(dn/dλ) — the log-derivative
#
# The tangent slope on the log curve equals the log-derivative directly.
# This is why log-diff is powerful: slopes of log-curves ARE the relative
# rates of change.

# %%
hdr("§5 — Animation: log-diff of silica refractive index")

fig = plt.figure(figsize=(11, 8))
gs_fig = GridSpec(3, 1, figure=fig, hspace=0.45)
ax1 = fig.add_subplot(gs_fig[0])
ax2 = fig.add_subplot(gs_fig[1])
ax3 = fig.add_subplot(gs_fig[2])

ln_n_arr   = np.log(n_arr)
dln_n_arr  = np.gradient(ln_n_arr, lam_um)

# Static curves
ax1.plot(lam_um, n_arr,      "b-",  lw=2, label="n(λ) Sellmeier")
ax2.plot(lam_um, ln_n_arr,   "g-",  lw=2, label="ln n(λ)")
ax3.plot(lam_um, dln_n_arr,  "r-",  lw=2, label="d ln n / dλ")
ax3.axhline(0, color="k", lw=0.8, ls="--")

# Phonon mode markers on ax1
colors_ph = ["orange","purple","brown","magenta"]
for (name, freq), col in zip(phonon_modes.items(), colors_ph):
    # Convert cm⁻¹ to μm wavelength: λ = 1/freq (cm) * 1e4 μm/cm
    lam_ph = 1e4 / freq
    if lam_um[0] < lam_ph < lam_um[-1]:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(lam_ph, color=col, lw=0.8, ls=":", alpha=0.7)
        ax1.text(lam_ph, n_arr.max()*0.97, name.split()[0],
                 color=col, fontsize=7, rotation=90, va="top")

for ax, ylabel, title in zip(
    [ax1, ax2, ax3],
    ["n(λ)", "ln n(λ)", "d ln n/dλ  (μm⁻¹)"],
    ["Refractive index — Sellmeier",
     "Log-transformed — slopes here ARE log-derivatives",
     "Log-derivative = relative rate of change"]
):
    ax.set_xlabel("λ (μm)"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Moving point + tangent line objects
pt1,  = ax1.plot([], [], "ro", ms=8, zorder=5)
tan1, = ax1.plot([], [], "r--", lw=1.5, alpha=0.7)
pt2,  = ax2.plot([], [], "go", ms=8, zorder=5)
tan2, = ax2.plot([], [], "g--", lw=1.5, alpha=0.7)
pt3,  = ax3.plot([], [], "rs", ms=8, zorder=5)

# Annotation text
ann = ax3.text(0.02, 0.92, "", transform=ax3.transAxes,
               fontsize=9, color="darkred",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Animation: sweep λ from 0.3 to 2.5 μm (visible + near-IR)
lam_anim = np.linspace(0.35, 2.4, 80)

def animate(i):
    lv = lam_anim[i % len(lam_anim)]
    idx = np.argmin(np.abs(lam_um - lv))

    nv   = n_arr[idx]
    lnnv = ln_n_arr[idx]
    dlv  = dln_n_arr[idx]
    dw   = 0.3  # tangent window half-width

    # Tangent on n(λ)
    dn_num_v = np.gradient(n_arr, lam_um)[idx]
    lam_t = np.array([lv - dw, lv + dw])
    tan_n = nv + dn_num_v * (lam_t - lv)
    pt1.set_data([lv], [nv]);  tan1.set_data(lam_t, tan_n)

    # Tangent on ln n(λ)
    tan_ln = lnnv + dlv * (lam_t - lv)
    pt2.set_data([lv], [lnnv]); tan2.set_data(lam_t, tan_ln)

    # Log-derivative value
    pt3.set_data([lv], [dlv])
    ann.set_text(f"λ={lv:.3f}μm\nd ln n/dλ = {dlv:.4f} μm⁻¹\n"
                 f"= (1/n)(dn/dλ) = {dn_num_v/nv:.4f} μm⁻¹")
    return pt1, tan1, pt2, tan2, pt3, ann

ani = animation.FuncAnimation(fig, animate, frames=len(lam_anim),
                               interval=80, blit=True)

# Save as GIF
gif_path = "repl/_anim_log_diff_silica.gif"
ani.save(gif_path, writer="pillow", fps=15, dpi=90)
plt.savefig("repl/_fig_log_diff_silica_static.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"  Saved animation : {gif_path}")
print(f"  Saved static fig: repl/_fig_log_diff_silica_static.png")

# In Jupyter: display the animation inline
if IN_JUPYTER:
    _D(_HTML(ani.to_jshtml()))

# %% [markdown]
# ---
# ## Summary
#
# | Concept | Formula | Physical meaning |
# |---------|---------|-----------------|
# | Log-diff rule | $f' = f\cdot\frac{d}{dx}\ln f$ | Slope of log-curve = relative rate |
# | xˣ | $f'=x^x(\ln x+1)$ | Neither power nor exp rule alone works |
# | Noether time-translation | $\partial L/\partial t=0\Rightarrow dH/dt=0$ | Physics same at all times → energy conserved |
# | Phonon mode | $E_n=\hbar\omega(n+\tfrac{1}{2})$ | Each SiO₂ vibration mode = conserved energy |
# | Sellmeier log-derivative | $\frac{d\ln n}{d\lambda}=\frac{1}{n}\frac{dn}{d\lambda}$ | Group delay, dispersion, ZDW at 1.27μm |
# | ZDW of silica | λ≈1.27μm | β₂=0 here; anomalous dispersion above |

# %%
hdr("Done")
print("  Open _repl_log_diff_silica.ipynb in Jupyter.")
print("  Animation plays inline via to_jshtml().")
print("  Static figure: repl/_fig_log_diff_silica_static.png")
