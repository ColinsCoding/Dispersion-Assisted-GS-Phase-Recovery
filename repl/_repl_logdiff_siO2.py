# %% [markdown]
# # Logarithmic Differentiation + Time Translation Symmetry of SiO₂
# `init_printing(use_latex="mathjax")` — LaTeX in Jupyter, unicode in terminal.
#
# **§1** Logarithmic differentiation — the trick, animated
# **§2** Why it works — chain rule proof
# **§3** Classic examples where log-diff is the only sane approach
# **§4** Time translation symmetry — Noether's theorem
# **§5** SiO₂ phonon modes — energy conservation from symmetry
# **§6** Animation: phonon oscillation + energy conservation live

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import sympy as sp
from sympy import (symbols, ln, exp, sqrt, diff, simplify, latex,
                   Eq, Symbol, sin, cos, pi, Rational, Function)
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
    if label: print(f"\n  {label}")
    if IN_JUPYTER: _D(expr)
    else: print("  " + sp.pretty(expr, use_unicode=True))

def hdr(s):
    bar = "─" * 64
    print(f"\n{bar}\n  {s}\n{bar}")

def chk(val, ref, label, tol=1e-6, absolute=False):
    try: v, r = float(val), float(ref)
    except: print(f"  [FAIL]  {label}"); return
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    print(f"  [{'PASS' if err<tol else 'FAIL'}]  {label}  got={v:.8g}  ref={r:.8g}")

print("=== Logarithmic Differentiation + SiO₂ Time-Translation Symmetry ===")

# %% [markdown]
# ---
# ## §1 · The Log-Diff Trick
#
# When f(x) is a **product of powers**, direct differentiation is painful.
# Log-diff shortcut:
#
# $$\ln f = \ln(\text{stuff}) \implies \frac{f'}{f} = \frac{d}{dx}\ln f
#   \implies \boxed{f' = f \cdot \frac{d}{dx}\ln f}$$
#
# **Why it's powerful**: ln turns products → sums, powers → multiplications.
# Differentiating a sum is always easier than differentiating a product.

# %%
hdr("§1 — Log-diff: the trick")

x = symbols("x", positive=True)

# Animated example: f(x) = x^x  (impossible without log-diff)
f_xx = x**x
lnf  = ln(f_xx)
show(Eq(Symbol("ln f"), sp.expand_log(lnf, force=True)),
     "Step 1 — take ln of f = xˣ:")
dlnf = diff(lnf, x)
show(Eq(Symbol("d/dx ln f"), simplify(dlnf)),
     "Step 2 — differentiate ln f:")
fprime = simplify(f_xx * dlnf)
show(Eq(Symbol("f'"), fprime),
     "Step 3 — multiply back by f:")

# Verify against direct diff (SymPy handles it)
direct = diff(x**x, x)
diff_check = simplify(fprime - direct)
chk(float(diff_check.subs(x, 2).evalf()), 0,
    "log-diff matches direct diff at x=2", absolute=True)
chk(float(diff_check.subs(x, 3).evalf()), 0,
    "log-diff matches direct diff at x=3", absolute=True)

# %% [markdown]
# ---
# ## §2 · Why It Works — Chain Rule Proof
#
# $$\frac{d}{dx}\ln f(x) = \frac{1}{f(x)} \cdot f'(x)$$
#
# Rearrange: $f'(x) = f(x) \cdot \frac{d}{dx}\ln f(x)$
#
# That's it. The chain rule applied to the composition ln∘f.
#
# **The ln turns**:
# - products → sums: $\ln(uv) = \ln u + \ln v$
# - quotients → differences: $\ln(u/v) = \ln u - \ln v$
# - powers → multiples: $\ln(u^n) = n\ln u$
#
# So you differentiate a **sum** instead of a **product** — always easier.

# %%
hdr("§2 — Proof via chain rule")

u, v = symbols("u v", positive=True)

# Show the log laws symbolically
f1 = u * v
f2 = u / v
f3 = u**3 * v**2

print("  Log laws that make diff easy:")
for f_e, lbl in [(f1,"u·v"), (f2,"u/v"), (f3,"u³v²")]:
    expanded = sp.expand_log(ln(f_e), force=True)
    show(Eq(Symbol(f"ln({lbl})"), expanded), f"ln({lbl}) =")

# Chain rule demonstration
f_chain = x**3 * sp.exp(x) * sp.sin(x)
ln_chain = sp.expand_log(ln(f_chain), force=True)
show(ln_chain, "ln(x³·eˣ·sin x) = sum of logs:")
d_ln = simplify(diff(ln_chain, x))
show(Eq(Symbol("d/dx ln f"), d_ln), "Differentiate the sum:")
fprime_chain = simplify(f_chain * d_ln)
direct_chain = diff(f_chain, x)
err_chain = simplify(fprime_chain - direct_chain)
chk(float(err_chain.subs(x, 1).evalf()), 0,
    "log-diff = direct for x³eˣsin(x) at x=1", absolute=True)

# %% [markdown]
# ---
# ## §3 · Classic Examples
#
# | f(x) | ln f | f′ via log-diff |
# |------|------|-----------------|
# | $x^x$ | $x\ln x$ | $x^x(\ln x + 1)$ |
# | $x^{\sin x}$ | $\sin x \cdot \ln x$ | $x^{\sin x}\left(\cos x \ln x + \frac{\sin x}{x}\right)$ |
# | $\frac{x^3(x+1)^2}{\sqrt{x^2+1}}$ | $3\ln x+2\ln(x+1)-\frac{1}{2}\ln(x^2+1)$ | one clean line |
# | $\prod_{k=1}^n f_k(x)$ | $\sum_{k=1}^n \ln f_k$ | $f\cdot\sum_k f_k'/f_k$ |

# %%
hdr("§3 — Classic examples verified")

examples = [
    (x**x,                               "xˣ"),
    (x**sp.sin(x),                       "x^{sin x}"),
    (x**3 * (x+1)**2 / sqrt(x**2+1),    "x³(x+1)²/√(x²+1)"),
    (x**Rational(1,3) * (2*x+1)**5,     "x^{1/3}(2x+1)⁵"),
]
for f_e, lbl in examples:
    ln_e   = sp.expand_log(ln(f_e), force=True)
    fprime_log = simplify(f_e * diff(ln_e, x))
    fprime_dir = diff(f_e, x)
    err = simplify(fprime_log - fprime_dir)
    v = float(err.subs(x, 2).evalf())
    chk(v, 0, f"log-diff for {lbl}", tol=1e-5, absolute=True)
    show(Eq(Symbol(f"({lbl})'"), simplify(fprime_log)), f"f′ for {lbl}:")

# %% [markdown]
# ---
# ## §4 · Time Translation Symmetry — Noether's Theorem
#
# **Noether's theorem** (1915): every continuous symmetry of the action
# corresponds to a conserved quantity.
#
# **Time translation symmetry**: if the laws of physics don't change with time
# (the Lagrangian has no explicit t dependence), then:
#
# $$\frac{\partial \mathcal{L}}{\partial t} = 0
#   \implies \frac{dE}{dt} = 0 \qquad \text{(energy is conserved)}$$
#
# **For a harmonic oscillator** (every phonon mode is one):
# $$\mathcal{L} = \tfrac{1}{2}m\dot{q}^2 - \tfrac{1}{2}m\omega^2 q^2$$
# No explicit t → energy $E = \tfrac{1}{2}m\dot{q}^2 + \tfrac{1}{2}m\omega^2 q^2$ = constant.

# %%
hdr("§4 — Noether: time translation → energy conservation")

t_s, m_s, omega_s, q0_s = symbols("t m omega q0", positive=True)

# Harmonic oscillator solution
q_t   = q0_s * cos(omega_s * t_s)
qdot  = diff(q_t, t_s)
T_kin = sp.Rational(1,2) * m_s * qdot**2
V_pot = sp.Rational(1,2) * m_s * omega_s**2 * q_t**2
E_tot = simplify(T_kin + V_pot)

show(Eq(Symbol("q(t)"), q_t),   "Position q(t):")
show(Eq(Symbol("T"), T_kin),    "Kinetic energy T:")
show(Eq(Symbol("V"), V_pot),    "Potential energy V:")
show(Eq(Symbol("E=T+V"), E_tot),"Total energy E=T+V:")

# Energy is constant (no t dependence in E)
dE_dt = simplify(diff(E_tot, t_s))
show(Eq(Symbol("dE/dt"), dE_dt), "dE/dt = 0 (time-translation symmetry):")
chk(float(dE_dt.evalf()), 0,
    "dE/dt = 0: energy conserved", tol=1e-10, absolute=True)

# Numerical verification
omega_n = 2*np.pi*1e12   # 1 THz phonon
m_n     = 28e-27         # Si mass kg (~28 amu)
q0_n    = 1e-11          # 0.01 nm amplitude
t_arr   = np.linspace(0, 3e-12, 10000)  # 3 ps
q_arr   = q0_n * np.cos(omega_n * t_arr)
v_arr   = -q0_n * omega_n * np.sin(omega_n * t_arr)
E_arr   = 0.5*m_n*v_arr**2 + 0.5*m_n*omega_n**2*q_arr**2
chk(np.std(E_arr)/np.mean(E_arr), 0,
    "E(t) = constant (numerical, <0.01% variation)", tol=1e-6, absolute=True)
print(f"  E = {np.mean(E_arr):.4e} J  (std/mean = {np.std(E_arr)/np.mean(E_arr):.2e})")

# %% [markdown]
# ---
# ## §5 · SiO₂ Phonon Modes
#
# Silica (SiO₂) has three atoms per primitive cell → 9 phonon branches.
# The IR-active modes (couple to light — relevant for THz + photonics):
#
# | Mode | Frequency | Physics |
# |------|-----------|---------|
# | Si-O-Si bending | ~7.5 THz (250 cm⁻¹) | ring deformation |
# | Si-O stretching | ~27 THz (900 cm⁻¹) | longitudinal optical |
# | Si-O-Si rocking | ~12 THz (400 cm⁻¹) | transverse |
#
# Each mode = independent harmonic oscillator.
# Time-translation symmetry → each mode's energy is separately conserved.
# This is why SiO₂ is a stable waveguide material — no energy leakage between modes.
#
# **Connection to photonics**: the dispersion of silica fiber (GVD, β₂) comes
# directly from the frequency dependence of these phonon resonances via
# the **Kramers-Kronig relations** (same ones from the causality notebook).

# %%
hdr("§5 — SiO₂ phonon modes and energy conservation")

# Three main IR modes of SiO2
modes = {
    "Si-O-Si bend":    {"freq_THz": 7.5,  "freq_cm": 250,  "color": "royalblue"},
    "Si-O-Si rock":    {"freq_THz": 12.0, "freq_cm": 400,  "color": "darkorange"},
    "Si-O stretch":    {"freq_THz": 27.0, "freq_cm": 900,  "color": "crimson"},
}

m_Si = 28e-27   # kg
m_O  = 16e-27   # kg
mu   = m_Si * m_O / (m_Si + m_O)   # reduced mass

t_phonon = np.linspace(0, 1e-12, 5000)  # 1 ps window

print("  Mode               freq(THz)  period(fs)  E per phonon (meV)")
print("  " + "─"*60)
for name, d in modes.items():
    omega_ph = 2*np.pi * d["freq_THz"] * 1e12
    T_ph     = 1 / (d["freq_THz"] * 1e12) * 1e15  # fs
    hbar     = 1.055e-34
    E_ph_meV = hbar * omega_ph / 1.6e-22   # meV
    print(f"  {name:<20} {d['freq_THz']:>6.1f}     {T_ph:>8.1f}      {E_ph_meV:>8.1f}")
    d["omega"] = omega_ph
    d["T_period"] = T_ph
    d["E_meV"] = E_ph_meV

# Verify energy conservation for each mode
print("\n  Energy conservation check (each mode independent):")
for name, d in modes.items():
    om = d["omega"]
    q  = 1e-11 * np.cos(om * t_phonon)
    v  = -1e-11 * om * np.sin(om * t_phonon)
    E  = 0.5 * mu * v**2 + 0.5 * mu * om**2 * q**2
    variation = np.std(E) / np.mean(E)
    chk(variation, 0, f"{name}: E conserved", tol=1e-6, absolute=True)

# Sellmeier dispersion of silica — consequence of these resonances
# n²(λ) = 1 + Σ Aᵢλ²/(λ²−λᵢ²)
print("\n  Sellmeier coefficients (silica):")
sellmeier = [
    (0.6961663, 0.0684043e-6),   # UV resonance
    (0.4079426, 0.1162414e-6),   # UV
    (0.8974794, 9.896161e-6),    # IR — the phonon resonances
]
lam_um = np.linspace(0.4, 2.0, 1000)   # wavelength in microns
lam_m  = lam_um * 1e-6
n2 = 1.0
for A, lam0 in sellmeier:
    n2 += A * lam_m**2 / (lam_m**2 - lam0**2)
n_arr = np.sqrt(n2)
# GVD: β₂ = (λ/2πc) d²n/dλ²
c_light = 3e8
dn  = np.gradient(n_arr, lam_m)
dn2 = np.gradient(dn, lam_m)
beta2 = (lam_m / (2*np.pi*c_light)) * dn2  # s²/m

# Zero dispersion wavelength
idx_zdw = np.argmin(np.abs(beta2))
zdw = lam_um[idx_zdw]
print(f"  Zero-dispersion wavelength: {zdw:.3f} μm  (theory ~1.27 μm)")
chk(zdw, 1.27, "ZDW of silica", tol=0.05)

# %% [markdown]
# ---
# ## §6 · Animations
#
# **Animation 1**: Logarithmic differentiation — watch f(x), ln f(x), and f′(x)
# build up simultaneously as x sweeps right.
#
# **Animation 2**: SiO₂ phonon modes oscillating — three frequencies,
# energy bar staying perfectly flat (time-translation symmetry live).

# %%
hdr("§6 — Animations")

# ─── Animation 1: Log differentiation of f(x) = x^x ─────────────────────────
x_anim = np.linspace(0.1, 3.0, 300)
f_vals  = x_anim ** x_anim
lnf_vals = np.log(f_vals)
fp_vals  = f_vals * (np.log(x_anim) + 1)   # x^x(ln x + 1)

fig1, axes1 = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
fig1.suptitle("Logarithmic Differentiation of f(x) = xˣ", fontsize=13, fontweight="bold")

titles = ["f(x) = xˣ", "ln f(x) = x·ln x", "f′(x) = xˣ(ln x + 1)"]
colors = ["royalblue", "darkorange", "crimson"]
data   = [f_vals, lnf_vals, fp_vals]
ylims  = [(0, 30), (-2, 4), (-5, 80)]

lines1 = []
for ax, title, color, ylim in zip(axes1, titles, colors, ylims):
    ax.set_xlim(0.1, 3.0)
    ax.set_ylim(ylim)
    ax.set_ylabel(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ln_obj, = ax.plot([], [], color=color, linewidth=2.5)
    lines1.append(ln_obj)
    # Ghost of full curve
    ax.plot(x_anim, data[titles.index(title)], color=color, alpha=0.12, linewidth=1)

axes1[-1].set_xlabel("x")

# Annotation showing the chain rule step
ann = axes1[0].annotate("", xy=(0.1, 1), fontsize=9, color="gray")

def init1():
    for ln_obj in lines1:
        ln_obj.set_data([], [])
    return lines1

def update1(frame):
    n = frame + 1
    for i, (ln_obj, d) in enumerate(zip(lines1, data)):
        ln_obj.set_data(x_anim[:n], d[:n])
    # Show current x value
    axes1[0].set_title(
        f"f(x)=xˣ   |   log-diff step by step   |   x = {x_anim[min(n-1,299)]:.2f}",
        fontsize=10)
    return lines1

anim1 = animation.FuncAnimation(
    fig1, update1, init_func=init1,
    frames=300, interval=20, blit=True)

anim1.save("repl/_anim_logdiff.gif", writer="pillow", fps=30)
plt.close(fig1)
print("  Saved: repl/_anim_logdiff.gif")

if IN_JUPYTER:
    _D(_HTML('<img src="repl/_anim_logdiff.gif">'))

# ─── Animation 2: SiO₂ phonon modes + energy conservation ────────────────────
t_anim  = np.linspace(0, 1e-12, 400)   # 1 ps
mode_list = list(modes.items())

fig2 = plt.figure(figsize=(11, 7))
gs2  = GridSpec(3, 2, figure=fig2, hspace=0.5, wspace=0.35)

ax_modes = [fig2.add_subplot(gs2[i, 0]) for i in range(3)]
ax_energy = fig2.add_subplot(gs2[:, 1])

fig2.suptitle("SiO₂ Phonon Modes — Time-Translation Symmetry → E = const",
              fontsize=11, fontweight="bold")

q0 = 1e-11   # 0.01 nm amplitude
phonon_lines  = []
energy_lines  = []
energy_data   = []

for i, (name, d) in enumerate(mode_list):
    om = d["omega"]
    ax = ax_modes[i]
    ax.set_xlim(0, 1e-12)
    ax.set_ylim(-1.5*q0, 1.5*q0)
    ax.set_ylabel(f"{name}\n({d['freq_THz']:.0f} THz)", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    # Ghost
    q_full = q0 * np.cos(om * t_anim)
    ax.plot(t_anim, q_full, color=d["color"], alpha=0.1, linewidth=1)
    ln_ph, = ax.plot([], [], color=d["color"], linewidth=2)
    phonon_lines.append(ln_ph)

    # Energy
    v_full = -q0 * om * np.sin(om * t_anim)
    E_full = 0.5*mu*v_full**2 + 0.5*mu*om**2*q_full**2
    energy_data.append(E_full)

ax_modes[-1].set_xlabel("time (s)")

# Energy panel
ax_energy.set_xlim(0, 1e-12)
all_E = np.array(energy_data)
E_max = np.max(all_E) * 1.3
ax_energy.set_ylim(0, E_max)
ax_energy.set_xlabel("time (s)")
ax_energy.set_ylabel("Energy (J)")
ax_energy.set_title("Each mode: E(t) = constant\n(Noether: time symmetry → E conserved)",
                     fontsize=8)
ax_energy.grid(True, alpha=0.3)

for i, (name, d) in enumerate(mode_list):
    # Ghost full energy curve
    ax_energy.plot(t_anim, energy_data[i], color=d["color"], alpha=0.1, linewidth=1)
    ln_e, = ax_energy.plot([], [], color=d["color"], linewidth=2,
                            label=f"{name}")
    energy_lines.append(ln_e)

ax_energy.legend(fontsize=7, loc="upper right")

def init2():
    for ln in phonon_lines + energy_lines:
        ln.set_data([], [])
    return phonon_lines + energy_lines

def update2(frame):
    n = frame + 1
    for i, (ln_ph, ln_e) in enumerate(zip(phonon_lines, energy_lines)):
        om = mode_list[i][1]["omega"]
        ln_ph.set_data(t_anim[:n], q0 * np.cos(om * t_anim[:n]))
        ln_e.set_data(t_anim[:n], energy_data[i][:n])
    return phonon_lines + energy_lines

anim2 = animation.FuncAnimation(
    fig2, update2, init_func=init2,
    frames=400, interval=25, blit=True)

anim2.save("repl/_anim_sio2_phonons.gif", writer="pillow", fps=30)
plt.close(fig2)
print("  Saved: repl/_anim_sio2_phonons.gif")

if IN_JUPYTER:
    _D(_HTML('<img src="repl/_anim_sio2_phonons.gif">'))

# %% [markdown]
# ---
# ## Summary
#
# | Concept | Key equation | Why it matters |
# |---------|-------------|----------------|
# | Log-diff | $f' = f \cdot \frac{d}{dx}\ln f$ | products/powers become sums |
# | Chain rule | $\frac{d}{dx}\ln f = f'/f$ | the proof behind the trick |
# | Noether | $\partial\mathcal{L}/\partial t=0 \implies dE/dt=0$ | symmetry → conservation |
# | Phonon modes | $E = \frac{1}{2}m\dot{q}^2 + \frac{1}{2}m\omega^2 q^2$ | each mode independent |
# | SiO₂ ZDW | β₂=0 at ~1.27 μm | Sellmeier from phonon resonances |
# | D-GS connection | GVD β₂ comes from IR phonons | the D in your D-GS |
#
# **The thread**: log-diff and Noether both say the same thing —
# *find the right transformation that turns hard structure into easy structure.*
# Log turns products into sums. Symmetry turns dynamics into conservation laws.

# %%
hdr("Done — animations saved")
print("  repl/_anim_logdiff.gif      — log differentiation of xˣ animated")
print("  repl/_anim_sio2_phonons.gif — SiO₂ phonon modes + energy conservation")
print()
print("  In Jupyter: animations display inline via HTML.")
print("  In VS Code: open the .gif files directly.")
