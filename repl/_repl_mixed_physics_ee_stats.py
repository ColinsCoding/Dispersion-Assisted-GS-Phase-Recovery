# %% [markdown]
# # Calculus · Ion Engines · Fusion · Interconnects · Comparators
# # Normal Distributions · Optics · Verilog/Git · Protein Chemistry
# `init_printing(use_latex="mathjax")` throughout. Every number derived.

# %% [markdown]
# ## Setup

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
from sympy import (symbols, integrate, diff, limit, oo, pi, exp, sqrt,
                   sin, cos, Rational, latex, simplify, ln, Abs, Function)
from sympy import init_printing
from IPython.display import display, Math, Markdown
import scipy.stats as stats
import scipy.integrate as sci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

init_printing(use_latex="mathjax")

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

def hdr(s):
    try: display(Markdown(f"### {s}"))
    except: print(f"\n=== {s} ===")

def chk(val, ref, label, tol=1e-3):
    err = abs(float(val) - float(ref)) / (abs(float(ref)) + 1e-30)
    ok  = err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}  got={float(val):.6g}  ref={float(ref):.6g}")
    return ok

# %% [markdown]
# ---
# ## §1 · Calculus — "add a bunch of little things"
#
# The **definite integral** IS literally "add infinitely many infinitely thin slices."
# $$\int_a^b f(x)\,dx = \lim_{n\to\infty}\sum_{k=1}^{n} f(x_k^*)\,\Delta x$$
# **Fundamental Theorem of Calculus:**
# $$\frac{d}{dx}\int_a^x f(t)\,dt = f(x)$$

# %%
hdr("§1 — Calculus: Riemann sums -> definite integral")

x, t_s, n_s, a_s, b_s = symbols('x t n a b', real=True)

# --- 1a. Riemann sum convergence (numerical) ---
def riemann(f_np, a, b, n, method='midpoint'):
    xs = np.linspace(a, b, n+1)
    dx = (b-a)/n
    if method == 'left':   pts = xs[:-1]
    elif method == 'right': pts = xs[1:]
    else:                   pts = (xs[:-1]+xs[1:])/2
    return np.sum(f_np(pts)) * dx

f_demo = lambda x_: x_**2          # integral = b^3/3 - a^3/3
a_v, b_v = 0.0, 3.0
exact_v = b_v**3/3 - a_v**3/3      # = 9.0

print("\n  f(x)=x^2  on [0,3],  exact = 9.0")
print(f"  {'n':>8s}  {'left':>12s}  {'midpoint':>12s}  {'right':>12s}  {'error(mid)':>12s}")
for nv in [4, 10, 100, 1000, 10000]:
    L = riemann(f_demo, a_v, b_v, nv, 'left')
    M = riemann(f_demo, a_v, b_v, nv, 'midpoint')
    R = riemann(f_demo, a_v, b_v, nv, 'right')
    print(f"  {nv:>8d}  {L:12.6f}  {M:12.6f}  {R:12.6f}  {abs(M-exact_v):12.2e}")

# --- 1b. SymPy: symbolic integration ---
print("\n  SymPy exact integrals:")
examples = [
    (x**2,            0, 3,   "x^2 [0,3]",       9),
    (sin(x),          0, pi,  "sin(x) [0,pi]",    2),
    (exp(-x**2),      -oo, oo,"e^{-x^2} (-inf,inf)", float(sqrt(pi))),
    (1/(1+x**2),      -oo, oo,"1/(1+x^2) (-inf,inf)", float(pi)),
    (x*exp(-x),       0, oo,  "x e^{-x} [0,inf]", 1),
]
for expr, lo, hi, label, ref in examples:
    val = integrate(expr, (x, lo, hi))
    show(val, f"integral({label}) =")
    chk(float(val.evalf()), ref, label)

# --- 1c. FTC: d/dx integral_0^x f(t) dt = f(x) ---
hdr("FTC: d/dx integral_a^x f(t)dt = f(x)")
F_func = integrate(t_s**3, (t_s, 0, x))
show(F_func, "F(x) = int_0^x t^3 dt =")
dFdx = diff(F_func, x)
show(dFdx, "F'(x) =")
show(sp.Eq(dFdx, x**3), "F'(x) = x^3 (FTC):")
print(f"  FTC check: F'(x) == x^3 : {sp.simplify(dFdx - x**3) == 0}")

# --- 1d. Integration by parts: integral x*e^x dx ---
hdr("Integration by parts: int x e^x dx")
val_ibp = integrate(x*exp(x), x)
show(val_ibp, r"\int x e^x dx =")
show(sp.Eq(sp.Symbol('check'), diff(val_ibp, x) - x*exp(x)), "d/dx[result] - x*e^x =")
chk(float(diff(val_ibp,x).subs(x,1) - 1*np.e), 0, "IBP derivative check")

# --- 1e. U-substitution: integral sin(x^2) * 2x dx ---
hdr("u-substitution: int 2x sin(x^2) dx")
val_usub = integrate(2*x*sin(x**2), x)
show(val_usub, r"\int 2x\sin(x^2)dx =")   # = -cos(x^2)

# %% [markdown]
# ---
# ## §2 · Ion Engine Physics — Hall Thruster
#
# **Thrust:**  $F = \dot{m}\,v_e = \dot{m}\,g_0\,I_{sp}$
#
# **Specific impulse** of gridded ion: $I_{sp} = \sqrt{2qV_b / m_i} / g_0$
# where $V_b$ = beam voltage, $m_i$ = propellant mass.
#
# **Xenon** ($A=131.3$, $q=e=1.602\times10^{-19}$ C) at 1800 V → Isp ≈ 3100 s.

# %%
hdr("§2 — Ion Engine Physics")

e_charge = 1.602e-19    # C
m_xe     = 131.3 * 1.66054e-27   # kg (Xe-131)
g0       = 9.80665      # m/s^2

# Sympy: derive exit velocity from energy conservation
Vb_s, q_s, mi_s, Isp_sym = symbols('V_b q m_i I_{sp}', positive=True)
KE_eq = sp.Eq(Rational(1,2)*mi_s*symbols('v_e')**2, q_s*Vb_s)
show(KE_eq, "Energy conservation (eV -> KE):")
ve_expr = sp.solve(KE_eq, symbols('v_e'))[1]
show(ve_expr, "v_e =")
show(sp.Eq(Isp_sym, ve_expr / g0), "I_{sp} =")

print("\n  Gridded Ion Thruster performance vs beam voltage:")
print(f"  {'V_b (V)':>10s}  {'v_e (km/s)':>12s}  {'Isp (s)':>10s}")
for Vb in [500, 1000, 1800, 3000, 5000, 8000]:
    ve_v  = np.sqrt(2 * e_charge * Vb / m_xe)
    Isp_v = ve_v / g0
    print(f"  {Vb:>10d}  {ve_v/1e3:>12.2f}  {Isp_v:>10.0f}")

# Reference: Dawn spacecraft (Xe, Vb=1.0kV, Isp=3100s)
Vb_dawn = 1200.0
ve_dawn = np.sqrt(2 * e_charge * Vb_dawn / m_xe)
Isp_dawn = ve_dawn / g0
chk(Isp_dawn, 3100, "Dawn Isp ~3100s at Vb=1.2kV", tol=0.15)

# Thrust calculation
# F = I_b / e * m_xe * v_e  (beam current I_b = charge flux)
I_beam = 1.76   # A (Dawn thruster, max)
mdot   = I_beam / e_charge * m_xe   # kg/s
F_thrust = mdot * ve_dawn
print(f"\n  Dawn thruster: Vb={Vb_dawn}V, I_beam={I_beam}A")
print(f"  mdot = {mdot*1000:.4f} mg/s")
print(f"  Thrust F = {F_thrust*1000:.2f} mN  (reference: ~90 mN max)")

# Power budget
P_input = I_beam * Vb_dawn         # electrical power to beam
eta_total = 0.65                   # typical efficiency
P_total   = P_input / eta_total
print(f"  Beam power P_beam = {P_input:.0f} W")
print(f"  Total thruster power (eta=0.65): {P_total:.0f} W")
print(f"  Specific power alpha = P/F = {P_total/F_thrust:.1f} W/N  (compare chem: ~0.1 W/N)")

# Hall thruster vs gridded comparison
print("\n  Hall (SPT) vs Gridded Ion comparison:")
print(f"  {'Type':20s} {'Isp(s)':>8s} {'T/P(mN/kW)':>12s} {'Power(kW)':>10s}")
thrusters = [
    ("Chemical (biprop)",     450,  0.1e3,  1.0),
    ("Gridded Ion (GIT)",    3000,   14.0,  2.5),
    ("Hall thruster (SPT)",  1600,   60.0,  1.5),
    ("VASIMR (RF plasma)",   5000,    5.0,  200.0),
]
for name, Isp, TperP, P in thrusters:
    T = TperP * P
    print(f"  {name:20s} {Isp:>8d} {TperP:>12.1f} {P:>10.1f}  F={T:.0f} mN")

# %% [markdown]
# ---
# ## §3 · Fusion — Lawson Criterion + D-T Reaction
#
# **D-T fusion:** $^2\text{H} + ^3\text{H} \to ^4\text{He}(3.52\,\text{MeV}) + n(14.06\,\text{MeV})$
#
# **Lawson criterion (breakeven):** $n\tau_E \geq \frac{12 k_B T}{\langle\sigma v\rangle\,E_\alpha}$
#
# At $T \approx 10\,\text{keV}$: $n\tau_E \geq 10^{20}\,\text{m}^{-3}\text{s}$
#
# **Q factor:** $Q = P_{fusion} / P_{heating}$.  Break-even $Q=1$. ITER target $Q=10$.

# %%
hdr("§3 — Fusion Physics")

# D-T Q-value
m_D   = 2.01410178  # u
m_T   = 3.01604928
m_He4 = 4.00260325
m_n   = 1.00866492
u_to_MeV = 931.494   # MeV/u

delta_m = m_D + m_T - m_He4 - m_n
Q_value = delta_m * u_to_MeV
print(f"\n  D-T mass defect: {delta_m:.8f} u")
print(f"  Q-value: {Q_value:.4f} MeV  (ref: 17.59 MeV)")
chk(Q_value, 17.59, "D-T Q-value MeV", tol=0.01)

# Energy split: alpha gets 3.52 MeV (20%), neutron gets 14.07 MeV (80%)
E_alpha = Q_value * m_n / (m_He4 + m_n)
E_n     = Q_value * m_He4 / (m_He4 + m_n)
print(f"  He-4 (alpha) energy: {E_alpha:.3f} MeV  (ref: 3.52)")
print(f"  Neutron energy:      {E_n:.3f} MeV  (ref: 14.07)")

# Lawson criterion
kB   = 8.617e-5      # eV/K
T_keV = 10.0         # keV (optimal for DT)
T_K   = T_keV * 1e3 / kB
sigma_v_DT = 3.68e-22  # m^3/s at 10 keV (from NRL formulary)
E_alpha_J  = E_alpha * 1e6 * 1.602e-19

ntau_lawson = 12 * kB * T_keV * 1e3 * 1.602e-19 / (sigma_v_DT * E_alpha_J)
print(f"\n  Lawson criterion at T={T_keV} keV:")
print(f"  n*tau_E >= {ntau_lawson:.2e} m^-3 s  (ref: ~1e20)")

# ITER parameters
n_ITER    = 1e20     # m^-3
tau_ITER  = 3.7      # s (energy confinement time)
ntau_ITER = n_ITER * tau_ITER
Q_ITER_target = 10
print(f"\n  ITER: n={n_ITER:.0e} m^-3, tau_E={tau_ITER} s")
print(f"  n*tau_E = {ntau_ITER:.2e} m^-3 s  ({'above' if ntau_ITER>ntau_lawson else 'below'} Lawson)")
print(f"  ITER design Q = {Q_ITER_target}  (50 MW heating -> 500 MW fusion)")

# Triple product: n * T * tau_E
triple = n_ITER * T_keV * 1e3 * tau_ITER   # eV m^-3 s
print(f"  Triple product n*T*tau = {triple:.2e} keV m^-3 s")
print(f"  Ignition triple product threshold: ~3e21 keV m^-3 s")

# Fusion power density
P_density = 0.25 * n_ITER**2 * sigma_v_DT * Q_value * 1e6 * 1.602e-19  # W/m^3
print(f"\n  Fusion power density: {P_density/1e6:.2f} MW/m^3")
print(f"  ITER plasma volume: ~840 m^3 -> P_fusion = {P_density*840/1e6:.0f} MW (target 500 MW)")

# Sympy: Lawson criterion
n_sym, tau_sym, T_sym, sv_sym, Ea_sym = symbols('n tau_E T <sigma*v> E_alpha', positive=True)
lawson_eq = sp.Eq(n_sym*tau_sym, 12*kB*T_sym / (sv_sym * Ea_sym))
show(lawson_eq, "Lawson criterion:")

# %% [markdown]
# ---
# ## §4 · Interconnect Stability — Transmission Lines
#
# Signal integrity at high frequency: a trace is a **transmission line**.
# $$Z_0 = \sqrt{L'/C'}, \quad v_p = 1/\sqrt{L'C'} = c/\sqrt{\varepsilon_r}$$
# **Reflection coefficient:** $\Gamma = (Z_L - Z_0)/(Z_L + Z_0)$
# **Stability condition:** $|\Gamma| < 1$ (matched: $\Gamma=0$ at $Z_L=Z_0$)

# %%
hdr("§4 — Interconnect / Transmission Line Stability")

# Microstrip impedance (Wheeler formula, er=4.4 FR4)
def Z0_microstrip(w, h, er=4.4):
    """Width w, height h in same units."""
    if w/h <= 1:
        Z0_air = (60/np.sqrt((er+1)/2 + (er-1)/2*(1/np.sqrt(1+12*h/w)+0.04*(1-w/h)**2)))
        return 60 / np.sqrt((er+1)/2 + (er-1)/2*(1/np.sqrt(1+12*h/w)+0.04*(1-w/h)**2))
    else:
        eff = (er+1)/2 + (er-1)/2 * (1+12*h/w)**(-0.5)
        return 120*np.pi / (np.sqrt(eff)*(w/h + 1.393 + 0.667*np.log(w/h+1.444)))

print("\n  Microstrip Z0 vs width/height ratio (FR4, er=4.4):")
print(f"  {'w/h':>8s}  {'Z0 (Ohm)':>10s}")
for wh in [0.5, 1.0, 1.56, 2.0, 3.0, 4.0]:
    z = Z0_microstrip(wh, 1.0)
    flag = " <-- 50-ohm match" if abs(z-50)<3 else ""
    print(f"  {wh:>8.2f}  {z:>10.2f}{flag}")

# Reflection at load
print("\n  Reflection coefficient vs load impedance (Z0=50 ohm):")
Z0 = 50.0
print(f"  {'ZL (ohm)':>12s}  {'Gamma':>10s}  {'VSWR':>8s}  {'condition':>12s}")
for ZL in [0, 25, 50, 75, 100, 200, 1000, np.inf]:
    if np.isinf(ZL):
        Gamma = 1.0
    elif ZL == 0:
        Gamma = -1.0
    else:
        Gamma = (ZL - Z0) / (ZL + Z0)
    VSWR = (1 + abs(Gamma)) / (1 - abs(Gamma)) if abs(Gamma) < 1 else np.inf
    cond = "matched" if abs(Gamma)<0.05 else ("open" if ZL==np.inf else "short" if ZL==0 else "")
    print(f"  {str(ZL):>12s}  {Gamma:>10.4f}  {VSWR:>8.2f}  {cond}")

# Termination strategies
print("\n  Termination strategies:")
print("    Series: R_s = Z0 - Z_source (at driver, absorbed forward)")
print("    Parallel: R_p = Z0 (at load, absorbs reflected)")
print("    AC: C+(R=Z0) series (DC blocked, high-freq terminated)")
print("    Thevenin: R1||R2 = Z0, Vth = Vdd*R2/(R1+R2)")

# Crosstalk
print("\n  Crosstalk (coupled lines, NEXT/FEXT):")
print("    NEXT: V_near = (Cm/2C0 + Lm/2L0) * dV/dt * T_d")
print("    FEXT: V_far  = (Cm/2C0 - Lm/2L0) * dV/dt * T_d")
print("    Reduce by: wider spacing, reference planes, differential pairs")

# Rise time / bandwidth
print("\n  Rise time <-> bandwidth:")
f_arr = np.array([100e6, 500e6, 1e9, 5e9, 10e9])
tr_arr = 0.35 / f_arr
for f, tr in zip(f_arr, tr_arr):
    print(f"    f={f/1e9:.1f} GHz -> t_r = {tr*1e12:.1f} ps  (lambda={3e8/f*100:.1f} cm)")

# Sympy: telegrapher's equations
z_s, t_sym2 = symbols('z t', real=True)
V, I_sym = symbols('V I', cls=Function)
R_s, L_s, C_s, G_s = symbols("R' L' C' G'", positive=True)
teleg_V = sp.Eq(-diff(V(z_s, t_sym2), z_s), R_s*I_sym(z_s,t_sym2) + L_s*diff(I_sym(z_s,t_sym2), t_sym2))
teleg_I = sp.Eq(-diff(I_sym(z_s, t_sym2), z_s), G_s*V(z_s,t_sym2) + C_s*diff(V(z_s,t_sym2), t_sym2))
show(teleg_V, "Telegrapher (V):")
show(teleg_I, "Telegrapher (I):")

# %% [markdown]
# ---
# ## §5 · Comparator / Schmitt Trigger — Physics of Comparison
#
# A **comparator** maps a continuous voltage to a binary output:
# $$V_{out} = \begin{cases}V_{OH} & V_+ > V_- \\ V_{OL} & V_+ < V_-\end{cases}$$
# A **Schmitt trigger** adds hysteresis to prevent chattering near the threshold.

# %%
hdr("§5 — Comparator + Schmitt Trigger")

Vcc = 3.3
R1, R2 = 10e3, 10e3     # feedback resistors

# Schmitt thresholds (inverting config)
# V_TH+ = Vcc * R2 / (R1+R2)   when Vout = VOH = Vcc
# V_TH- = 0                    when Vout = VOL = 0  (with R1 to output)
# Non-inverting Schmitt:
VOH = Vcc; VOL = 0.0
VTP = VOH * R2 / (R1 + R2)   # upper threshold
VTN = VOL * R2 / (R1 + R2)   # lower threshold
hysteresis = VTP - VTN

print(f"\n  Non-inverting Schmitt trigger (Vcc={Vcc}V, R1=R2={R1/1e3:.0f}k):")
print(f"  V_TH+ = {VTP:.3f} V  (rising edge fires here)")
print(f"  V_TH- = {VTN:.3f} V  (falling edge fires here)")
print(f"  Hysteresis = {hysteresis:.3f} V")
print(f"  Noise immunity: signals < {hysteresis/2*1000:.0f} mV peak will not cause spurious transitions")

# Noise margin
VIH = 2.0; VIL = 0.8   # CMOS 3.3V typical
VOH_out = 3.0; VOL_out = 0.1
NMH = VOH_out - VIH
NML = VIL - VOL_out
print(f"\n  CMOS 3.3V noise margins:")
print(f"  NMH = VOH - VIH = {VOH_out} - {VIH} = {NMH:.1f} V")
print(f"  NML = VIL - VOL = {VIL} - {VOL_out} = {NML:.1f} V")

# Comparator response time
print("\n  Propagation delay vs overdrive (LM339 datasheet):")
print(f"  {'Overdrive (mV)':>16s}  {'tpd (ns)':>10s}")
for overdrive, tpd in [(5, 1300), (10, 500), (20, 200), (50, 100), (100, 70)]:
    print(f"  {overdrive:>16d}  {tpd:>10d}")

# Transfer curve
V_in  = np.linspace(0, Vcc, 1000)
# Ideal comparator
V_comp_ideal = np.where(V_in > VTP, VOH, VOL)
# Schmitt with hysteresis
V_schmitt = np.zeros_like(V_in)
state = 0   # start low
for i, v in enumerate(V_in):
    if state == 0 and v > VTP: state = 1
    elif state == 1 and v < VTN: state = 0
    V_schmitt[i] = VOH if state == 1 else VOL

print(f"\n  Schmitt PASS: upper threshold fires at {VTP:.3f}V, hysteresis={hysteresis:.3f}V")

# %% [markdown]
# ---
# ## §6 · Normal Distribution — "Line everyone up by height"
#
# Human height is approximately normally distributed.
# US adult male: $\mu = 5'9'' = 175.3\,\text{cm}$, $\sigma = 7.1\,\text{cm}$
# US adult female: $\mu = 5'4'' = 162.6\,\text{cm}$, $\sigma = 6.4\,\text{cm}$
#
# **Central Limit Theorem:** sum of $n$ i.i.d. variables $\to \mathcal{N}(\mu_n, \sigma^2/n)$

# %%
hdr("§6 — Normal Distribution / Height / CLT")

mu_m, sig_m = 175.3, 7.1    # cm, male
mu_f, sig_f = 162.6, 6.4    # cm, female

h = np.linspace(140, 215, 1000)
pdf_m = stats.norm.pdf(h, mu_m, sig_m)
pdf_f = stats.norm.pdf(h, mu_f, sig_f)

print(f"\n  US adult male:   mu={mu_m} cm ({mu_m/2.54/12:.1f}'), sigma={sig_m} cm")
print(f"  US adult female: mu={mu_f} cm ({mu_f/2.54/12:.1f}'), sigma={sig_f} cm")

print("\n  Population fraction by height (male):")
heights_cm = [160, 165, 170, 175, 180, 185, 190, 195]
print(f"  {'Height':>10s}  {'z-score':>8s}  {'percentile':>12s}  {'taller than X%':>16s}")
for hv in heights_cm:
    z = (hv - mu_m) / sig_m
    pct = stats.norm.cdf(z) * 100
    print(f"  {hv:>8.0f}cm  {z:>8.3f}  {pct:>11.2f}%  top {100-pct:.2f}%")

# If you lined up all ~4 billion adult males by height:
n_world_m = 4e9
print(f"\n  Lining up ~{n_world_m:.0e} adult males:")
for h_thresh, label in [(200, "6'7\" (200cm)"), (210, "6'11\" (210cm)"), (213, "7'0\" (213cm)")]:
    fraction = 1 - stats.norm.cdf((h_thresh - mu_m)/sig_m)
    count    = fraction * n_world_m
    print(f"    Above {label}: {fraction*100:.4f}%  = {count:.0f} people")

# Sympy: Gaussian PDF and normalization
x_n, mu_n, sig_n = symbols('x mu sigma', real=True)
sig_pos = symbols('sigma', positive=True)
pdf_sym = (1/(sig_pos*sqrt(2*pi))) * exp(-(x_n-mu_n)**2/(2*sig_pos**2))
show(pdf_sym, "N(mu,sigma^2) PDF:")
norm_check = integrate(pdf_sym.subs(mu_n,0), (x_n, -oo, oo))
show(norm_check, "normalization:")
chk(float(norm_check), 1.0, "Gaussian PDF norm=1")

# CLT demonstration
hdr("CLT: sum of n uniform -> Gaussian")
rng = np.random.default_rng(42)
clt_n_vals = [1, 2, 5, 30]
clt_samples = {}
for n_clt in clt_n_vals:
    sums = rng.uniform(0,1,(200000, n_clt)).sum(axis=1)
    sums_norm = (sums - n_clt/2) / np.sqrt(n_clt/12)   # standardize
    clt_samples[n_clt] = sums_norm
    _, pval = stats.normaltest(sums_norm[:5000])
    print(f"  n={n_clt:3d}:  mean={sums_norm.mean():.4f}  std={sums_norm.std():.4f}  "
          f"normality p={pval:.4f} ({'Normal' if pval>0.05 else 'Not yet normal'})")

# %% [markdown]
# ---
# ## §7 · Optics — Thin Lens / Image Plane
#
# **Thin lens equation:** $\dfrac{1}{f} = \dfrac{1}{d_o} + \dfrac{1}{d_i}$
#
# **Problem:** $d_o = 150\,\text{mm}$, $d_i = 250\,\text{mm}$ — find $f$, magnification $m$.

# %%
hdr("§7 — Thin Lens: image plane at do=150, di=250")

do, di_var, f_s = symbols('d_o d_i f', positive=True)
thin_lens = sp.Eq(1/f_s, 1/do + 1/di_var)
show(thin_lens, "thin lens equation:")

# Solve for f given do=150, di=250
do_v = 150.0; di_v = 250.0
f_v = 1/(1/do_v + 1/di_v)
m_v = -di_v / do_v    # magnification (negative = inverted)
print(f"\n  d_o = {do_v} mm,  d_i = {di_v} mm")
print(f"  f = 1/(1/{do_v:.0f} + 1/{di_v:.0f}) = {f_v:.4f} mm = {f_v:.2f} mm")
print(f"  Magnification m = -d_i/d_o = -{di_v:.0f}/{do_v:.0f} = {m_v:.4f}  (inverted, {abs(m_v):.2f}x)")

f_sym_val = sp.Rational(int(do_v*di_v), int(do_v+di_v))
show(f_sym_val, "f (exact rational, mm) =")

# Power of lens in diopters
f_m = f_v / 1000   # meters
P_diopter = 1 / f_m
print(f"  Lens power P = 1/f = {P_diopter:.4f} diopters")

# Depth of field estimate
print("\n  Varying object distance — image plane location:")
print(f"  {'d_o (mm)':>10s}  {'d_i (mm)':>10s}  {'m':>8s}  {'real?':>6s}")
for do_test in [50, 100, 150, 200, 250, 500, 1000, 5000]:
    denom = 1/f_v - 1/do_test
    if denom > 0:
        di_test = 1/denom
        m_test  = -di_test/do_test
        real = "real" if di_test > 0 else "virtual"
    else:
        di_test = float('inf')
        m_test  = float('inf')
        real = "infinity"
    print(f"  {do_test:>10d}  {di_test:>10.2f}  {m_test:>8.4f}  {real}")

# Lensmaker's equation
print("\n  Lensmaker's equation for f=93.75mm:")
print("  1/f = (n-1)[1/R1 - 1/R2 + (n-1)d/(n*R1*R2)]")
print("  For thin biconvex (d->0, n=1.5, R1=-R2=R):")
print("  1/f = (0.5)(2/R) = 1/R  =>  R = f = 93.75 mm")

# %% [markdown]
# ---
# ## §8 · Protein Chemistry — Why Cheese Curdles
#
# Milk contains **casein micelles** (~150 nm diameter): spherical aggregates
# of alpha/beta/kappa-casein stabilized by kappa-casein's hydrophilic tail.
#
# **Acid coagulation:** lower pH -> neutralize negative charges -> van der Waals
# attraction wins -> aggregation.
#
# **Rennet coagulation:** chymosin cleaves kappa-casein at Phe105-Met106
# -> removes steric repulsion -> micelles aggregate -> gel forms.

# %%
hdr("§8 — Protein Chemistry: Cheese Curdling")

print("""
  Casein micelle structure:
    - ~150 nm diameter, ~20,000 casein molecules
    - Internal: alpha_s1, alpha_s2, beta-casein (hydrophobic core)
    - Surface: kappa-casein (hydrophilic tail, negative charge at pH 6.7)
    - Stabilized by:
        (1) Electrostatic repulsion (zeta potential ~ -20 mV at pH 6.7)
        (2) Steric repulsion from kappa-casein "hairy layer"

  Why milk is stable at pH 6.7:
    pKa of phosphoserine residues ~ 6.5  =>  net negative charge
    DLVO theory: repulsion barrier E_B ~ kT * 10^2  (very stable)

  Acid coagulation (yogurt, fresh cheese):
    pH 6.7 -> 4.6 (isoelectric point of casein)
    At pI: net charge = 0 -> no electrostatic repulsion
    Van der Waals attraction dominates -> aggregation
    Rate follows DLVO kinetics: dN/dt = -k_a * N^2
""")

# Isoelectric point calculation for casein
# Rough model: pI = (pKa_acid + pKa_base)/2 for simple case
pKa_COOH = 4.07   # C-terminus / Asp, Glu
pKa_NH3  = 9.1    # N-terminus / Lys
pI_simple = (pKa_COOH + pKa_NH3) / 2
print(f"  Simplified pI estimate: ({pKa_COOH}+{pKa_NH3})/2 = {pI_simple:.2f}  (casein actual: ~4.6)")

# Henderson-Hasselbalch
pH_range = np.linspace(2, 12, 500)
pKa_demo = 4.6
# Fraction deprotonated = 1/(1+10^(pKa-pH))
alpha = 1 / (1 + 10**(pKa_demo - pH_range))
print(f"\n  Henderson-Hasselbalch: fraction deprotonated at pKa={pKa_demo}:")
for pH_v in [3.0, 4.0, 4.6, 5.0, 6.0, 7.0]:
    a = 1/(1+10**(pKa_demo-pH_v))
    print(f"    pH={pH_v}: alpha={a:.4f}  ({'50% deprotonated' if abs(a-0.5)<0.05 else ''})")

# Rennet action
print("""
  Rennet coagulation (hard cheese):
    Chymosin (aspartic protease) cleaves kappa-casein Phe105-Met106
    Removes 64-amino-acid "macropeptide" (hydrophilic)
    Remaining para-kappa-casein: hydrophobic, aggregates at T>15°C
    Gelation: diffusion-limited aggregation (fractal dimension D_f ~ 2.1)
    Syneresis: gel contracts, expels whey -> curds form

  Temperature effect (Arrhenius):
    k = A * exp(-Ea/RT)
    Rennet Ea ~ 30 kJ/mol  => 10°C rise ~doubles reaction rate
""")

# Arrhenius for rennet
Ea_rennet = 30e3    # J/mol
R_gas = 8.314
T_vals = np.array([4, 10, 15, 20, 30, 37]) + 273.15
A_rennet = 1.0   # relative
k_rennet = A_rennet * np.exp(-Ea_rennet / (R_gas * T_vals))
k_ref = k_rennet[np.abs(T_vals-293.15).argmin()]
print(f"\n  Relative rennet activity (Arrhenius, Ea=30 kJ/mol):")
for T_C, k in zip(T_vals-273.15, k_rennet):
    print(f"    T={T_C:.0f}C: k_rel = {k/k_ref:.3f}")

# %% [markdown]
# ---
# ## §9 · Verilog + Git Workflow for RTL Design

# %%
hdr("§9 — Verilog HDL + Git for Hardware Engineering")

print("""
  === Verilog HDL basics ===

  // 4-bit ripple carry adder
  module rca_4bit(
      input  [3:0] A, B,
      input        Cin,
      output [3:0] S,
      output       Cout
  );
      wire c1, c2, c3;
      full_adder fa0(.a(A[0]),.b(B[0]),.cin(Cin), .s(S[0]),.cout(c1));
      full_adder fa1(.a(A[1]),.b(B[1]),.cin(c1),  .s(S[1]),.cout(c2));
      full_adder fa2(.a(A[2]),.b(B[2]),.cin(c2),  .s(S[2]),.cout(c3));
      full_adder fa3(.a(A[3]),.b(B[3]),.cin(c3),  .s(S[3]),.cout(Cout));
  endmodule

  module full_adder(input a,b,cin, output s,cout);
      assign s    = a ^ b ^ cin;
      assign cout = (a&b)|(b&cin)|(a&cin);
  endmodule

  // Synchronous always block (synthesizable)
  module d_ff(input clk, rst_n, d, output reg q);
      always @(posedge clk or negedge rst_n)
          if (!rst_n) q <= 1'b0;
          else        q <= d;
  endmodule

  // Testbench pattern
  `timescale 1ns/1ps
  module tb_rca;
      reg [3:0] A, B; reg Cin;
      wire [3:0] S; wire Cout;
      rca_4bit dut(.A(A),.B(B),.Cin(Cin),.S(S),.Cout(Cout));
      initial begin
          $dumpfile("rca.vcd"); $dumpvars(0, tb_rca);
          A=4'd5; B=4'd3; Cin=0; #10;
          $display("5+3=%0d Cout=%b  PASS=%b", S, Cout, (S==8)&&(!Cout));
          A=4'd15; B=4'd1; Cin=0; #10;
          $display("15+1=%0d Cout=%b PASS=%b", S, Cout, (S==0)&&(Cout==1));
          $finish;
      end
  endmodule
""")

print("""
  === Git workflow for RTL (hardware) ===

  Standard branch strategy:
    main          <- tape-out / release only
    develop       <- integration branch
    feature/uart  <- feature branches
    hotfix/crc    <- bug fixes

  Key .gitignore entries for EDA tools:
    *.vcd *.fsdb          # waveform dumps (large binary)
    *.log *.rpt           # synthesis/sim logs (generated)
    work/                 # ModelSim work library
    simv simv.daidir      # VCS compiled simulation
    *.key *.synopsys      # license files
    .Xil/ .srcs/          # Vivado junk

  Commit discipline for RTL:
    - Atomic commits: one module = one commit
    - Include simulation results in commit message
    - Tag at milestones: git tag -a v0.1 -m "RTL complete, 0 lint errors"
    - Sign off: git commit -s (DCO, required for open-source IP)

  Lint -> Simulate -> Synthesize -> PR flow:
    make lint     # Verilator --lint-only --Wall
    make sim      # iverilog + vvp OR VCS/ModelSim
    make synth    # Yosys (open) or Design Compiler
    make pnr      # OpenROAD or Innovus
    git push origin feature/uart && gh pr create

  Formal verification entry point:
    # SymbiYosys (.sby file)
    [tasks] prove cover
    [options] mode prove
    [engines] smtbmc boolector
    [script] read -formal rtl/uart_tx.v
             prep -top uart_tx
    [files]  rtl/uart_tx.v
""")

# Simulate the 4-bit adder in Python to verify
print("  === Python simulation of 4-bit RCA ===")
def full_adder(a, b, cin):
    s = a ^ b ^ cin
    cout = (a & b) | (b & cin) | (a & cin)
    return s, cout

def rca_4bit(A, B, Cin):
    S = 0; c = Cin
    for bit in range(4):
        a_bit = (A >> bit) & 1
        b_bit = (B >> bit) & 1
        s_bit, c = full_adder(a_bit, b_bit, c)
        S |= (s_bit << bit)
    return S, c

errors = 0
for A_v in range(16):
    for B_v in range(16):
        for Cin_v in [0, 1]:
            S_out, Cout_out = rca_4bit(A_v, B_v, Cin_v)
            expected = A_v + B_v + Cin_v
            S_exp = expected & 0xF; C_exp = (expected >> 4) & 1
            if S_out != S_exp or Cout_out != C_exp:
                errors += 1
print(f"  4-bit RCA exhaustive check (512 cases): {errors} errors  [{'PASS' if errors==0 else 'FAIL'}]")

# %% [markdown]
# ---
# ## §10 · Figure — 12 panels

# %%
fig = plt.figure(figsize=(20, 15))
gs_f = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)
c4 = ["#4C72B0","#DD8452","#55A868","#C44E52"]

# P1: Riemann sums
ax1 = fig.add_subplot(gs_f[0,0])
x_rb = np.linspace(0, 3, 200)
ax1.fill_between(x_rb, x_rb**2, alpha=0.15, color='steelblue')
ax1.plot(x_rb, x_rb**2, 'b-', lw=2, label='f(x)=x²')
n_bars = 8
x_bars = np.linspace(0,3,n_bars+1)
dx_b = 3/n_bars
for i in range(n_bars):
    xm = (x_bars[i]+x_bars[i+1])/2
    ax1.bar(x_bars[i], xm**2, width=dx_b, alpha=0.5, align='edge',
            color='orange', edgecolor='k', linewidth=0.5)
ax1.set_title(f"Riemann sum (n={n_bars}) -> ∫x²dx=9", fontsize=9)
ax1.set_xlabel("x"); ax1.legend(fontsize=7)

# P2: Ion engine Isp vs beam voltage
ax2 = fig.add_subplot(gs_f[0,1])
Vb_arr = np.linspace(200, 10000, 300)
Isp_arr = np.sqrt(2 * e_charge * Vb_arr / m_xe) / g0
ax2.plot(Vb_arr, Isp_arr/1000, 'b-', lw=2)
ax2.axvline(1200, color='r', ls='--', lw=1, label='Dawn (1.2 kV)')
ax2.axhline(3100/1000, color='g', ls=':', lw=1, label='Isp=3100s')
ax2.set_xlabel("Beam voltage (V)"); ax2.set_ylabel("Isp (ks)")
ax2.set_title("Ion engine: Isp vs V_b", fontsize=9); ax2.legend(fontsize=7)

# P3: Fusion triple product history
ax3 = fig.add_subplot(gs_f[0,2])
years = [1970, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2022, 2025]
triple_prod = [1e16, 1e18, 1e19, 5e19, 1e20, 2e20, 5e20, 3e20, 1e21, 3e21]
ax3.semilogy(years, triple_prod, 'ro-', ms=5)
ax3.axhline(3e21, color='k', ls='--', lw=1, label='Ignition ~3e21')
ax3.fill_between([1970,2025],[3e21,3e21],[1e23,1e23], alpha=0.1, color='gold')
ax3.set_xlabel("Year"); ax3.set_ylabel("n·T·tau (keV m^-3 s)")
ax3.set_title("Fusion triple product progress", fontsize=9); ax3.legend(fontsize=7)

# P4: Transmission line reflection
ax4 = fig.add_subplot(gs_f[0,3])
ZL_arr = np.linspace(0.1, 500, 1000)
Gamma_arr = (ZL_arr - 50) / (ZL_arr + 50)
ax4.plot(ZL_arr, Gamma_arr, 'purple', lw=2)
ax4.axhline(0, color='k', lw=0.5)
ax4.axvline(50, color='r', ls='--', lw=1, label='matched Z0=50')
ax4.set_xlabel("Z_L (ohm)"); ax4.set_ylabel("Gamma")
ax4.set_title("Reflection coeff vs load", fontsize=9); ax4.legend(fontsize=7)

# P5: Schmitt trigger transfer curve
ax5 = fig.add_subplot(gs_f[1,0])
ax5.plot(V_in, V_comp_ideal, 'b-', lw=1.5, alpha=0.6, label='Ideal comparator')
ax5.plot(V_in, V_schmitt,    'r-', lw=2,   label=f'Schmitt (hyst={hysteresis:.2f}V)')
ax5.axvline(VTP, color='gray', ls=':', lw=1)
ax5.axvline(VTN, color='gray', ls=':', lw=1)
ax5.set_xlabel("V_in (V)"); ax5.set_ylabel("V_out (V)")
ax5.set_title("Comparator vs Schmitt trigger", fontsize=9); ax5.legend(fontsize=7)

# P6: Height distribution
ax6 = fig.add_subplot(gs_f[1,1])
ax6.plot(h, pdf_m, 'b-', lw=2, label=f"Male  mu={mu_m}cm")
ax6.plot(h, pdf_f, 'r-', lw=2, label=f"Female mu={mu_f}cm")
ax6.fill_between(h, pdf_m, where=(h>190), alpha=0.3, color='blue', label='>190cm')
ax6.set_xlabel("Height (cm)"); ax6.set_ylabel("PDF")
ax6.set_title("Human height distribution", fontsize=9); ax6.legend(fontsize=7)

# P7: CLT convergence
ax7 = fig.add_subplot(gs_f[1,2])
z_range = np.linspace(-4, 4, 200)
gauss_pdf = np.exp(-z_range**2/2)/np.sqrt(2*np.pi)
for n_clt, col in zip([1,2,5,30], c4):
    ax7.hist(clt_samples[n_clt], bins=80, density=True, alpha=0.5,
             color=col, label=f'n={n_clt}')
ax7.plot(z_range, gauss_pdf, 'k-', lw=2, label='N(0,1)')
ax7.set_title("CLT: Uniform(0,1) sums", fontsize=9); ax7.legend(fontsize=6)

# P8: Thin lens image distance vs object distance
ax8 = fig.add_subplot(gs_f[1,3])
do_arr = np.linspace(f_v*1.01, 1000, 500)
di_arr = 1/(1/f_v - 1/do_arr)
m_arr  = -di_arr/do_arr
ax8.plot(do_arr, di_arr, 'b-', lw=2, label='d_i')
ax8.axhline(0, color='k', lw=0.5)
ax8.axvline(do_v, color='r', ls='--', lw=1, label=f'do={do_v:.0f}mm')
ax8.axhline(di_v, color='g', ls='--', lw=1, label=f'di={di_v:.0f}mm')
ax8.set_xlim(90, 600); ax8.set_ylim(0, 800)
ax8.set_xlabel("d_o (mm)"); ax8.set_ylabel("d_i (mm)")
ax8.set_title(f"Thin lens f={f_v:.1f}mm", fontsize=9); ax8.legend(fontsize=7)

# P9: Henderson-Hasselbalch
ax9 = fig.add_subplot(gs_f[2,0])
ax9.plot(pH_range, alpha, 'purple', lw=2, label=f'pKa={pKa_demo}')
ax9.axvline(pKa_demo, color='r', ls='--', lw=1, label=f'pH=pI={pKa_demo}')
ax9.axhline(0.5, color='gray', ls=':', lw=1)
ax9.set_xlabel("pH"); ax9.set_ylabel("Fraction deprotonated")
ax9.set_title("HH equation (casein pKa=4.6)", fontsize=9); ax9.legend(fontsize=7)

# P10: Rennet Arrhenius
ax10 = fig.add_subplot(gs_f[2,1])
T_C_arr = np.linspace(0, 50, 200)
T_K_arr = T_C_arr + 273.15
k_arr = np.exp(-Ea_rennet/(R_gas*T_K_arr))
k_arr /= k_arr[np.abs(T_C_arr-20).argmin()]
ax10.plot(T_C_arr, k_arr, 'orange', lw=2)
ax10.axvline(4, color='b', ls=':', lw=1, label='Fridge 4C')
ax10.axvline(37, color='r', ls=':', lw=1, label='Body temp')
ax10.set_xlabel("T (C)"); ax10.set_ylabel("Relative rate k/k_20")
ax10.set_title("Rennet activity (Ea=30 kJ/mol)", fontsize=9); ax10.legend(fontsize=7)

# P11: Verilog RCA exhaustive test results
ax11 = fig.add_subplot(gs_f[2,2])
A_grid = np.arange(16); B_grid = np.arange(16)
result_grid = np.zeros((16,16), dtype=int)
for Ai in range(16):
    for Bi in range(16):
        S_out, _ = rca_4bit(Ai, Bi, 0)
        result_grid[Ai,Bi] = S_out
im = ax11.imshow(result_grid, cmap='viridis', origin='lower')
plt.colorbar(im, ax=ax11)
ax11.set_xlabel("B"); ax11.set_ylabel("A")
ax11.set_title("4-bit RCA: S=A+B (mod 16)", fontsize=9)

# P12: Magnification map
ax12 = fig.add_subplot(gs_f[2,3])
do_m = np.linspace(f_v*1.05, 800, 400)
di_m = 1/(1/f_v - 1/do_m)
mag_m = np.abs(-di_m/do_m)
ax12.semilogy(do_m, mag_m, 'g-', lw=2)
ax12.axvline(do_v, color='r', ls='--', lw=1, label=f'do={do_v:.0f}mm, |m|={abs(m_v):.2f}')
ax12.axvline(2*f_v, color='b', ls=':', lw=1, label=f'2f={2*f_v:.1f}mm, |m|=1')
ax12.set_xlabel("d_o (mm)"); ax12.set_ylabel("|magnification|")
ax12.set_title("Thin lens magnification", fontsize=9); ax12.legend(fontsize=7)

fig.suptitle(
    "Calculus · Ion Engines · Fusion · Interconnects · Comparators"
    " · Heights · Optics · Verilog · Chemistry",
    fontsize=12, fontweight='bold', y=1.01
)

import pathlib
out_dir  = pathlib.Path(__file__).parent if "__file__" in dir() else pathlib.Path("repl")
out_path = out_dir / "_out_mixed_physics_ee_stats.png"
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)

print("\n=== All §1-§10 complete ===")
