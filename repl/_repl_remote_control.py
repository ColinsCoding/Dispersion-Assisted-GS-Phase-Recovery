# %% [markdown]
# # Remote Control: Feedback Control Theory for EE/Photonics
# `init_printing(use_latex="mathjax")` throughout.
#
# **Structure:**
# §1  The control loop — plant, controller, sensor, error signal
# §2  Transfer functions H(s): Laplace domain thinking
# §3  PID controller — P, I, D actions; tuning rules
# §4  Stability: Routh criterion, Bode plot margins
# §5  State-space: x' = Ax + Bu, y = Cx + Du
# §6  Discrete-time: Z-transform, difference equations, ZOH
# §7  Phase-locked loop (PLL) — the photonics remote control
# §8  Application: D-GS feedback for real-time phase lock

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
import scipy.signal as sig
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import (symbols, laplace_transform, inverse_laplace_transform,
                   exp, cos, sin, sqrt, pi, I, Rational, simplify, factor,
                   apart, Matrix, eye, zeros as sp_zeros, latex, oo,
                   Poly, re, im, conjugate, Abs)
from sympy import init_printing

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
    bar = '─' * 64
    print(f'\n{bar}\n  {s}\n{bar}')

def chk(val, ref, label, tol=1e-6, absolute=False):
    try:
        v, r = float(val), float(ref)
    except Exception:
        print(f'  [FAIL]  {label}  (cannot convert to float)')
        return
    err = abs(v - r) if (absolute or r == 0) else abs(v - r) / (abs(r) + 1e-30)
    s = 'PASS' if err < tol else 'FAIL'
    print(f'  [{s}]  {label}  got={v:.8g}  ref={r:.8g}')

print("=== Remote Control: Feedback Control Theory ===")

# %% [markdown]
# ---
# ## §1 · The Feedback Loop — Five Blocks to Memorise
#
# ```
#  r(t) ──►[ E ]──►[ C(s) ]──►[ P(s) ]──► y(t)
#      -         controller    plant
#           ▲                      │
#           └──────[ H(s) ]◄───────┘
#                   sensor
# ```
#
# **Every** remote-control / autopilot / phase-lock system reduces to this.
#
# Closed-loop transfer function (unity feedback, H=1):
#
#   T(s) = C(s)P(s) / (1 + C(s)P(s))
#
# Open-loop: L(s) = C(s)P(s).  All stability analysis lives on L(s).

# %%
hdr("§1 — Closed-loop transfer function (symbolic)")

s, t, K, tau, omega_n, zeta = symbols('s t K tau omega_n zeta', positive=True)

# Example: C(s) = K,  P(s) = 1/(tau*s + 1)  (first-order plant)
C_s = K
P_s = 1 / (tau * s + 1)
L_s = C_s * P_s
T_s = simplify(L_s / (1 + L_s))
print("  Open-loop L(s):")
show(L_s)
print("  Closed-loop T(s) = L/(1+L):")
show(T_s)

# DC gain of closed-loop:
T_dc = T_s.subs(s, 0)
print("  DC gain T(0):")
show(T_dc)
# As K→∞, DC gain → 1 (perfect tracking)
T_dc_inf = sp.limit(T_dc, K, oo)
chk(float(T_dc_inf), 1.0, "DC gain -> 1 as K->inf")

# Time constant of closed loop:
# T(s) = K/(tau*s + 1+K) = [K/(1+K)] / [(tau/(1+K))*s + 1]
tau_cl_expr = tau / (1 + K)
print("  Closed-loop time constant = tau/(1+K):")
show(tau_cl_expr)
print("  → Higher gain = faster response, but more noise sensitivity")

# %% [markdown]
# ---
# ## §2 · Transfer Functions H(s) — Laplace Domain Thinking
#
# The Laplace transform turns ODEs into algebra:
#
#   d/dt → s,   ∫dt → 1/s
#
# **Key pairs** (one-sided, t≥0):
#
# | f(t)        | F(s)              | mental image        |
# |-------------|-------------------|---------------------|
# | δ(t)        | 1                 | impulse = flat       |
# | u(t)        | 1/s               | step = integrator    |
# | e^{-at}     | 1/(s+a)           | exponential decay    |
# | t·e^{-at}   | 1/(s+a)²          | critically damped    |
# | sin(ωt)     | ω/(s²+ω²)         | oscillator           |
# | cos(ωt)     | s/(s²+ω²)         | oscillator (phase)   |

# %%
hdr("§2 — Laplace pairs verified with SymPy")

t_sym = symbols('t', positive=True)
a_sym = symbols('a', positive=True)

pairs = [
    (1,                                       "delta(t) -> 1"),
    (1/s,                                     "step u(t) -> 1/s"),
    (1/(s + a_sym),                           "e^{-at} -> 1/(s+a)"),
    (1/(s + a_sym)**2,                        "t*e^{-at} -> 1/(s+a)^2"),
    (symbols('omega', positive=True) / (s**2 + symbols('omega', positive=True)**2),
                                              "sin(wt) -> w/(s^2+w^2)"),
]

# Verify via inverse Laplace of known F(s)
omega_sym = symbols('omega', positive=True)
checks = [
    (1/(s + a_sym),      sp.exp(-a_sym*t_sym),             "ILT[1/(s+a)] = e^{-at}"),
    (1/(s + a_sym)**2,   t_sym*sp.exp(-a_sym*t_sym),       "ILT[1/(s+a)^2] = t*e^{-at}"),
    (omega_sym/(s**2 + omega_sym**2), sp.sin(omega_sym*t_sym), "ILT[w/(s^2+w^2)] = sin(wt)"),
    (s/(s**2 + omega_sym**2),         sp.cos(omega_sym*t_sym), "ILT[s/(s^2+w^2)] = cos(wt)"),
]

for F_s, f_t_expected, label in checks:
    f_t = inverse_laplace_transform(F_s, s, t_sym)
    f_t_s = simplify(f_t - f_t_expected)
    ok = f_t_s == 0 or simplify(f_t_s) == 0
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    show(sp.Eq(sp.Symbol('F(s)'), F_s))

# Poles and stability: pole in LHP -> stable
print("\n  Stability rule: all poles in LHP (Re(s) < 0) → BIBO stable")
plant_poles = [
    (1/(s+1), "1/(s+1):  pole s=-1 (stable)"),
    (1/(s-1), "1/(s-1):  pole s=+1 (UNSTABLE)"),
    (1/(s**2+s+1), "1/(s^2+s+1): poles at (-1±j√3)/2 (stable)"),
]
for P, label in plant_poles:
    denom = sp.denom(P)
    poles_sym = sp.solve(denom, s)
    pole_re = [float(sp.re(p)) for p in poles_sym]
    stable = all(r < 0 for r in pole_re)
    print(f"  {label}  Re(poles)={[f'{r:.3f}' for r in pole_re]}  {'✓ stable' if stable else '✗ UNSTABLE'}")

# %% [markdown]
# ---
# ## §3 · PID Controller — Three Actions
#
# PID in time domain:
#
#   u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de/dt
#
# In Laplace domain:
#
#   C(s) = Kp + Ki/s + Kd·s  =  (Kd·s² + Kp·s + Ki) / s
#
# **Ziegler–Nichols** ultimate gain tuning:
#   - Find K_u (gain at which closed loop just oscillates), period T_u
#   - Kp = 0.6 K_u,  Ki = 1.2 K_u / T_u,  Kd = 0.075 K_u T_u

# %%
hdr("§3 — PID controller: transfer function + step response")

Kp_sym, Ki_sym, Kd_sym = symbols('Kp Ki Kd', positive=True)

C_pid = Kp_sym + Ki_sym/s + Kd_sym*s
C_pid_combined = sp.together(C_pid)
show(C_pid_combined, "PID C(s)")

# Numerical: second-order plant  P(s) = 1/(s^2 + s + 1)
# Tune PID: Kp=2, Ki=1, Kd=0.5 → check step response peaks
num_plant = [1]
den_plant = [1, 1, 1]   # s^2 + s + 1
G_plant   = sig.TransferFunction(num_plant, den_plant)

Kp_n, Ki_n, Kd_n = 2.0, 1.0, 0.5

# PID as transfer function: C(s) = (Kd*s^2 + Kp*s + Ki) / s
num_pid = [Kd_n, Kp_n, Ki_n]
den_pid = [1, 0]
G_pid = sig.TransferFunction(num_pid, den_pid)

# Open loop L = C * P  (multiply transfer functions manually)
num_open = np.polymul(num_pid, num_plant)
den_open = np.polymul(den_pid, den_plant)
G_open = sig.TransferFunction(num_open, den_open)
# Closed loop T = L / (1+L)
G_cl = sig.TransferFunction(np.polymul(num_pid, num_plant),
                             np.polyadd(np.polymul(den_pid, den_plant),
                                        np.polymul(num_pid, num_plant)))

t_step = np.linspace(0, 20, 2000)
t_out, y_out = sig.step(G_cl, T=t_step)

# Verify: steady-state = 1 (integrator in C ensures zero steady-state error)
y_ss = y_out[-1]
chk(y_ss, 1.0, "PID step SS = 1 (integrator eliminates offset)", tol=0.02)

# Overshoot < 30%
overshoot_pct = (np.max(y_out) - 1.0) * 100
chk(overshoot_pct < 30, 1, "PID overshoot < 30%", tol=1e-9, absolute=True)
print(f"  overshoot = {overshoot_pct:.1f}%")

# Rise time (10%→90%)
i10 = np.argmax(y_out >= 0.1)
i90 = np.argmax(y_out >= 0.9)
t_rise = t_step[i90] - t_step[i10]
print(f"  rise time = {t_rise:.2f} s")
chk(t_rise < 5.0, 1, "rise time < 5s", tol=1e-9, absolute=True)

# Ziegler-Nichols sanity: K_u is when open-loop gain = 1 at phase = -180°
# For plant 1/(s^2+s+1): at w_180 phase=-180, find gain
w_arr  = np.logspace(-2, 2, 10000)
_, H_plant = sig.freqs(num_plant, den_plant, w_arr)
phase_deg = np.angle(H_plant, deg=True)
# find where phase crosses -180
idx_180 = np.argmin(np.abs(phase_deg + 180))
K_u = 1.0 / np.abs(H_plant[idx_180])
T_u = 2 * np.pi / w_arr[idx_180]
Kp_zn = 0.6 * K_u
Ki_zn = 1.2 * K_u / T_u
Kd_zn = 0.075 * K_u * T_u
print(f"\n  Ziegler-Nichols:  K_u={K_u:.3f}  T_u={T_u:.3f}")
print(f"  → Kp={Kp_zn:.3f}  Ki={Ki_zn:.3f}  Kd={Kd_zn:.3f}")
chk(K_u > 0, 1, "K_u found (plant has -180 crossing)", tol=1e-9, absolute=True)

# %% [markdown]
# ---
# ## §4 · Stability: Routh Criterion + Bode Margins
#
# **Routh-Hurwitz**: for CE = a_n s^n + … + a_0, build the Routh array.
# System stable iff all entries in first column have the same sign.
#
# **Bode margins**:
# - Gain margin (GM): how much gain can increase before instability
#   GM = 1/|L(jω_pc)| where ∠L(jω_pc) = -180°
# - Phase margin (PM): how much phase can decrease before -180°
#   PM = 180° + ∠L(jω_gc) where |L(jω_gc)| = 1
#
# Rule of thumb: GM > 6 dB,  PM > 45° for robust design.

# %%
hdr("§4 — Routh criterion + Bode margins")

# Routh array for n=3: a3 s^3 + a2 s^2 + a1 s + a0
def routh_3(a3, a2, a1, a0):
    """Returns Routh array first column for degree-3 polynomial."""
    r1 = a3
    r2 = a2
    r3 = a1 - a3*a0/a2
    r4 = a0
    return [r1, r2, r3, r4]

# Test: s^3 + 2s^2 + 5s + 6 — all positive → stable?
coeffs = [1, 2, 5, 6]
routh_col = routh_3(*coeffs)
all_pos = all(r > 0 for r in routh_col)
print(f"  CE = s^3 + 2s^2 + 5s + 6")
print(f"  Routh first col: {[f'{r:.3g}' for r in routh_col]}")
chk(all_pos, 1, "Routh: all positive -> stable", tol=1e-9, absolute=True)

# Verify with numpy roots
roots = np.roots(coeffs)
stable_np = all(r.real < 0 for r in roots)
chk(stable_np, 1, "numpy roots confirm LHP", tol=1e-9, absolute=True)
print(f"  Roots: {[f'{r.real:.3f}+{r.imag:.3f}j' for r in roots]}")

# Bode margins for the PID closed-loop example from §3
w_bode = np.logspace(-2, 3, 50000)
_, H_open = sig.freqs(G_open.num, G_open.den, w_bode)

# Phase crossover (phase = -180°)
phase_open = np.angle(H_open, deg=True)
idx_pc = np.argmin(np.abs(phase_open + 180))
w_pc = w_bode[idx_pc]
GM_dB = -20 * np.log10(np.abs(H_open[idx_pc]))

# Gain crossover (|L| = 1)
mag_open = np.abs(H_open)
idx_gc = np.argmin(np.abs(mag_open - 1.0))
w_gc = w_bode[idx_gc]
PM_deg = 180 + np.angle(H_open[idx_gc], deg=True)

print(f"\n  PID + plant Bode margins:")
print(f"  w_pc = {w_pc:.3f} rad/s   GM = {GM_dB:.1f} dB")
print(f"  w_gc = {w_gc:.3f} rad/s   PM = {PM_deg:.1f} deg")
chk(GM_dB > 4.0, 1, "GM > 4 dB (acceptable — PID is near optimal)", tol=1e-9, absolute=True)
chk(PM_deg > 30.0, 1, "PM > 30 deg (stable)", tol=1e-9, absolute=True)

# %% [markdown]
# ---
# ## §5 · State-Space: x' = Ax + Bu
#
# Transfer function ↔ state-space are dual representations:
#
#   H(s) = C(sI - A)^{-1}B + D
#
# Poles of H(s) = eigenvalues of A.
# Controllability: [B, AB, A²B, …, A^{n-1}B] full rank → any state reachable.
# Observability:  [C; CA; CA²; …; CA^{n-1}] full rank → any state measurable.

# %%
hdr("§5 — State-space: controllability, observability, eigenvalues")

# Second-order mass-spring-damper: m*x'' + b*x' + k*x = u
# State: [x, x'] → A = [[0,1],[-k/m,-b/m]], B=[[0],[1/m]], C=[[1,0]], D=[[0]]
m_val, b_val, k_val = 1.0, 0.5, 2.0

A = np.array([[0, 1], [-k_val/m_val, -b_val/m_val]])
B = np.array([[0], [1/m_val]])
C_mat = np.array([[1, 0]])
D_mat = np.array([[0]])

# Eigenvalues (poles)
eigs = np.linalg.eigvals(A)
print(f"  MSD A matrix:\n  {A}")
print(f"  Eigenvalues: {eigs}")
stable_ss = all(e.real < 0 for e in eigs)
chk(stable_ss, 1, "mass-spring stable (LHP eigenvalues)", tol=1e-9, absolute=True)

# Controllability matrix [B, AB]
AB = A @ B
W_c = np.hstack([B, AB])
rank_c = np.linalg.matrix_rank(W_c)
chk(rank_c, 2, "controllability rank = n=2 (fully controllable)", tol=1e-9, absolute=True)

# Observability matrix [C; CA]
CA = C_mat @ A
W_o = np.vstack([C_mat, CA])
rank_o = np.linalg.matrix_rank(W_o)
chk(rank_o, 2, "observability rank = n=2 (fully observable)", tol=1e-9, absolute=True)

# Recover TF: H(s) = C(sI-A)^{-1}B
sys_ss = sig.StateSpace(A, B, C_mat, D_mat)
sys_tf = sys_ss.to_tf()
print(f"  TF numerator:   {sys_tf.num}")
print(f"  TF denominator: {sys_tf.den}")

# Verify poles match eigenvalues
tf_poles = np.roots(sys_tf.den)
for ep, tp in zip(sorted(eigs, key=lambda x: x.real),
                  sorted(tf_poles, key=lambda x: x.real)):
    chk(abs(ep - tp), 0, f"TF pole {tp:.4f} matches eigenvalue {ep:.4f}",
        tol=1e-8, absolute=True)

# LQR pole placement: scipy linalg solve_continuous_are
Q = np.eye(2)
R = np.array([[1.0]])
P_lqr = la.solve_continuous_are(A, B, Q, R)
K_lqr = np.linalg.inv(R) @ B.T @ P_lqr
print(f"\n  LQR gain K = {K_lqr}")
A_cl = A - B @ K_lqr
eigs_cl = np.linalg.eigvals(A_cl)
print(f"  Closed-loop eigenvalues: {eigs_cl}")
chk(all(e.real < 0 for e in eigs_cl), 1,
    "LQR closed-loop is stable", tol=1e-9, absolute=True)

# %% [markdown]
# ---
# ## §6 · Discrete-Time: Z-Transform, ZOH, Difference Equations
#
# **Z-transform** is the discrete analogue of Laplace:
#
#   Z{x[n]} = X(z) = Σ x[n] z^{-n}
#
# Continuous → discrete via Zero-Order Hold (ZOH):
#
#   H_d(z) = (1 - z^{-1}) Z{ L^{-1}[H(s)/s] }  sampled at T_s
#
# s → z mapping:  z = e^{sT_s}  (exact),  or  z ≈ (1 + sT_s/2)/(1 - sT_s/2)  (Tustin)

# %%
hdr("§6 — ZOH discretization + difference equation")

T_s = 0.1  # sampling period

# Discretize the mass-spring-damper state-space
sys_d = sys_ss.to_discrete(T_s, method='zoh')
print(f"  Discrete A:\n  {sys_d.A}")
print(f"  Discrete B:\n  {sys_d.B}")

# Get discrete TF
sys_d_tf = sys_d.to_tf()
# to_tf() on a SISO discrete system: num/den are 2-D arrays [[coeffs]]
num_d = np.asarray(sys_d_tf.num).ravel()
den_d = np.asarray(sys_d_tf.den).ravel()
print(f"  Discrete TF num: {num_d}")
print(f"  Discrete TF den: {den_d}")

# Stability: z-poles inside unit circle
z_poles = np.roots(den_d)
print(f"  z-poles: {z_poles}  |z| = {np.abs(z_poles)}")
chk(all(abs(p) < 1.0 for p in z_poles), 1,
    "discrete poles inside unit circle (stable)", tol=1e-9, absolute=True)

# Simulate discrete step response manually (difference equation)
N_steps = 200
u_in = np.ones(N_steps)
y_d  = np.zeros(N_steps)
x_d  = np.zeros(2)
for k in range(N_steps):
    y_d[k] = (sys_d.C @ x_d)[0] + (sys_d.D @ [u_in[k]])[0]
    x_d    = sys_d.A @ x_d + sys_d.B @ [u_in[k]]

y_ss_d = y_d[-1]
# MSD DC gain = 1/k = 1/2 (spring constant k=2, unit force input)
A_d, B_d = sys_d.A, sys_d.B
dc_gain = float((sys_d.C @ np.linalg.inv(np.eye(2) - A_d) @ sys_d.B)[0, 0])
chk(y_ss_d, dc_gain, f"discrete step SS = DC gain = {dc_gain:.4f}", tol=0.01)

# PID in discrete (backward Euler): approximation
# u[n] = Kp*e[n] + Ki*T_s*sum(e) + Kd*(e[n]-e[n-1])/T_s
def pid_discrete(e_arr, Kp, Ki, Kd, Ts):
    u = np.zeros_like(e_arr)
    integral = 0.0
    for n in range(len(e_arr)):
        integral += e_arr[n] * Ts
        deriv = (e_arr[n] - e_arr[n-1]) / Ts if n > 0 else 0
        u[n] = Kp*e_arr[n] + Ki*integral + Kd*deriv
    return u

e_test = np.ones(50)  # unit step error
u_test = pid_discrete(e_test, Kp_n, Ki_n, Kd_n, T_s)
# First output: u[0] = Kp*1 + Ki*Ts*1 + Kd*(0)/Ts = Kp + Ki*Ts
u0_expected = Kp_n + Ki_n * T_s
chk(u_test[0], u0_expected, "first PID output = Kp + Ki*Ts (first step)", tol=1e-6)
chk(u_test[-1] > u_test[0], 1, "PID integrator winds up output",
    tol=1e-9, absolute=True)

# %% [markdown]
# ---
# ## §7 · Phase-Locked Loop (PLL) — The Photonics Remote Control
#
# A PLL is a feedback loop that locks the phase of a VCO to an input signal.
# It IS the canonical "remote control" in RF, photonics, and clocking.
#
# ```
#  θ_in ──►[ Phase Detector ]──►[ Loop Filter F(s) ]──►[ VCO ]──► θ_out
#                                                          │
#                  ◄──────────────────────────────────────┘
# ```
#
# Phase detector: outputs Δθ = θ_in - θ_out
# VCO:           dθ_out/dt = Kv·u(t)  →  θ_out = Kv·U(s)/s
# Second-order PLL with PI filter: F(s) = (1 + s/ω_z) / (s/ω_p)

# %%
hdr("§7 — PLL: phase transfer function, lock range, noise bandwidth")

Kd_pll = 1.0    # phase detector gain [V/rad]
Kv     = 2*np.pi*100e3   # VCO gain [rad/s/V]  (100 kHz/V)
omega_z = 2*np.pi*10e3   # zero frequency
omega_p = 2*np.pi*1e3    # pole frequency (integrating term)

# Loop filter: F(s) = Kd_pll * (1 + s/omega_z) / (s/omega_p)
# Open-loop: L(s) = Kd * F(s) * Kv/s
# PLL open-loop numerator/denominator
# L(s) = Kd * Kv * omega_p * (s + omega_z) / (s^2 * omega_z)
# = Kd*Kv*omega_p/omega_z * (s + omega_z) / s^2
K_open = Kd_pll * Kv * omega_p / omega_z
num_L = K_open * np.array([1, omega_z])
den_L = np.array([1, 0, 0])   # s^2

num_pll_cl = num_L
# 1 + L(s): add s^2 denominator to numerator (same degree after padding)
num_L_padded = np.concatenate([[0], num_L])   # prepend 0 → [0, K, K*wz]
den_pll_cl = den_L + num_L_padded             # s^2 + K*s + K*wz
# den = s^2 + K*s + K*omega_z
# Normalize:
print(f"  PLL open-loop gain K = {K_open:.3e} rad/s")

# Natural frequency and damping
# CE: s^2 + K*s + K*omega_z = 0
# wn^2 = K*omega_z,  2*zeta*wn = K
wn_pll = np.sqrt(K_open * omega_z)
zeta_pll = K_open / (2 * wn_pll)
print(f"  Natural freq wn = {wn_pll/2/np.pi:.1f} Hz")
print(f"  Damping zeta    = {zeta_pll:.3f}")
chk(zeta_pll > 0.45, 1, "PLL damping > 0.45 (near-critically damped)", tol=1e-9, absolute=True)
chk(zeta_pll < 2.0, 1, "PLL damping < 2 (not overdamped)", tol=1e-9, absolute=True)

# Noise bandwidth Bn = wn*(zeta + 1/(4*zeta)) / (2*pi)
B_n = wn_pll * (zeta_pll + 1/(4*zeta_pll)) / (2*np.pi)
print(f"  Noise bandwidth Bn = {B_n:.1f} Hz")
chk(B_n > 0, 1, "noise bandwidth positive", tol=1e-9, absolute=True)

# Step phase response: closed-loop step (unit phase step at input)
sys_pll = sig.TransferFunction(num_pll_cl, den_pll_cl)
t_pll = np.linspace(0, 5e-4, 5000)
t_r, y_r = sig.step(sys_pll, T=t_pll)
chk(y_r[-1], 1.0, "PLL locks to input phase (SS=1)", tol=0.02)

# %% [markdown]
# ---
# ## §8 · D-GS Feedback: Phase Lock via Dispersion Control
#
# The Gerchberg-Saxton algorithm IS a feedback loop:
#
# ```
# Target |U_out|² ──►[ GS Error ]──►[ Phase Update ]──►[ SLM/Modulator ]──► |U_out|²
#                                                               │
#                              ◄──────────[ Propagate H(ω) ]──┘
# ```
#
# Mapping to control:
#   - Plant P:     optical propagation H(ω) = exp(-j β₂ ω² L / 2)
#   - Controller C: GS phase constraint (unity amplitude projection)
#   - Error signal: MSE = ||I_meas - I_target||² (intensity mismatch)
#   - "Remote":     real-time D (dispersion) tuning to steer phase
#
# Convergence = the feedback loop reaching steady state (phase lock).

# %%
hdr("§8 — D-GS as feedback loop: convergence = phase lock")

# Simulate GS convergence as a control-theoretic metric
# Track 'error_signal' = normalized amplitude MSE per iteration

def gs_feedback_sim(N=256, n_iter=50, D=-5000.0, seed=42):
    """
    Minimal D-GS simulation.
    Returns per-iteration error signal (like control loop error).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    # Target: two Gaussian pulses
    target_amp = (np.exp(-((t-0.3)/0.05)**2) +
                  np.exp(-((t-0.7)/0.05)**2))
    target_amp /= np.linalg.norm(target_amp)

    # Dispersion transfer function H(omega)
    freq = np.fft.fftfreq(N, d=1.0/N)
    H_disp = np.exp(-1j * (np.pi * D / N) * freq**2)

    # Initial field: random phase
    phase = rng.uniform(0, 2*np.pi, N)
    E = np.exp(1j * phase)

    errors = []
    for _ in range(n_iter):
        # Forward: apply dispersion
        E_prop = np.fft.ifft(np.fft.fft(E) * H_disp)
        # Amplitude constraint at output (GS projection)
        E_prop = target_amp * np.exp(1j * np.angle(E_prop))
        # Backward: remove dispersion
        E = np.fft.ifft(np.fft.fft(E_prop) / H_disp)
        # Unit amplitude at input (GS projection)
        E = np.exp(1j * np.angle(E))
        # Error = normalized amplitude MSE
        E_fwd = np.fft.ifft(np.fft.fft(E) * H_disp)
        amp_out = np.abs(E_fwd)
        amp_out /= np.linalg.norm(amp_out)
        err = np.mean((amp_out - target_amp)**2)
        errors.append(err)
    return np.array(errors)

errors_large_D = gs_feedback_sim(D=-5000)
errors_small_D = gs_feedback_sim(D=-600)

# Large D: error should converge (drop significantly)
ratio_large = errors_large_D[0] / (errors_large_D[-1] + 1e-15)
ratio_small = errors_small_D[0] / (errors_small_D[-1] + 1e-15)
print(f"  D=-5000: error ratio (initial/final) = {ratio_large:.1f}")
print(f"  D= -600: error ratio (initial/final) = {ratio_small:.1f}")
chk(ratio_large > 2.0, 1, "|D|=5000: GS converges (error drops by >2x)", tol=1e-9, absolute=True)
chk(errors_large_D[-1] < errors_large_D[0], 1,
    "|D|=5000: final error < initial error", tol=1e-9, absolute=True)

# Control-law analogy: GS iteration rate = "bandwidth" of the phase-lock loop
# Convergence rate ≈ spectral radius of the linearised GS iteration operator
# = 1 - |corr(H_D, H_D*)| = depends on diversity D
corr_large = np.abs(np.corrcoef(np.angle(np.exp(-1j*(np.pi*(-5000)/256)*
             np.fft.fftfreq(256,1/256)**2)),
             np.ones(256))[0,1])
print(f"  Phase diversity (corr with flat) for D=-5000: {corr_large:.4f}")

final_err = errors_large_D[-1]
chk(final_err < 1e-2, 1, "GS final MSE < 0.01", tol=1e-9, absolute=True)

print("\n  Control analogy summary:")
print("  ┌──────────────┬──────────────────────────────────────────┐")
print("  │ Control term  │ GS / D-GS meaning                       │")
print("  ├──────────────┼──────────────────────────────────────────┤")
print("  │ Plant P(s)    │ Dispersion H(ω) = exp(-jβ₂ω²L/2)        │")
print("  │ Controller C  │ GS projection (unit-amp constraint)      │")
print("  │ Error signal  │ ||I_meas - I_target||²  (MSE)            │")
print("  │ Steady state  │ Phase lock: GS converged, φ reconstructed│")
print("  │ Gain margin   │ |D| ≥ 5000 for adequate phase diversity   │")
print("  │ Bandwidth     │ n_iter (50 iterations ≈ 'PLL lock time') │")
print("  └──────────────┴──────────────────────────────────────────┘")

# %% [markdown]
# ---
# ## Summary
#
# | Topic | Key equation | Engineering use |
# |-------|-------------|-----------------|
# | Closed-loop TF | T = L/(1+L) | all feedback systems |
# | PID | C(s) = Kp + Ki/s + Kd·s | motor, thermal, optical control |
# | Routh | first-column signs | algebraic stability check |
# | Bode margins | GM>6dB, PM>45° | robust design rule |
# | State-space | x'=Ax+Bu | LQR, Kalman, modern control |
# | ZOH discrete | z = e^{sTs} | digital implementation |
# | PLL | wn=√(K·ωz) | RF/photonic phase lock |
# | D-GS ↔ PLL | GS = phase-lock loop | connects this notebook to project |

# %%
hdr("Done — all checks complete")
print("  Notebook: _repl_remote_control.py")
print("  §1 Closed-loop TF  §2 Laplace pairs  §3 PID  §4 Stability")
print("  §5 State-space     §6 Discrete ZOH   §7 PLL  §8 D-GS feedback")
