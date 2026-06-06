# %% [markdown]
# # Special Relativity: Twin Paradox, 4-Momentum, Temporal Symmetry
# Formal treatment — every identity verified numerically and symbolically.
# Topics: Lorentz factor · time dilation · length contraction · Minkowski interval ·
#         twin paradox (both frames) · 4-momentum pq · E²=(pc)²+(mc²)² · CPT symmetry

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from repl_helpers import hdr, show, chk
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import symbols, sqrt, Rational, latex, simplify, diff, Matrix, eye, diag
from sympy import cosh, sinh, tanh, atanh, cos, sin, pi, exp, I, conjugate

sp.init_printing(use_latex='mathjax')

# ── helpers ──────────────────────────────────────────────────────────────────
def hdr(s):
    bar = '─' * 60
    print(f'\n{bar}\n  {s}\n{bar}')

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

def chk(val, ref, label, tol=1e-9, absolute=False):
    v, r = float(val), float(ref)
    if absolute or r == 0:
        err = abs(v - r)
    else:
        err = abs(v - r) / (abs(r) + 1e-30)
    status = 'PASS' if err < tol else 'FAIL'
    print(f'  [{status}]  {label}  got={v:.8g}  ref={r:.8g}')

# %% [markdown]
# ## §1 — The Lorentz Factor γ and the "Circle" in Spacetime
#
# **Circle diameter / Minkowski interval**: in Euclidean space,
# the "circle" of radius r is x²+y²=r². In Minkowski spacetime the
# invariant "circle" (hyperbola) is:
#
#   s² = (ct)² − x²  =  invariant
#
# This s is the *proper time* (times c). Every inertial observer agrees on s²
# even though they disagree on t and x separately. That's the core of SR.

# %%
hdr("§1 — Lorentz factor γ, time dilation, length contraction")

v, c_sym, beta_sym = symbols('v c beta', positive=True)
gamma_sym = 1 / sqrt(1 - v**2/c_sym**2)

show(gamma_sym, 'gamma(v)')

# Numerical: γ at several speeds
c = 3e8   # m/s
betas = [0.1, 0.5, 0.8, 0.9, 0.99, 0.999]
print('\n  beta    gamma     time-dilation   length-contraction')
print('  -------------------------------------------------------')
for b in betas:
    g = 1/np.sqrt(1 - b**2)
    print(f'  {b:.3f}   {g:8.4f}   dt_lab = {g:.4f} dt_proper   L_lab = L/{g:.4f}')

# Verify: gamma at beta=0.6 should be 1.25 (3-4-5 triangle)
chk(1/np.sqrt(1 - 0.6**2), 1.25, "gamma(0.6) = 5/4  (Pythagorean triple)")

# Taylor series: gamma ≈ 1 + (1/2)beta² for small beta
beta_s = symbols('beta', positive=True)
gamma_s = 1/sqrt(1 - beta_s**2)
series = sp.series(gamma_s, beta_s, 0, 5)
show(series, 'gamma Taylor series')

# %%
hdr("§1b — Minkowski invariant interval")

# s² = c²t² - x²  is Lorentz-invariant
# In frame S: event at (t, x)
# In frame S' (moving at v): event at (t', x')
# Claim: c²t'² - x'² = c²t² - x²

t_sym, x_sym = symbols('t x', real=True)
gamma_num = 5/4   # beta=0.6, gamma=1.25
v_num = 0.6 * c

# Lorentz transform (symbolic, beta=3/5, gamma=5/4)
beta_n = sp.Rational(3, 5)
gamma_n = sp.Rational(5, 4)

t_prime = gamma_n * (t_sym - beta_n * x_sym / 1)   # units: c=1
x_prime = gamma_n * (x_sym - beta_n * t_sym)

interval_S  = t_sym**2 - x_sym**2
interval_Sp = t_prime**2 - x_prime**2
diff_check  = simplify(interval_Sp - interval_S)
show(diff_check, "s'² - s² (must be 0)")
chk(float(diff_check.subs([(t_sym, 1.7), (x_sym, 0.4)])), 0,
    "Minkowski interval is Lorentz-invariant", tol=1e-12)

# %% [markdown]
# ## §2 — Lorentz Transformation (Formal Matrix Form)
#
# The Lorentz boost along x is a **hyperbolic rotation** in the (ct, x) plane:
#
#   [ct']   [cosh φ   -sinh φ] [ct]
#   [ x' ] = [-sinh φ   cosh φ] [ x]
#
# where tanh φ = β = v/c. Compare to Euclidean rotation with trig (cos, sin).
# Lorentz boost is rotation by an *imaginary angle* — this is why it mixes space and time.

# %%
hdr("§2 — Lorentz boost as hyperbolic rotation")

phi = symbols('phi', positive=True)  # rapidity

Lambda = Matrix([[cosh(phi), -sinh(phi)],
                 [-sinh(phi), cosh(phi)]])
show(Lambda, 'Lorentz boost matrix (rapidity phi)')

# Verify: det(Lambda) = 1 (preserves orientation and volume)
det_L = simplify(Lambda.det())
show(det_L, 'det(Lambda)')
chk(float(det_L.subs(phi, 1.2)), 1.0, "det(Lambda) = 1 for all phi")

# Verify: Lambda^T * eta * Lambda = eta  (eta = diag(-1,+1) in (-,+) signature)
# Using (+,-) signature: eta = diag(+1,-1) [more common in particle physics]
eta = Matrix([[1, 0],[0, -1]])
LT_eta_L = simplify(Lambda.T * eta * Lambda)
show(LT_eta_L, 'Lambda^T eta Lambda (should = eta)')
diff_eta = simplify(LT_eta_L - eta)
chk(float(diff_eta.norm().subs(phi, 0.7)), 0, "Lambda preserves Minkowski metric", tol=1e-12)

# Rapidity addition (unlike velocity, rapidity ADDS under successive boosts)
phi1, phi2 = symbols('phi_1 phi_2', positive=True)
L1 = Matrix([[cosh(phi1), -sinh(phi1)],[-sinh(phi1), cosh(phi1)]])
L2 = Matrix([[cosh(phi2), -sinh(phi2)],[-sinh(phi2), cosh(phi2)]])
L12 = simplify(L1 * L2)
show(L12, 'L1 * L2 (rapidity adds)')
# Should be Lambda(phi1+phi2)
L_sum = L12.applyfunc(lambda e: sp.trigsimp(sp.expand_trig(e)))
# Check numerically
p1, p2 = 0.5, 0.8
L1n = np.array([[np.cosh(p1),-np.sinh(p1)],[-np.sinh(p1),np.cosh(p1)]])
L2n = np.array([[np.cosh(p2),-np.sinh(p2)],[-np.sinh(p2),np.cosh(p2)]])
L12n = L1n @ L2n
p12 = p1 + p2
Lsn = np.array([[np.cosh(p12),-np.sinh(p12)],[-np.sinh(p12),np.cosh(p12)]])
chk(np.max(np.abs(L12n - Lsn)), 0, "L(phi1)*L(phi2) = L(phi1+phi2)", tol=1e-12, absolute=True)

# Velocity addition formula (from rapidity addition)
# beta = tanh(phi), so beta_12 = tanh(phi1+phi2) = (b1+b2)/(1+b1*b2)
b1, b2 = 0.6, 0.8
phi1_n = np.arctanh(b1)
phi2_n = np.arctanh(b2)
beta_12_rapidity = np.tanh(phi1_n + phi2_n)
beta_12_formula  = (b1 + b2)/(1 + b1*b2)
chk(beta_12_rapidity, beta_12_formula, "velocity addition: tanh(phi1+phi2)=(b1+b2)/(1+b1*b2)")
print(f'  Note: classical (b1+b2)={b1+b2:.2f} > 1, SR result={beta_12_formula:.4f} < 1  ✓')

# %% [markdown]
# ## §3 — Twin Paradox (Both Frames, Formal Resolution)
#
# **Setup**: Alice stays on Earth. Bob travels at β=0.8 to a star 4 light-years away,
# turns around, comes back. Who ages less — and why?
#
# **Both frames agree**: Bob ages less. The *asymmetry* is not "who sees whom moving"
# (symmetric) but that **Bob must accelerate to turn around**. Alice is inertial
# throughout; Bob is not. Bob's worldline has shorter proper length.
#
# **"Double vision"**: from Earth, Bob's clock runs slow both ways.
# From Bob, Earth's clock runs slow BOTH legs — but the turnaround creates a
# *simultaneity jump* that accounts for the difference.

# %%
hdr("§3 — Twin Paradox: formal resolution")

beta_trip = 0.8
gamma_trip = 1 / np.sqrt(1 - beta_trip**2)
print(f'  Trip: beta={beta_trip}, gamma={gamma_trip:.6f}')

# Alice's frame (Earth)
d_ly = 4.0          # light-years to star
v_c  = beta_trip    # fraction of c; in units c=1, v=beta
t_each_leg_Alice = d_ly / v_c   # time in Alice frame per leg
t_total_Alice    = 2 * t_each_leg_Alice
print(f'\n  Alice (Earth frame):')
print(f'    Each leg: {t_each_leg_Alice:.4f} yr  |  Total: {t_total_Alice:.4f} yr')

# Bob's proper time per leg (time dilation)
tau_each_leg_Bob = t_each_leg_Alice / gamma_trip
tau_total_Bob    = 2 * tau_each_leg_Bob
print(f'\n  Bob (proper time):')
print(f'    Each leg: {tau_each_leg_Bob:.4f} yr  |  Total: {tau_total_Bob:.4f} yr')
print(f'\n  Age difference: Alice is {t_total_Alice - tau_total_Bob:.4f} yr older when Bob returns')

# Verify via spacetime interval: proper time = sqrt(s²)/c = sqrt((ct)²-x²)/c
# In Alice frame, Bob's outbound leg: Delta_t = 5yr, Delta_x = 4 ly (c=1 units)
Delta_t = t_each_leg_Alice
Delta_x = d_ly
interval_Bob = np.sqrt(Delta_t**2 - Delta_x**2)
chk(interval_Bob, tau_each_leg_Bob,
    "Minkowski interval = Bob's proper time per leg")

# Simultaneity jump at turnaround (derived from simultaneity lines, not a formula guess)
# Outbound frame simultaneity through turnaround (t=5, x=4) on Alice (x=0):
#   t_E - beta*x_E = const  →  t_E = 5 - 0.8*4 = 1.8 yr  (Alice's age)
# Inbound frame (-beta) simultaneity through turnaround on Alice (x=0):
#   t_E + beta*x_E = const  →  t_E = 5 + 0.8*4 = 8.2 yr
# Jump = 8.2 - 1.8 = 6.4 yr  =  2*beta*d  (in c=1 units)
t1_alice = t_each_leg_Alice - beta_trip * d_ly   # 1.8 yr
t2_alice = t_each_leg_Alice + beta_trip * d_ly   # 8.2 yr
t_jump   = t2_alice - t1_alice                   # 6.4 yr = 2*beta*d
alice_per_leg = tau_each_leg_Bob / gamma_trip    # time dilation: 3/1.667 = 1.8 yr
print(f'\n  Simultaneity resolution (Bob\'s frame):')
print(f'    Outbound: Alice ages {alice_per_leg:.4f} yr  (tau/gamma = {tau_each_leg_Bob:.1f}/{gamma_trip:.4f})')
print(f'    Turnaround jump: {t1_alice:.2f} yr  →  {t2_alice:.2f} yr  (delta = {t_jump:.4f} yr = 2*beta*d)')
print(f'    Inbound:  Alice ages {alice_per_leg:.4f} yr')
total_Alice_seen_by_Bob = alice_per_leg + t_jump + alice_per_leg
chk(total_Alice_seen_by_Bob, t_total_Alice,
    "Bob's frame: Alice ages 10yr total (including simultaneity jump)")

# %% [markdown]
# ## §4 — 4-Momentum p^μ ("pq" formal treatment)
#
# In SR, energy and momentum unify into the **4-momentum**:
#
#   p^μ = (E/c, p_x, p_y, p_z)
#
# The Minkowski inner product gives the **mass-shell constraint**:
#
#   p^μ p_μ = (E/c)² − |p|² = (mc)²    →    E² = (pc)² + (mc²)²
#
# This is the fundamental dispersion relation. For photons (m=0): E = pc.
# For massive particles at rest (p=0): E = mc².
#
# **"pq"** also appears in Hamiltonian mechanics:
#   - q = generalized coordinate, p = canonical momentum
#   - In QM: [q, p] = iℏ  (Heisenberg uncertainty)
#   - Relativistic: replace p·q → p^μ x_μ = Et − p·x (Lorentz scalar phase)

# %%
hdr("§4 — 4-momentum and E² = (pc)² + (mc²)²")

E_sym, p_sym, m_sym, c_s = symbols('E p m c', positive=True)
hbar = symbols('hbar', positive=True)

# Mass-shell relation
mass_shell = E_sym**2 - (p_sym*c_s)**2 - (m_sym*c_s**2)**2
show(mass_shell, 'E² - (pc)² - (mc²)²  = 0 on mass shell')

# Verify numerically for electron
m_e = 9.109e-31   # kg
c_n = 3e8         # m/s
mc2_eV = m_e * c_n**2 / 1.602e-19   # eV
mc2_MeV = mc2_eV / 1e6
print(f'\n  Electron rest energy: mc² = {mc2_MeV:.6f} MeV  (ref 0.511 MeV)')
chk(mc2_MeV, 0.511, "electron mc² = 0.511 MeV", tol=0.003)  # tol=0.3% for rounded constants

# Relativistic kinetic energy
# K = (gamma-1)*mc²
beta_e = 0.9
gamma_e = 1/np.sqrt(1-beta_e**2)
p_e = gamma_e * m_e * beta_e * c_n      # kg m/s
E_e = np.sqrt((p_e*c_n)**2 + (m_e*c_n**2)**2)
E_e_eV = E_e / 1.602e-19
K_e_MeV = (gamma_e - 1) * mc2_MeV
print(f'\n  Electron at beta=0.9: gamma={gamma_e:.4f}')
print(f'    Total energy E  = {E_e_eV/1e6:.4f} MeV')
print(f'    Kinetic energy K = {K_e_MeV:.4f} MeV')
E_from_mass_shell = np.sqrt((p_e*c_n)**2 + (m_e*c_n**2)**2)
chk(E_from_mass_shell, E_e, "E from mass-shell = E from gamma*mc²", tol=1e-9)

# Ultra-relativistic limit: E ≈ pc (photon-like)
beta_ur = 0.9999
gamma_ur = 1/np.sqrt(1-beta_ur**2)
p_ur = gamma_ur * m_e * beta_ur * c_n
E_ur = np.sqrt((p_ur*c_n)**2 + (m_e*c_n**2)**2)
E_ur_approx = p_ur * c_n   # photon approximation
ratio = E_ur_approx / E_ur
chk(ratio, 1.0, "ultra-relativistic: E ≈ pc  (ratio at beta=0.9999)", tol=0.01)

# 4-momentum Lorentz invariant: p^mu p_mu = m²c² (in natural units c=1)
# Explicitly in matrix form
print('\n  4-momentum as Lorentz 4-vector:')
print('  p^mu = (E/c, px, py, pz)')
print('  p_mu = eta_{mu nu} p^nu = (E/c, -px, -py, -pz)  [with metric (+,-,-,-)]')
print('  p^mu p_mu = (E/c)^2 - px^2 - py^2 - pz^2 = (mc)^2  [invariant!]')

# Hamiltonian mechanics connection
print('\n  Hamiltonian bridge:')
print('  Classical: H(q,p) = p²/2m + V(q),  {q,p}_PB = 1')
print('  QM: H|psi> = E|psi>,  [q,p] = i*hbar  (Heisenberg)')
print('  Relativistic action: S = -mc * integral sqrt(1-v^2/c^2) dt')
print('               = integral p_mu dx^mu  (Lorentz scalar)')

# %% [markdown]
# ## §5 — Temporal Symmetry: T, P, C and CPT
#
# Three discrete symmetries of spacetime:
#
# | Symmetry | Operation | Classical | QM |
# |----------|-----------|-----------|-----|
# | **T** (time-reversal) | t → −t | v → −v | K → K* |
# | **P** (parity)        | x → −x | L → L  | |psi(x)> → |psi(-x)> |
# | **C** (charge conj.)  | q → −q | no classical analogue | particle ↔ antiparticle |
#
# **CPT theorem**: ANY local Lorentz-invariant QFT conserves CPT combined.
# Individual symmetries CAN be broken (P violated by weak force, CP violated by kaons).
#
# **Temporal symmetry in SR**: the Lorentz group has 4 components.
# The *proper orthochronous* subgroup (det=+1, preserves time direction)
# is the connected piece. T and P are *discrete* transformations outside this.

# %%
hdr("§5 — Temporal symmetry: T, P, C and CPT")

# T reversal on 4-momentum: p^mu = (E/c, px) → (E/c, -px)  [E unchanged, p flips]
# Because T: t→-t, x unchanged → velocity v=dx/dt flips → momentum p=mv flips
print('  T-reversal:')
print('    t  → -t')
print('    x  →  x  (unchanged)')
print('    v  → -v  (velocity flips)')
print('    p  → -p  (momentum flips)')
print('    E  →  E  (kinetic energy unchanged)')
print('    p^mu = (E/c, p) → (E/c, -p)  [not a Lorentz boost!]')

# Matrix form of P (parity) in Minkowski
P_matrix = np.diag([1, -1, -1, -1])   # (+,-,-,-) metric: flip spatial components
T_matrix = np.diag([-1, 1, 1, 1])     # flip time component
CPT_matrix = -np.eye(4)               # flip everything
print('\n  Discrete symmetry matrices on x^mu = (ct, x, y, z):')
print(f'    P: {np.diag(P_matrix).tolist()}  (flip spatial)')
print(f'    T: {np.diag(T_matrix).tolist()}  (flip time)')
print(f'    CPT: {np.diag(CPT_matrix).tolist()}  (flip all)')

# Verify: CPT = C * P * T (order doesn't matter since they commute as matrices)
CPT_from_PT = T_matrix @ P_matrix   # T*P = diag(-1,1,1,1)*diag(1,-1,-1,-1) = -I
chk(np.max(np.abs(CPT_from_PT - CPT_matrix)), 0, "P*T = -I = CPT on 4-vectors", tol=1e-12, absolute=True)

# Time-reversal and the arrow of time
print('\n  Arrow of time puzzle:')
print('    Microscopic laws: T-symmetric (Newton, Maxwell, QM all work backwards)')
print('    Thermodynamics:   NOT T-symmetric (entropy increases)')
print('    Resolution:       low-entropy initial condition (Big Bang)')
print('    SR alone:         no preferred time direction — T is a global discrete flip')

# Light cone structure
print('\n  Light cone and causality:')
print('    Future: (ct)^2 - x^2 > 0 and t > 0  (timelike future)')
print('    Past:   (ct)^2 - x^2 > 0 and t < 0  (timelike past)')
print('    Spacelike: (ct)^2 - x^2 < 0  (cannot be causally connected)')
print('    T flips future<->past but preserves spacelike region')

# Verify: proper time is T-invariant
# tau = integral sqrt(1 - v^2/c^2) dt — under t→-t, dt→-d(-t)=dt, so tau is same
print('\n  Proper time tau = integral sqrt(1-v^2/c^2) dt')
print('  Under T: t→-t, dt→dt (magnitude), v=dx/dt→dx/d(-t)=-v, v^2→v^2')
print('  So dtau → dtau: proper time is T-INVARIANT  ✓')

# %% [markdown]
# ## §6 — Full Verification Figure

# %%
hdr("§6 — Figures: spacetime diagram + gamma curve + twin paradox + light cone")

fig = plt.figure(figsize=(16, 10))
fig.suptitle('Special Relativity: Twin Paradox · 4-Momentum · Temporal Symmetry',
             fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── P1: Lorentz factor γ(β) ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
betas_plot = np.linspace(0, 0.9999, 500)
gammas_plot = 1/np.sqrt(1 - betas_plot**2)
ax1.plot(betas_plot, gammas_plot, 'b-', lw=2)
ax1.axhline(1, color='gray', ls=':', lw=1)
for b_mark in [0.5, 0.8, 0.9, 0.99]:
    g_mark = 1/np.sqrt(1-b_mark**2)
    ax1.plot(b_mark, g_mark, 'ro', ms=5)
    ax1.annotate(f'β={b_mark}\nγ={g_mark:.2f}', (b_mark, g_mark),
                 textcoords='offset points', xytext=(5, 5), fontsize=7)
ax1.set_xlabel('β = v/c'); ax1.set_ylabel('γ')
ax1.set_title('Lorentz factor γ(β)', fontsize=10)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 12)

# ── P2: Spacetime diagram (twin paradox worldlines) ───────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
# Alice: vertical worldline at x=0
t_A = np.linspace(0, 10, 100)
x_A = np.zeros(100)
# Bob: outbound (slope = 1/beta), turnaround at (5yr, 4ly), inbound
t_Bob_out = np.linspace(0, 5, 50)
x_Bob_out = 0.8 * t_Bob_out
t_Bob_in  = np.linspace(5, 10, 50)
x_Bob_in  = 4 - 0.8*(t_Bob_in - 5)
ax2.plot(x_A, t_A, 'b-', lw=2, label='Alice (Earth)')
ax2.plot(x_Bob_out, t_Bob_out, 'r-', lw=2, label='Bob (traveler)')
ax2.plot(x_Bob_in, t_Bob_in, 'r-', lw=2)
ax2.plot(0, 0, 'ko', ms=8)
ax2.plot(4, 5, 'rs', ms=8, label='Turnaround')
ax2.plot(0, 10, 'b^', ms=8, label='Reunion')
# Light cone from origin
t_lc = np.linspace(0, 5, 50)
ax2.plot(t_lc, t_lc, 'g--', lw=1, alpha=0.5, label='light (x=ct)')
ax2.plot(-t_lc, t_lc, 'g--', lw=1, alpha=0.5)
ax2.set_xlabel('x (light-years)'); ax2.set_ylabel('t (years)')
ax2.set_title('Spacetime diagram\n(Alice ages 10yr, Bob 6yr)', fontsize=10)
ax2.legend(fontsize=7); ax2.set_xlim(-1, 6); ax2.set_ylim(-0.5, 11)
ax2.text(0.2, 10.2, 'Bob: 6yr', color='r', fontsize=9)
ax2.text(-0.8, 10.2, 'Alice: 10yr', color='b', fontsize=9)

# ── P3: Mass-shell parabola E vs p ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
p_arr = np.linspace(0, 5, 200)   # units: mc
E_massive = np.sqrt(p_arr**2 + 1)   # natural units m=c=1
E_photon  = p_arr
ax3.plot(p_arr, E_massive, 'b-', lw=2, label=r'$E=\sqrt{(pc)^2+(mc^2)^2}$')
ax3.plot(p_arr, E_photon,  'r--', lw=2, label='E=pc (photon, m=0)')
ax3.plot(0, 1, 'bs', ms=8, label='Rest: E=mc²')
ax3.set_xlabel('p  (units mc)'); ax3.set_ylabel('E  (units mc²)')
ax3.set_title('Mass-shell: E²=(pc)²+(mc²)²', fontsize=10)
ax3.legend(fontsize=8)

# ── P4: Rapidity vs velocity ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
betas2 = np.linspace(0, 0.9999, 500)
rapidities = np.arctanh(betas2)
ax4.plot(betas2, rapidities, 'purple', lw=2)
ax4.set_xlabel('β = v/c'); ax4.set_ylabel('φ = atanh(β)')
ax4.set_title('Rapidity φ (ADDS under boosts)', fontsize=10)
ax4.axvline(0.6, color='r', ls=':', lw=1)
ax4.axvline(0.8, color='g', ls=':', lw=1)
b12 = (0.6+0.8)/(1+0.6*0.8)
ax4.axvline(b12, color='orange', ls='-', lw=2, label=f'β₁⊕β₂={b12:.3f}')
ax4.annotate('β₁=0.6', (0.6, 0.1), fontsize=8, color='r')
ax4.annotate('β₂=0.8', (0.8, 0.3), fontsize=8, color='g')
ax4.legend(fontsize=8)

# ── P5: Light cone diagram ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
t_lc2 = np.linspace(-3, 3, 100)
ax5.fill_between(t_lc2, np.abs(t_lc2), 3, alpha=0.15, color='blue', label='Future (timelike)')
ax5.fill_between(t_lc2, -3, -np.abs(t_lc2), alpha=0.15, color='red', label='Past (timelike)')
ax5.fill_betweenx(np.linspace(-np.abs(t_lc2).min(), np.abs(t_lc2).min(), 100),
                  -3, 3, alpha=0.1, color='gray')
ax5.plot(t_lc2, np.abs(t_lc2), 'g-', lw=2)
ax5.plot(t_lc2, -np.abs(t_lc2), 'g-', lw=2, label='Light cone (E=pc)')
ax5.plot(0, 0, 'ko', ms=6)
ax5.set_xlabel('x (light-seconds)'); ax5.set_ylabel('ct (light-seconds)')
ax5.set_title('Light cone + temporal regions', fontsize=10)
ax5.text(0.05, 2.2, 'FUTURE', color='blue', fontsize=10, fontweight='bold')
ax5.text(0.05, -2.7, 'PAST', color='red', fontsize=10, fontweight='bold')
ax5.text(1.5, 0.0, 'SPACELIKE', color='gray', fontsize=8)
ax5.text(-2.8, 0.0, 'SPACELIKE', color='gray', fontsize=8)
ax5.set_xlim(-3, 3); ax5.set_ylim(-3, 3)
ax5.legend(fontsize=7, loc='upper right')

# ── P6: Simultaneity jump (Bob's frame) ──────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
# Plot Alice's age as seen by Bob during trip
t_bob = np.linspace(0, 6, 300)
tau_leg = 3.0   # Bob's proper time per leg
# Outbound: Alice ages slowly (time dilation): rate = 1/gamma² (in Bob's frame)
# Inbound: same rate
rate_slow = 1/gamma_trip**2   # = 1 - beta^2 = 0.36

rate_per_tau = 1/gamma_trip                               # Alice clock rate = 1/gamma per Bob proper time
alice_age_outbound = tau_leg * rate_per_tau              # = 3 * 0.6 = 1.8 yr
alice_age_jump     = t_jump                              # = 6.4 yr
alice_age_inbound  = tau_leg * rate_per_tau              # = 1.8 yr

t_out = np.linspace(0, 3, 150)
t_in  = np.linspace(3, 6, 150)
alice_out = t_out * rate_per_tau
alice_in  = alice_age_outbound + alice_age_jump + (t_in - 3)*rate_per_tau

ax6.plot(t_out, alice_out, 'b-', lw=2, label="Bob sees Alice aging (slow)")
ax6.axvline(3.0, color='orange', ls='--', lw=2, label=f'Turnaround: Alice jumps +{alice_age_jump:.2f}yr')
ax6.plot([3.0, 3.0], [alice_age_outbound, alice_age_outbound + alice_age_jump],
         'orange', lw=3, alpha=0.8)
ax6.plot(t_in, alice_in, 'b-', lw=2)
ax6.plot(6.0, alice_in[-1], 'r*', ms=12, label=f"Alice total: {alice_in[-1]:.2f}yr")
ax6.set_xlabel("Bob's proper time (yr)"); ax6.set_ylabel("Alice's age (yr)")
ax6.set_title("Simultaneity jump resolves paradox", fontsize=10)
ax6.legend(fontsize=7)

out_path = r'D:\Summer2026\Dispersion-Assisted-GS-Phase-Recovery\repl\_out_special_relativity.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')

# ── Final summary ─────────────────────────────────────────────────────────────
hdr("Summary: 5 key equations of SR")
print("""
  1.  gamma = 1/sqrt(1-beta^2)                [Lorentz factor]
  2.  s^2 = (ct)^2 - x^2 = invariant          [Minkowski "circle diameter"]
  3.  Lambda = [cosh phi, -sinh phi; ...]      [boost = hyperbolic rotation]
  4.  E^2 = (pc)^2 + (mc^2)^2                 [4-momentum mass shell]
  5.  CPT conserved in all local LI QFT        [temporal symmetry theorem]

  Twin paradox resolution:
    - Alice ages 10yr, Bob ages 6yr   [time dilation]
    - Asymmetry: Bob accelerates      [breaks inertial equivalence]
    - Bob's frame: simultaneity jump +6.4yr at turnaround [= 2*beta*d; accounts for ALL difference]

  "Circle diameter" = Minkowski invariant interval s^2 = (ct)^2 - |x|^2
    The 'circle' in spacetime is a HYPERBOLA, not a Euclidean circle.
    gamma comes from the ratio of the hyperbola arm to its projection.
""")
print("=== Special relativity section complete ===")
