"""
_repl_radiation_bessel.py
Cell radiation response + Bessel difference equations + LoL cylindrical beam damage
Run: py -3.12 repl/_repl_radiation_bessel.py
"""

import numpy as np
import sympy as sp
from scipy.special import jn, jn_zeros, jnp_zeros, i0, i1, k0, k1, gamma
from scipy.linalg import null_space

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: BESSEL RECURRENCE AS DIFFERENCE EQUATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: BESSEL RECURRENCE = DIFFERENCE EQUATION")
print("=" * 65)

# Bessel recurrence (3-term, second-order linear difference eq):
#   J_{n+1}(x) = (2n/x) * J_n(x) - J_{n-1}(x)
# This IS a difference equation in the order n:
#   y[n+1] - (2n/x)*y[n] + y[n-1] = 0
# Analogous to ODE: y'' + (1/x)*y' + (1 - n^2/x^2)*y = 0
# The recurrence connects the "spatial" ODE to a "discrete" ladder structure

print("\nRecurrence:  J_{n+1}(x) = (2n/x)*J_n(x) - J_{n-1}(x)")
print("This is a 2nd-order LINEAR DIFFERENCE EQUATION in n.")
print("Stability: forward recurrence is UNSTABLE for large n (amplifies errors).")
print("Solution:  use Miller's backward algorithm (start from large N, normalize).\n")

x_test = 5.0
N_max  = 15

# Forward recurrence (unstable for large n -- shows error growth)
J_fwd = np.zeros(N_max + 1)
J_fwd[0] = jn(0, x_test)
J_fwd[1] = jn(1, x_test)
for n in range(1, N_max):
    J_fwd[n+1] = (2*n / x_test) * J_fwd[n] - J_fwd[n-1]

# Backward (Miller) algorithm -- stable
J_bwd = np.zeros(N_max + 2)
J_bwd[N_max + 1] = 0.0
J_bwd[N_max]     = 1e-30    # arbitrary seed
for n in range(N_max - 1, -1, -1):
    J_bwd[n] = (2*(n+1) / x_test) * J_bwd[n+1] - J_bwd[n+2]
# Normalize using J_0 sum rule: sum_n J_n^2 = 1/2 for large x... use J_0 exact
J_bwd *= jn(0, x_test) / J_bwd[0]

print(f"  x = {x_test},  comparing forward vs backward vs exact:")
print(f"  {'n':>3}  {'exact':>12}  {'forward':>12}  {'backward':>12}  fwd_err")
for n in range(N_max + 1):
    exact = jn(n, x_test)
    fwd_err = abs(J_fwd[n] - exact)
    print(f"  {n:>3}  {exact:>12.6f}  {J_fwd[n]:>12.6f}  {J_bwd[n]:>12.6f}  {fwd_err:.1e}")

print("\n  Forward recurrence error explodes at n>8 (amplification ~2n/x per step).")
print("  Backward (Miller) stays exact to machine precision.")

# ── SymPy: difference equation operator form ─────────────────────────────────
print("\n--- SymPy: Bessel recurrence as Z-transform / generating function ---")
n_s, x_s, z_s = sp.symbols('n x z', positive=True)
# Recurrence: J_{n+1} = (2n/x)*J_n - J_{n-1}
# In Z-domain: z*F(z) = (2n/x)*F(z) ... (operator form)
# Generating function: sum_{n=-inf}^{inf} J_n(x)*t^n = exp(x/2*(t - 1/t))
t_s = sp.Symbol('t')
gen_fn = sp.exp(x_s/2 * (t_s - 1/t_s))
print(f"\n  Generating function: exp(x/2*(t-1/t)) = {gen_fn}")
print("  Series coefficient of t^n -> J_n(x)")
# Extract a few coefficients
# The generating function is a LAURENT series (negative + positive powers of t)
# exp(x/2*(t-1/t)) = sum_{n=-inf}^{inf} J_n(x) * t^n
# SymPy's series() only handles positive powers; use the integral definition instead:
# J_n(x) = (1/2pi) * integral_0^{2pi} cos(n*theta - x*sin(theta)) d_theta
print(f"\n  Generating fn is a LAURENT series: sum J_n(x)*t^n for n=-inf..inf")
print(f"  Verify J_n(1) from integral definition vs scipy:")
theta = sp.Symbol('theta', real=True)
for k in range(4):
    integrand = sp.cos(k*theta - 1*sp.sin(theta)) / (2*sp.pi)
    # numerical check
    th_vals = np.linspace(0, 2*np.pi, 10000)
    J_num = np.trapezoid(np.cos(k*th_vals - np.sin(th_vals)), th_vals) / (2*np.pi)
    exact_k = float(jn(k, 1.0))
    print(f"    J_{k}(1): numeric_integral={J_num:.6f}  exact={exact_k:.6f}  err={abs(J_num-exact_k):.1e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CELL RADIATION RESPONSE — LINEAR-QUADRATIC MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: CELL RADIATION RESPONSE (Linear-Quadratic Model)")
print("=" * 65)

# LQ survival fraction: S(d) = exp(-alpha*d - beta*d^2)
# alpha = single-hit lethal damage (DSB directly lethal)
# beta  = two-hit lethal damage (2 sublethal lesions interact)
# alpha/beta ratio = crossover dose (equal contribution)

cell_lines = {
    "HeLa (cervical)":    (0.30, 0.030),   # alpha/beta = 10 Gy
    "Prostate tumor":      (0.15, 0.050),   # alpha/beta = 3 Gy (late-responding)
    "Lung (A549)":         (0.20, 0.020),   # alpha/beta = 10 Gy
    "Neurons (late resp)": (0.05, 0.025),   # alpha/beta = 2 Gy -- radiosensitive!
    "Bone marrow stem":    (0.35, 0.035),   # alpha/beta = 10 Gy
}

print(f"\n  {'Cell line':25s}  {'alpha':>6}  {'beta':>6}  {'a/b':>5}  "
      + "  ".join([f"{d}Gy" for d in [1, 2, 4, 8]]))
print("-" * 85)

for name, (alpha, beta) in cell_lines.items():
    ab = alpha / beta
    survivals = [np.exp(-alpha*d - beta*d**2) for d in [1, 2, 4, 8]]
    surv_str = "  ".join([f"{s:.4f}" for s in survivals])
    print(f"  {name:25s}  {alpha:>6.3f}  {beta:>6.3f}  {ab:>5.1f}  {surv_str}")

# ── Fractionation: 2Gy x 30 fractions (standard RT) vs 8Gy x 5 (SBRT) ───────
print("\n--- Fractionation comparison ---")
alpha_tumor, beta_tumor  = 0.30, 0.030   # HeLa-like
alpha_normal, beta_normal = 0.05, 0.025  # late-responding normal tissue

def BED(alpha, beta, d, n):
    """Biologically Effective Dose = n*d*(1 + d/(alpha/beta))"""
    return n * d * (1 + d / (alpha / beta))

def TCP(alpha, beta, d, n, N0=1e8):
    """Tumor Control Probability: Poisson prob that 0 cells survive"""
    S_per_fraction = np.exp(-alpha*d - beta*d**2)
    S_total = S_per_fraction**n
    N_surviving = N0 * S_total
    return np.exp(-N_surviving)

schedules = [
    ("Conventional  2Gy x 30", 2.0, 30),
    ("Hypofrx       4Gy x 15", 4.0, 15),
    ("SBRT          8Gy x 5 ", 8.0, 5),
    ("SRS           24Gy x 1 ", 24.0, 1),
]

print(f"\n  {'Schedule':28s}  {'Total':>6}  {'BED tumor':>10}  "
      f"{'BED normal':>10}  {'TCP':>8}")
print("-" * 75)
for name, d, n in schedules:
    total = d * n
    bed_t = BED(alpha_tumor, beta_tumor, d, n)
    bed_n = BED(alpha_normal, beta_normal, d, n)
    tcp   = TCP(alpha_tumor, beta_tumor, d, n)
    print(f"  {name:28s}  {total:>6.0f}  {bed_t:>10.1f}  {bed_n:>10.1f}  {tcp:>8.4f}")

print("\n  BED tumor should be high; BED normal should be low.")
print("  SBRT achieves high TCP with fewer fractions BUT more normal tissue BED.")

# ── Cell cycle checkpoints ────────────────────────────────────────────────────
print("\n--- Cell cycle checkpoints after radiation ---")
print("""
  G1/S checkpoint  (p53/p21 axis):
    ATM -> p53 -> p21 -> CDK2 inhibited -> G1 arrest
    Purpose: repair before replication

  Intra-S checkpoint (ATR/CHK1):
    Replication fork stalls at DSB -> CHK1 -> CDC25A degraded -> S-phase slowdown

  G2/M checkpoint (CHK1/CHK2):
    CHK1/CHK2 -> CDC25C phosphorylation/sequestration -> CDK1 inactive -> G2 arrest
    Purpose: most important clinically (majority of cells in G2 after 2 Gy)

  Mitotic checkpoint (spindle assembly):
    Not radiation-specific; prevents aneuploidy

  Dose-response of arrest duration (empirical):
    G2 arrest t(d) ~ 2 + 0.8*d  hours  (for d in Gy, HeLa)
""")

d_doses = [0, 1, 2, 4, 8]
print(f"  {'Dose(Gy)':>9}  {'G2 arrest (hr)':>15}  {'S(d) HeLa':>12}")
for d in d_doses:
    t_g2 = 2 + 0.8*d if d > 0 else 0
    s_d  = np.exp(-0.30*d - 0.030*d**2)
    print(f"  {d:>9}  {t_g2:>15.1f}  {s_d:>12.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CYLINDRICAL DOSE DISTRIBUTION — BESSEL APPEARS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: CYLINDRICAL RADIATION BEAM — BESSEL IN THE DOSE KERNEL")
print("=" * 65)

# Pencil beam kernel in cylindrical geometry:
# Scatter dose: D_scatter(r) ~ (1/r)*exp(-r/lambda) for lateral scatter
# Exact solution of diffusion equation in cylindrical coords:
#   D(r) = D0 * K_0(r/lambda)  (modified Bessel K_0)
# K_0 is the Green's function of the cylindrical diffusion equation:
#   (1/r)*d/dr[r*dD/dr] - (1/lambda^2)*D = -S*delta(r)

lambda_scatter = 3.0  # cm lateral scatter length in tissue
r_cm = np.linspace(0.01, 15.0, 300)

D_scatter = k0(r_cm / lambda_scatter)   # K_0: modified Bessel, falls off as exp(-r)/sqrt(r)
D_primary = np.exp(-r_cm / 10.0)        # crude primary beam attenuation

# Normalize
D_scatter /= D_scatter[0]
print(f"\n  Scatter dose kernel D(r) = K_0(r/lambda), lambda={lambda_scatter} cm")
print(f"  {'r (cm)':>8}  {'D_scatter':>12}  {'K_0 asymptotic':>16}")
r_sample = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for r in r_sample:
    K0_exact  = k0(r / lambda_scatter) / k0(0.1/lambda_scatter)
    K0_asymp  = np.sqrt(np.pi*lambda_scatter/(2*r)) * np.exp(-r/lambda_scatter)
    K0_asymp /= k0(0.1/lambda_scatter)
    print(f"  {r:>8.1f}  {K0_exact:>12.6f}  {K0_asymp:>16.6f}")

print("\n  K_0(r/lambda) is the cylindrical analog of exp(-r) in 1D.")
print("  It satisfies: (1/r)*d/dr[r*dD/dr] = (1/lambda^2)*D")
print("  This is the modified Bessel equation of order 0.")

# ── Normal tissue complication: dose-volume integral ─────────────────────────
print("\n--- Dose-volume histogram (DVH) effect on TCP/NTCP ---")
# Parallel organ (lung): NTCP = 1 - prod_{i}(1 - P_i(d_i))
# Series organ (spinal cord): NTCP = 1 - prod_{i}(1 - P_i(d_i))  same formula
# Lyman-Kutcher-Burman (LKB) model for lung:
#   NTCP = Phi[(D_eff/TD50 - 1)/m]
#   D_eff = (sum_i v_i * d_i^(1/n))^n

TD50_lung = 30.0   # Gy — dose for 50% pneumonitis
m_lung    = 0.36   # slope
n_lung    = 0.99   # volume exponent (parallel organ, near 1)

def lkb_ntcp(d_eff, TD50, m):
    from scipy.special import erf
    t = (d_eff/TD50 - 1) / m
    return 0.5 * (1 + erf(t / np.sqrt(2)))

# Simulate DVH: lung receives graded dose
lung_vols  = np.array([0.10, 0.20, 0.30, 0.20, 0.10, 0.10])  # fraction of organ
lung_doses = np.array([20.0, 15.0, 10.0,  5.0,  2.0,  0.5])  # Gy per volume bin

D_eff = (np.sum(lung_vols * lung_doses**(1/n_lung)))**n_lung
ntcp  = lkb_ntcp(D_eff, TD50_lung, m_lung)
print(f"\n  Lung DVH: D_eff = {D_eff:.2f} Gy,  NTCP(pneumonitis) = {ntcp:.4f}")
print(f"  V20 (fraction receiving >20 Gy) = {lung_vols[lung_doses>=20].sum():.0%}")
print(f"  Clinical constraint: V20 < 30% -- {'PASS' if lung_vols[lung_doses>=20].sum()<0.30 else 'FAIL'}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: LEAGUE OF LEGENDS — CYLINDRICAL LASER BEAMS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: LOL VEL'KOZ LASER BEAM (cylindrical Bessel modes)")
print("=" * 65)

print("""
  Vel'Koz (The Eye of the Void) fires a continuous laser.
  The beam is a TM01 mode in a cylindrical waveguide (vacuum "void tube").

  Field profile of TM01 inside waveguide of radius a:
    E_z(r) = E0 * J_0(p01 * r/a)      p01 = 2.4048 (first J_0 zero)
    H_phi(r) = -i*E0*(epsilon_0/mu_0)^0.5 * J_1(p01 * r/a)

  Damage = Poynting flux ~ |E_z|^2 + |E_r|^2
         = E0^2 * [J_0^2 + J_1^2](p01 * r/a)

  At r=0: J_0=1, J_1=0 -> max central damage
  At r=a: J_0=0 (by boundary condition) -> zero field at wall
""")

a_vel = 150.0    # beam radius in LoL units
r_beam = np.linspace(0, a_vel, 200)
p01    = jn_zeros(0, 1)[0]   # 2.4048

E_z  = jn(0, p01 * r_beam / a_vel)
H_phi = jn(1, p01 * r_beam / a_vel)
damage = E_z**2 + H_phi**2

print(f"  J_0 first zero (cutoff radius): p01 = {p01:.6f}")
print(f"  Beam radius a = {a_vel:.0f} units\n")
print(f"  {'r (units)':>10}  {'J_0(p01*r/a)':>14}  {'J_1(p01*r/a)':>14}  {'Damage':>10}")
for r in [0, 30, 60, 90, 120, 150]:
    idx = int(r / a_vel * 199)
    print(f"  {r:>10}  {E_z[idx]:>14.6f}  {H_phi[idx]:>14.6f}  {damage[idx]:>10.6f}")

# ── Orianna (ball splash) — J_0 zeros = zero-damage rings ──────────────────
print("\n--- Orianna Command: Shockwave splash profile ---")
a_ori = 410.0    # LoL units, shockwave radius
zeros = jn_zeros(0, 3)
print(f"  Shockwave damage D(r) = J_0(2.4048 * r / {a_ori:.0f})")
print(f"  Zero-damage rings at r = {', '.join([f'{z/p01*a_ori:.0f}' for z in zeros])} units")

# ── Lux laser: Gaussian approx vs true Bessel ────────────────────────────────
print("\n--- Lux laser: Bessel TEM00 vs Gaussian approximation ---")
# Gaussian beam: I(r) = I0 * exp(-2*r^2/w^2)
# Bessel beam:   I(r) = I0 * J_0^2(k_r * r)  [non-diffracting!]
# Gaussian spreads; Bessel doesn't (infinite energy -- physical bessel-gauss used)
w_lux = 80.0    # beam waist units
k_r   = 2*np.pi / (2*w_lux)
r_lux = np.linspace(0, 300, 200)
I_gaussian = np.exp(-2*r_lux**2 / w_lux**2)
I_bessel   = jn(0, k_r * r_lux)**2

print(f"\n  {'r':>6}  {'Gaussian I':>12}  {'Bessel I':>12}  diff%")
for r in [0, 40, 80, 120, 160, 200]:
    idx = int(r/300*199)
    g   = I_gaussian[idx]
    b   = I_bessel[idx]
    d   = abs(g-b)/max(g,1e-9)*100
    print(f"  {r:>6}  {g:>12.4f}  {b:>12.4f}  {d:>6.1f}%")

print("\n  Bessel beam is non-diffracting (J_0^2 oscillates, no spreading).")
print("  Gaussian decays faster -> Lux laser clips at shorter range.")
print("  Real laser pointers use Bessel-Gauss (Bessel * Gaussian envelope).")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: DNA DOUBLE-STRAND BREAK KINETICS + REPAIR (Bessel? No -- but ODE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: DNA DSB REPAIR KINETICS (two-component exponential)")
print("=" * 65)

# DSB repair: two-component model
# dN/dt = -lambda_fast * N_fast - lambda_slow * N_slow
# N_fast(t) = f * N0 * exp(-lambda_fast * t)
# N_slow(t) = (1-f) * N0 * exp(-lambda_slow * t)
# f ~ 0.65 (fast component), t_half_fast ~ 0.25 hr, t_half_slow ~ 4 hr

N0_dsb       = 40.0    # DSBs per cell per 2 Gy (rough average)
f_fast       = 0.65
t_half_fast  = 0.25    # hr
t_half_slow  = 4.0     # hr
lam_fast     = np.log(2) / t_half_fast
lam_slow     = np.log(2) / t_half_slow

t_hr = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16, 24])
N_fast = f_fast       * N0_dsb * np.exp(-lam_fast * t_hr)
N_slow = (1-f_fast)   * N0_dsb * np.exp(-lam_slow * t_hr)
N_total = N_fast + N_slow
pct_repaired = (1 - N_total / N0_dsb) * 100

print(f"\n  Initial DSBs: {N0_dsb:.0f} per cell (2 Gy dose)")
print(f"  Fast component: {f_fast*100:.0f}%,  t_half={t_half_fast:.2f} hr")
print(f"  Slow component: {(1-f_fast)*100:.0f}%,  t_half={t_half_slow:.1f} hr\n")
print(f"  {'t (hr)':>7}  {'N_fast':>8}  {'N_slow':>8}  {'Total':>8}  {'Repaired%':>10}")
for i, t in enumerate(t_hr):
    print(f"  {t:>7.2f}  {N_fast[i]:>8.2f}  {N_slow[i]:>8.2f}  "
          f"{N_total[i]:>8.2f}  {pct_repaired[i]:>10.1f}%")

# ── Misrepair -> mutation -> cancer ──────────────────────────────────────────
print("\n--- Misrepair probability and oncogenic risk ---")
# Probability of misrepair per DSB: p_mis ~ 1e-4 to 1e-3
# P(at least one oncogenic event) ~ 1 - (1-p_mis)^N0
p_mis = 5e-4
P_oncogenic = 1 - (1 - p_mis)**N0_dsb
print(f"\n  p_misrepair per DSB = {p_mis:.1e}")
print(f"  N0 = {N0_dsb:.0f} DSBs per 2 Gy")
print(f"  P(>=1 oncogenic misrepair per cell) = {P_oncogenic:.4f} = {P_oncogenic*100:.2f}%")
print(f"\n  With 10^8 tumor cells irradiated:")
print(f"    Expected misrepairs = {N0_dsb*p_mis*1e8:.0f}")
print(f"  This is WHY radiotherapy is carcinogenic at low doses but curative at high doses:")
print(f"    Low dose -> insufficient kill, some misrepairs survive -> RISK")
print(f"    High dose -> most cells killed before they can divide -> CURE")

print("\nDone.")
