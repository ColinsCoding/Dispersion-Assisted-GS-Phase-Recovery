"""
_repl_lennard_jones.py
Lennard-Jones potential + DOM tree traversal + Smith chart + Black-Scholes + caffeine PK
Run: py -3.12 repl/_repl_lennard_jones.py
"""

import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import odeint

kB  = 1.380649e-23
NA  = 6.02214076e23
e_c = 1.602176634e-19

# =============================================================================
# SECTION 1: LENNARD-JONES POTENTIAL
# =============================================================================
print("=" * 65)
print("SECTION 1: LENNARD-JONES 12-6 POTENTIAL")
print("=" * 65)

print("""
  V(r) = 4*eps * [(sigma/r)^12 - (sigma/r)^6]

  (sigma/r)^12  repulsion  (Pauli exclusion, orbital overlap)
  (sigma/r)^6   attraction (London dispersion, induced dipole)

  Minimum: r_min = 2^(1/6) * sigma   (equilibrium bond length)
  V(r_min)    = -eps                  (well depth = binding energy)
  V(sigma)    = 0                     (hard-sphere contact)

  Force: F(r) = -dV/dr = 24*eps/r * [2*(sigma/r)^12 - (sigma/r)^6]
  F > 0 (repulsive) for r < r_min
  F < 0 (attractive) for r > r_min
""")

r_s, eps_s, sigma_s = sp.symbols('r epsilon sigma', positive=True)
V_LJ = 4 * eps_s * ((sigma_s/r_s)**12 - (sigma_s/r_s)**6)
F_LJ = -sp.diff(V_LJ, r_s)
r_eq = sp.solve(sp.diff(V_LJ, r_s), r_s)[0]
V_eq = V_LJ.subs(r_s, r_eq)

print(f"  V(r) = {V_LJ}")
print(f"  F(r) = -dV/dr = {sp.simplify(F_LJ)}")
print(f"  r_min = {sp.simplify(r_eq)} = 2^(1/6)*sigma = {2**(1/6):.6f}*sigma")
print(f"  V(r_min) = {sp.simplify(V_eq)}")

# Numerical LJ for Argon (noble gas, well-characterized)
eps_Ar   = 1.654e-21   # J  (119.8 K * kB)
sigma_Ar = 3.405e-10   # m  (3.405 Angstrom)
r_min_Ar = 2**(1/6) * sigma_Ar

print(f"\n  Argon LJ parameters:")
print(f"  eps = {eps_Ar/kB:.1f} K * kB = {eps_Ar/e_c*1000:.4f} meV")
print(f"  sigma = {sigma_Ar*1e10:.3f} Angstrom")
print(f"  r_min = 2^(1/6)*sigma = {r_min_Ar*1e10:.3f} Angstrom")
print(f"  Boiling point Ar = 87.3 K   (eps/kB = {eps_Ar/kB:.1f} K -- same order, good)")

# V(r) table
r_arr = np.linspace(0.85*sigma_Ar, 4.0*sigma_Ar, 12)
print(f"\n  {'r (A)':>8}  {'r/sigma':>8}  {'V (meV)':>10}  {'F (pN)':>10}  {'regime'}")
print("-" * 55)
for r in r_arr:
    rs    = r / sigma_Ar
    V_num = 4 * eps_Ar * (rs**-12 - rs**-6)
    F_num = 24 * eps_Ar / r * (2*rs**-12 - rs**-6)
    reg   = "repulsive" if F_num > 0 else ("equilib." if abs(F_num) < 1e-13 else "attractive")
    print(f"  {r*1e10:>8.3f}  {rs:>8.4f}  {V_num/e_c*1000:>10.4f}  {F_num*1e12:>10.4f}  {reg}")

# Cutoff radius: truncate and shift at r_c = 2.5*sigma (MD standard)
r_c    = 2.5 * sigma_Ar
V_rc   = 4 * eps_Ar * ((sigma_Ar/r_c)**12 - (sigma_Ar/r_c)**6)
print(f"\n  MD cutoff r_c = 2.5*sigma = {r_c*1e10:.3f} A")
print(f"  V(r_c) = {V_rc/e_c*1000:.5f} meV  (small -- truncation error minimal)")
print(f"  Truncate-and-shift: V_ts(r) = V(r) - V(r_c) for r < r_c, else 0")
print(f"  This removes discontinuity in V but F still discontinuous")

# Reduced units (LJ units): e* = eps, sigma* = sigma, tau* = sigma*sqrt(m/eps)
m_Ar = 39.948e-3 / NA   # kg
tau_LJ = sigma_Ar * np.sqrt(m_Ar / eps_Ar)
T_star_melt = 0.694   # reduced melting temp of LJ solid
T_melt_K    = T_star_melt * eps_Ar / kB
print(f"\n  LJ reduced units (Argon):")
print(f"  Time unit tau* = sigma*sqrt(m/eps) = {tau_LJ*1e12:.3f} ps")
print(f"  T* = kBT/eps:  T*_melt = {T_star_melt}  -> T_melt = {T_melt_K:.1f} K  (actual: 83.8 K)")

# =============================================================================
# SECTION 2: DOM TREE — TREE TRAVERSAL AS GRAPH STRUCTURE
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 2: DOM TREE -- TREE TRAVERSAL (DFS/BFS/XPath)")
print("=" * 65)

print("""
  DOM (Document Object Model): HTML parsed into a tree
  Each node: element tag, attributes, children, parent

  Traversal algorithms:
    DFS preorder:  visit node, then children left-to-right
    DFS postorder: children first, then node (used in deletion)
    BFS (level):   queue-based, level by level

  Why trees matter in physics/signal processing:
    - Wavelet transform = binary tree of frequency bands
    - Octree = 3D spatial partitioning (MD neighbor lists!)
    - B-tree = database index (FASTQ genomic lookups)
    - Merkle tree = blockchain / genomic integrity hash
""")

# Simple tree implementation in Python
class Node:
    def __init__(self, tag, val=None):
        self.tag      = tag
        self.val      = val
        self.children = []
    def add(self, child):
        self.children.append(child)
        return child

# Build a small HTML-like DOM
html  = Node("html")
head  = html.add(Node("head"))
head.add(Node("title", "RogueGuard Spectrum"))
head.add(Node("meta",  "charset=utf-8"))
body  = html.add(Node("body"))
div   = body.add(Node("div", "id=main"))
h1    = div.add(Node("h1",  "Phase Retrieval Dashboard"))
canvas = div.add(Node("canvas", "id=spectrogram width=800 height=400"))
script = body.add(Node("script", "src=gs_core.js"))

def dfs_preorder(node, depth=0):
    indent = "  " * depth
    print(f"  {indent}<{node.tag}> {node.val or ''}")
    for child in node.children:
        dfs_preorder(child, depth + 1)

def bfs(root):
    from collections import deque
    q = deque([(root, 0)])
    level_prev = -1
    while q:
        node, level = q.popleft()
        if level != level_prev:
            print(f"\n  Level {level}:", end="")
            level_prev = level
        print(f" <{node.tag}>", end="")
        for child in node.children:
            q.append((child, level + 1))
    print()

def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.children)

def max_depth(node):
    if not node.children:
        return 0
    return 1 + max(max_depth(c) for c in node.children)

print("  DOM tree (DFS preorder):")
dfs_preorder(html)
print(f"\n  BFS (level order):")
bfs(html)
print(f"\n  Nodes: {count_nodes(html)},  Max depth: {max_depth(html)}")

# XPath-like selector: find all nodes with tag
def find_all(node, tag):
    results = []
    if node.tag == tag:
        results.append(node)
    for child in node.children:
        results.extend(find_all(child, tag))
    return results

canvases = find_all(html, "canvas")
print(f"\n  find_all('canvas'): {[n.val for n in canvases]}")

# Complexity
N = count_nodes(html)
print(f"\n  DFS time O(N), BFS time O(N), space O(depth) DFS / O(width) BFS")
print(f"  This tree: N={N}, depth={max_depth(html)} -- tiny")
print(f"  Real DOM (gmail): N~10000, depth~30, BFS queue ~200 nodes/level")

# Wavelet tree connection
print("\n  Wavelet/octree connection:")
print("  N=1024 signal -> DWT binary tree: log2(1024)=10 levels")
print("  Level 0: 512 detail coeffs (high freq)")
print("  Level 9: 2 approx coeffs  (low freq DC)")
print("  Octree for MD neighbor list: subdivide until < 32 atoms/cell")
print("  Query time: O(log N) vs O(N^2) brute force -> 1e6 atoms feasible")

# =============================================================================
# SECTION 3: SMITH CHART -- RF IMPEDANCE
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 3: SMITH CHART -- COMPLEX IMPEDANCE NAVIGATION")
print("=" * 65)

print("""
  Smith chart: complex reflection coefficient Gamma plotted on unit circle
    Gamma = (Z - Z0) / (Z + Z0)    Z0 = 50 ohm system impedance
    |Gamma| = 1: perfect reflection (open/short)
    |Gamma| = 0: matched (Z = Z0, center of chart)

  Normalized impedance: z = Z/Z0 = r + j*x
    r = resistance circles  (constant r -> circles tangent at Gamma=+1)
    x = reactance arcs      (constant x -> arcs from Gamma=+1)

  Key points:
    Center   (Gamma=0):   Z = 50 ohm  (matched)
    Right    (Gamma=+1):  Z = inf     (open circuit)
    Left     (Gamma=-1):  Z = 0       (short circuit)
    Top      (Im>0):      inductive
    Bottom   (Im<0):      capacitive
""")

Z0 = 50.0   # ohms

impedances = {
    "Matched (50 ohm)":          50 + 0j,
    "Open circuit":              1e6 + 0j,
    "Short circuit":             0.001 + 0j,
    "Inductive (50+j50)":        50 + 50j,
    "Capacitive (50-j50)":       50 - 50j,
    "Coax connector ~75 ohm":    75 + 0j,
    "Dipole antenna ~73+j42":    73 + 42j,
    "Laser diode (~5+j2 ohm)":    5 + 2j,
}

print(f"  {'Impedance':28s}  {'Z':>14}  {'Gamma':>18}  {'|Gamma|':>8}  {'VSWR':>7}  {'RL(dB)':>8}")
print("-" * 85)
for name, Z in impedances.items():
    Gamma    = (Z - Z0) / (Z + Z0)
    mag_G    = abs(Gamma)
    VSWR     = (1 + mag_G) / (1 - mag_G) if mag_G < 0.9999 else np.inf
    RL_dB    = -20 * np.log10(mag_G) if mag_G > 1e-9 else np.inf
    print(f"  {name:28s}  {Z.real:>5.1f}+j{Z.imag:>5.1f}  "
          f"{Gamma.real:>+7.4f}+j{Gamma.imag:>+7.4f}  "
          f"{mag_G:>8.4f}  {VSWR:>7.2f}  {RL_dB:>8.2f}")

# L-network impedance matching
print("\n--- L-network: match laser diode Z=5+j2 to 50 ohm ---")
print("  Two-element LC network: series L + shunt C (or vice versa)")
print("  Q = sqrt((R_s/R_L) - 1)  where R_s=50, R_L=5")
R_s, R_L = 50.0, 5.0
Q_match = np.sqrt(R_s/R_L - 1)
f_RF    = 2.4e9   # Hz (WiFi band example)
omega   = 2*np.pi*f_RF
X_s     = Q_match * R_L          # series reactance
X_p     = R_s / Q_match          # parallel reactance
L_ser   = X_s / omega
C_par   = 1 / (omega * X_p)

print(f"  R_source=50, R_load=5, f={f_RF/1e9:.1f}GHz")
print(f"  Q = sqrt(50/5 - 1) = {Q_match:.4f}")
print(f"  Series L = X_s/omega = {X_s:.2f}/omega = {L_ser*1e9:.2f} nH")
print(f"  Shunt  C = 1/(omega*X_p) = {C_par*1e12:.2f} pF")
print(f"  After matching: |Gamma| -> 0 at f={f_RF/1e9:.1f} GHz")

# =============================================================================
# SECTION 4: BLACK-SCHOLES OPTIONS (connects to rogue_market REPL)
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 4: BLACK-SCHOLES OPTIONS PRICING")
print("=" * 65)

print("""
  Black-Scholes model: stock price S follows geometric Brownian motion
    dS = mu*S*dt + sigma*S*dW    (Ito SDE)

  Call option C(S,t): right to BUY at strike K by expiry T
  Put  option P(S,t): right to SELL at strike K by expiry T

  Black-Scholes formula:
    C = S*N(d1) - K*exp(-r*T)*N(d2)
    P = K*exp(-r*T)*N(-d2) - S*N(-d1)

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

  N(x) = standard normal CDF
  sigma = implied volatility (NOT LJ sigma!)
  Connection to GS: both solve for hidden state from observable intensity
""")

from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option='call'):
    if T <= 0:
        return max(S-K, 0) if option == 'call' else max(K-S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    vega  = S*norm.pdf(d1)*np.sqrt(T) / 100   # per 1% vol change
    return delta, gamma, theta, vega

# NVDA example (photonics/AI stock)
S0    = 950.0    # current stock price
K_atm = 950.0    # at-the-money strike
r_rf  = 0.053    # risk-free rate
sigma_v = 0.55   # implied vol (NVDA is volatile)
T_vals  = [1/52, 1/12, 3/12, 6/12, 1.0]  # weeks to years

print(f"  NVDA: S={S0}, sigma_iv={sigma_v*100:.0f}%, r={r_rf*100:.1f}%")
print(f"  {'T':>8}  {'C (ATM)':>9}  {'P (ATM)':>9}  {'delta':>7}  {'gamma':>7}  "
      f"{'theta/day':>10}  {'vega/1%':>9}")
print("-" * 70)
for T in T_vals:
    C = black_scholes(S0, K_atm, T, r_rf, sigma_v, 'call')
    P = black_scholes(S0, K_atm, T, r_rf, sigma_v, 'put')
    d, g, th, v = greeks(S0, K_atm, T, r_rf, sigma_v)
    T_str = f"{T*52:.1f}wk" if T < 0.1 else f"{T*12:.1f}mo" if T < 0.3 else f"{T:.1f}yr"
    print(f"  {T_str:>8}  ${C:>8.2f}  ${P:>8.2f}  {d:>7.4f}  {g:>7.5f}  {th:>10.4f}  ${v:>8.4f}")

# Implied vol surface (volatility smile)
print("\n  Volatility smile: IV varies with strike (market reality != BS)")
K_range = np.array([700, 800, 850, 900, 950, 1000, 1050, 1100, 1200])
T_fixed = 1/12   # 1 month
iv_smile = sigma_v + 0.15*(K_range/S0 - 1)**2 - 0.05*(K_range/S0 - 1)  # skew + smile
print(f"  T=1mo:")
print(f"  {'K':>6}  {'K/S':>6}  {'IV':>8}  {'Call':>9}")
for K, iv in zip(K_range, iv_smile):
    C = black_scholes(S0, K, T_fixed, r_rf, iv)
    print(f"  {K:>6}  {K/S0:>6.3f}  {iv*100:>7.1f}%  ${C:>8.2f}")

print("\n  Put-call parity: C - P = S - K*exp(-r*T)")
T_pc = 0.25
C_check = black_scholes(S0, K_atm, T_pc, r_rf, sigma_v, 'call')
P_check = black_scholes(S0, K_atm, T_pc, r_rf, sigma_v, 'put')
parity  = S0 - K_atm*np.exp(-r_rf*T_pc)
print(f"  C-P = {C_check-P_check:.4f},  S-K*e^(-rT) = {parity:.4f}  "
      f"{'OK' if abs(C_check-P_check-parity)<0.01 else 'FAIL'}")

# =============================================================================
# SECTION 5: CAFFEINE PHARMACOKINETICS (energy drink)
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 5: CAFFEINE PK -- ENERGY DRINK MODEL")
print("=" * 65)

print("""
  Caffeine: 1,3,7-trimethylxanthine  MW=194.19 g/mol
  Mechanism: adenosine receptor antagonist (blocks fatigue signal)
  Also: inhibits PDE -> raises cAMP -> PKA activation -> alert

  1-compartment PK model (oral):
    dC/dt = -k_el * C   (after absorption peak)
    C(t)  = C_max * exp(-k_el * (t - t_max))

  Parameters (average adult, 70 kg):
    t_half    = 5.0 hr  (range 3-7 hr, CYP1A2 genotype-dependent)
    Vd        = 0.6 L/kg = 42 L
    F         = 99%  (nearly complete oral bioavailability)
    t_max     = 45 min (time to peak plasma concentration)
    Therapeutic effect: C > 2-5 ug/mL (alertness)
    Toxicity:  C > 80 ug/mL (tachycardia, anxiety)
    LD50 (rat): ~200 mg/kg -> human estimate ~10g (150 cans of Red Bull)
""")

# Energy drink caffeine content
drinks = {
    "Red Bull (250mL)":     80,
    "Monster (473mL)":     160,
    "Bang (473mL)":        300,
    "NOS (473mL)":         160,
    "Celsius (355mL)":     200,
    "Drip coffee (237mL)": 120,
    "Espresso (30mL)":      63,
    "Green tea (237mL)":    28,
    "Coca-Cola (355mL)":    34,
}

t_half_hr = 5.0
k_el      = np.log(2) / t_half_hr   # hr^-1
Vd_L      = 42.0
F_bio     = 0.99
t_max_hr  = 0.75

print(f"  k_el = ln(2)/t_half = {k_el:.4f} hr^-1  (t_half={t_half_hr} hr)")

print(f"\n  {'Drink':28s}  {'Caffeine(mg)':>13}  {'C_max(ug/mL)':>13}  {'C@4hr':>9}  {'C@8hr':>9}")
print("-" * 80)
for name, dose_mg in drinks.items():
    C_max = F_bio * dose_mg / Vd_L   # ug/mL = mg/L
    C_4hr = C_max * np.exp(-k_el * (4 - t_max_hr)) if 4 > t_max_hr else C_max
    C_8hr = C_max * np.exp(-k_el * (8 - t_max_hr))
    alert = "alerting" if C_max > 2 else "mild"
    print(f"  {name:28s}  {dose_mg:>13}  {C_max:>13.3f}  {C_4hr:>9.3f}  {C_8hr:>9.3f}  {alert}")

# Stacking: Bang + Monster within 1 hour
print("\n--- Caffeine stacking: Bang (t=0) + Monster (t=1hr) ---")
t_arr = np.linspace(0, 24, 500)

def C_caffeine(t, dose_mg, t_dose_hr=0.0):
    t_eff = t - t_dose_hr
    C_max = F_bio * dose_mg / Vd_L
    if hasattr(t_eff, '__len__'):
        C = np.where(t_eff < 0, 0,
            np.where(t_eff < t_max_hr,
                     C_max * (t_eff / t_max_hr),
                     C_max * np.exp(-k_el * (t_eff - t_max_hr))))
    else:
        if t_eff < 0: return 0
        elif t_eff < t_max_hr: return C_max * t_eff / t_max_hr
        else: return C_max * np.exp(-k_el * (t_eff - t_max_hr))
    return C

C_bang    = C_caffeine(t_arr, 300, 0.0)
C_monster = C_caffeine(t_arr, 160, 1.0)
C_total   = C_bang + C_monster

t_peak_idx = np.argmax(C_total)
print(f"  Bang 300mg at t=0 + Monster 160mg at t=1hr:")
print(f"  Peak concentration: {C_total[t_peak_idx]:.2f} ug/mL at t={t_arr[t_peak_idx]:.1f} hr")
print(f"  Hours above 2 ug/mL (alerting threshold): "
      f"{np.sum(C_total > 2.0) * 24/500:.1f} hr")
print(f"  Hours above 5 ug/mL (strong stimulation): "
      f"{np.sum(C_total > 5.0) * 24/500:.1f} hr")
print(f"  {'Bang alone peak':30s}: {np.max(C_bang):.2f} ug/mL")

print(f"\n  {'t (hr)':>8}  {'C_Bang':>8}  {'C_Monster':>10}  {'C_total':>9}  {'effect'}")
for t_show in [0, 0.75, 1.75, 3, 5, 8, 12, 18, 24]:
    idx = np.argmin(np.abs(t_arr - t_show))
    ct  = C_total[idx]
    eff = "PEAK" if abs(t_show - t_arr[t_peak_idx]) < 0.5 else (
          "alerting" if ct > 2 else "mild" if ct > 0.5 else "baseline")
    print(f"  {t_show:>8.1f}  {C_bang[idx]:>8.3f}  {C_monster[idx]:>10.3f}  "
          f"{ct:>9.3f}  {eff}")

# CYP1A2 genotype effect
print("\n--- CYP1A2 genotype: fast vs slow metabolizers ---")
print("  CYP1A2*1F homozygous (fast): t_half ~ 3 hr  (~50% population)")
print("  Wild-type (normal):          t_half ~ 5 hr  (~35% population)")
print("  CYP1A2*1C slow:              t_half ~ 9 hr  (~15% population)")
print()
for geno, t_h in [("Fast (3hr)", 3.0), ("Normal (5hr)", 5.0), ("Slow (9hr)", 9.0)]:
    ke = np.log(2)/t_h
    C_max_rb = F_bio * 80 / Vd_L
    C_8 = C_max_rb * np.exp(-ke * (8 - t_max_hr))
    sleep_ok = "sleep affected" if C_8 > 1.0 else "OK to sleep at 10pm"
    print(f"  {geno:18s}: C at 8hr = {C_8:.3f} ug/mL  -> {sleep_ok}")
print("  Red Bull at noon:")
print("  Slow metabolizer at 10pm (8hr later) still has >1 ug/mL -> disrupts sleep onset")

# BLT: food energy thermodynamics
print("\n--- BLT sandwich: food energy (kJ, kcal, ATP yield) ---")
print("  Bacon Lettuce Tomato on whole wheat")
blt = {
    "Bacon (2 strips, 14g)":   (61,  5.0, 4.5, 0.1),   # kcal, prot, fat, carb (g)
    "Lettuce (15g)":            (3,   0.2, 0.0, 0.5),
    "Tomato (60g)":            (11,   0.6, 0.1, 2.4),
    "Whole wheat bread (2sl)": (138,  6.0, 2.0,26.0),
    "Mayo (1 tbsp, 14g)":       (94,  0.1,10.5, 0.1),
}
total_kcal = sum(v[0] for v in blt.values())
total_prot = sum(v[1] for v in blt.values())
total_fat  = sum(v[2] for v in blt.values())
total_carb = sum(v[3] for v in blt.values())

# ATP yield: ~4 kcal/g carb, ~4 kcal/g protein, ~9 kcal/g fat
# Actual ATP: ~38 ATP/glucose, ~129 ATP/palmitate
ATP_per_glucose  = 38
kcal_per_glucose = 686/1000  # thermodynamic combustion per g glucose * ~0.686
ATP_equiv        = total_kcal * 4.184e3 / 30.5e3 * 6.02e23  # kJ -> J, /30.5kJ/mol, * Avogadro

print(f"  {'Item':30s}  {'kcal':>6}  {'prot(g)':>8}  {'fat(g)':>7}  {'carb(g)':>8}")
for item, (kcal, p, f, c) in blt.items():
    print(f"  {item:30s}  {kcal:>6}  {p:>8.1f}  {f:>7.1f}  {c:>8.1f}")
print(f"  {'TOTAL':30s}  {total_kcal:>6}  {total_prot:>8.1f}  {total_fat:>7.1f}  {total_carb:>8.1f}")
print(f"\n  {total_kcal} kcal = {total_kcal*4.184:.0f} kJ")
print(f"  Equivalent ATP molecules: ~{ATP_equiv:.2e}")
print(f"  Caffeine from energy drink prolongs use of this ATP by ~{total_kcal/300:.0f}% longer subjectively")

print("\nDone.")
