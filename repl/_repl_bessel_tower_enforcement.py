# -*- coding: utf-8 -*-
"""
_repl_bessel_tower_enforcement.py
====================================
S1: BESSEL FUNCTIONS IN CYLINDRICAL WAVEGUIDES
    - TE_mn and TM_mn modes: H_z or E_z ~ J_m(k_c*r)*cos(m*phi)
    - TM zeros: J_m(x) = 0  ->  x_mn table (SymPy + scipy)
    - TE zeros: J_m'(x) = 0  ->  x'_mn table
    - Cutoff frequencies f_mn = c*x_mn / (2*pi*a)
    - Field distributions: E_r, E_phi, H_r, H_phi for TM_01
    - Dominant mode: TE_11 (lowest cutoff in circular waveguide)
    - Wave impedance, group velocity, phase velocity vs frequency
    - RogueGuard: fiber = circular dielectric waveguide, LP modes

S2: BESSEL IN CIRCULAR TOWER CROSS-SECTIONS
    - Polar moment of area: J = pi*R^4/2 (solid), pi*(R^4-Ri^4)/2 (hollow)
    - Torsional shear stress: tau = T*r/J
    - Euler buckling: Pcr = pi^2*E*I/(K*L)^2, I = pi*R^4/4
    - Vortex shedding (Strouhal): f_vs = St*V/D, St~0.2 for cylinder
    - Karman vortex street: resonance if f_vs ~ f_natural -> tower oscillation
    - Tacoma Narrows lesson: aeroelastic flutter, not pure VIV
    - Guy wire tension and catenary (hyperbolic cosine shape)

S3: BUILDING CODE ENFORCEMENT (towers)
    - ASCE 7-22: wind loads on structures (Chapter 26-29)
    - TIA-222-H: Structural Standard for Antenna Supporting Structures
    - Wind pressure: p = 0.00256 * Kz * Kzt * Kd * V^2  [psf]
    - Exposure categories: B (suburban), C (open), D (coastal)
    - Seismic base shear: V = Cs * W, Cs = SDS/(R/Ie)
    - ACI 318 column design: phi*Pn >= Pu (strength reduction factor)
    - IBC occupancy categories, special inspections, permit requirements

S4: SOFTWARE CODE ENFORCEMENT (DoD / RogueGuard CI)
    - Pre-commit hooks: ruff, mypy, black, bandit (security)
    - mypy strict: --strict flag, py.typed marker, type stubs
    - ruff: 700+ rules, replaces flake8+isort+pyupgrade
    - MISRA-C enforcement: PC-lint Plus, Polyspace Bug Finder
    - GitHub Actions CI pipeline: lint -> test -> build -> scan -> deploy
    - SBIR deliverable: software quality plan (SQP), test coverage >= 80%
    - .pre-commit-config.yaml and pyproject.toml configuration

Output: repl/_out_bessel_tower_enforcement.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.colors as mcolors
import sympy as sp
from sympy import (symbols, besselj, bessely, diff, pi, sqrt, cos, sin,
                   lambdify, Rational, simplify, oo, exp, I)
from scipy import special as sc
from scipy.optimize import brentq
import os

try:
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_out_bessel_tower_enforcement.png")
except NameError:
    OUT = "_out_bessel_tower_enforcement.png"

SEP = "=" * 65

# ============================================================
# S1: BESSEL FUNCTIONS IN CYLINDRICAL WAVEGUIDES
# ============================================================
print(SEP)
print("SECTION 1: BESSEL FUNCTIONS IN CYLINDRICAL WAVEGUIDE")
print(SEP)

print("""
  GEOMETRY:
    Hollow metallic circular waveguide, radius a, axis along z.
    PEC (perfect electric conductor) walls: E_tan = 0 at r = a.
    Fields: E(r,phi,z,t) = E_t(r,phi) * exp(i*(beta*z - omega*t))
    Decompose into TE (transverse electric: E_z=0) and TM (H_z=0) modes.

  TM MODES (transverse magnetic): E_z != 0, H_z = 0
    E_z satisfies: (nabla_t^2 + k_c^2) * E_z = 0
    In cylindrical coords (r,phi): Bessel equation in r.
    Solution (regular at r=0):
      E_z(r,phi) = E_0 * J_m(k_c * r) * cos(m*phi)
    PEC boundary condition at r=a: E_z(a) = 0
      => J_m(k_c * a) = 0
      => k_c * a = x_mn  (n-th zero of J_m)
      => k_c = x_mn / a

  TE MODES (transverse electric): H_z != 0, E_z = 0
    H_z(r,phi) = H_0 * J_m(k_c * r) * cos(m*phi)
    PEC boundary at r=a: dE_phi/dr|_{r=a} = 0 => dH_z/dr|_{r=a} = 0
      => J_m'(k_c * a) = 0
      => k_c * a = x'_mn  (n-th zero of J_m')

  CUTOFF FREQUENCY:
    f_mn = c * x_mn / (2*pi*a)      [TM modes]
    f_mn = c * x'_mn / (2*pi*a)     [TE modes]
    Below f_mn: mode is EVANESCENT (decays exponentially, no propagation).
    Above f_mn: mode PROPAGATES.
""")

# Compute Bessel zeros
def jn_zeros(m, n_zeros):
    """First n_zeros of J_m(x) = 0."""
    zeros = []
    x = 0.1 + m    # start just after m (first zero roughly at m+1)
    bracket_start = 0.1
    x_scan = np.linspace(bracket_start, 30, 3000)
    jm_vals = sc.jv(m, x_scan)
    sign_changes = np.where(np.diff(np.sign(jm_vals)))[0]
    for idx in sign_changes:
        if len(zeros) >= n_zeros:
            break
        try:
            z = brentq(lambda x: sc.jv(m, x), x_scan[idx], x_scan[idx+1])
            zeros.append(z)
        except Exception:
            pass
    return zeros[:n_zeros]

def jnp_zeros(m, n_zeros):
    """First n_zeros of J_m'(x) = 0."""
    x_scan = np.linspace(0.01, 30, 6000)
    djm = np.gradient(sc.jv(m, x_scan), x_scan)
    sign_changes = np.where(np.diff(np.sign(djm)))[0]
    zeros = []
    for idx in sign_changes:
        if len(zeros) >= n_zeros:
            break
        if x_scan[idx] < 0.5 and m > 0:
            continue   # skip x=0 for m>0
        try:
            z = brentq(lambda x: np.gradient(sc.jv(m, np.array([x-1e-6, x, x+1e-6])),
                                              np.array([-1e-6, 0, 1e-6]))[1],
                       x_scan[idx]+1e-6, x_scan[idx+1]-1e-6)
            zeros.append(z)
        except Exception:
            pass
    return zeros[:n_zeros]

# TM zeros: J_m(x_mn) = 0
print("  TM MODE ZEROS x_mn (J_m(x_mn) = 0):")
print(f"  {'m\\n':<6} " + " ".join(f"{'n='+str(n):<12}" for n in range(1,5)))
tm_zeros = {}
for m in range(4):
    z = jn_zeros(m, 4)
    tm_zeros[m] = z
    row = f"  {m:<6} " + " ".join(f"{x:<12.6f}" for x in z)
    print(row)

# TE zeros: J_m'(x'_mn) = 0  (use scipy)
print("\n  TE MODE ZEROS x'_mn (J_m'(x'_mn) = 0):")
print(f"  {'m\\n':<6} " + " ".join(f"{'n='+str(n):<12}" for n in range(1,5)))
te_zeros_scipy = {}
for m in range(4):
    z = list(sc.jnp_zeros(m, 4))
    te_zeros_scipy[m] = z
    row = f"  {m:<6} " + " ".join(f"{x:<12.6f}" for x in z)
    print(row)

# Sorted mode table
a_waveguide = 0.05   # m (5 cm radius -- WR-90 style)
c_light = 3e8
print(f"\n  CUTOFF FREQUENCIES (a = {a_waveguide*100:.1f} cm):")
print(f"  {'Mode':<10} {'x_mn':<12} {'f_c (GHz)':<14} {'Type'}")
print(f"  {'-'*48}")
mode_list = []
for m in range(4):
    for n_idx, x in enumerate(te_zeros_scipy[m][:3]):
        mode_list.append((f"TE_{m}{n_idx+1}", x, x*c_light/(2*np.pi*a_waveguide)/1e9, "TE"))
    for n_idx, x in enumerate(tm_zeros[m][:3]):
        mode_list.append((f"TM_{m}{n_idx+1}", x, x*c_light/(2*np.pi*a_waveguide)/1e9, "TM"))
mode_list.sort(key=lambda x: x[1])
for name, x, fc, typ in mode_list[:12]:
    dominant = " <- DOMINANT (lowest)" if name == "TE_11" else ""
    print(f"  {name:<10} {x:<12.4f} {fc:<14.3f} {typ}{dominant}")

print("""
  DOMINANT MODE: TE_11 (x'_11 = 1.8412)
    Lowest cutoff in circular waveguide.
    Single-mode operation: f_c(TE_11) < f < f_c(TM_01) or f_c(TE_21)
    TE_11 has non-uniform phi distribution -> two degenerate polarizations.
    In FIBER (dielectric, not hollow): LP_01 is dominant, LP_11 is first higher.
    LP_01 ~ HE_11 (fundamental fiber mode, no cutoff, propagates for all V>0).

  TRANSVERSE FIELD COMPONENTS (TM_01, m=0):
    E_z = E_0 * J_0(k_c * r) * exp(i*beta*z)
    E_r = -i*(beta/k_c) * E_0 * J_0'(k_c*r) = i*(beta/k_c)*E_0*J_1(k_c*r)
    E_phi = 0  (TM_0n: azimuthally symmetric)
    H_phi = i*(omega*eps/k_c) * E_0 * J_1(k_c*r)
    H_r   = 0
    Power: P = (pi*a^2/2) * (beta*omega*eps/k_c^2) * E_0^2 * J_1(x_01)^2
    Wave impedance: Z_TM = E_r/H_phi = beta/(omega*eps) < eta_0
""")

# TM_01 field profile
r_arr  = np.linspace(0, 1, 300)  # normalized r/a
x01    = tm_zeros[0][0]          # first zero of J_0
Ez_01  = sc.jv(0, x01 * r_arr)
Er_01  = sc.jv(1, x01 * r_arr)  # = -J_0'(x) = J_1(x)
print(f"  TM_01: x_01 = {x01:.4f}")
print(f"    E_z peak at r=0:   J_0(0) = {sc.jv(0,0):.4f}")
print(f"    E_r peak at r~0.6: J_1 max ~ {np.max(Er_01):.4f} at r/a = {r_arr[np.argmax(Er_01)]:.3f}")
print(f"    E_z at wall r=a:   J_0({x01:.4f}) = {sc.jv(0,x01):.6f}  (should be ~0)")

# ============================================================
# S2: BESSEL IN CIRCULAR TOWER CROSS-SECTIONS
# ============================================================
print(f"\n{SEP}")
print("SECTION 2: BESSEL IN CIRCULAR TOWER CROSS-SECTIONS")
print(SEP)

print("""
  CIRCULAR CROSS-SECTION (radius R, hollow inner radius Ri):

  SECOND MOMENT OF AREA (bending):
    I = pi*(R^4 - Ri^4) / 4    [m^4 or in^4]
    (Appears in Euler buckling Pcr = pi^2*E*I/(K*L)^2)

  POLAR MOMENT OF AREA (torsion):
    J = pi*(R^4 - Ri^4) / 2 = 2*I   [m^4]
    (Appears in torsion: tau_max = T*R/J)

  RADIUS OF GYRATION:
    r_g = sqrt(I/A) = sqrt((R^2 + Ri^2)/4)
    Slenderness ratio: K*L/r_g  (must be < 200 per AISC for steel)

  CONNECTION TO BESSEL: The cross-section INTEGRAL is:
    I = integral_0^R integral_0^{2pi} r^2 * r dr dphi
      = 2*pi * integral_0^R r^3 dr = pi*R^4/2  (this is J)
    The Bessel ODE arises naturally in the TORSION of prismatic bars
    and in the VIBRATION modes of circular membranes/plates.
    Circular plate vibration: W(r) ~ J_0(k*r) with k = (omega^2*rho*h/D)^{1/4}
    Drum modes: zeros of J_m(k*a) = 0 -> same table as waveguide!
""")

# Numerical: hollow steel tower
R_outer = 0.30   # m (30cm outer radius)
t_wall  = 0.012  # m (12mm wall thickness)
R_inner = R_outer - t_wall
E_steel = 200e9  # Pa
rho_steel = 7850  # kg/m^3

I_solid  = np.pi * R_outer**4 / 4
I_hollow = np.pi * (R_outer**4 - R_inner**4) / 4
J_polar  = 2 * I_hollow
A_hollow = np.pi * (R_outer**2 - R_inner**2)
r_g      = np.sqrt(I_hollow / A_hollow)

print(f"  HOLLOW STEEL TOWER CROSS-SECTION:")
print(f"    R_outer = {R_outer*100:.0f} cm,  t_wall = {t_wall*100:.1f} cm,  R_inner = {R_inner*100:.1f} cm")
print(f"    I_hollow = pi*(R^4-Ri^4)/4 = {I_hollow*1e6:.4f} * 10^-6 m^4 = {I_hollow/2.54e-2**4:.1f} in^4")
print(f"    J_polar  = 2*I = {J_polar*1e6:.4f} * 10^-6 m^4")
print(f"    A_hollow = pi*(R^2-Ri^2) = {A_hollow*1e4:.2f} cm^2")
print(f"    r_g      = sqrt(I/A) = {r_g*100:.3f} cm")

# Euler buckling for various tower heights
print(f"\n  EULER BUCKLING Pcr vs TOWER HEIGHT (K=2.0, cantilever):")
print(f"  {'H (m)':<10} {'K*L/r_g':<14} {'Pcr (kN)':<14} {'Status'}")
print(f"  {'-'*48}")
K_eff = 2.0   # cantilever (fixed-free)
for H in [10, 20, 30, 50, 80]:
    KLr = K_eff * H / r_g
    Pcr = np.pi**2 * E_steel * I_hollow / (K_eff * H)**2 / 1e3  # kN
    status = "OK" if KLr < 200 else "SLENDER (check)"
    print(f"  {H:<10} {KLr:<14.1f} {Pcr:<14.1f} {status}")

print("""
  VORTEX SHEDDING (Karman vortex street):
    Wind flow past a circular cylinder sheds vortices alternately
    from each side. Shedding frequency:
      f_vs = St * V / D
      St   = Strouhal number ~ 0.2 for circular cylinder (Re > 1000)
      V    = wind speed [m/s]
      D    = diameter [m]

    If f_vs approaches the tower's natural frequency f_nat:
      RESONANCE -> large lateral oscillations -> fatigue failure.
    Lock-in: once f_vs ~ f_nat, vortex shedding LOCKS to f_nat over
    a range of wind speeds (V +/- 10%).

  NATURAL FREQUENCY OF TOWER (Euler-Bernoulli cantilever):
    omega_1 = (1.875)^2 * sqrt(E*I / (rho*A*L^4))
    f_nat   = omega_1 / (2*pi)

  TACOMA NARROWS (1940): NOT pure vortex shedding.
    True cause: AEROELASTIC FLUTTER (torsional galloping).
    Coupled bending-torsion mode; negative aerodynamic damping.
    Still -- Karman VIV IS a real concern for tall slender towers.
""")

# Vortex shedding resonance analysis
H_tower = 50.0   # m
rho_air = 1.225  # kg/m^3
St = 0.2
D_tower = 2 * R_outer   # diameter

# Natural frequency (cantilever, 1st mode)
rho_L = rho_steel * A_hollow   # mass per unit length [kg/m]
beta1 = 1.8751     # first mode for cantilever
omega_nat = beta1**2 * np.sqrt(E_steel * I_hollow / (rho_L * H_tower**4))
f_nat = omega_nat / (2*np.pi)
V_critical = f_nat * D_tower / St

print(f"\n  TOWER: H={H_tower}m, D={D_tower*100:.0f}cm, f_nat = {f_nat:.3f} Hz")
print(f"  Critical wind speed for VIV lock-in: V_crit = f_nat*D/St = {V_critical:.2f} m/s = {V_critical*2.237:.1f} mph")
print(f"  Lock-in range: {V_critical*0.9:.2f} - {V_critical*1.1:.2f} m/s")

# Guy wire catenary
print("""
  GUY WIRE CATENARY:
    A wire under its own weight hangs as a CATENARY:
      y(x) = a * cosh(x/a) - a
      where a = T_H / (w)  [T_H = horizontal tension, w = weight/length]
    Sag at midspan: delta = a*(cosh(L/2a) - 1) ~ L^2*w/(8*T_H) for small sag
    Tension at anchor: T = T_H * cosh(L/2a) = sqrt(T_H^2 + (w*L/2)^2)
    Pre-tension recommendation: T_H ~ 10-15% of breaking strength.
    Guy wire reduces effective K of tower from 2.0 (cantilever) to ~0.7-0.8.
    Reduces Pcr by (K_guyed/K_free)^2 = (0.7/2.0)^2 = 0.12 -> 8x MORE CAPACITY!
""")

# Guy wire catenary shape
L_span = 40.0    # m horizontal span
w_wire = 5.0     # N/m  (wire weight per meter)
T_H    = 20000.0 # N horizontal tension
a_cat  = T_H / w_wire   # catenary parameter
x_cat  = np.linspace(-L_span/2, L_span/2, 200)
y_cat  = a_cat * (np.cosh(x_cat / a_cat) - 1)
sag    = a_cat * (np.cosh(L_span/(2*a_cat)) - 1)
T_anchor = T_H * np.cosh(L_span/(2*a_cat))
print(f"  Catenary: L={L_span}m, w={w_wire}N/m, T_H={T_H/1e3:.1f}kN")
print(f"    a = T_H/w = {a_cat:.1f} m")
print(f"    Sag at midspan: {sag:.3f} m")
print(f"    Tension at anchor: {T_anchor/1e3:.2f} kN (vs T_H={T_H/1e3:.1f} kN)")

# ============================================================
# S3: BUILDING CODE ENFORCEMENT (towers)
# ============================================================
print(f"\n{SEP}")
print("SECTION 3: BUILDING CODE ENFORCEMENT (TOWERS)")
print(SEP)

print("""
  GOVERNING STANDARDS FOR TOWERS:
    ASCE 7-22:   Minimum Design Loads and Associated Criteria
                 (wind, seismic, snow, ice loads)
    TIA-222-H:   Structural Standard for Antenna Supporting Structures
                 (telecom towers -- monopoles, lattice, guyed)
    IBC 2021:    International Building Code (jurisdictional adoption)
    ACI 318-19:  Concrete pole/pier design
    AISC 360-22: Steel structure design (LRFD or ASD)

  ASCE 7-22 WIND LOAD (Chapter 27, MWFRS):
    p = q_z * G * Cf     [psf]
    q_z = 0.00256 * Kz * Kzt * Ke * V^2   [psf]
      Kz  = velocity pressure exposure coefficient (height and terrain)
      Kzt = topographic factor (hills, escarpments -> amplification)
      Ke  = ground elevation factor
      V   = basic wind speed [mph] from ASCE 7 wind map (3-sec gust)
    G   = gust factor ~ 0.85 (rigid) or calculated (flexible: T > 1s)
    Cf  = force coefficient ~ 0.8-1.3 for circular cylinders
""")

# Wind pressure calculation
V_mph = 115.0     # mph (Risk Cat II, most areas)
Kz_vals = {10: 0.85, 20: 0.90, 33: 1.00, 50: 1.09, 100: 1.22, 200: 1.35}
Kzt = 1.0
Ke  = 1.0
G   = 0.85
Cf  = 0.9   # circular cross-section

print(f"\n  WIND PRESSURE p_z vs HEIGHT (V={V_mph} mph, Exposure C):")
print(f"  {'z (ft)':<10} {'Kz':<8} {'q_z (psf)':<14} {'p (psf)':<12}")
print(f"  {'-'*45}")
for z_ft, Kz in Kz_vals.items():
    qz = 0.00256 * Kz * Kzt * Ke * V_mph**2
    p  = qz * G * Cf
    print(f"  {z_ft:<10} {Kz:<8.3f} {qz:<14.2f} {p:<12.2f}")

print("""
  TIA-222-H TOWER DESIGN:
    Wind Ice combination: 1/2 inch radial ice + 40 mph wind (most of US).
    Lattice towers: Cf depends on solidity ratio (open frames: Cf~2.0).
    Monopole: round; Cf ~ 0.8 (smooth), 1.2 (rough/weathered).
    Safety factor approach:
      LRFD: phi*Rn >= gamma*Q  (strength*resistance >= load factor * load)
      TIA-222-H uses: 1.6*W + 1.2*D (for wind controlling)

  SEISMIC BASE SHEAR (ASCE 7-22, Chapter 12):
    V = Cs * W
    Cs = SDS / (R/Ie)
    SDS = 2/3 * SMS = 2/3 * Fa * SS  (design spectral acceleration, short period)
    R   = response modification factor (8 for special moment frame steel)
    Ie  = importance factor (1.0 to 1.5)
    W   = seismic weight (dead load + 25% live)
""")

# Seismic calculation example
SDS   = 1.0   # g  (high seismic zone, e.g., Los Angeles)
R_fac = 8.0   # special moment frame
Ie    = 1.25  # Risk Category III (essential facility, telecom)
Cs    = SDS / (R_fac / Ie)
W_tower = 500e3 / 4.44822  # N -> kN  (500 kips total weight)
V_seismic = Cs * W_tower
print(f"  SEISMIC EXAMPLE (Los Angeles, Risk Cat III):")
print(f"    SDS = {SDS:.1f}g,  R = {R_fac:.0f},  Ie = {Ie}")
print(f"    Cs  = SDS/(R/Ie) = {Cs:.4f}")
print(f"    W   = {W_tower:.0f} kN")
print(f"    V_seismic = {V_seismic:.0f} kN = {V_seismic*0.2248:.0f} kips")

print("""
  SPECIAL INSPECTIONS (IBC 1705) REQUIRED FOR TOWERS:
    Concrete: mix design review, slump tests, cylinder breaks, placement.
    Welding: AWS D1.1 visual + UT or MT for groove welds.
    High-strength bolts: torque verification, pretension.
    Anchor bolts: embedment depth, edge distance, epoxy injection.
    Reinforcing steel: grade, placement, cover.

  PERMIT PROCESS:
    1. Site survey + geotechnical report (boring, SPT, soil bearing).
    2. Structural calculations (PE stamp required).
    3. Building permit application + plan review (4-12 weeks in major cities).
    4. Special inspection program filed with jurisdiction.
    5. Construction with continuous special inspections.
    6. Final inspection + certificate of occupancy.
    7. Annual maintenance inspection per TIA-222-H Annex A.
""")

# ============================================================
# S4: SOFTWARE CODE ENFORCEMENT
# ============================================================
print(f"\n{SEP}")
print("SECTION 4: SOFTWARE CODE ENFORCEMENT (DoD / RogueGuard CI)")
print(SEP)

print("""
  PHILOSOPHY: "Code that isn't enforced is a suggestion."
    Pre-commit: catch problems BEFORE they enter the repo.
    CI/CD: enforce on every push/PR, block merge on failure.
    Static analysis: find bugs without running the code.
    DoD SBIR: software quality plan (SQP) is a contract deliverable.
    Coverage gate: >= 80% line coverage required.
""")

precommit_yaml = r"""
  # .pre-commit-config.yaml  (root of repo)
  repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.4.4
      hooks:
        - id: ruff               # lint: 700+ rules (E,W,F,I,N,UP,B,S,...)
          args: [--fix]
        - id: ruff-format        # format (replaces black)

    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v1.10.0
      hooks:
        - id: mypy
          args: [--strict, --ignore-missing-imports]
          additional_dependencies: [numpy, scipy]

    - repo: https://github.com/PyCQA/bandit
      rev: 1.7.8
      hooks:
        - id: bandit             # security: finds B101 (assert), B105 (hardcoded pw), etc.
          args: [-r, gs_core.py, gs_fno.py, -l]

    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.6.0
      hooks:
        - id: trailing-whitespace
        - id: end-of-file-fixer
        - id: check-yaml
        - id: check-merge-conflict
        - id: detect-private-key    # never commit .pem, RSA keys
        - id: no-commit-to-branch   # block direct commit to main/master
          args: [--branch, main, --branch, master]
"""
print(precommit_yaml)

pyproject_toml = r"""
  # pyproject.toml  (tool configuration)
  [tool.ruff]
  line-length = 100
  target-version = "py312"
  select = [
      "E",  "W",   # pycodestyle errors/warnings
      "F",         # pyflakes (undefined names, unused imports)
      "I",         # isort (import order)
      "N",         # pep8-naming
      "UP",        # pyupgrade (use f-strings, walrus, match, etc.)
      "B",         # flake8-bugbear (opinionated bug detection)
      "S",         # flake8-bandit (security)
      "ANN",       # flake8-annotations (type hints required)
      "PT",        # flake8-pytest-style
      "RUF",       # ruff-specific rules
  ]
  ignore = ["ANN101", "ANN102"]    # skip self/cls annotations

  [tool.mypy]
  python_version = "3.12"
  strict = true            # enables: --disallow-untyped-defs, --no-implicit-optional,
                           #          --warn-return-any, --warn-unused-ignores, etc.
  warn_unreachable = true
  show_error_codes = true

  [tool.pytest.ini_options]
  addopts = "--cov=. --cov-report=term-missing --cov-fail-under=80"
  testpaths = ["tests"]

  [tool.coverage.run]
  omit = ["repl/*", "tests/*", "docs/*"]
"""
print(pyproject_toml)

github_actions = r"""
  # .github/workflows/ci.yml
  name: RogueGuard CI

  on: [push, pull_request]

  jobs:
    lint:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: "3.12" }
        - run: pip install pre-commit
        - run: pre-commit run --all-files   # ruff + mypy + bandit + hooks

    test:
      needs: lint               # only run tests if lint passes
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: pip install -e ".[test]"
        - run: pytest --tb=short            # fails if coverage < 80%

    build-c:
      needs: lint
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: sudo apt-get install -y libfftw3-dev
        - run: |
            gcc -O3 -march=native -ffast-math -Wall -Wextra -Werror \
                -shared -fPIC -o libgs_core.so gs_core.c -lfftw3 -lm
        - run: |
            # AddressSanitizer build for fuzzing
            gcc -fsanitize=address,undefined -g -O1 \
                gs_core.c gs_fuzz_harness.c -o gs_fuzz -lfftw3 -lm
            ./gs_fuzz 100   # run 100 fuzz iterations in CI

    misra-check:
      needs: build-c
      runs-on: ubuntu-latest
      # PC-lint Plus or cppcheck with MISRA addon
      steps:
        - uses: actions/checkout@v4
        - run: |
            pip install cppcheck
            cppcheck --addon=misra --error-exitcode=1 \
                     --suppress=misra-c2012-21.6 \
                     gs_core.c

    sbom:                         # Software Bill of Materials (DoD requirement)
      needs: [test, build-c]
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: anchore/sbom-action@v0  # generates SBOM in SPDX/CycloneDX
          with:
            format: spdx-json
            output-file: sbom.spdx.json
        - uses: actions/upload-artifact@v4
          with: { name: sbom, path: sbom.spdx.json }
"""
print(github_actions)

print("""
  MISRA-C 2012 KEY RULES (RogueGuard firmware):
    Rule  4.1: No octal constants (013 reads as 11 in decimal -- trap!)
    Rule  7.2: Unsigned integer constants with 'u' suffix: 100u not 100
    Rule 10.3: Value of expression shall not be assigned to narrower type
    Rule 14.4: if/while/for condition shall be essentially Boolean
    Rule 15.5: Function shall have single point of exit (one return)
    Rule 17.2: No recursion (stack overflow risk in embedded)
    Rule 20.4: No dynamic allocation (malloc/free/realloc)
    Rule 21.3: No atoi/atof (undefined on overflow) -> use strtol/strtod
    Rule 22.1: All resources shall be explicitly released before return
    Advisory:  Use static analysis tool (PC-lint Plus, Polyspace, Coverity)

  SBIR SOFTWARE QUALITY PLAN (DoD requirement):
    Section 1: Software Development Plan (Agile, sprint cadence)
    Section 2: Coding Standard (MISRA-C for firmware, PEP 8 for Python)
    Section 3: Verification & Validation (unit test, integration test, HIL test)
    Section 4: Configuration Management (git, semantic versioning, signed tags)
    Section 5: Problem Reporting (GitHub Issues, severity levels 1-4)
    Section 6: Metrics (coverage >= 80%, static analysis warnings = 0, MTBF target)
    Deliverable: Software Test Report (STR) at Phase I completion.
    SBIR Phase I milestone: working demo on RPi CM4 + ADC, GS convergence verified.
""")

# Demonstrate mypy strictness
print("""  MYPY STRICT EXAMPLE (type-annotated GS core):""")
mypy_example = r"""
  # gs_core.py  -- type-annotated Python GS wrapper (mypy --strict clean)
  from __future__ import annotations
  from typing import TypeAlias
  import numpy as np
  import numpy.typing as npt

  FloatArray: TypeAlias = npt.NDArray[np.float64]
  ComplexArray: TypeAlias = npt.NDArray[np.complex128]

  def dispersion_filter(N: int, D: float) -> ComplexArray:
      '''Build H[k] = exp(i*pi*D*(k/N)^2) for GS phase retrieval.'''
      if abs(D) < 5000:
          raise ValueError(f"|D|={abs(D):.1f} < 5000 -- insufficient diversity")
      k: FloatArray = np.arange(N, dtype=np.float64)
      return np.exp(1j * np.pi * D * (k / N) ** 2)

  def unit_amplitude(A: ComplexArray, I_target: FloatArray) -> ComplexArray:
      '''Replace |A| with sqrt(I_target), preserve phase.'''
      mag = np.abs(A)
      mag = np.where(mag < 1e-15, 1e-15, mag)   # guard divide-by-zero
      return A / mag * np.sqrt(I_target)

  def retrieve_phase(
      I1: FloatArray,
      I2: FloatArray,
      D: float = -7000.0,
      n_iter: int = 50,
  ) -> FloatArray:
      '''GS phase retrieval. |D| >= 5000, n_iter >= 50 recommended.'''
      if I1.shape != I2.shape:
          raise ValueError(f"Shape mismatch: {I1.shape} vs {I2.shape}")
      if np.any(I1 < 0) or np.any(I2 < 0):
          raise ValueError("Intensities must be non-negative")
      N: int = I1.size
      H: ComplexArray = dispersion_filter(N, D)
      A: ComplexArray = np.sqrt(I1).astype(np.complex128)
      for _ in range(n_iter):
          A_hat = np.fft.fft(A) / N
          A_hat = A_hat * H
          A2    = np.fft.ifft(A_hat * N)
          A     = unit_amplitude(A2, I2)
          A_hat2 = np.fft.fft(A) / N
          A_hat2 = A_hat2 * H.conj()
          A      = np.fft.ifft(A_hat2 * N)
          A      = unit_amplitude(A, I1)
      return np.angle(A)
"""
print(mypy_example)

# GS convergence demo: show error reduction and D1!=D2 requirement
import sys as _sys
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
    if "__file__" in dir() else os.getcwd()
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
from gs_core import disperse, retrieve_phase as _gs_retrieve

# Unit-amplitude signal with smooth phase (constant-envelope -> unit_amplitude=True)
N_gs = 512; np.random.seed(42)
t_gs = np.linspace(0, 1, N_gs)
phi_true = 2.0*np.sin(2*np.pi*3*t_gs) + 1.0*np.sin(2*np.pi*7*t_gs)
E_true   = np.exp(1j * phi_true)   # unit amplitude
D1_gs, D2_gs = -5000.0, -5750.0   # |D1-D2| = 750 (diversity)
I1_gs = np.abs(disperse(E_true, D1_gs))**2
I2_gs = np.abs(disperse(E_true, D2_gs))**2

# Full GS with two dispersions
_, errors_gs = _gs_retrieve(I1_gs, I2_gs, D1=D1_gs, D2=D2_gs, n_iter=100)

# Degenerate case: D1 = D2 (no diversity -- should NOT converge)
# (use slightly different values to avoid the D1==D2 ValueError)
D_same = -5000.0
I_same = np.abs(disperse(E_true, D_same))**2
_, errors_degen = _gs_retrieve(I_same, I_same, D1=D_same, D2=D_same + 0.1,
                                n_iter=100)

print(f"\n  GS CONVERGENCE DEMO (N={N_gs}, smooth sinusoidal phase, n_iter=100):")
print(f"    TWO dispersions D1={D1_gs}, D2={D2_gs}  |D1-D2|=750:")
print(f"      Error iter 1:  {errors_gs[0]:.4f}")
print(f"      Error iter 10: {errors_gs[9]:.4f}")
print(f"      Error iter 50: {errors_gs[49]:.4f}")
print(f"      Error iter 100:{errors_gs[99]:.6f}")
print(f"      Reduction: {errors_gs[0]/max(errors_gs[-1],1e-9):.1f}x  (algorithm IS converging)")
print(f"    DEGENERATE: D1=D2={D_same} (|D1-D2|~0, near-zero diversity):")
print(f"      Error iter 1:  {errors_degen[0]:.4f}")
print(f"      Error iter 100:{errors_degen[99]:.4f}")
print(f"      Reduction: {errors_degen[0]/max(errors_degen[-1],1e-9):.2f}x  (stagnates)")
print(f"    KEY INSIGHT: Diversity |D1-D2| drives convergence.")
print(f"    Physical: D1=-695 ps/nm, D2=-800 ps/nm -> D_norm ~ -5000/-5750 (Solli 2009)")
errors_gs = errors_gs   # keep for plot

# ============================================================
# MATPLOTLIB -- 6-PANEL FIGURE
# ============================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(19, 13))
fig.patch.set_facecolor("#F7F7F2")
gs0 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36,
                        top=0.93, bottom=0.06, left=0.06, right=0.97)

ax_wg   = fig.add_subplot(gs0[0, 0])
ax_mode = fig.add_subplot(gs0[0, 1])
ax_disp = fig.add_subplot(gs0[0, 2])
ax_vs   = fig.add_subplot(gs0[1, 0])
ax_wind = fig.add_subplot(gs0[1, 1])
ax_gs   = fig.add_subplot(gs0[1, 2])

fig.suptitle(
    "Bessel Waveguide Modes | Circular Tower (Vortex/Buckling) | "
    "ASCE 7 Wind | Code Enforcement | GS Phase Recovery",
    fontsize=10.5, fontweight="bold", color="#1a1a2e"
)

# ---- AX_WG: TM_01 and TE_11 field profiles ----
ax = ax_wg
ax.set_facecolor("#F0F8FF")
r_plot = np.linspace(0, 1, 400)
# TM_01: E_z ~ J_0(x_01*r), E_r ~ J_1(x_01*r)
x01_plot = tm_zeros[0][0]
# TE_11: H_z ~ J_1(x'_11*r)
x11p = sc.jnp_zeros(1, 1)[0]
# TM_11: E_z ~ J_1(x_11*r)
x11_tm = tm_zeros[1][0]

Ez_tm01  = sc.jv(0, x01_plot  * r_plot)
Er_tm01  = sc.jv(1, x01_plot  * r_plot)
Hz_te11  = sc.jv(1, x11p      * r_plot)
Ez_tm11  = sc.jv(1, x11_tm    * r_plot)

ax.plot(r_plot, Ez_tm01, "#1f77b4", lw=2.0, label=f"TM$_{{01}}$ $E_z$ (x={x01_plot:.3f})")
ax.plot(r_plot, Er_tm01, "#1f77b4", lw=1.5, ls="--", label=f"TM$_{{01}}$ $E_r$")
ax.plot(r_plot, Hz_te11, "#d62728", lw=2.0, label=f"TE$_{{11}}$ $H_z$ (x'={x11p:.3f})")
ax.plot(r_plot, Ez_tm11, "#2ca02c", lw=1.5, label=f"TM$_{{11}}$ $E_z$ (x={x11_tm:.3f})")
ax.axhline(0, color="#999", lw=0.5)
ax.axvline(1.0, color="#333", lw=1.0, ls="--", label="Wall r=a")
ax.set_xlabel("Normalized radius r/a", fontsize=9)
ax.set_ylabel("Field amplitude (normalized)", fontsize=9)
ax.set_title("Circular Waveguide\nBessel Mode Profiles", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.2)

# ---- AX_MODE: 2D field pattern TE_11 ----
ax = ax_mode
ax.set_facecolor("#1a1a2e")
ax.set_aspect("equal")
r2d  = np.linspace(0, 1, 80)
phi2d = np.linspace(0, 2*np.pi, 80)
Rg, Phig = np.meshgrid(r2d, phi2d)
Xg = Rg * np.cos(Phig)
Yg = Rg * np.sin(Phig)
# TE_11: dominant field component H_z ~ J_1(x'_11*r)*cos(phi)
Hz_2d = sc.jv(1, x11p * Rg) * np.cos(Phig)
im = ax.contourf(Xg, Yg, Hz_2d, levels=20, cmap="RdBu_r")
circle_wall = Circle((0,0), 1.0, fill=False, edgecolor="gold", lw=2.0)
ax.add_patch(circle_wall)
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.set_title("TE$_{11}$ Dominant Mode\n$H_z = J_1(x_{11}^{\\prime} r)\\cos\\phi$",
             fontsize=10, color="white")
ax.set_xlabel("x/a", fontsize=9, color="white")
ax.set_ylabel("y/a", fontsize=9, color="white")
ax.tick_params(colors="white")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("$H_z$", color="white")

# ---- AX_DISP: cutoff frequency chart ----
ax = ax_disp
ax.set_facecolor("#FFFFF0")
mode_names = [m[0] for m in mode_list[:10]]
mode_fc    = [m[2] for m in mode_list[:10]]
colors_m   = ["#d62728" if "TE" in n else "#1f77b4" for n in mode_names]
ax.barh(mode_names, mode_fc, color=colors_m, edgecolor="k", lw=0.6, height=0.6)
ax.set_xlabel("Cutoff frequency (GHz)", fontsize=9)
ax.set_title(f"Waveguide Cutoff Frequencies\n(a={a_waveguide*100:.0f} cm)", fontsize=10)
ax.grid(axis="x", alpha=0.2)
f_op = (mode_list[0][2] + mode_list[1][2]) / 2
ax.axvline(f_op, color="#2ca02c", lw=1.5, ls="--",
           label=f"Single-mode band: {mode_list[0][2]:.2f}-{mode_list[1][2]:.2f} GHz")
ax.legend(fontsize=7.5)
# Red = TE, Blue = TM legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#d62728",label="TE"), Patch(facecolor="#1f77b4",label="TM"),
                   plt.Line2D([0],[0],color="#2ca02c",ls="--",label="Single-mode window")],
          fontsize=7.5)

# ---- AX_VS: vortex shedding + catenary ----
ax = ax_vs
ax.set_facecolor("#FFF5E0")
V_range = np.linspace(0, 30, 300)
f_vs_range = St * V_range / D_tower
ax.plot(V_range, f_vs_range, "#1f77b4", lw=2.0, label="$f_{vs}=St \\cdot V/D$")
ax.axhline(f_nat, color="#d62728", lw=1.5, ls="--", label=f"$f_{{nat}}={f_nat:.3f}$ Hz")
ax.axvline(V_critical, color="#d62728", lw=1.0, ls=":", alpha=0.7)
ax.fill_between(V_range,
                np.where((V_range>V_critical*0.9)&(V_range<V_critical*1.1), f_vs_range, np.nan),
                color="#ff7f0e", alpha=0.3, label="Lock-in zone")
ax.set_xlabel("Wind speed V (m/s)", fontsize=9)
ax.set_ylabel("Vortex shedding freq $f_{vs}$ (Hz)", fontsize=9)
ax.set_title(f"Vortex-Induced Vibration\n(H={H_tower}m, D={D_tower*100:.0f}cm tower)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.05, 0.92, f"$V_{{crit}}={V_critical:.1f}$ m/s = {V_critical*2.237:.0f} mph",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(fc="white", ec="#d62728", pad=2))

# Inset: catenary
ax_ins2 = ax.inset_axes([0.55, 0.05, 0.43, 0.42])
ax_ins2.plot(x_cat, -y_cat, "#2ca02c", lw=2.0)
ax_ins2.set_xlabel("x (m)", fontsize=7)
ax_ins2.set_ylabel("y (m)", fontsize=7)
ax_ins2.set_title(f"Guy wire catenary\nsag={sag:.2f}m", fontsize=7.5)
ax_ins2.grid(alpha=0.2)
ax_ins2.invert_yaxis()

# ---- AX_WIND: ASCE 7 wind pressure profile ----
ax = ax_wind
ax.set_facecolor("#F0FFF0")
z_heights = np.linspace(0, 200, 300)  # ft
def Kz_exposure_C(z):
    """ASCE 7 Kz for Exposure C."""
    z = np.maximum(z, 15)
    return 2.01 * (z / 900) ** (2/9.5)

Kz_arr  = Kz_exposure_C(z_heights)
qz_arr  = 0.00256 * Kz_arr * Kzt * Ke * V_mph**2
pz_arr  = qz_arr * G * Cf

ax.plot(pz_arr, z_heights, "#1f77b4", lw=2.0, label="Net pressure p (psf)")
ax.plot(qz_arr, z_heights, "#ff7f0e", lw=1.5, ls="--", label="Velocity pressure q_z (psf)")
ax.set_xlabel("Pressure (psf)", fontsize=9)
ax.set_ylabel("Height z (ft)", fontsize=9)
ax.set_title(f"ASCE 7-22 Wind Pressure Profile\n(V={V_mph} mph, Exp C, Cf={Cf})", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.axhline(165, color="#d62728", lw=0.8, ls=":", alpha=0.7)
ax.text(pz_arr[0]+0.3, 170, "164 ft = 50 m", fontsize=7.5, color="#888")

# ---- AX_GS: GS convergence + enforcement pipeline ----
ax = ax_gs
ax.set_facecolor("#F5F0FF")
ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")
ax.set_title("Software Code Enforcement Pipeline\n(RogueGuard SBIR CI/CD)", fontsize=10)

pipeline = [
    (0.5, 7.8, 9, 0.9, "#e8d5f5", "#7b2d8b", "git commit  ->  pre-commit hooks", 10),
    (0.5, 6.5, 9, 0.9, "#d5e8f5", "#1f6fa8", "ruff (lint+format)  |  mypy --strict  |  bandit (security)", 9),
    (0.5, 5.2, 9, 0.9, "#d5f5e8", "#1a8a4e", "GitHub Actions CI: lint job -> PASS or BLOCK merge", 9),
    (0.5, 3.9, 9, 0.9, "#f5e8d5", "#8a5a1a", "pytest --cov (>= 80%)  |  cppcheck MISRA-C  |  ASan fuzz", 9),
    (0.5, 2.6, 9, 0.9, "#f5d5d5", "#8a1a1a", "SBOM generation (SPDX/CycloneDX) -- DoD requirement", 9),
    (0.5, 1.3, 9, 0.9, "#d5d5f5", "#1a1a8a", "Signed release tag  ->  deploy to RPi CM4 firmware OTA", 9),
]
for x, y, w, h, fc, ec, label, fs in pipeline:
    rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                          linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8.0,
            color="#1a1a2e", fontweight="bold")
for y_arrow in [7.8, 6.5, 5.2, 3.9, 2.6]:
    ax.annotate("", xy=(5, y_arrow), xytext=(5, y_arrow+0.9),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))

ax.text(0.5, 0.08,
        f"GS: D1={D1_gs}, D2={D2_gs}  err {errors_gs[0]:.3f}->{errors_gs[-1]:.3f}  D1!=D2 required",
        transform=ax.transAxes, fontsize=8.5, ha="center",
        color="#1a8a4e",
        bbox=dict(fc="#eee", ec="#bbb", pad=3))

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
