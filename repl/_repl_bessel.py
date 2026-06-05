"""
repl/_repl_bessel.py
Bessel functions in cylindrical coordinates.
3-year specialization track: fiber optics -> pressure vessels -> ballistics -> aerospace.
Game anchors: LoL ability splash, Civ7 city radius, spin-stabilized projectile.
"""
import math
import numpy as np
import sympy as sp
from scipy.special import jn, yn, jn_zeros, kn, iv
from scipy.linalg import eigh
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 62)
print("BESSEL FUNCTIONS + CYLINDRICAL COORDINATES")
print("=" * 62)
print()

# ============================================================
# 0. WHY BESSEL FUNCTIONS APPEAR
# ============================================================
print("""=== 0. WHY BESSEL: ANY PDE WITH CYLINDRICAL SYMMETRY ===

  Laplacian in cylindrical coords (r, phi, z):
    d^2f/dr^2 + (1/r)*df/dr + (1/r^2)*d^2f/dphi^2 + d^2f/dz^2 = 0

  Separate variables: f = R(r) * Phi(phi) * Z(z)
  Phi(phi) = exp(i*n*phi)  (periodic, n = 0,1,2,...)
  Z(z)     = exp(i*kz*z)

  R(r) satisfies BESSEL'S EQUATION:
    r^2 R'' + r R' + (k^2 r^2 - n^2) R = 0

  Solutions:
    J_n(kr)   Bessel of 1st kind  -- finite at r=0  (inside)
    Y_n(kr)   Bessel of 2nd kind  -- diverges at r=0 (outside shells)
    I_n(kr)   Modified 1st kind   -- exponentially growing
    K_n(kr)   Modified 2nd kind   -- exponentially decaying (evanescent)

  APPEARS IN:
    Fiber optics:         LP modes J_n inside, K_n outside
    Pressure vessels:     stress in thick-walled cylinder
    Heat equation:        temperature in cylindrical fuel rod
    EM waveguide:         TE/TM modes in circular pipe
    Acoustics:            drum head vibration, speaker cone
    Ballistics:           spin-stabilized projectile yaw modes
    Quantum:              hydrogen atom radial wavefunction
    LoL/game engine:      circular AOE splash damage falloff
""")

# ============================================================
# 1. BESSEL FUNCTION PROPERTIES
# ============================================================
print("=== 1. Key properties ===")
r_arr = np.linspace(0, 20, 1000)

print("  Zeros of J_n(x) -- critical for mode cutoff:")
for n in range(4):
    zeros = jn_zeros(n, 5)
    zstr = "  ".join(f"{z:.4f}" for z in zeros)
    print(f"    J_{n} zeros: {zstr}")
print()

# recurrence relation
print("  Recurrence: J_{n-1}(x) + J_{n+1}(x) = (2n/x) * J_n(x)")
x_test = 3.5
lhs = jn(0, x_test) + jn(2, x_test)
rhs = (2*1/x_test) * jn(1, x_test)
print(f"    n=1, x={x_test}: LHS={lhs:.6f}  RHS={rhs:.6f}  diff={abs(lhs-rhs):.2e}")

# Wronskian
W = jn(0, x_test) * (jn(1, x_test) * (-1)) - jn(1, x_test) * jn(0, x_test)
# actual Wronskian: J_n Y_n' - J_n' Y_n = 2/(pi*x)
from scipy.special import yv
Wr = jn(0, x_test) * (-yv(1, x_test)) - (-jn(1, x_test)) * yv(0, x_test)
theory = 2 / (np.pi * x_test)
print(f"  Wronskian W[J_0, Y_0](x={x_test}) = {Wr:.6f}  theory 2/(pi*x) = {theory:.6f}")
print()

# ============================================================
# 2. FIBER OPTIC LP MODES (the Jalali application)
# ============================================================
print("=== 2. Fiber LP modes -- Jalali lab core ===")
print("""
  Step-index fiber: n_core > n_clad
  Inside (r < a):   E ~ J_n(k_r * r) * exp(i*n*phi)
  Outside (r > a):  E ~ K_n(gamma * r) * exp(i*n*phi)

  Boundary condition at r=a:
    J_n(k_r * a) / (k_r * J_{n+1}(k_r * a)) = K_n(gamma*a) / (gamma * K_{n+1}(gamma*a))

  V-number (normalized frequency):
    V = (2*pi*a/lambda) * sqrt(n_core^2 - n_clad^2)

  Single-mode condition: V < 2.405  (first zero of J_0)
  LP01: fundamental,  LP11: first higher order (cutoff at V=2.405)
""")

# SMF-28 parameters
lam_um = 1.55       # wavelength um
a_um   = 4.1        # core radius um (SMF-28)
n_core = 1.4504
n_clad = 1.4447
NA     = math.sqrt(n_core**2 - n_clad**2)
V      = (2*math.pi*a_um/lam_um) * NA

print(f"  SMF-28 at 1550nm:")
print(f"    Core radius a   = {a_um} um")
print(f"    NA              = {NA:.4f}")
print(f"    V-number        = {V:.4f}")
print(f"    Single-mode?    {'YES (V < 2.405)' if V < 2.405 else 'NO -- multimode'}")
print()

# mode field diameter via Gaussian approx
w0 = a_um * (0.65 + 1.619/V**1.5 + 2.879/V**6)
print(f"    MFD (Marcuse):  {2*w0:.2f} um  (SMF-28 spec: ~10.4 um)")
print()

# V-number sweep: which modes cut on
print("  LP mode cutoff V-numbers:")
lp_modes = [
    ("LP01", 0,      0.000),
    ("LP11", 1,      2.405),
    ("LP21", 2,      3.832),
    ("LP02", 0,      3.832),
    ("LP31", 3,      5.136),
    ("LP12", 1,      5.520),
]
for name, n, v_cut in lp_modes:
    status = "ON " if V > v_cut else "off"
    print(f"    {name}  (n={n})  V_cutoff={v_cut:.3f}  [{status}] at V={V:.3f}")
print()

# ============================================================
# 3. PRESSURE VESSEL: Lame equations in cylindrical coords
# ============================================================
print("=== 3. Thick-walled pressure vessel (Lame) ===")
print("""
  Problem: cylinder, inner radius a, outer radius b
           internal pressure p_i, external pressure p_o

  Stress state (plane strain):
    sigma_r(r)   = A - B/r^2    (radial stress)
    sigma_phi(r) = A + B/r^2    (hoop/circumferential stress)

  Boundary conditions:
    sigma_r(a) = -p_i
    sigma_r(b) = -p_o

  Solution (p_o = 0 for simplicity):
    A = p_i * a^2 / (b^2 - a^2)
    B = p_i * a^2 * b^2 / (b^2 - a^2)

  Max stress: sigma_phi(a) = p_i * (a^2 + b^2) / (b^2 - a^2)
  Thin wall limit (b-a << a): sigma_phi ~ p_i * a / t  (hoop stress)
""")

# example: gun barrel, hydraulic cylinder, submarine hull
cases = [
    ("Gun barrel",      0.020,  0.060,  400e6,  "MPa (4000 bar smokeless powder)"),
    ("Hydraulic cyl",   0.025,  0.035,  70e6,   "MPa (700 bar hydraulic)"),
    ("Deep sub hull",   1.000,  1.050,  60e6,   "MPa (6000m ocean)"),
    ("Rocket nozzle",   0.050,  0.080,  20e6,   "MPa (200 bar combustion)"),
]
print(f"  {'Case':18s}  {'a(m)':6s}  {'b(m)':6s}  {'p_i':10s}  {'sigma_max(MPa)':14s}  {'factor'}")
for name, a, b, p_i, unit in cases:
    A_lame = p_i * a**2 / (b**2 - a**2)
    B_lame = p_i * a**2 * b**2 / (b**2 - a**2)
    sigma_max = A_lame + B_lame / a**2  # hoop at inner radius
    steel_ult = 800e6  # MPa typical structural steel
    factor = steel_ult / sigma_max
    print(f"  {name:18s}  {a:6.3f}  {b:6.3f}  {p_i/1e6:10.1f}  {sigma_max/1e6:14.1f}  {factor:.2f}x margin")
print()

# ============================================================
# 4. HEAT EQUATION IN CYLINDER (nuclear fuel rod)
# ============================================================
print("=== 4. Heat equation in cylinder: nuclear fuel rod ===")
print("""
  Steady-state heat equation (cylindrical, no phi/z dependence):
    (1/r) d/dr (r * dT/dr) + Q/k = 0

  With uniform heat generation Q (W/m^3):
    T(r) = T_surface + (Q/4k) * (a^2 - r^2)

  T_max at r=0 (center):
    T_center = T_surface + Q*a^2 / (4k)

  This is a PARABOLIC profile -- NOT Bessel
  Bessel enters for TIME-DEPENDENT cooling:
    dT/dt = alpha * (1/r) d/dr(r dT/dr)
    Solution: T(r,t) = sum_n C_n * J_0(mu_n r/a) * exp(-alpha*mu_n^2*t/a^2)
    where mu_n are zeros of J_0
""")

# fuel rod parameters
Q_heat   = 500e6     # W/m^3 (UO2 nuclear fuel typical)
k_UO2    = 3.0       # W/m/K
a_fuel   = 0.004     # m (4mm radius pellet)
T_surf   = 400.0     # C (cladding-coolant interface)

T_center = T_surf + Q_heat * a_fuel**2 / (4 * k_UO2)
print(f"  Nuclear fuel rod:")
print(f"    Q = {Q_heat/1e6:.0f} MW/m^3   a = {a_fuel*1000:.1f} mm   k = {k_UO2} W/mK")
print(f"    T_surface = {T_surf:.0f} C")
print(f"    T_center  = {T_center:.0f} C  (UO2 melts at ~2865 C -> {'OK' if T_center < 2865 else 'MELT'})")
print()

# transient: first 3 Bessel modes, time to cool
print("  Transient cooling time constants (Bessel modes):")
alpha_UO2 = k_UO2 / (10970 * 250)   # k/(rho*Cp): rho=10970 kg/m^3, Cp=250 J/kgK
zeros_J0  = jn_zeros(0, 3)
for i, mu in enumerate(zeros_J0):
    tau = a_fuel**2 / (alpha_UO2 * mu**2)
    print(f"    Mode {i+1}: mu_{i+1} = {mu:.4f}  tau = {tau:.2f} s")
print()

# ============================================================
# 5. BALLISTICS: spin-stabilized projectile (gyroscopic)
# ============================================================
print("=== 5. Spin-stabilized projectile (Civ 7 / ballistics) ===")
print("""
  Rifled barrel imparts spin omega_s to projectile.
  Gyroscopic stability requires spin rate high enough
  to resist overturning moment from aerodynamic forces.

  Gyroscopic stability factor:
    S_g = I_axial * omega_s^2 / (2 * rho_air * V^2 * Cl_alpha * d^2 * I_transverse)

  S_g > 1  -> gyroscopically stable
  S_g >> 1 -> over-stabilized (won't follow trajectory)
  Optimal:   1.3 < S_g < 2.0

  Cylindrical symmetry -> moments of inertia:
    I_axial      = (1/2) m r^2        (solid cylinder)
    I_transverse = m(3r^2 + L^2)/12  (solid cylinder, L=length)

  Bessel functions enter in: barrel resonance modes, stability eigenvalues
""")

# projectile: 5.56mm NATO round
m_proj   = 0.004     # kg
r_proj   = 0.00278   # m (5.56mm caliber)
L_proj   = 0.023     # m (23mm long)
V_muz    = 960.0     # m/s muzzle velocity
twist    = 0.178     # m per turn (1:7 inch = 177.8mm)
omega_s  = 2*np.pi * V_muz / twist  # rad/s

I_ax  = 0.5 * m_proj * r_proj**2
I_tr  = m_proj * (3*r_proj**2 + L_proj**2) / 12

# simplified S_g (aerodynamic moment approximation)
rho_air = 1.225
Cl_a    = 2.0    # typical lift curve slope
d       = 2*r_proj
S_g = (I_ax * omega_s**2) / (2 * rho_air * V_muz**2 * Cl_a * d**2 * I_tr)

print(f"  5.56mm NATO at muzzle:")
print(f"    Muzzle velocity:  {V_muz:.0f} m/s")
print(f"    Spin rate:        {omega_s/1000:.1f} krad/s  ({omega_s/(2*np.pi):.0f} rev/s)")
print(f"    I_axial:          {I_ax:.3e} kg*m^2")
print(f"    I_transverse:     {I_tr:.3e} kg*m^2")
print(f"    S_g (stability):  {S_g:.2f}  ({'STABLE' if S_g > 1 else 'UNSTABLE'})")
print()

# ============================================================
# 6. LOL GAME: circular AOE -- Bessel zeros as damage rings
# ============================================================
print("=== 6. LoL/Civ7: circular AOE as Bessel physics ===")
print("""
  League of Legends: Orianna ball, Ziggs bombs, Morgana Q
  All have circular damage regions.

  If damage falls off as a WAVE pattern (not just cutoff):
    I(r) = I_0 * J_0(k * r)^2    (standing wave pattern)

  This appears in:
    - Circular drum head (Bessel modes = overtones)
    - Laser beam cross-section (Laguerre-Gaussian modes)
    - Diffraction through circular aperture (Airy disk)
    - WiFi/antenna radiation pattern near circular array

  Zeros of J_0 = rings of zero intensity:
    r_1 = 2.405/k,  r_2 = 5.520/k,  r_3 = 8.654/k

  For a spell with 400 unit radius, k = 2.405/400:
""")

R_spell = 400.0   # game units
k_spell = 2.405 / R_spell
r_pts   = np.linspace(0, R_spell, 200)
I_splash = jn(0, k_spell * r_pts)**2

zeros_400 = jn_zeros(0, 3) / k_spell
print(f"  Orianna Q: R={R_spell:.0f} units, k={k_spell:.4f}")
for i, rz in enumerate(zeros_400):
    print(f"    Zero {i+1}: r = {rz:.1f} units  (ring of zero intensity)")
print(f"  Peak I at center: J_0(0)^2 = {jn(0,0)**2:.1f}")
print(f"  I at edge r={R_spell}: J_0({k_spell*R_spell:.3f})^2 = {jn(0,k_spell*R_spell)**2:.4f}")
print()

# Civ7: city radius growth
print("  Civ7: city territory expansion")
print("  Each ring adds tiles ~ 2*pi*r -> area grows as pi*r^2")
print("  Culture spread modeled as diffusion in 2D disk:")
print("  C(r,t) = sum_n A_n * J_0(mu_n r/R) * exp(-D*mu_n^2*t/R^2)")
R_civ = 6.0  # tiles
D_civ = 0.1
zeros_civ = jn_zeros(0, 4)
print(f"  {'Mode':6s}  {'mu_n':8s}  {'tau (turns)':12s}  {'relative amplitude'}")
for i, mu in enumerate(zeros_civ):
    tau = R_civ**2 / (D_civ * mu**2)
    amp = 1.0 / (i+1)**2  # modes decay faster
    print(f"  n={i+1}       {mu:.4f}    {tau:12.2f}    {amp:.4f}")
print()

# ============================================================
# 7. AEROSPACE: nozzle throat Bessel modes + shock structure
# ============================================================
print("=== 7. Aerospace: circular nozzle throat ===")
print("""
  Rocket nozzle throat: circular cross-section, area A*
  Flow is choked (M=1) at throat.

  Acoustic modes in cylindrical duct (no flow):
    p'(r,phi,z,t) = J_n(k_r r) exp(i*n*phi) exp(i*kz*z) exp(-i*omega*t)

  Cut-on frequency for mode (n,m):
    omega_nm = c * sqrt(k_r_nm^2 + kz^2)
    where J_n'(k_r_nm * R) = 0  (hard wall BC)

  Rotating instability in turbomachinery:
    Modes with n>0 rotate at omega/n -> resonance causes blade flutter

  Zeros of J_n'(x) = J_{n-1}(x) - (n/x)*J_n(x):
""")

from scipy.special import jnp_zeros  # zeros of J_n'
print(f"  {'Mode (n,m)':12s}  {'Zero of J_n prime':18s}  {'Cut-on f (kHz, R=0.1m, c=340m/s)'}")
for n in range(3):
    for m in range(1, 3):
        try:
            z = jnp_zeros(n, m)[m-1]
        except Exception:
            continue
        R_noz = 0.10   # m
        c_val = 340.0
        f_cuton = c_val * z / (2*np.pi*R_noz) / 1000
        print(f"  ({n},{m}):         {z:18.4f}  {f_cuton:12.2f} kHz")
print()

# ============================================================
# 8. FULL CONNECTION BACK TO JALALI / GS
# ============================================================
print("=== 8. Cylindrical symmetry -> fiber -> GS ===")
print("""
  FIBER OPTIC PIPELINE:
  +-----------+   Bessel J_n   +-----------+   Dispersion   +----------+
  | LP modes  | ------------> | E(t) pulse| ------------> | I1,I2    |
  | (r,phi)   |               | time domain               | measured |
  +-----------+               +-----------+               +----------+
        ^                                                       |
        |  cylindrical                                     GS recovery
        |  coordinates                                          |
        |                                               +-------v------+
        +-----------------------------------------------+ phi(t) phase|
                                                        +--------------+

  The Bessel function IS the fiber mode profile.
  The fiber mode carries the pulse E(t).
  The pulse acquires phase phi(t) through propagation.
  GS recovers phi(t) from I1,I2 = intensities.

  SBIR pitch connects all three layers:
    Physics:    Bessel modes, GVD, solitons
    Algorithm:  TD-GS + FNO phase recovery
    Hardware:   RogueGuard 1U sensor (RPi CM4)
    Market:     DoD fiber network monitoring, undersea cables

  "You specialized in cylindrical coordinates for 3 years"
  -> that IS the fiber optics curriculum
  -> V-number, LP modes, MFD, splice loss, cutoff
  -> all reduces to zeros of Bessel functions
""")

# sanity check: J_0 zero = SMF-28 single-mode cutoff
J0_z1 = jn_zeros(0, 1)[0]
print(f"  J_0 first zero = {J0_z1:.4f}  <->  SMF-28 V-number < {J0_z1:.4f} for single-mode")
print(f"  SMF-28 at 1550nm: V = {V:.4f}  -> {'single-mode confirmed' if V < J0_z1 else 'MULTIMODE'}")
