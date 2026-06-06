# -*- coding: utf-8 -*-
"""
_repl_tise_sbessel_microfluidics.py
====================================
Time-independent Schrodinger equation, spherical Bessel functions,
and microfluidics -- with SymPy, scipy, matplotlib.

S1: Time-independent Schrodinger equation (TISE)
    - Derivation from TDSE by separation of variables
    - Stationary states, energy eigenstates
    - General solutions: infinite square well, finite square well,
      harmonic oscillator, free particle
    - WKB approximation
    - Numerical shooting method for arbitrary V(x)

S2: Spherical Bessel functions (precise)
    - Derivation from 3D TISE in spherical coordinates
    - j_n(x), y_n(x) -- closed-form expressions for all n
    - Recurrence relations, Wronskian
    - Rayleigh's formula: j_n(x) = (-x)^n (1/x d/dx)^n (sin x / x)
    - Applications: hydrogen atom radial equation, Mie scattering,
      acoustic cavity modes, quantum well in spherical geometry

S3: Microfluidics
    - Reynolds number Re = rho*v*L/mu -- laminar vs turbulent
    - Hagen-Poiseuille flow: parabolic profile in cylinder
    - Electroosmotic flow (EOF): Debye layer, zeta potential
    - Diffusion in microchannels: Peclet number Pe = v*L/D
    - Droplet microfluidics: capillary number Ca = mu*v/gamma
    - Lab-on-chip: pressure-driven vs electric-field-driven
    - RogueGuard connection: optofluidic sensing, photonic readout

Output: repl/_out_tise_sbessel_microfluidics.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import special as sc
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from sympy import (symbols, sqrt, exp, pi, oo, I, Rational, sin, cos,
                   diff, integrate, simplify, series, factorial,
                   besselj, bessely, limit, Function, Eq, solve,
                   tanh, sinh, cosh, tan)
import os

try:
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_out_tise_sbessel_microfluidics.png")
except NameError:
    OUT = "_out_tise_sbessel_microfluidics.png"

SEP = "=" * 65

# ============================================================
# S1: TIME-INDEPENDENT SCHRODINGER EQUATION
# ============================================================
print(SEP)
print("SECTION 1: TIME-INDEPENDENT SCHRODINGER EQUATION (TISE)")
print(SEP)

print("""
  SEPARATION OF VARIABLES FROM TDSE:
    TDSE: i*hbar * d|Psi>/dt = H_hat |Psi>

    Try: Psi(x,t) = psi(x) * phi(t)
    i*hbar * psi(x) * phi'(t) = H_hat[psi(x)] * phi(t)
    Divide both sides by psi(x)*phi(t):
    i*hbar * phi'(t)/phi(t) = H_hat[psi(x)] / psi(x) = E  (const)

    Left side (time only):
      phi'(t)/phi(t) = -iE/hbar
      phi(t) = exp(-iEt/hbar)    [phase factor]

    Right side (space only):
      H_hat psi(x) = E * psi(x)  <-- TISE

    Full solution: Psi(x,t) = psi(x) * exp(-iEt/hbar)
    |Psi|^2 = |psi(x)|^2  (TIME-INDEPENDENT -- stationary state)

  THE TISE:
    [-hbar^2/(2m) * d^2/dx^2 + V(x)] psi(x) = E * psi(x)
    OR: H_hat psi = E psi  (eigenvalue equation)

    psi = energy eigenfunction  (stationary state)
    E   = energy eigenvalue

  BOUNDARY CONDITIONS:
    psi must be: normalizable (INT|psi|^2 < inf)
                 continuous
                 d_psi/dx continuous (except where V -> inf)
    => quantization: only discrete E allowed in bound states.
""")

# ---- 1a: Infinite square well ----
print("  INFINITE SQUARE WELL (particle in box):")
print("""
    V(x) = 0 for 0<x<L,   V = inf outside.
    TISE: -hbar^2/(2m) psi'' = E*psi  inside
    General solution: psi = A*sin(kx) + B*cos(kx), k = sqrt(2mE)/hbar
    BC: psi(0)=0 => B=0;  psi(L)=0 => sin(kL)=0 => kL = n*pi
    => k_n = n*pi/L,   n = 1,2,3,...  (n=0 gives psi=0, unphysical)

    Energy:    E_n = hbar^2*k_n^2/(2m) = n^2 * pi^2*hbar^2 / (2mL^2)
    Wavefunction: psi_n(x) = sqrt(2/L) * sin(n*pi*x/L)
    <x>   = L/2  (symmetric)
    <x^2> = L^2*(1/3 - 1/(2n^2*pi^2))
    <p>   = 0    (equal left/right momentum)
    <p^2> = (n*pi*hbar/L)^2 = hbar^2*k_n^2
    sigma_x * sigma_p = (hbar/2)*sqrt(n^2*pi^2/3 - 2)  >= hbar/2 (HUP)
    At n=1: sigma_x*sigma_p = 0.568*hbar > hbar/2  (above minimum)
""")

x_s, L_s, n_s = symbols("x L n", positive=True)
hbar_s, m_s   = symbols("hbar m", positive=True)

psi_n = sqrt(2/L_s) * sin(n_s*pi*x_s/L_s)
E_n   = n_s**2 * pi**2 * hbar_s**2 / (2*m_s*L_s**2)

norm = simplify(integrate(psi_n**2, (x_s, 0, L_s)))
xexp = simplify(integrate(x_s * psi_n**2, (x_s, 0, L_s)))
print(f"  SymPy normalization INT|psi_n|^2 = {norm}")
print(f"  SymPy <x> = {xexp}  (= L/2)")

# Finite square well
print("""
  FINITE SQUARE WELL:
    V(x) = -V0 for |x|<a,  V=0 outside  (V0>0)
    Inside: psi = A*cos(kx) [even] or A*sin(kx) [odd],   k=sqrt(2m(E+V0))/hbar
    Outside: psi = B*exp(-kappa*x) [x>a],                 kappa=sqrt(-2mE)/hbar
    Match psi and psi' at x=a:
    Even: k*tan(ka) = kappa
    Odd:  -k*cot(ka) = kappa
    Transcendental -- solve graphically or numerically.
    Always at least ONE bound state (no matter how small V0).
    Finite well has FEWER bound states than infinite well (penetration).
    As V0 -> inf: reduces to infinite square well.
""")

# Graphical solution: k*tan(ka) = kappa
V0  = 10.0   # eV * m^{-2} * hbar^2/(2m) units
a   = 1.0    # half-width (arbitrary units)
k_arr = np.linspace(0.001, 4*np.pi, 2000)
# z = k*a, z0 = a*sqrt(2mV0)/hbar
z0 = np.sqrt(V0) * a  # normalized depth
z  = k_arr * a
kappa_z = np.sqrt(np.maximum(z0**2 - z**2, 0))

even_lhs = z * np.tan(z)
even_rhs = kappa_z

# Find crossings (even solutions)
crossings = []
for i in range(len(z)-1):
    if (even_lhs[i] > 0 and even_rhs[i] > 0 and
            np.sign(even_lhs[i]-even_rhs[i]) != np.sign(even_lhs[i+1]-even_rhs[i+1])):
        crossings.append(z[i]/a)
print(f"  Finite well (V0={V0}, a={a}): even-parity crossings at k =",
      [f"{c:.4f}" for c in crossings[:3]])

# ---- 1b: WKB approximation ----
print("""
  WKB APPROXIMATION (Wentzel-Kramers-Brillouin):
    Valid when potential varies slowly compared to de Broglie wavelength.
    Condition: |dV/dx| << hbar*k^3 / m  (slowly varying)

    Inside classically allowed region (E > V):
      psi(x) ~ (1/sqrt(k(x))) * [A*exp(i*INT k dx) + B*exp(-i*INT k dx)]
      k(x) = sqrt(2m(E-V(x)))/hbar  [local wavenumber]
    Inside classically forbidden region (E < V):
      psi(x) ~ (1/sqrt(kappa(x))) * [C*exp(INT kappa dx) + D*exp(-INT kappa dx)]
      kappa(x) = sqrt(2m(V(x)-E))/hbar  [local decay rate]

    WKB QUANTIZATION (Bohr-Sommerfeld):
      OINT p dx = (n + 1/2) * h  for n = 0,1,2,...
      INT_{x1}^{x2} sqrt(2m(E-V)) dx = (n+1/2)*pi*hbar
      (x1, x2 are classical turning points where E=V)

    EXACT for: linear V (Airy functions), harmonic oscillator.
    Approximate for: smooth potentials far from turning points.

    TUNNELING: amplitude through barrier ~ exp(-2*INT_{x1}^{x2} kappa dx)
    Gamma_tunnel ~ exp(-2/hbar * INT sqrt(2m(V-E)) dx)
    Used for: alpha decay, Josephson junction, STM, FET gate tunneling.
""")

# WKB example: harmonic oscillator
# V = (1/2)*m*omega^2*x^2
# INT_{-a}^{a} sqrt(2m(E - mw^2x^2/2)) dx = (n+1/2)*pi*hbar
# with a=sqrt(2E/(mw^2)):
# Result: 2*E/omega * pi/2 = (n+1/2)*pi*hbar => E = (n+1/2)*hbar*omega  EXACT
print("  WKB for SHO: gives E_n = (n+1/2)*hbar*omega EXACTLY (correct!)")

# ---- 1c: Numerical shooting method ----
print("""
  NUMERICAL SHOOTING METHOD:
    Convert 2nd-order ODE to system: psi' = phi, phi' = 2m/hbar^2*(V-E)*psi
    Shoot from left (psi=0, psi'=1 at x=x_min)
    Shoot from right (psi=0, psi'=-1 at x=x_max)
    Eigenvalue E: find where left and right solutions match smoothly.
    Bisect on E until psi_left(x_match) = psi_right(x_match).
""")

# Numerov method for infinite square well verification
N = 1000
L_num = 1.0
x_arr = np.linspace(0, L_num, N+2)[1:-1]  # exclude endpoints
dx = x_arr[1] - x_arr[0]

# Build tridiagonal Hamiltonian (V=0 inside well)
# H_{ij} = -hbar^2/(2m) * d^2/dx^2 delta_{ij}
# In units hbar^2/(2m)=1, L=1:
diag     = np.full(N, 2.0/dx**2)
off_diag = np.full(N-1, -1.0/dx**2)
E_num, vecs = eigh_tridiagonal(diag, off_diag, select="i", select_range=(0,4))

E_exact = [(n**2 * np.pi**2) for n in range(1,6)]
print("  Numerical vs exact eigenvalues (units hbar^2/2mL^2):")
print(f"  {'n':<5} {'E_num':<15} {'E_exact':<15} {'error'}")
for i in range(5):
    print(f"  {i+1:<5} {E_num[i]:<15.6f} {E_exact[i]:<15.6f} "
          f"{abs(E_num[i]-E_exact[i]):.2e}")

# ============================================================
# S2: SPHERICAL BESSEL FUNCTIONS (PRECISE)
# ============================================================
print(f"\n{SEP}")
print("SECTION 2: SPHERICAL BESSEL FUNCTIONS")
print(SEP)

print("""
  ORIGIN: 3D TISE WITH SPHERICAL SYMMETRY
    TISE: [-hbar^2/(2m) * nabla^2 + V(r)] Psi = E * Psi
    In spherical coordinates:
    nabla^2 = d^2/dr^2 + (2/r)*d/dr + (1/r^2)*L^2_angular

    Separate: Psi(r,theta,phi) = R(r) * Y_l^m(theta,phi)
    L^2 Y_l^m = hbar^2 * l*(l+1) * Y_l^m

    Radial equation:
    [-hbar^2/(2m)] [R'' + (2/r)R' - l(l+1)/r^2 * R] + V(r)*R = E*R

    For FREE PARTICLE (V=0):
    R'' + (2/r)R' + [k^2 - l(l+1)/r^2]*R = 0,  k = sqrt(2mE)/hbar
    Let u(r) = r*R(r):  u'' + [k^2 - l(l+1)/r^2]*u = 0
    Substitute x = kr:
    x^2 u'' + [x^2 - l(l+1)] u = 0  (spherical Bessel equation)

  SPHERICAL BESSEL ODE:
    x^2 y'' + 2x y' + [x^2 - l(l+1)] y = 0
    (different from regular Bessel: coefficient of y' is 2, not 1)

  SOLUTIONS:
    j_l(x) = sqrt(pi/(2x)) * J_{l+1/2}(x)   [1st kind, finite at x=0]
    y_l(x) = sqrt(pi/(2x)) * Y_{l+1/2}(x)   [2nd kind, singular at x=0]

  RAYLEIGH'S FORMULA (exact closed forms):
    j_l(x) = (-x)^l * (1/x * d/dx)^l * [sin(x)/x]
    y_l(x) = -(-x)^l * (1/x * d/dx)^l * [cos(x)/x]
""")

# Derive closed forms via Rayleigh formula
x_b = symbols("x", positive=True)
sinc = sin(x_b)/x_b

print("  Closed forms via Rayleigh formula:")
# j_0: l=0, (-x)^0 * (1/x d/dx)^0 [sinx/x] = sinx/x
j0_sym = sin(x_b)/x_b
# j_1: l=1, (-x)^1 * (1/x d/dx)^1 [sinx/x]
#   = (-x) * (1/x) * d/dx[sinx/x]
#   = (-x) * (1/x) * (cos(x)*x - sin(x))/x^2
#   = (-x) * (cos(x)*x - sin(x))/x^3
#   = (sin(x) - x*cos(x))/x^2
d_sinc = diff(sin(x_b)/x_b, x_b)
j1_sym = simplify(-x_b * (1/x_b) * d_sinc)
# j_2: l=2
d2 = diff(d_sinc/x_b, x_b)
j2_sym = simplify(x_b**2 * (1/x_b) * d2)  # (-x)^2 * (1/x d/dx)^2

for l_val, name, sym_expr in [
    (0, "j_0", j0_sym),
    (1, "j_1", j1_sym),
]:
    print(f"    {name}(x) = {sym_expr}")

print("""
  ALL CLOSED FORMS:
    j_0(x) = sin(x)/x
    j_1(x) = sin(x)/x^2 - cos(x)/x
    j_2(x) = (3/x^2 - 1)*sin(x)/x - 3*cos(x)/x^2
    j_3(x) = (15/x^3 - 6/x)*sin(x)/x - (15/x^2 - 1)*cos(x)/x
    y_0(x) = -cos(x)/x
    y_1(x) = -cos(x)/x^2 - sin(x)/x
    y_2(x) = -(3/x^2-1)*cos(x)/x - 3*sin(x)/x^2

  RECURRENCE RELATIONS:
    f_{l+1}(x) = (2l+1)/x * f_l(x) - f_{l-1}(x)   [upward from j_0,j_1]
    f_{l-1}(x) = (2l+1)/x * f_l(x) - f_{l+1}(x)   [downward -- more stable]
    d/dx f_l(x) = f_{l-1}(x) - (l+1)/x * f_l(x)
    d/dx [x^{l+1} f_l(x)] = x^{l+1} f_{l-1}(x)

  WRONSKIAN:
    W[j_l, y_l] = j_l*y_l' - j_l'*y_l = 1/x^2  (for all l, x>0)
    Confirms j_l and y_l are always linearly independent.

  ORTHOGONALITY (on [0,a]):
    INT_0^a r^2 j_l(k_{ln}*r) j_l(k_{ln'}*r) dr = (a^3/2)*[j_{l+1}(k_{ln}*a)]^2 * delta_{nn'}
    where k_{ln} = x_{ln}/a, x_{ln} = nth zero of j_l.

  SPHERICAL HANKEL FUNCTIONS (outgoing/incoming waves):
    h_l^(1)(x) = j_l(x) + i*y_l(x)    [outgoing: ~ e^(ix)/x as x->inf]
    h_l^(2)(x) = j_l(x) - i*y_l(x)    [incoming: ~ e^(-ix)/x as x->inf]
    Used in scattering: partial wave expansion f(theta) = SUM a_l * P_l(cos theta)

  ASYMPTOTICS:
    x -> 0:  j_l(x) ~ x^l / (2l+1)!!        [finite, = 1 only for l=0]
             y_l(x) ~ -(2l-1)!! / x^{l+1}   [singular, ~ -1/x for l=0]
    x -> inf: j_l(x) ~ sin(x - l*pi/2) / x  [oscillating, decaying]
              y_l(x) ~ -cos(x - l*pi/2) / x
""")

# Numerical verification of recurrence
print("  Recurrence verification at x=5.0:")
x_val = 5.0
j_vals = [sc.spherical_jn(l, x_val) for l in range(5)]
for l in range(1, 4):
    recur = (2*l+1)/x_val * j_vals[l] - j_vals[l-1]
    exact = j_vals[l+1]
    print(f"    j_{l+1}(5): recurrence={recur:.8f}, exact={exact:.8f}, "
          f"err={abs(recur-exact):.2e}")

# Zeros of j_0, j_1, j_2
print("\n  Zeros of spherical Bessel functions (NOT same as J_n zeros):")
# j_l zeros = zeros of J_{l+1/2}(x) scaled by sqrt(pi/2x)
# = zeros of J_{l+1/2}, which equal zeros of j_l
for l in range(3):
    # Find zeros numerically from sign changes
    x_scan = np.linspace(0.5, 30, 10000)
    jl_scan = sc.spherical_jn(l, x_scan)
    sign_changes = np.where(np.diff(np.sign(jl_scan)))[0]
    zero_approx = [(x_scan[i] + x_scan[i+1])/2 for i in sign_changes[:5]]
    print(f"    j_{l} zeros: {[f'{z:.4f}' for z in zero_approx]}")

# Hydrogen atom connection
print("""
  HYDROGEN ATOM RADIAL SOLUTION:
    Free particle in Coulomb potential uses LAGUERRE not spherical Bessel,
    but the BOUND free spherical cavity uses j_l.
    Spherical infinite well (V=0, r<R; V=inf, r>=R):
      psi_nlm = j_l(x_{ln}*r/R) * Y_l^m(theta,phi)
      E_nl = hbar^2*x_{ln}^2 / (2mR^2)
      x_{ln} = nth zero of j_l
    Lowest state (l=0,n=1): j_0 zero = pi => E_10 = hbar^2*pi^2/(2mR^2)
    Same as 1D infinite well with L=R (makes sense by symmetry).
""")

# ============================================================
# S3: MICROFLUIDICS
# ============================================================
print(f"\n{SEP}")
print("SECTION 3: MICROFLUIDICS")
print(SEP)

print("""
  LENGTH SCALES:
    Macro:  L > 1 mm  -- turbulent flow possible, gravity matters
    Micro:  L ~ 1-100 um  -- laminar flow dominant, surface effects dominate
    Nano:   L < 100 nm  -- molecular effects, electric double layer comparable

  DIMENSIONLESS NUMBERS (govern microfluidic behavior):

  REYNOLDS NUMBER:  Re = rho*v*L / mu
    rho = fluid density [kg/m^3]
    v   = characteristic velocity [m/s]
    L   = channel dimension [m]
    mu  = dynamic viscosity [Pa*s]
    Physical meaning: inertial forces / viscous forces
    Re << 1: viscous dominates -> LAMINAR (smooth, predictable)
    Re > 2300: turbulent (chaotic, mixing)
    Microfluidics: Re typically 0.001 - 10  (always laminar!)
    Water in 100um channel at 1mm/s: Re = 1000*0.001*100e-6 / 1e-3 = 0.1

  PECLET NUMBER:  Pe = v*L / D
    D = diffusion coefficient [m^2/s]
    Physical meaning: convective transport / diffusive transport
    Pe >> 1: convection dominates (species carried by flow, stratified)
    Pe << 1: diffusion dominates (well-mixed quickly)
    Diffusion-limited mixing is a key challenge in microfluidics.
    DNA (D~1e-12 m^2/s) at 1mm/s in 100um channel: Pe = 0.001*100e-6/1e-12 = 1e5
    => DNA barely mixes! Need chaotic advection, serpentine channels.

  CAPILLARY NUMBER:  Ca = mu*v / gamma
    gamma = surface tension [N/m]
    Physical meaning: viscous forces / surface tension forces
    Ca << 1: surface tension dominates -> droplets form, stable
    Ca >> 1: viscous forces win -> droplet breakup, jetting
    Water-oil at v=1mm/s: Ca ~ 1e-3*1e-3/0.02 = 5e-5 (droplets stable)
    Droplet microfluidics operates at Ca ~ 1e-3 to 1e-1.

  BOND NUMBER:  Bo = rho*g*L^2 / gamma
    Physical meaning: gravity / surface tension
    L=100um water: Bo = 1000*9.8*(100e-6)^2/0.072 = 1.4e-3 << 1
    => Gravity NEGLIGIBLE in microfluidics. Surface tension rules.
""")

rho = 1000.0   # kg/m^3 water
mu  = 1e-3     # Pa*s water at 20C
D   = 1e-9     # m^2/s small molecule diffusion
gamma_val = 0.072  # N/m water-air
g   = 9.81

print("  KEY DIMENSIONLESS NUMBERS FOR WATER IN MICROCHANNELS:")
print(f"  {'Parameter':<20} {'100um/1mm/s':<18} {'10um/0.1mm/s':<18} {'1mm/10mm/s'}")
print(f"  {'-'*68}")
for label, L, v in [("100um, 1mm/s", 100e-6, 1e-3),
                     ("10um, 0.1mm/s", 10e-6, 0.1e-3),
                     ("1mm, 10mm/s",  1e-3,  10e-3)]:
    Re = rho*v*L/mu
    Pe = v*L/D
    Ca = mu*v/gamma_val
    Bo = rho*g*L**2/gamma_val
    print(f"  {label:<20} Re={Re:.3f}  Pe={Pe:.0f}  Ca={Ca:.1e}  Bo={Bo:.1e}")

print("""
  HAGEN-POISEUILLE FLOW (pressure-driven, cylindrical channel):
    Governing equation (Stokes flow, Re<<1):
    mu * nabla^2 v = grad(P)
    In cylindrical coordinates, v_z(r) only:
    mu * (1/r) * d/dr [r * dv_z/dr] = dP/dz = -DeltaP/L

    Solution (parabolic velocity profile):
    v_z(r) = (DeltaP/(4*mu*L)) * (R^2 - r^2)
    v_max  = DeltaP*R^2 / (4*mu*L)        [at r=0, center]
    v_mean = v_max / 2                     [= DeltaP*R^2/(8muL)]
    Q      = pi*R^4*DeltaP / (8*mu*L)     [Hagen-Poiseuille law]

    Hydraulic resistance: R_hyd = 8*mu*L / (pi*R^4)
    Analogy to Ohm's law: DeltaP = R_hyd * Q  (P=V, Q=I, R_hyd=R)
    Networks: resistors in series/parallel!

    IMPORTANT: Q ~ R^4 (fourth power!)
    Halving radius: 16x increase in resistance
    => Small channels require large pressure drops.

  RECTANGULAR CHANNEL (most common in chip fabrication):
    Width W >> Height H (aspect ratio >> 1):
    v(y) ~ (H^2/(8muL)) * DeltaP * [1 - (2y/H)^2]
    Q ~ W*H^3*DeltaP / (12*mu*L)   [R_hyd ~ 12muL/(W*H^3)]
""")

# Numerical: HP flow
R_ch = 50e-6    # 50 um radius
L_ch = 1e-2     # 1 cm long
dP   = 1000.0   # 1 kPa pressure drop

Q_HP    = np.pi * R_ch**4 * dP / (8 * mu * L_ch)
v_max   = dP * R_ch**2 / (4*mu*L_ch)
R_hyd   = 8 * mu * L_ch / (np.pi * R_ch**4)

print(f"\n  HP FLOW EXAMPLE: R={R_ch*1e6:.0f}um, L={L_ch*100:.0f}cm, dP={dP:.0f}Pa:")
print(f"    v_max  = {v_max*1000:.2f} mm/s")
print(f"    Q      = {Q_HP*1e12:.4f} nL/s = {Q_HP*1e9*60:.4f} uL/min")
print(f"    R_hyd  = {R_hyd:.3e} Pa*s/m^3")
print(f"    Re     = {rho*v_max*2*R_ch/mu:.4f}  (laminar, << 2300)")

print("""
  ELECTROOSMOTIC FLOW (EOF):
    In microchannels, walls are usually negatively charged (silica: SiO^-).
    This attracts a layer of positive ions -> ELECTRIC DOUBLE LAYER (EDL).
    Apply electric field E along channel: ions move -> drag fluid.

    DEBYE LENGTH (EDL thickness):
    lambda_D = sqrt(eps_r*eps_0*kT / (2*n_0*z^2*e^2))
    For 10mM NaCl: lambda_D ~ 3 nm
    For 0.1mM NaCl: lambda_D ~ 30 nm
    lambda_D << channel size (100um) -> thin EDL approximation valid.

    EOF VELOCITY (Helmholtz-Smoluchowski):
    v_EOF = -eps_r*eps_0*zeta*E / mu
    zeta = zeta potential (surface charge effect, ~ -50 mV for SiO2)
    E = applied electric field [V/m]

    PLUG FLOW: EOF has FLAT velocity profile (not parabolic)!
    v_z(r) = v_EOF = const (for lambda_D << R)
    => No Taylor dispersion: bands stay sharp (great for separations!)
    => Electrophoresis + EOF = basis of capillary electrophoresis (CE).

    COMBINED PRESSURE + EOF:
    v_total = v_HP(r) + v_EOF
    Can cancel HP backpressure with EOF (useful for isoelectric focusing).
""")

# EOF calculation
eps_r   = 80.0     # water
eps_0   = 8.854e-12
zeta    = -50e-3   # V
E_field = 1000.0   # V/m (1 kV over 1m)

v_EOF = -eps_r * eps_0 * zeta * E_field / mu
print(f"\n  EOF EXAMPLE: zeta={zeta*1000:.0f}mV, E={E_field:.0f}V/m:")
print(f"    v_EOF = {v_EOF*1000:.4f} mm/s")
Re_EOF = rho * v_EOF * 100e-6 / mu
print(f"    Re    = {Re_EOF:.4f}  (laminar)")

kT = 1.38e-23 * 293  # room temp
n0 = 0.01 * 6.022e23 * 1000  # 10mM = 6.022e24 ions/m^3
e  = 1.602e-19
lambda_D = np.sqrt(eps_r*eps_0*kT / (2*n0*e**2))
print(f"    Debye length (10mM NaCl): {lambda_D*1e9:.2f} nm")

print("""
  DIFFUSION IN MICROCHANNELS:
    Fick's law: J = -D * grad(c)
    1D: dc/dt = D * d^2c/dx^2   (diffusion equation)
    Solution for point source: c(x,t) = (1/sqrt(4piDt)) * exp(-x^2/(4Dt))
    Diffusion length: delta = sqrt(4Dt) = sqrt(2*D*t) (std dev of Gaussian)

    MIXING TIME (diffusive): t_mix ~ L^2 / D
    Molecule (D=1e-9): t_mix = (100um)^2/1e-9 = 10 sec
    DNA (D=1e-12): t_mix = 10000 sec! -- very slow
    => Need active mixing: serpentine, herringbone grooves, acoustic mixing.

  DROPLET MICROFLUIDICS:
    Two immiscible fluids: aqueous in oil.
    Droplets formed at T-junction or flow-focusing geometry.
    Droplet generation rate: f ~ v/lambda (lambda = droplet spacing)
    Droplet volume: V ~ Ca^{-0.4} * (W*H)^{3/2} / L_orifice
    Applications:
    - Digital PCR: one DNA copy per droplet
    - Drug encapsulation: single-cell assay
    - High-throughput screening: 10^6 droplets/hour

  ROGUEGUARD / OPTOFLUIDIC CONNECTION:
    Optical fiber in microfluidic channel = optofluidic platform.
    Evanescent wave from fiber core extends ~100nm into cladding/fluid.
    If fluid contains particles: scattering changes I(t) detected by ADC.
    Phase shift from particle passage: delta_phi ~ Delta_n * L / lambda
    This delta_phi encoded in I(t) -> GS recovers phi(nu) -> particle size!
    Rogue wave detector: same physics as flow cytometer + phase retrieval.

    PECLET NUMBER for optical readout:
    Pe_optical = v_flow * L_interaction / D_particle_diffusion
    For 1um bead (D~0.5e-12): Pe = 1e-3 * 100e-6 / 0.5e-12 = 2e5
    >> 1: particle carried by flow, not diffusing across sensor -> clean signal.
""")

# ============================================================
# MATPLOTLIB -- 6-PANEL FIGURE
# ============================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#F8F8F0")
gs0 = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                        top=0.93, bottom=0.05, left=0.06, right=0.97)

ax_well   = fig.add_subplot(gs0[0, 0])
ax_wkb    = fig.add_subplot(gs0[0, 1])
ax_num    = fig.add_subplot(gs0[0, 2])
ax_sph    = fig.add_subplot(gs0[1, 0])
ax_sph2   = fig.add_subplot(gs0[1, 1])
ax_hank   = fig.add_subplot(gs0[1, 2])
ax_hp     = fig.add_subplot(gs0[2, 0])
ax_eof    = fig.add_subplot(gs0[2, 1])
ax_droplet= fig.add_subplot(gs0[2, 2])

fig.suptitle("TISE + Spherical Bessel + Microfluidics",
             fontsize=14, fontweight="bold", color="#1a1a2e")

colors5 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]

# ---- AX_WELL: infinite square well wavefunctions ----
ax = ax_well
ax.set_facecolor("#F0F4FF")
x_w = np.linspace(0, 1, 500)
for n in range(1, 6):
    psi = np.sqrt(2) * np.sin(n*np.pi*x_w)
    E   = n**2 * np.pi**2
    ax.plot(x_w, psi/8 + E, color=colors5[n-1], lw=1.5, label=f"n={n}")
    ax.axhline(E, color=colors5[n-1], lw=0.6, ls="--", alpha=0.4)
ax.set_xlabel("x/L"); ax.set_ylabel(r"$E_n / ({\hbar^2}/{2mL^2})$")
ax.set_title(r"Infinite Square Well $\psi_n$", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.2)
ax.text(0.55, 0.05,
        r"$E_n = n^2\pi^2\hbar^2/2mL^2$" + "\n" + r"$\psi_n=\sqrt{2/L}\sin(n\pi x/L)$",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_WKB: WKB potential barrier tunneling ----
ax = ax_wkb
ax.set_facecolor("#FFF5F0")
x_wkb = np.linspace(-3, 3, 500)
V_wkb = 5 * np.exp(-x_wkb**2)    # Gaussian barrier
E_wkb = 2.0                       # particle energy

ax.fill_between(x_wkb, 0, V_wkb, alpha=0.2, color="#d62728", label="V(x)")
ax.plot(x_wkb, V_wkb, "#d62728", lw=1.5)
ax.axhline(E_wkb, color="#1f77b4", lw=1.5, ls="--", label=f"E={E_wkb}")

# WKB tunneling integral
forbidden = x_wkb[V_wkb > E_wkb]
if len(forbidden) > 1:
    kappa_arr = np.sqrt(np.maximum(2*(V_wkb - E_wkb), 0))
    # approximate: Gamma ~ exp(-2 * integral)
    dx_wkb = x_wkb[1]-x_wkb[0]
    integral_kappa = np.trapezoid(kappa_arr, x_wkb)
    Gamma = np.exp(-2*integral_kappa)
    ax.text(0, V_wkb.max()*0.5,
            f"Gamma~exp(-2*INT kappa dx)\n= {Gamma:.4f}",
            ha="center", fontsize=8,
            bbox=dict(fc="#fff8f0", ec="#bbb", pad=2))

ax.set_title("WKB Tunneling Through Barrier", fontsize=10)
ax.legend(fontsize=8)
ax.set_xlabel("x"); ax.set_ylabel("Energy")
ax.grid(alpha=0.2)
ax.set_ylim(-0.5, 6)

# ---- AX_NUM: numerical eigenvalues ----
ax = ax_num
ax.set_facecolor("#F0FFF0")
# Show first 5 eigenfunctions from numerical solution
x_num_plot = np.linspace(0, L_num, N+2)[1:-1]
for i in range(5):
    psi_i = vecs[:, i]
    norm_i = np.sqrt(np.trapezoid(psi_i**2, x_num_plot))
    psi_i /= norm_i
    ax.plot(x_num_plot, psi_i + (i+1)**2*np.pi**2,
            color=colors5[i], lw=1.5, label=f"n={i+1}")
    ax.axhline((i+1)**2*np.pi**2, color=colors5[i], lw=0.6, ls="--", alpha=0.4)
ax.set_title("Numerical FDM Eigenfunctions\n(tridiagonal matrix, N=1000)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_xlabel("x/L"); ax.set_ylabel("E + psi (offset)")
ax.grid(alpha=0.2)

# ---- AX_SPH: spherical Bessel j_l ----
ax = ax_sph
ax.set_facecolor("#FFF0F0")
x_sp = np.linspace(0.001, 20, 1000)
for l in range(5):
    jl = sc.spherical_jn(l, x_sp)
    ax.plot(x_sp, jl, color=colors5[l], lw=1.5, label=f"j_{l}(x)")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(-0.4, 1.05)
ax.set_title(r"Spherical Bessel $j_l(x)$ (1st kind)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.set_xlabel("x")
ax.text(0.02, 0.05,
        r"$j_0=\sin x/x$" + "\n" + r"Rayleigh: $j_l=(-x)^l(x^{-1}\partial_x)^l(\sin x/x)$",
        transform=ax.transAxes, fontsize=7.5,
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_SPH2: spherical Bessel y_l (singular) ----
ax = ax_sph2
ax.set_facecolor("#F5F0FF")
x_sp2 = np.linspace(0.2, 20, 1000)
for l in range(4):
    yl = sc.spherical_yn(l, x_sp2)
    ax.plot(x_sp2, np.clip(yl, -4, 1), color=colors5[l], lw=1.5, label=f"y_{l}(x)")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(-3, 1)
ax.set_title(r"Spherical Bessel $y_l(x)$ (2nd kind, singular)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.set_xlabel("x")
ax.text(0.55, 0.05,
        r"$y_0=-\cos x/x$" + "\n" + "Excluded if r=0 in domain",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_HANK: Hankel function |h_l^(1)| ----
ax = ax_hank
ax.set_facecolor("#F0FFF8")
x_h = np.linspace(0.5, 20, 500)
for l in range(4):
    jl = sc.spherical_jn(l, x_h)
    yl = sc.spherical_yn(l, x_h)
    hl = np.abs(jl + 1j*yl)
    ax.plot(x_h, hl, color=colors5[l], lw=1.5, label=f"|h^(1)_{l}|")
# Asymptotic: |h_l| ~ 1/x
ax.plot(x_h, 1/x_h, "k--", lw=1.0, label="1/x asymptote")
ax.set_yscale("log")
ax.set_title(r"Spherical Hankel $|h_l^{(1)}| = |j_l + iy_l|$", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.2, which="both")
ax.set_xlabel("x")
ax.text(0.02, 0.05, "Outgoing spherical wave\n~exp(ix)/x as x->inf",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_HP: Hagen-Poiseuille parabolic profile ----
ax = ax_hp
ax.set_facecolor("#FFFFF0")
R_plot = 1.0
r_arr  = np.linspace(-R_plot, R_plot, 300)
dP_vals = [500, 1000, 2000]
for dP_v, col in zip(dP_vals, ["#4a90d9","#1a3a8c","#d62728"]):
    v_max_v = dP_v * R_plot**2 / (4*mu*L_ch*1e6)  # scaled
    v_r     = v_max_v * (1 - (r_arr/R_plot)**2)
    ax.plot(v_r, r_arr, color=col, lw=1.8, label=f"dP={dP_v}Pa")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("v_z (scaled)"); ax.set_ylabel("r/R")
ax.set_title("Hagen-Poiseuille:\n Parabolic Velocity Profile", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.55, 0.05,
        r"$v_z(r)=\frac{\Delta P}{4\mu L}(R^2-r^2)$" + "\n" +
        r"$Q=\frac{\pi R^4 \Delta P}{8\mu L}$",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="#fffff0", ec="#bbb", pad=2))

# ---- AX_EOF: EOF flat profile vs HP parabolic ----
ax = ax_eof
ax.set_facecolor("#F0F4FF")
r_eof = np.linspace(-1, 1, 300)
v_HP_norm  = 1 - r_eof**2         # parabolic
v_EOF_norm = np.ones_like(r_eof)  # flat plug flow
# Near-wall correction for EOF (EDL region, exaggerated for visibility)
lambda_ratio = 0.05  # lambda_D / R (exaggerated)
v_EOF_real = 1 - np.exp(-np.abs(1-np.abs(r_eof))/lambda_ratio)

ax.plot(v_HP_norm,  r_eof, "#d62728", lw=2.0, label="HP (parabolic)")
ax.plot(v_EOF_norm, r_eof, "#1f77b4", lw=2.0, ls="--", label="EOF (plug, ideal)")
ax.plot(v_EOF_real, r_eof, "#2ca02c", lw=1.5, ls=":", label="EOF (with EDL)")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("v / v_max"); ax.set_ylabel("r/R")
ax.set_title("HP (parabolic) vs EOF (plug flow)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.02, 0.05,
        "EOF: flat profile -> no\nTaylor dispersion -> sharp bands",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(fc="white", ec="#1f77b4", pad=2))

# ---- AX_DROPLET: dimensionless numbers ----
ax = ax_droplet
ax.set_facecolor("#FFF0F0")
ax.axis("off")
ax.set_title("Microfluidics Dimensionless Numbers", fontsize=10)

rows = [
    ("Number", "Formula", "Micro value", "Meaning", "#222"),
    ("------", "-------", "-----------", "-------", "#aaa"),
    ("Reynolds Re", "rho*v*L/mu", "0.001-10", "Inertia/viscous", "#1f77b4"),
    ("Peclet Pe",   "v*L/D",    "1-10^5",   "Convect/diffuse", "#ff7f0e"),
    ("Capillary Ca","mu*v/gamma","1e-5-0.1", "Viscous/surface", "#2ca02c"),
    ("Bond Bo",     "rho*g*L^2/g","1e-4",   "Gravity/surface", "#d62728"),
    ("Damkohler Da","k*L/v",    "varies",    "React/convect",  "#9467bd"),
    ("Knudsen Kn",  "lambda/L", "<<1 liquid","Continuum valid", "#8c564b"),
    ("", "", "", "", ""),
    ("HP Flow:", "Q=pi*R^4*dP/8muL", "R^4 dependence!", "", "#333"),
    ("EOF:",     "v=eps*zeta*E/mu",  "flat profile",   "", "#333"),
    ("Diffusion:","delta=sqrt(4Dt)", "slow for DNA",   "", "#333"),
]

col_x = [0.01, 0.25, 0.50, 0.68]
row_y  = np.linspace(0.97, 0.03, len(rows))
for j, (name, form, val, meaning, color) in enumerate(rows):
    y = row_y[j]
    if not name:
        continue
    ax.text(col_x[0], y, name,    fontsize=7.5, va="top", color=color,
            transform=ax.transAxes, fontweight="bold" if j <= 1 else "normal")
    ax.text(col_x[1], y, form,    fontsize=7.0, va="top", color="#1a1a4e" if j>1 else "#555",
            transform=ax.transAxes, fontfamily="monospace")
    ax.text(col_x[2], y, val,     fontsize=7.0, va="top", color="#2c7c2c" if j>1 else "#555",
            transform=ax.transAxes)
    ax.text(col_x[3], y, meaning, fontsize=7.0, va="top", color="#555",
            transform=ax.transAxes)

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
