# -*- coding: utf-8 -*-
"""
_repl_soliton_dielectric_mm_pyc.py
====================================
S1: SOLITONS IN ELECTRODYNAMICS
    - Nonlinear Schrodinger equation (NLS/NLSE) derivation
    - Bright soliton: balance of GVD and SPM
    - Peregrine soliton = prototype rogue wave (exact analytic solution)
    - N-soliton interaction, soliton number N = sqrt(gamma*P0*Ld/pi)
    - Numerical split-step Fourier method (SSFM) -- propagate fiber soliton
    - RogueGuard connection: rogue waves as extreme Peregrine events

S2: DIELECTRIC ELECTRODYNAMICS
    - Maxwell equations in linear dielectric: D = eps*E, B = mu*H
    - Complex permittivity eps(omega) = eps' - i*eps''
    - Lorentz oscillator model: eps(omega) from bound electrons
    - Refractive index n = sqrt(eps_r*mu_r), absorption alpha = 2*omega*k/c
    - Kramers-Kronig relations (causality -> dispersion)
    - Silica fiber: Sellmeier equation n(lambda) 3-term, dispersion D(lambda)

S3: MICHELSON-MORLEY EXPERIMENT (michel moorley)
    - 1887 aether hypothesis: light speed c +/- v_earth (30 km/s)
    - Interferometer geometry: arm length L, expected fringe shift Delta
    - Calculation: Delta = 2*L*v^2 / (lambda*c^2) ~ 0.4 fringes
    - NULL RESULT: < 0.01 fringes observed
    - Consequences: Lorentz contraction, Einstein SR (c=const)
    - Modern analogs: LIGO, atom interferometers, fiber Sagnac (RogueGuard)

S4: PYTHON -> C DESIGN PATTERNS (DoD firmware)
    - Philosophy: design the API in Python, implement hot path in C
    - ctypes: call existing C libraries from Python
    - CFFI: cleaner, ABI or API mode
    - Writing a C extension module (_gs_core.c)
    - Struct layout: pack(1) vs natural alignment
    - Error propagation: errno, return codes, Python exceptions
    - DoD considerations: MISRA-C, no dynamic allocation in safety-critical

S5: CONCRETE + STRUCTURAL (3-story building)
    - Compressive strength fc': normal 3000-5000 psi (20-35 MPa)
    - Tensile weakness: ft ~ 0.1*fc', steel rebar provides tension
    - Load path: floor -> beam -> column -> footing -> soil
    - 3-story dead + live load calculation (ACI 318 simplified)
    - Euler column buckling: Pcr = pi^2*E*I / (K*L)^2
    - Japan/Germany/Columbia: high-seismic vs wind-dominated design

Output: repl/_out_soliton_dielectric_mm_pyc.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from sympy import (symbols, Function, exp, I, pi, sqrt, diff, simplify,
                   conjugate, Abs, oo, integrate, cos, sin, tanh, sech,
                   latex, Rational)
import os

try:
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_out_soliton_dielectric_mm_pyc.png")
except NameError:
    OUT = "_out_soliton_dielectric_mm_pyc.png"

SEP = "=" * 65

# ============================================================
# S1: SOLITONS IN ELECTRODYNAMICS
# ============================================================
print(SEP)
print("SECTION 1: SOLITONS IN ELECTRODYNAMICS")
print(SEP)

print("""
  NONLINEAR SCHRODINGER EQUATION (NLSE):
  In optical fiber, the slowly-varying envelope A(z,t) of the electric
  field obeys the NLSE:

    i * dA/dz - (beta_2/2) * d^2A/dt^2 + gamma * |A|^2 * A = 0

  where:
    A(z,t)  = complex field envelope  [sqrt(W)]
    z       = propagation distance    [m]
    t       = retarded time (co-moving frame)  [s]
    beta_2  = GVD coefficient  [s^2/m]   (anomalous: beta_2 < 0)
    gamma   = nonlinear coefficient = n2*omega/(c*Aeff)  [1/(W*m)]
    n2      = nonlinear index of silica ~ 2.6e-20 m^2/W

  PHYSICAL ORIGIN OF TWO TERMS:
    beta_2 * d^2A/dt^2  : GROUP VELOCITY DISPERSION (linear)
      Disperses pulse: different freq components travel at different speeds.
      For anomalous dispersion (beta_2 < 0): RED components faster.
      Acts like: A_chirped(t) = A(t) * exp(i*phi_chirp(t))

    gamma * |A|^2 * A   : SELF-PHASE MODULATION (nonlinear)
      High-intensity center of pulse acquires MORE phase.
      Acts like: phi_NL(t) = gamma * P(t) * z   (instantaneous frequency shift)
      phi_NL < 0 at center -> RED shifts leading edge, BLUE shifts trailing.
      Opposite of anomalous GVD chirp -> THEY CAN CANCEL.

  SOLITON CONDITION (balance GVD and SPM):
    Characteristic lengths:
      L_D = T0^2 / |beta_2|    (dispersion length)
      L_NL = 1 / (gamma * P0)  (nonlinear length)
    Soliton number: N^2 = L_D / L_NL = gamma * P0 * T0^2 / |beta_2|
    N = 1: fundamental soliton (GVD and SPM exactly balance, no broadening)
    N = 2,3,...: higher-order solitons (periodic breathing)
""")

# Fundamental bright soliton
z_sym, t_sym, z0_sym = symbols('z t z_0', real=True)
T0, P0_sym = symbols('T_0 P_0', positive=True)
beta2_sym = symbols('beta_2', negative=True)
gamma_sym = symbols('gamma', positive=True)

# Bright soliton solution (anomalous dispersion: beta_2 < 0)
# A(z,t) = sqrt(P0) * sech(t/T0) * exp(i*z/(2*L_D))
# where L_D = T0^2/|beta_2|
# Use sech = 1/cosh
A_soliton_str = "A(z,t) = sqrt(P0) * sech(t/T0) * exp(i * |beta_2|*z / (2*T0^2))"
print(f"  BRIGHT SOLITON (fundamental, N=1):")
print(f"    {A_soliton_str}")
print(f"    Temporal profile: |A|^2 = P0 * sech^2(t/T0)  [hyperbolic secant]")
print(f"    Peak power: P0 = |beta_2| / (gamma * T0^2)")
print(f"    Propagates without distortion for ANY distance!")

# Numerical values for standard SMF-28 at 1550nm
T0_ps    = 1.0        # ps  (pulse FWHM ~ 1.76 * T0 ~ 1.76 ps)
beta2_val= -21e-27    # s^2/m = -21 ps^2/km  (anomalous)
gamma_val= 1.3e-3     # 1/(W*m) = 1.3 1/(W*km)
P0_val   = abs(beta2_val) / (gamma_val * (T0_ps*1e-12)**2)
LD_val   = (T0_ps*1e-12)**2 / abs(beta2_val)
LNL_val  = 1 / (gamma_val * P0_val)

print(f"""
  NUMERICAL EXAMPLE (SMF-28 @ 1550nm):
    T0       = {T0_ps} ps
    beta_2   = {beta2_val*1e27:.1f} ps^2/km  (anomalous)
    gamma    = {gamma_val*1e3:.2f} 1/(W*km)
    P0       = {P0_val:.2f} W  (soliton peak power)
    L_D      = {LD_val/1e3:.2f} km  (dispersion length)
    L_NL     = {LNL_val/1e3:.2f} km  (nonlinear length)
    N        = {np.sqrt(LD_val/LNL_val):.4f}  (must = 1 for fundamental soliton)
    FWHM     = {1.763 * T0_ps:.3f} ps
""")

print("""
  PEREGRINE SOLITON (ROGUE WAVE PROTOTYPE):
    The Peregrine soliton is an exact analytic solution of the NLSE
    that is LOCALIZED IN BOTH SPACE AND TIME -- appears from nowhere,
    disappears without trace (Akhmediev 1985; Peregrine 1983).

    On a continuous wave background A0 = sqrt(P0) * exp(i*gamma*P0*z):

    A_P(z,t) = sqrt(P0) * [1 - (4*(1 + 2*i*gamma*P0*z)) /
                               (1 + 4*(gamma*P0)^2*t^2 +
                                    4*(gamma*P0*z)^2)] * exp(i*gamma*P0*z)

    Peak amplitude: 3*sqrt(P0)   (3x the background -- FACTOR OF 3 ROGUE CRITERION)
    Peak intensity: 9*P0         (9x the background)
    Appears at:     z=0, t=0
    Duration:       ~1/(gamma*P0) in z, ~1/sqrt(gamma*P0) in t

    WHY THIS IS A ROGUE WAVE:
    - Modulation instability (MI) generates sidebands from noise
    - MI sidebands grow exponentially: gain g = gamma*P0 for freq <= sqrt(2*gamma*P0/|beta2|)
    - Extreme realizations of MI produce Peregrine-like events
    - Statistical: Peregrine amplitude 3x -> appears in tail of distribution
    - RogueGuard measures: I1(t), I2(t) from two dispersed copies
      GS reconstructs phase -> identifies Peregrine events real-time

    AKHMEDIEV BREATHER (periodic in z, localized in t):
    - Between Peregrine (one-shot) and Kuznetsov-Ma (periodic in t, localized in z)
    - General Akhmediev: A_AB involves elliptic functions
    - All are solutions of same NLSE -- same fiber, different initial conditions
""")

# Split-step Fourier method for soliton propagation
print(f"  SPLIT-STEP FOURIER METHOD (SSFM) -- numerical NLSE solver:")
print(f"    Operator form: dA/dz = (D_hat + N_hat) * A")
print(f"    D_hat: linear dispersion = -i*beta_2/2 * (d/dt)^2  (diagonal in freq domain)")
print(f"    N_hat: nonlinear       = i*gamma*|A|^2             (diagonal in time domain)")
print(f"    Split-step: A(z+h) ~ exp(h*D_hat) * exp(h*N_hat) * A(z)")
print(f"    Error: O(h^2) symmetric Strang splitting -- O(h^3) error")
print(f"    Algorithm per step:")
print(f"      1. Half-step dispersion: A_half = IFFT[exp(-i*beta2/2*(2pi*f)^2*(h/2)) * FFT[A]]")
print(f"      2. Full-step nonlinear: A_nl    = A_half * exp(i*gamma*|A_half|^2*h)")
print(f"      3. Half-step dispersion: A_next = IFFT[exp(-i*beta2/2*(2pi*f)^2*(h/2)) * FFT[A_nl]]")

# Implement SSFM
def ssfm_propagate(A0, t_grid, z_total, n_steps, beta2, gamma):
    """Split-step Fourier for NLSE. Returns A at z_total."""
    dt  = t_grid[1] - t_grid[0]
    N   = len(t_grid)
    dz  = z_total / n_steps
    freq = np.fft.fftfreq(N, d=dt)
    omega= 2 * np.pi * freq

    # Dispersion operator in frequency domain
    disp_half = np.exp(-1j * beta2/2 * omega**2 * dz/2)

    A = A0.copy().astype(complex)
    for _ in range(n_steps):
        # Half dispersion
        A = np.fft.ifft(disp_half * np.fft.fft(A))
        # Full nonlinear
        A = A * np.exp(1j * gamma * np.abs(A)**2 * dz)
        # Half dispersion
        A = np.fft.ifft(disp_half * np.fft.fft(A))
    return A

# Time grid for soliton simulation
N_t    = 2048
t_max  = 50e-12   # 50 ps window
t_grid = np.linspace(-t_max/2, t_max/2, N_t)
dt     = t_grid[1] - t_grid[0]

# Fundamental soliton
T0_s   = T0_ps * 1e-12
A_sol  = np.sqrt(P0_val) * (1/np.cosh(t_grid / T0_s))   # sech profile
# Add small noise to see MI/Peregrine
np.random.seed(42)
A_noisy = A_sol + 0.05*np.sqrt(P0_val) * (np.random.randn(N_t) + 1j*np.random.randn(N_t))

# Propagate 5 soliton periods
z_period = np.pi/2 * LD_val  # soliton period
z_prop   = 2 * z_period
n_steps  = 500

A_out_clean = ssfm_propagate(A_sol,    t_grid, z_prop, n_steps, beta2_val, gamma_val)
A_out_noisy = ssfm_propagate(A_noisy,  t_grid, z_prop, n_steps, beta2_val, gamma_val)

print(f"\n  SSFM simulation: propagated {z_prop/1e3:.2f} km ({2} soliton periods)")
print(f"    Input peak power:  {np.max(np.abs(A_sol)**2):.3f} W")
print(f"    Output peak power (clean): {np.max(np.abs(A_out_clean)**2):.3f} W")
max_rogue = np.max(np.abs(A_out_noisy)**2)
bg_power  = np.mean(np.abs(A_noisy)**2)
print(f"    Output peak power (noisy): {max_rogue:.3f} W  "
      f"(rogue factor: {max_rogue/bg_power:.1f}x background)")

# Peregrine soliton analytical
P0_per = 1.0   # normalized
z_arr  = np.linspace(-3, 3, 200)
t_arr  = np.linspace(-5, 5, 200)
Zg, Tg = np.meshgrid(z_arr, t_arr)
denom  = 1 + 4*Zg**2 + 4*Tg**2
A_per  = np.sqrt(P0_per) * np.abs(1 - 4*(1 + 2j*Zg) / denom) * np.exp(1j*Zg)
I_per  = np.abs(A_per)**2
print(f"\n  Peregrine soliton: max intensity = {np.max(I_per):.2f} * P0  (theory: 9.0)")

# ============================================================
# S2: DIELECTRIC ELECTRODYNAMICS
# ============================================================
print(f"\n{SEP}")
print("SECTION 2: DIELECTRIC ELECTRODYNAMICS")
print(SEP)

print("""
  MAXWELL EQUATIONS IN LINEAR DIELECTRIC MEDIA:
    D = epsilon_0 * epsilon_r * E      (electric displacement)
    B = mu_0 * mu_r * H               (magnetic field)
    Free charges/currents: rho_f, J_f  (usually zero in dielectric)

    Curl-H = dD/dt + J_f              (Ampere-Maxwell)
    Curl-E = -dB/dt                   (Faraday)
    Div-D  = rho_f                    (Gauss electric)
    Div-B  = 0                        (Gauss magnetic, no monopoles)

    Wave equation in dielectric (no free charges):
      nabla^2 E - (n^2/c^2) * d^2E/dt^2 = 0
      where n = sqrt(epsilon_r * mu_r)  (refractive index)
             c = 1/sqrt(epsilon_0 * mu_0)  (speed of light in vacuum)
    In dielectric: phase velocity v = c/n
    For silica: n ~ 1.45, v ~ 2.07e8 m/s

  COMPLEX PERMITTIVITY:
    Real dielectrics have absorption: epsilon(omega) = epsilon'(omega) - i*epsilon''(omega)
    Refractive index: N = n + i*k   (complex, k = extinction coefficient)
    N^2 = epsilon_r * mu_r   (mu_r ~ 1 for most optical materials)
    n^2 - k^2 = epsilon'
    2*n*k     = epsilon''

    Plane wave in lossy dielectric:
    E(z,t) = E0 * exp(i*(N*omega/c)*z - i*omega*t)
           = E0 * exp(-alpha*z/2) * exp(i*(n*omega/c)*z - i*omega*t)
    where alpha = 2*omega*k/c   [absorption coefficient, 1/m]
    Beer-Lambert: I(z) = I0 * exp(-alpha*z)

  LORENTZ OSCILLATOR MODEL:
    Bound electrons: damped harmonic oscillators driven by E-field.
    Equation of motion: m*(x'' + gamma*x' + omega0^2*x) = -e*E
    Chi(omega) = (n_e * e^2 / m*epsilon_0) * 1/(omega0^2 - omega^2 - i*gamma*omega)
    epsilon_r(omega) = 1 + Chi(omega)
      Real part:  1 + chi' = 1 + A*(omega0^2-omega^2)/[(omega0^2-omega^2)^2+(gamma*omega)^2]
      Imag part:  chi''    = A*gamma*omega/[(omega0^2-omega^2)^2+(gamma*omega)^2]
    NORMAL dispersion: dn/domega > 0  (away from resonance, most of visible)
    ANOMALOUS dispersion: dn/domega < 0  (near resonance peak)
    ANOMALOUS in fiber @ 1550nm: silica has zero-dispersion at 1310nm,
      anomalous at 1550nm -> enables soliton propagation!
""")

# Lorentz oscillator
omega_sym  = symbols('omega', real=True, positive=True)
omega0_sym = symbols('omega_0', positive=True)
A_sym, gam_sym = symbols('A gamma_L', positive=True)

chi_lorentz = A_sym * 1 / (omega0_sym**2 - omega_sym**2 - I*gam_sym*omega_sym)
chi_real = sp.re(chi_lorentz.rewrite(sp.re))

print(f"  Lorentz chi(omega) = A / (omega0^2 - omega^2 - i*gamma*omega)")
print(f"  At omega = omega0 (resonance): |chi| = A/(gamma*omega0) -> MAXIMUM absorption")
print(f"  For omega << omega0 (static): chi_0 = A/omega0^2  (static susceptibility)")

# Kramers-Kronig (qualitative print -- don't do full integral symbolically)
print("""
  KRAMERS-KRONIG RELATIONS (causality):
    Response function must be CAUSAL: chi(t) = 0 for t < 0
    (effect cannot precede cause)
    In frequency domain: chi(omega) must be analytic in upper half plane.
    This forces a CONSTRAINT between real and imaginary parts:

    n(omega) - 1 = (2/pi) * P.V. integral_0^oo [omega'*k(omega')/(omega'^2-omega^2)] domega'
    k(omega)     = -(2*omega/pi) * P.V. integral_0^oo [(n(omega')-1)/(omega'^2-omega^2)] domega'

    Physical meaning: if you know the ABSORPTION spectrum k(omega) at ALL omega,
    you can compute the REFRACTIVE INDEX n(omega) -- and vice versa.
    Used in: optical constants of materials, femtosecond pulse shaping, fiber design.
    RogueGuard: Sellmeier (empirical form of KK) gives n(lambda) for silica.
""")

# Sellmeier equation for silica (3-term)
# n^2 = 1 + sum_i A_i*lambda^2/(lambda^2 - B_i)
sellmeier_A = [0.6961663, 0.4079426, 0.8974794]
sellmeier_B = [0.0684043**2, 0.1162414**2, 9.896161**2]  # B in um^2

lam_um = np.linspace(0.5, 2.0, 500)  # wavelength in um
n2_sellmeier = np.ones_like(lam_um)
for A_s, B_s in zip(sellmeier_A, sellmeier_B):
    n2_sellmeier += A_s * lam_um**2 / (lam_um**2 - B_s)
n_sellmeier = np.sqrt(n2_sellmeier)

# Group velocity dispersion D = -lambda/c * d^2n/dlambda^2
dn_dlam    = np.gradient(n_sellmeier, lam_um)
d2n_dlam2  = np.gradient(dn_dlam, lam_um)
c_light    = 3e8   # m/s
lam_m      = lam_um * 1e-6
D_fiber    = -lam_m / c_light * d2n_dlam2 * 1e6  # ps/(nm*km)  [convert units]
# Actually: D [ps/(nm*km)] = -(lam[um]*1e3 / c[m/s]) * d2n/dlam2[1/um^2] * 1e12/1e9
lam_nm     = lam_um * 1e3  # nm
D_fiber_psnmkm = -(lam_nm * 1e-9) / c_light * np.gradient(np.gradient(n_sellmeier, lam_nm*1e-9), lam_nm*1e-9) * 1e12 / 1e-3

print(f"\n  SELLMEIER EQUATION (3-term Malitson fit for fused silica):")
print(f"    n^2 = 1 + 0.6962*lam^2/(lam^2-0.00468) + 0.4079*lam^2/(lam^2-0.01350)")
print(f"              + 0.8975*lam^2/(lam^2-97.93)   [lam in um]")
for lam_check in [0.633, 1.064, 1.310, 1.550]:
    idx = np.argmin(np.abs(lam_um - lam_check))
    print(f"    n({lam_check:.3f} um) = {n_sellmeier[idx]:.6f}")
idx1310 = np.argmin(np.abs(lam_um - 1.31))
idx1550 = np.argmin(np.abs(lam_um - 1.55))
print(f"    Zero-dispersion wavelength ~ 1.31 um (D crosses 0)")
print(f"    D(1.55 um) ~ {D_fiber_psnmkm[idx1550]:.1f} ps/(nm*km)  (anomalous, enables solitons)")

# ============================================================
# S3: MICHELSON-MORLEY EXPERIMENT
# ============================================================
print(f"\n{SEP}")
print("SECTION 3: MICHELSON-MORLEY EXPERIMENT (1887)")
print(SEP)

print("""
  HISTORICAL CONTEXT:
    1880s: light was known to be a wave. Waves need a medium.
    Proposed medium: luminiferous AETHER -- fills all space, rest frame of universe.
    Earth orbits Sun at v_E ~ 30 km/s relative to aether.
    If aether exists: light speed in Earth frame is NOT isotropic.
    Prediction: c + v_E  (heading with aether wind)
                c - v_E  (heading against aether wind)
    Albert Michelson (1881) and Edward Morley (1887) built the definitive test.

  INTERFEROMETER SETUP:
    Light source -> beamsplitter -> splits into two PERPENDICULAR arms.
    Arm 1 (length L): along aether wind (v_E)
    Arm 2 (length L): perpendicular to aether wind
    Each arm has a mirror at the end. Light reflects back, recombines.
    Interference fringes observed. Rotate apparatus 90 degrees.
    If c is isotropic: no fringe shift on rotation.
    If aether exists: fringe shift Delta = 2*L*v_E^2 / (lambda*c^2)

  TIME CALCULATION (classical):
    Arm 1 (parallel to v_E):
      t1 = L/(c-v) + L/(c+v) = 2*L*c/(c^2-v^2) = (2L/c)/(1-beta^2)
      where beta = v/c
    Arm 2 (perpendicular to v_E):
      Light travels diagonally. Effective speed = sqrt(c^2-v^2).
      t2 = 2*L / sqrt(c^2-v^2) = (2L/c)/sqrt(1-beta^2)
    Time difference:
      Delta_t = t1 - t2 = (2L/c) * [1/(1-beta^2) - 1/sqrt(1-beta^2)]
              ~ (L*v^2/c^3)  for v << c
    Path length difference: c*Delta_t ~ L*v^2/c^2
    Fringe shift (both arms counted twice -- rotate 90 deg):
      N_fringes = 2*L*v^2 / (lambda*c^2)
""")

# Fringe shift calculation
v_earth  = 30e3    # m/s (Earth orbital speed)
c_val    = 3e8     # m/s
L_arm    = 11.0    # m  (Michelson-Morley effective arm length with multiple reflections)
lam_val  = 590e-9  # m  (sodium D line)
beta_val = v_earth / c_val
N_fringes = 2 * L_arm * v_earth**2 / (lam_val * c_val**2)

print(f"\n  NUMERICAL PREDICTION:")
print(f"    v_Earth = {v_earth/1e3:.0f} km/s  (orbital velocity)")
print(f"    c       = {c_val/1e8:.1f} * 10^8 m/s")
print(f"    beta    = v/c = {beta_val:.2e}")
print(f"    beta^2  = {beta_val**2:.2e}")
print(f"    L_arm   = {L_arm:.1f} m  (effective, with mirrors)")
print(f"    lambda  = {lam_val*1e9:.0f} nm  (sodium D line)")
print(f"    Expected fringe shift N = 2*L*v^2/(lambda*c^2) = {N_fringes:.3f} fringes")
print(f"    Instrument precision: ~0.01 fringes")
print(f"    OBSERVED:             < 0.01 fringes  (NULL RESULT)")
print(f"    Ratio observed/expected < {0.01/N_fringes:.2f}  (< 2.5% of prediction)")

print("""
  CONSEQUENCES OF NULL RESULT:
    Lorentz-FitzGerald (1892): proposed moving bodies CONTRACT in direction of motion.
    L' = L * sqrt(1 - v^2/c^2)   --> this "explains" MM but is ad hoc.

    Einstein (1905): SPECIAL RELATIVITY (correct explanation).
    Postulate 1: Laws of physics are IDENTICAL in all inertial frames.
    Postulate 2: Speed of light c is CONSTANT in all inertial frames, regardless
                 of motion of source or observer.
    No aether needed. c is not relative to any medium. It's a fundamental constant.

    Consequences:
    Time dilation:     Delta_t' = Delta_t / sqrt(1-beta^2)  (moving clocks run slow)
    Length contraction: L'      = L * sqrt(1-beta^2)  (moving objects shorten)
    Relativistic mass: m = m0/sqrt(1-beta^2)
    E = m*c^2  (mass-energy equivalence -- most famous result)

  MODERN ANALOGS:
    LIGO gravitational wave detector (2015): Michelson interferometer, arm L=4km,
      sensitivity Delta_L/L = 10^-21  (1/1000 the proton radius)
      Uses laser power recycling, squeezed light (quantum noise limit)

    Atom interferometry: replace light with matter waves (de Broglie).
      Sensitivity to gravity gradients, tests of equivalence principle.

    Fiber Sagnac (RogueGuard): RELATED -- measures rotation, not translation.
      Phase difference = 4*pi*A*Omega / (lambda*c)
      where A = enclosed area, Omega = rotation rate.
      Same mathematical structure as MM, but measures ROTATION not wind.
""")

# ============================================================
# S4: PYTHON -> C DESIGN PATTERNS (DoD)
# ============================================================
print(f"\n{SEP}")
print("SECTION 4: PYTHON -> C DESIGN PATTERNS (DoD firmware)")
print(SEP)

print("""
  PHILOSOPHY: "Python at the top, C at the bottom"
    Design the API in Python (clear, type-annotated, testable).
    Profile to find the hot path.
    Rewrite ONLY the hot path in C (or Cython, or Rust/PyO3).
    Keep Python wrapper for safety, logging, orchestration.
    Never rewrite in C prematurely -- premature optimization is the root of all evil.

  LAYER ARCHITECTURE (for RogueGuard firmware):
    Python (control plane):
      - GS phase retrieval orchestration
      - ML inference (FNO, CNN alert classifier)
      - Logging, telemetry, SBIR demo interface
      - Unit tests, validation, rapid iteration
    C (data plane):
      - ADC DMA ring buffer management
      - FFT (using FFTW or ARM CMSIS-DSP)
      - Dispersion filter H(f) = exp(i*pi*D*f^2)
      - Triggering (rogue wave detection in real time)
      - Hard real-time constraints: < 1us interrupt latency

  METHOD 1: CTYPES (call existing compiled C library)
""")

ctypes_example = r"""
  # Python side: ctypes example
  import ctypes
  import numpy as np

  # Load compiled shared library (GS core, compiled with gcc -O3 -shared -fPIC)
  lib = ctypes.CDLL("./libgs_core.so")

  # Declare function signatures (CRITICAL: wrong types -> crash or silent UB)
  lib.gs_retrieve_phase.restype  = ctypes.c_int       # return: error code
  lib.gs_retrieve_phase.argtypes = [
      ctypes.POINTER(ctypes.c_double),  # I1: measured intensity 1
      ctypes.POINTER(ctypes.c_double),  # I2: measured intensity 2
      ctypes.c_int,                      # N: number of samples
      ctypes.c_double,                   # D: dispersion parameter
      ctypes.c_int,                      # n_iter: iterations
      ctypes.POINTER(ctypes.c_double),  # phi_out: recovered phase [output]
  ]

  # Call from Python
  N = 4096
  I1 = np.random.rand(N).astype(np.float64)  # MUST be contiguous C double array
  I2 = np.random.rand(N).astype(np.float64)
  phi_out = np.zeros(N, dtype=np.float64)

  ret = lib.gs_retrieve_phase(
      I1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
      I2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
      ctypes.c_int(N),
      ctypes.c_double(-7000.0),   # D in units of samples^2
      ctypes.c_int(50),           # n_iter
      phi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  if ret != 0:
      raise RuntimeError(f"gs_retrieve_phase returned error {ret}")
"""
print(ctypes_example)

print("""  /* C side: gs_core.c (DoD-informed style) */""")

c_gs_core = r"""
  /* gs_core.c - Gerchberg-Saxton phase retrieval core
   * Compiled: gcc -O3 -march=native -ffast-math -shared -fPIC -o libgs_core.so gs_core.c -lfftw3 -lm
   * MISRA-C subset: no dynamic allocation, no recursion, no VLAs.
   */

  #include <stdint.h>
  #include <math.h>
  #include <string.h>
  #include <fftw3.h>

  #define GS_MAX_N       65536    /* max buffer size -- static allocation */
  #define GS_OK          0
  #define GS_ERR_ARGS   -1
  #define GS_ERR_D_ZERO -2

  /* Static workspace -- no malloc in critical path (MISRA-C Rule 20.4) */
  static double _complex_buf1[GS_MAX_N * 2];   /* [re, im, re, im, ...] */
  static double _complex_buf2[GS_MAX_N * 2];

  int gs_retrieve_phase(
      const double *I1,      /* [N] intensity from fiber 1 (unit-amplitude source) */
      const double *I2,      /* [N] intensity from fiber 2 (dispersed copy) */
      int N,                 /* number of samples */
      double D,              /* dispersion parameter |D| >= 5000 required */
      int n_iter,            /* GS iterations, >= 50 recommended */
      double *phi_out)       /* [N] recovered phase [output] */
  {
      /* Input validation -- public API always validates (defense in depth) */
      if (I1 == NULL || I2 == NULL || phi_out == NULL) return GS_ERR_ARGS;
      if (N <= 0 || N > GS_MAX_N)                      return GS_ERR_ARGS;
      if (n_iter <= 0)                                  return GS_ERR_ARGS;
      if (fabs(D) < 5000.0)                             return GS_ERR_D_ZERO;

      /* Initialize FFTW plans (plan once, execute many -- expensive to create) */
      /* In production: plans are created once at startup, stored in struct */
      fftw_complex *in  = (fftw_complex *)_complex_buf1;
      fftw_complex *out = (fftw_complex *)_complex_buf2;
      fftw_plan fwd = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD,  FFTW_ESTIMATE);
      fftw_plan inv = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

      /* Build dispersion filter H[k] = exp(i*pi*D*k^2/N^2) */
      double H_re[GS_MAX_N], H_im[GS_MAX_N];
      for (int k = 0; k < N; k++) {
          double kn   = (double)k / N;
          double phase = M_PI * D * kn * kn;
          H_re[k]     = cos(phase);
          H_im[k]     = sin(phase);
      }

      /* Initialize: A = sqrt(I1) * exp(i*0)  (unit amplitude, zero phase) */
      double A_re[GS_MAX_N], A_im[GS_MAX_N];
      for (int n = 0; n < N; n++) {
          A_re[n] = sqrt(I1[n]);   /* unit-amplitude constraint */
          A_im[n] = 0.0;
      }

      /* GS iteration */
      for (int iter = 0; iter < n_iter; iter++) {
          /* 1. FFT: A(t) -> A_hat(f) */
          for (int n = 0; n < N; n++) { in[n][0]=A_re[n]; in[n][1]=A_im[n]; }
          fftw_execute(fwd);
          /* Normalize */
          for (int k = 0; k < N; k++) { out[k][0]/=N; out[k][1]/=N; }

          /* 2. Apply dispersion filter: A_hat *= H */
          for (int k = 0; k < N; k++) {
              double ar = out[k][0], ai = out[k][1];
              out[k][0] = ar*H_re[k] - ai*H_im[k];
              out[k][1] = ar*H_im[k] + ai*H_re[k];
          }

          /* 3. IFFT: back to time domain */
          for (int k = 0; k < N; k++) { in[k][0]=out[k][0]; in[k][1]=out[k][1]; }
          fftw_execute(inv);

          /* 4. CONSTRAINT: replace amplitude with sqrt(I2), keep phase */
          for (int n = 0; n < N; n++) {
              double re = out[n][0], im = out[n][1];
              double mag = sqrt(re*re + im*im);
              if (mag < 1e-15) mag = 1e-15;   /* avoid divide by zero */
              double scale = sqrt(I2[n]) / mag;
              A_re[n] = re * scale;
              A_im[n] = im * scale;
          }

          /* 5. IFFT dispersion: back to reference frame */
          /* (apply H* = conj(H) to undo dispersion) */
          for (int n = 0; n < N; n++) { in[n][0]=A_re[n]; in[n][1]=A_im[n]; }
          fftw_execute(fwd);
          for (int k = 0; k < N; k++) { out[k][0]/=N; out[k][1]/=N; }
          for (int k = 0; k < N; k++) {
              double ar = out[k][0], ai = out[k][1];
              out[k][0] = ar*H_re[k] + ai*H_im[k];    /* H* = (re, -im) */
              out[k][1] =-ar*H_im[k] + ai*H_re[k];
          }
          for (int k = 0; k < N; k++) { in[k][0]=out[k][0]; in[k][1]=out[k][1]; }
          fftw_execute(inv);

          /* 6. UNIT-AMPLITUDE constraint: replace with sqrt(I1), keep phase */
          for (int n = 0; n < N; n++) {
              double re = out[n][0], im = out[n][1];
              double mag = sqrt(re*re + im*im);
              if (mag < 1e-15) mag = 1e-15;
              double scale = sqrt(I1[n]) / mag;
              A_re[n] = re * scale;
              A_im[n] = im * scale;
          }
      }

      /* Output: phase of final A */
      for (int n = 0; n < N; n++) {
          phi_out[n] = atan2(A_im[n], A_re[n]);
      }

      fftw_destroy_plan(fwd);
      fftw_destroy_plan(inv);
      return GS_OK;
  }
"""
print(c_gs_core)

print("""
  DoD / MISRA-C RULES DEMONSTRATED ABOVE:
    Rule 20.4: No dynamic allocation (malloc/free) in safety-critical code.
               Static buffers with fixed MAX_N instead.
    Rule 17.2: No recursion. (GS iteration is a loop, not recursive)
    Rule 15.5: Single point of exit. (multiple returns at top for validation only)
    Rule 10.x: Explicit casts when mixing integer/float.
    Defensive: every pointer checked for NULL before dereference.
    Error codes: int return with #defined error constants (not magic numbers).
    Constants: #define GS_MAX_N -- no magic numbers in code body.
    Bounds: N > GS_MAX_N rejected -- no buffer overrun possible.
    Fuzzing: add libfuzzer harness for CI/CD: send random N, D, n_iter.
    Formal verification: Frama-C WP plugin can verify absence of UB for C99.

  CFFI ALTERNATIVE (cleaner Python bindings):
    from cffi import FFI
    ffi = FFI()
    ffi.cdef(open("gs_core.h").read())   # parse the C header
    lib = ffi.dlopen("./libgs_core.so")
    # Call exactly as above but no ctypes.POINTER boilerplate
    # CFFI handles NULL checks, type safety at Python level

  CYTHON ALTERNATIVE (C extension module, import as pure Python):
    gs_core.pyx:
      import numpy as np
      cimport numpy as cnp
      def retrieve_phase(cnp.ndarray[double,ndim=1] I1, ...):
          # Cython generates C code, compiled to .pyd/.so
          # Typesafe, debuggable, fastest Python-C bridge
""")

# ============================================================
# S5: CONCRETE + STRUCTURAL (3-story building)
# ============================================================
print(f"\n{SEP}")
print("SECTION 5: CONCRETE + STRUCTURAL (3-STORY BUILDING)")
print(SEP)

print("""
  WHY CONCRETE:
    Compressive strength: fc' = 3000-8000 psi (20-55 MPa) depending on mix.
    Tensile strength:     ft  ~ 0.1 * fc'  (WEAK -- needs rebar)
    Young's modulus:      Ec  = 57000 * sqrt(fc' [psi]) psi  (ACI 318)
    Density:              ~150 pcf (2400 kg/m^3)
    Portland cement + aggregate + water. Water/cement ratio controls strength.
    Curing time: 28-day strength (fc' is measured at 28 days).

  LOAD PATH IN A 3-STORY BUILDING:
    FLOOR SLAB -> BEAM -> COLUMN -> FOOTING -> SOIL
    Each element must carry all loads ABOVE it plus its own weight.
    "Should've not been cast" -- concrete that wasn't properly cured, has
    voids (honeycombing), or was placed in frozen conditions: DEFECTIVE.

  LOADS (ACI 318 / IBC):
    Dead load (D): weight of structure itself.
      Slab (6"): 75 psf.  Beam+column: 20 psf.  Total dead: ~95 psf.
    Live load (L): occupancy.
      Office: 50 psf.  Residential: 40 psf.  Assembly: 100 psf.
    Design load: U = 1.2*D + 1.6*L  (ACI factored strength design, USD/LRFD)

  3-STORY COLUMN LOAD EXAMPLE (10x10 ft tributary area per column):
""")

# Column load calculation
area_trib   = 10 * 10        # ft^2
D_psf       = 95.0           # psf dead load
L_psf       = 50.0           # psf live load (office)
n_floors    = 3              # 3 stories
U_psf       = 1.2*D_psf + 1.6*L_psf  # factored load per floor

print(f"  Tributary area: {area_trib} ft^2")
print(f"  Dead load D:    {D_psf:.0f} psf")
print(f"  Live load L:    {L_psf:.0f} psf")
print(f"  Factored U:     1.2*{D_psf:.0f} + 1.6*{L_psf:.0f} = {U_psf:.0f} psf")
print(f"  Column carries {n_floors} floors:")
for floor in range(1, n_floors+1):
    load_floor = U_psf * area_trib * floor
    print(f"    Floors above floor {floor}: P = {U_psf:.0f}*{area_trib}*{floor} = {load_floor:.0f} lb = {load_floor/1000:.1f} kips")

P_total = U_psf * area_trib * n_floors  # lb, at ground floor column
print(f"\n  Ground floor column carries: {P_total:.0f} lb = {P_total/1000:.1f} kips")

# Euler column buckling
E_concrete = 57000 * (4000**0.5)  # psi (fc'=4000 psi)
b_col = 12    # inches (square column)
I_col = b_col**4 / 12   # in^4
L_col_ft = 12   # ft story height
L_col_in = L_col_ft * 12   # inches
K_eff = 1.0    # fixed-free = 2, pin-pin = 1, fixed-pin = 0.7
Pcr   = np.pi**2 * E_concrete * I_col / (K_eff * L_col_in)**2  # lb

print(f"""
  EULER COLUMN BUCKLING:
    Pcr = pi^2 * E * I / (K*L)^2
    E_concrete = 57000*sqrt({4000}) = {E_concrete/1e6:.2f} * 10^6 psi
    Column: {b_col}x{b_col} in square, I = {b_col}^4/12 = {I_col:.1f} in^4
    Story height L = {L_col_ft} ft = {L_col_in} in,  K = {K_eff} (pin-pin)
    Pcr = {Pcr/1e3:.0f} kips  (plain concrete, before rebar)
    Applied P = {P_total/1000:.1f} kips
    Safety factor = Pcr/P = {Pcr/P_total:.1f}  (need > 2.0, add rebar if insufficient)
""")

print("""
  INTERNATIONAL SEISMIC/WIND CODES:
    JAPAN (Tokyo): JIS/BCJ -- very high seismic zone. Moment frames, base isolation.
      Base shear V = Cs * W,  Cs ~ 0.2g for zone 3 (Tokyo).
      Buildings must survive Kobe-scale (7.2M) earthquakes.
      Innovative: tuned mass dampers in Tokyo skyscrapers (Shinjuku Park Tower).

    GERMANY (Frankfurt/Berlin): Eurocode 8 (EC8) -- moderate seismic, high wind.
      EC8 design: q-factor (behavior factor) ~ 3.0-4.5 for ductile RC frames.
      DIN 1045: German concrete standard, tight mix design control.
      German engineering tradition: conservative, over-engineered, long service life.

    COLOMBIA (Bogota/Medellin): NSR-10 -- very high seismic (Pacific Ring of Fire).
      Zone Aa: peak ground acceleration 0.25g-0.50g.
      SCBF (special concentric braced frame) or special moment frames required.
      Low-income construction: unreinforced masonry (adobe) KILLS PEOPLE.
      NSR-10 mandates seismic design even for small buildings after 1999 Armenia quake.
""")

# ============================================================
# MATPLOTLIB -- 6-PANEL FIGURE
# ============================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(19, 13))
fig.patch.set_facecolor("#F6F6F0")
gs0 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36,
                        top=0.93, bottom=0.06, left=0.06, right=0.97)

ax_sol  = fig.add_subplot(gs0[0, 0])
ax_per  = fig.add_subplot(gs0[0, 1])
ax_diel = fig.add_subplot(gs0[0, 2])
ax_mm   = fig.add_subplot(gs0[1, 0])
ax_c    = fig.add_subplot(gs0[1, 1])
ax_col  = fig.add_subplot(gs0[1, 2])

fig.suptitle("Solitons (NLS/Peregrine) | Dielectric EM | Michelson-Morley | "
             "Python->C | Concrete Structures",
             fontsize=11, fontweight="bold", color="#1a1a2e")

# ---- AX_SOL: soliton SSFM propagation ----
ax = ax_sol
ax.set_facecolor("#F0F8FF")
t_ps = t_grid * 1e12
ax.plot(t_ps, np.abs(A_sol)**2,      "#1f77b4", lw=1.5, alpha=0.7, label="Input sech^2 (clean)")
ax.plot(t_ps, np.abs(A_out_clean)**2,"#2ca02c", lw=1.5, label="Output (clean, N=1 soliton)")
ax.plot(t_ps, np.abs(A_out_noisy)**2,"#d62728", lw=1.0, alpha=0.8, label="Output (5% noise -> rogue)")
ax.set_xlim(-20, 20)
ax.set_xlabel("Time (ps)", fontsize=9)
ax.set_ylabel("Power (W)", fontsize=9)
ax.set_title("NLSE SSFM: Fiber Soliton + Rogue Onset", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(alpha=0.2)
ax.text(0.02, 0.97, f"beta2={beta2_val*1e27:.0f} ps^2/km\ngamma={gamma_val*1e3:.1f}/(W*km)\nP0={P0_val:.1f}W",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_PER: Peregrine soliton 2D intensity ----
ax = ax_per
im = ax.contourf(z_arr, t_arr, I_per, levels=30, cmap="inferno")
ax.set_xlabel("z (normalized)", fontsize=9)
ax.set_ylabel("t (normalized)", fontsize=9)
ax.set_title("Peregrine Soliton |A|^2\n(Rogue Wave Prototype: 9x background)", fontsize=10)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Intensity", fontsize=8)
ax.axhline(0, color="w", lw=0.8, ls="--", alpha=0.5)
ax.axvline(0, color="w", lw=0.8, ls="--", alpha=0.5)
ax.text(0.05, 0.95, "Appears/disappears\nfrom nowhere",
        transform=ax.transAxes, fontsize=8, va="top", color="white",
        bbox=dict(fc="#333", ec="none", alpha=0.6, pad=2))

# ---- AX_DIEL: Sellmeier n(lambda) and D(lambda) ----
ax = ax_diel
ax.set_facecolor("#FFFFF0")
color_n  = "#1f77b4"
color_D  = "#d62728"
ax2_diel = ax.twinx()
ax.plot(lam_um, n_sellmeier, color_n, lw=2.0, label="n(lambda)")
ax2_diel.plot(lam_um, D_fiber_psnmkm, color_D, lw=2.0, label="D(lambda)")
ax2_diel.axhline(0, color="#999", lw=0.8, ls="--")
ax2_diel.axvline(1.31, color="#888", lw=0.8, ls=":", alpha=0.7)
ax.set_xlabel("Wavelength (um)", fontsize=9)
ax.set_ylabel("Refractive index n", color=color_n, fontsize=9)
ax2_diel.set_ylabel("D (ps/nm/km)", color=color_D, fontsize=9)
ax.set_title("Silica Sellmeier n(lambda)\n& GVD D(lambda)", fontsize=10)
ax.tick_params(axis="y", labelcolor=color_n)
ax2_diel.tick_params(axis="y", labelcolor=color_D)
ax.set_xlim(0.5, 2.0)
ax2_diel.set_ylim(-200, 100)
ax.text(1.32, 1.451, "ZDW\n1.31um", fontsize=7.5, color="#888")
ax.text(1.56, 1.445, "1550nm\nn=1.444", fontsize=7.5, color=color_n)

# ---- AX_MM: Michelson-Morley fringe shift calculation ----
ax = ax_mm
ax.set_facecolor("#FFF5E0")
v_range  = np.linspace(0, 50e3, 200)   # m/s
N_fr_range = 2 * L_arm * v_range**2 / (lam_val * c_val**2)
ax.plot(v_range/1e3, N_fr_range, "#1f77b4", lw=2.0)
ax.axvline(30, color="#d62728", lw=1.2, ls="--", label="v_Earth=30km/s")
ax.axhline(N_fringes, color="#d62728", lw=0.8, ls="--")
ax.axhline(0.01, color="#2ca02c", lw=1.2, ls=":", label="Instrument limit 0.01")
ax.fill_between(v_range/1e3, 0, 0.01, color="#2ca02c", alpha=0.15)
ax.set_xlabel("Aether wind speed (km/s)", fontsize=9)
ax.set_ylabel("Expected fringe shift N", fontsize=9)
ax.set_title("Michelson-Morley (1887)\nNull Result -> Special Relativity", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.05, 0.92, f"Predicted: {N_fringes:.2f} fringes\nObserved: < 0.01 fringes\n-> Aether REJECTED",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(fc="#fff0d0", ec="#d62728", pad=3))

# ---- AX_C: GS phase retrieval architecture diagram ----
ax = ax_c
ax.set_facecolor("#F5F0FF")
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
ax.set_title("Python->C GS Firmware Architecture", fontsize=10)

layers = [
    (1, 6.5, 8, 1.0, "#4e79a7", "Python (control plane): GS orchestration, FNO inference, logging"),
    (1, 5.2, 8, 1.0, "#f28e2b", "ctypes / CFFI: Python-C bridge (type-safe, validated inputs)"),
    (1, 3.9, 8, 1.0, "#e15759", "C gs_core (data plane): FFT, H(f), GS loop -- MISRA-C, no malloc"),
    (1, 2.6, 8, 1.0, "#76b7b2", "FFTW3 / ARM CMSIS-DSP: optimized FFT kernel"),
    (1, 1.3, 8, 1.0, "#59a14f", "Hardware: RPi CM4, dual ADC DMA, fiber tap -- RogueGuard 1U"),
]
for x, y, w, h, color, label in layers:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333",
                          linewidth=1.2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=7.8, color="white", fontweight="bold", wrap=True)
    # Arrow connecting layers
    if y > 1.3:
        ax.annotate("", xy=(5, y), xytext=(5, y+0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0))

for y_arr in [6.5, 5.2, 3.9, 2.6]:
    ax.annotate("", xy=(5, y_arr), xytext=(5, y_arr+1.0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))

ax.text(0.5, 0.08, "DoD: MISRA-C | No malloc | Formal verify (Frama-C) | SBIR Phase I",
        transform=ax.transAxes, fontsize=7.5, ha="center", color="#444",
        bbox=dict(fc="#eee", ec="#bbb", pad=2))

# ---- AX_COL: column load by story + buckling ----
ax = ax_col
ax.set_facecolor("#F0FFF0")
stories    = np.arange(1, n_floors+2)
loads_kips = np.array([U_psf * area_trib * (n_floors - s + 1) / 1000 for s in stories])
loads_kips[-1] = 0  # top
bar_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
for i in range(n_floors):
    ax.barh(i, loads_kips[i], color=bar_colors[i], edgecolor="k", lw=0.8, height=0.6)
    ax.text(loads_kips[i]+0.5, i, f"{loads_kips[i]:.1f} k",
            va="center", fontsize=9)

ax.axvline(Pcr/1e3, color="#d62728", lw=1.5, ls="--", label=f"Pcr={Pcr/1e3:.0f} k (Euler)")
ax.set_yticks(range(n_floors))
ax.set_yticklabels([f"Floor {i+1} column" for i in range(n_floors)])
ax.set_xlabel("Column axial load (kips)", fontsize=9)
ax.set_title(f"3-Story Column Loads\n(10x10 ft trib, D={D_psf:.0f}, L={L_psf:.0f} psf)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="x", alpha=0.2)
ax.set_xlim(0, max(loads_kips)*1.4)

# Inset: concrete stress-strain
ax_ins = ax.inset_axes([0.55, 0.05, 0.43, 0.38])
eps_arr = np.linspace(0, 0.004, 200)
# Parabola-linear concrete model (ACI)
eps_c0 = 0.002
fc_prime = 4000  # psi
sig_arr  = np.where(eps_arr <= eps_c0,
                    0.85*fc_prime * (2*eps_arr/eps_c0 - (eps_arr/eps_c0)**2),
                    0.85*fc_prime * np.ones_like(eps_arr))
sig_arr[eps_arr > 0.003] = 0   # crushing strain
ax_ins.plot(eps_arr*1e3, sig_arr/1e3, "#2ca02c", lw=1.5)
ax_ins.axvline(2, color="#888", lw=0.8, ls="--")
ax_ins.axvline(3, color="#d62728", lw=0.8, ls="--")
ax_ins.set_xlabel("Strain (x10^-3)", fontsize=7)
ax_ins.set_ylabel("f'c (ksi)", fontsize=7)
ax_ins.set_title("Concrete\nStress-Strain", fontsize=7.5)
ax_ins.grid(alpha=0.2)

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
