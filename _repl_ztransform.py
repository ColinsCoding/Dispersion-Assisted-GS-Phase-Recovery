"""
_repl_ztransform.py -- Z-transform, Laplace, Fourier: one unified picture
+ Noether's theorem: symmetry -> conservation law (numerical demo)
"""
import numpy as np
import sympy as sp
import pandas as pd

# ============================================================
# 1. The big picture: continuous vs discrete
# ============================================================
print("=== The Unified Transform Picture ===")
print("""
Signal domain     Transform         Variable    Poles live on
-----------       ---------         --------    -------------
Continuous time   Laplace           s = sigma + j*omega   s-plane
Continuous time   Fourier (CTFT)    j*omega     imaginary axis (s-plane slice)
Discrete time     Z-transform       z = r*exp(j*omega)    z-plane
Discrete time     DTFT              exp(j*omega)          unit circle (z-plane slice)
Discrete time     DFT/FFT           exp(j*2*pi*k/N)       N points on unit circle

Key substitution:  z = exp(s*T)   where T = sampling period
  -> unit circle (|z|=1)  ==  imaginary axis (sigma=0, stable boundary)
  -> inside unit circle   ==  left half s-plane (stable)
  -> outside unit circle  ==  right half s-plane (unstable)
""")

# ============================================================
# 2. Z-transform of common sequences (SymPy)
# ============================================================
print("=== Z-Transform pairs (SymPy symbolic) ===")
z, n, a, T_s = sp.symbols('z n a T', positive=True)

pairs = [
    ("delta[n]",        "1",                     "z/(z-1) at n=0 -> 1"),
    ("u[n] (step)",     "z/(z-1)",               "pole at z=1"),
    ("a^n * u[n]",      "z/(z-a)",               "pole at z=a"),
    ("n * a^n * u[n]",  "a*z/(z-a)^2",           "double pole at z=a"),
    ("cos(w0*n)*u[n]",  "z(z-cos w0)/(z^2-2z*cos(w0)+1)", "poles at z=exp(+-j*w0)"),
    ("exp(-a*n)*u[n]",  "z/(z-exp(-a))",         "discrete decay, pole at exp(-a)"),
]
print(f"  {'x[n]':30s}  {'X(z)':40s}  note")
print("  " + "-"*90)
for xn, Xz, note in pairs:
    print(f"  {xn:30s}  {Xz:40s}  {note}")
print()

# ============================================================
# 3. Calculus <-> Z-transform operators
# ============================================================
print("=== Differentiation analogy ===")
print("""
Continuous (Laplace)          Discrete (Z-transform)
--------------------          ----------------------
d/dt  <->  multiply by s      forward diff x[n+1]-x[n]  <->  (z-1)*X(z)
integral  <->  divide by s    accumulator sum x[k]       <->  X(z)/(1-z^-1)
delay by T  <->  exp(-sT)     delay by 1 sample          <->  z^-1 * X(z)
""")
print("Euler method: s ~= (z-1)/T   (first order, forward)")
print("Tustin (bilinear): s = (2/T)*(z-1)/(z+1)  (second order, preserves freq)")
print()

# ============================================================
# 4. Numerical: pole-zero of a simple IIR filter
# ============================================================
print("=== Pole-zero: H(z) = z / (z - 0.9) ===")
omega = np.linspace(0, 2*np.pi, 512)
z_vals = np.exp(1j * omega)
pole = 0.9

H = z_vals / (z_vals - pole)
mag_dB = 20 * np.log10(np.abs(H))

peak_idx = np.argmax(mag_dB)
print(f"  Peak gain: {mag_dB[peak_idx]:.1f} dB at omega = {omega[peak_idx]:.3f} rad/sample")
print(f"  DC gain (omega=0): {20*np.log10(abs(1/(1-pole))):.1f} dB")
print(f"  Nyquist (omega=pi): {20*np.log10(abs(-1/(-1-pole))):.1f} dB")
print(f"  Pole at z={pole} -> time constant tau = {-1/np.log(pole):.1f} samples")
print()

# ============================================================
# 5. de Broglie + Z-transform: sampling a quantum wave
# ============================================================
print("=== de Broglie breakdown: when does discrete sampling matter? ===")
print("""
de Broglie:  lambda = h/p = h / (m*v)

Nyquist says: sample at T <= lambda/2 to avoid aliasing.
Below that, the discrete Z-transform and continuous Fourier are equivalent.
Above that, you get aliasing -- the particle 'looks' like it has lower momentum.

Jalali lab: temporal sampling at T ~ 1 ps  (1 THz ADC equivalent)
  -> Nyquist limit: lambda_min ~ 2 * c * T ~ 0.6 mm  (RF, not optical)
  -> Optical (lambda ~ 1550 nm): way above Nyquist, sampled in INTENSITY not field
  -> That's why you need GS: intensity |E|^2 loses phase, GS recovers it

The 14.4 V at 1 Angstrom:
  k*e / (1e-10 m) = 8.988e9 * 1.602e-19 / 1e-10 = 14.4 V
  This is the natural voltage unit in atomic physics.
  Electron binding energies in eV -> directly in volts at angstrom scale.
  Chemistry = electrostatics at 1-10 V across 1-10 Angstrom gaps.
""")

# ============================================================
# 6. Noether's theorem: symmetry -> conservation (numerical)
# ============================================================
print("=== Noether's Theorem: symmetry -> conservation law ===")
print("""
Theorem: every continuous symmetry of the action S = integral L dt
         has a corresponding conserved quantity.

Symmetry              Conserved quantity     In GS/ML
-----------           ------------------     --------
Time translation      Energy                 Loss landscape time-invariant
Space translation     Linear momentum        Shift-invariant conv filters (CNN)
Rotation              Angular momentum       Rotational equiv (equivariant nets)
U(1) phase rotation   Electric charge        Global phase ambiguity in GS
U(1): E -> E*exp(ij)  |E|^2 unchanged        wrapped-phase loss invariant

GS has U(1) symmetry: phi and phi+C recover the same intensities I1, I2.
That's why RMS vs ground truth needs an offset correction:
  offset = angle(mean(exp(i*(phi_true - phi_hat))))
  phi_aligned = phi_hat + offset
""")

# numerical demo: conserved quantity under time translation
print("Numerical: harmonic oscillator, check energy conservation")
dt = 0.01
t_end = 10.0
steps = int(t_end / dt)
omega0 = 2.0   # rad/s
x  = np.zeros(steps); v = np.zeros(steps)
x[0] = 1.0; v[0] = 0.0

# symplectic Euler (energy-conserving integrator)
for i in range(steps - 1):
    v[i+1] = v[i] - omega0**2 * x[i] * dt
    x[i+1] = x[i] + v[i+1] * dt

KE   = 0.5 * v**2
PE   = 0.5 * omega0**2 * x**2
E_total = KE + PE

print(f"  Initial energy:  {E_total[0]:.6f}")
print(f"  Final energy:    {E_total[-1]:.6f}")
print(f"  Max drift:       {np.max(np.abs(E_total - E_total[0])):.2e}  <- conserved")
print(f"  Time symmetry -> energy conservation. Noether confirmed.")
print()

# U(1) symmetry in GS
print("U(1) phase symmetry in GS:")
rng = np.random.default_rng(0)
phi = rng.uniform(0, 2*np.pi, 64)
E   = np.exp(1j * phi)
for theta in [0.0, np.pi/4, np.pi/2, np.pi]:
    E_rot = E * np.exp(1j * theta)
    I_orig = np.abs(E)**2
    I_rot  = np.abs(E_rot)**2
    diff = np.max(np.abs(I_orig - I_rot))
    print(f"  theta={theta:.3f} rad  ->  max |I_rot - I_orig| = {diff:.2e}  (invariant)")

print()
print("This is why GS always has a global phase ambiguity.")
print("Noether: U(1) symmetry of |E|^2 -> phase is not an observable.")
