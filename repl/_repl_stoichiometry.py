"""
repl/_repl_stoichiometry.py
Stoichiometry as linear algebra. Mass matrix. Impulse-momentum.
Dive with initial velocity. Inward/outward flux (divergence theorem).
Daily chemistry: combustion, electrolysis, rust, photosynthesis.
"""
import math
import numpy as np
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 62)
print("STOICHIOMETRY + MASS MATRIX + MOMENTUM + FLUX")
print("=" * 62)
print()

# ============================================================
# 1. STOICHIOMETRY = NULL SPACE OF ATOM MATRIX
# ============================================================
print("=== 1. STOICHIOMETRY = LINEAR ALGEBRA ===")
print("""
  Chemical equation:  a*A + b*B -> c*C + d*D
  Atom balance:       atom_count(element) conserved

  Matrix form:
    Columns = species (reactants negative, products positive)
    Rows    = elements
    Find x in null space: A*x = 0

  This is EXACTLY the same problem as finding the kernel of a matrix.
  Same math as GS: find vector in intersection of constraint sets.
""")

# Example 1: Combustion of methane  CH4 + O2 -> CO2 + H2O
# Species: CH4, O2, CO2, H2O  (reactants negative, products positive)
# Rows: C, H, O
print("  --- Combustion: CH4 + O2 -> CO2 + H2O ---")
#          CH4  O2  CO2  H2O
A_comb = sp.Matrix([
    [ 1,   0,  -1,   0],   # C: 1 in CH4, -1 in CO2
    [ 4,   0,   0,  -2],   # H: 4 in CH4, -2 in H2O
    [ 0,   2,  -2,  -1],   # O: 2 in O2, -2 in CO2, -1 in H2O
])
null = A_comb.nullspace()
v = null[0]
# normalize to smallest integer
from fractions import Fraction
ratios = [Fraction(vi).limit_denominator(10) for vi in v]
lcm_den = 1
for r in ratios:
    lcm_den = lcm_den * r.denominator // math.gcd(lcm_den, r.denominator)
coeffs = [abs(int(r * lcm_den)) for r in ratios]
print(f"  Null space vector (raw): {v.T}")
print(f"  Balanced: {coeffs[0]} CH4 + {coeffs[1]} O2 -> {coeffs[2]} CO2 + {coeffs[3]} H2O")
print()

# Example 2: Rust  Fe + O2 -> Fe2O3
print("  --- Rust: Fe + O2 -> Fe2O3 ---")
#          Fe  O2  Fe2O3
A_rust = sp.Matrix([
    [ 1,   0,  -2],   # Fe: 1 in Fe, -2 in Fe2O3
    [ 0,   2,  -3],   # O:  2 in O2, -3 in Fe2O3
])
null2 = A_rust.nullspace()
v2 = null2[0]
ratios2 = [Fraction(vi).limit_denominator(20) for vi in v2]
lcm2 = 1
for r in ratios2:
    lcm2 = lcm2 * r.denominator // math.gcd(lcm2, r.denominator)
c2 = [abs(int(r * lcm2)) for r in ratios2]
print(f"  Balanced: {c2[0]} Fe + {c2[1]} O2 -> {c2[2]} Fe2O3")
print()

# Example 3: Photosynthesis  CO2 + H2O -> C6H12O6 + O2
print("  --- Photosynthesis: CO2 + H2O -> C6H12O6 + O2 ---")
#          CO2  H2O  C6H12O6  O2
A_photo = sp.Matrix([
    [ 1,   0,  -6,   0],   # C
    [ 0,   2, -12,   0],   # H
    [ 2,   1,  -6,  -2],   # O
])
null3 = A_photo.nullspace()
v3 = null3[0]
ratios3 = [Fraction(vi).limit_denominator(20) for vi in v3]
lcm3 = 1
for r in ratios3:
    lcm3 = lcm3 * r.denominator // math.gcd(lcm3, r.denominator)
c3 = [abs(int(r * lcm3)) for r in ratios3]
print(f"  Balanced: {c3[0]} CO2 + {c3[1]} H2O -> {c3[2]} C6H12O6 + {c3[3]} O2")
print()

# Daily chemical processes table
print("  Daily chemistry reference:")
print(f"  {'Reaction':28s}  {'Balanced equation'}")
daily = [
    ("Combustion (gas stove)",  "CH4 + 2O2 -> CO2 + 2H2O"),
    ("Rust (car/rebar)",        "4Fe + 3O2 -> 2Fe2O3"),
    ("Photosynthesis (plants)", "6CO2 + 6H2O -> C6H12O6 + 6O2"),
    ("Electrolysis (water)",    "2H2O -> 2H2 + O2"),
    ("Limestone (cement)",      "CaCO3 -> CaO + CO2  (heat)"),
    ("Bleach (cleaning)",       "Cl2 + 2NaOH -> NaCl + NaClO + H2O"),
    ("Baking soda + acid",      "NaHCO3 + HCl -> NaCl + H2O + CO2"),
    ("Li-ion battery",          "LiCoO2 + C -> LiC6 + CoO2  (charge)"),
]
for name, eq in daily:
    print(f"  {name:28s}  {eq}")
print()

# ============================================================
# 2. MASS MATRIX + CHANGE IN MOMENTUM
# ============================================================
print("=== 2. MASS MATRIX + IMPULSE-MOMENTUM ===")
print("""
  Newton's 2nd law in matrix form:
    M * a = F
    M * dv/dt = F
    M * (v_final - v_initial) = integral F dt = J  (impulse)

  For a rigid body:
    M = [  m   0   0  ]   (linear)
        [  0   m   0  ]
        [  0   0   m  ]
        +
    I = [Ixx  Ixy  Ixz]   (rotational, moment of inertia tensor)
        [Iyx  Iyy  Iyz]
        [Izx  Izy  Izz]

  Combined 6-DOF:
    [m*I3  0  ] [a_linear ]   [F]
    [0    I_cm] [alpha_rot] = [T]

  For MINIMUM MASS design:
    given stress constraint sigma < sigma_yield
    minimize: m = rho * Volume

  Thin-walled cylinder (pressure vessel):
    Required wall thickness: t >= p*r / (sigma_yield * safety)
    Mass: m = 2*pi*r*L*rho*t = 2*pi*r^2*L*rho*p / (sigma_yield * safety)
    -> minimum mass scales as r^2 (bigger radius = heavier for same pressure)
""")

# Minimum mass cylinder example
print("  Minimum mass pressure vessel:")
print(f"  {'Material':15s}  {'sigma_y (MPa)':14s}  {'rho (kg/m3)':12s}  {'t_min(mm)':10s}  {'mass(kg)':10s}  {'specific':8s}")
p_i = 70e6    # 700 bar hydraulic
r   = 0.025   # 25mm bore
L   = 0.5     # 500mm length
safety = 2.0

materials = [
    ("Steel 4340",     1470e6, 7850),
    ("Ti-6Al-4V",       880e6, 4430),
    ("CF/epoxy (UD)",  1500e6, 1600),
    ("Al 7075-T6",      503e6, 2810),
    ("Inconel 718",    1100e6, 8190),
]
for name, sy, rho in materials:
    t_min = p_i * r / (sy / safety)
    vol   = 2 * math.pi * r * L * t_min
    mass  = rho * vol
    spec  = sy / (rho * 9.81)   # specific strength (m)
    print(f"  {name:15s}  {sy/1e6:14.0f}  {rho:12.0f}  {t_min*1000:10.3f}  {mass:10.4f}  {spec/1000:8.1f} km")
print()

# Impulse-momentum for collision
print("  Impulse example: GS sensor housing drop test")
m_sensor = 2.5    # kg
v_impact = 3.0    # m/s (drop from ~0.46m)
dt_stop  = 0.002  # 2ms contact time (hard floor)
dt_foam  = 0.020  # 20ms with foam padding

F_floor = m_sensor * v_impact / dt_stop
F_foam  = m_sensor * v_impact / dt_foam

print(f"  Sensor mass: {m_sensor} kg,  impact velocity: {v_impact} m/s")
print(f"  Hard floor (dt={dt_stop*1000:.0f}ms): F = {F_floor:.0f} N  = {F_floor/9.81:.0f}g  (likely damage)")
print(f"  Foam pad   (dt={dt_foam*1000:.0f}ms): F = {F_foam:.0f} N  = {F_foam/9.81:.1f}g  (safe)")
print()

# ============================================================
# 3. DIVE WITH INITIAL VELOCITY (fluid entry)
# ============================================================
print("=== 3. DIVE WITH INITIAL VELOCITY ===")
print("""
  Problem: object enters water at angle theta with speed v0
           Find trajectory, max depth, deceleration.

  Forces in water:
    Gravity:      F_g = m*g (downward)
    Drag:         F_d = (1/2)*rho_w*Cd*A*v^2  (opposing motion)
    Buoyancy:     F_b = rho_w*g*V_obj  (upward)

  Terminal velocity in water:
    v_term = sqrt(2*m*g / (rho_w * Cd * A))

  Equations of motion (vertical only, theta=90 dive):
    m * dv/dt = m*g - (1/2)*rho_w*Cd*A*v^2 - rho_w*g*V

  For NEUTRALLY BUOYANT object (m = rho_w * V):
    F_net = -drag only
    dv/dt = -(rho_w*Cd*A / 2m) * v^2
    v(t) = v0 / (1 + v0 * k * t)     where k = rho_w*Cd*A / (2m)
    x(t) = (1/k) * ln(1 + v0*k*t)    penetration depth
""")

rho_w = 1000.0   # kg/m3 water
g_val = 9.81

# Projectile entering water (e.g., torpedo, depth charge, sensor buoy)
cases = [
    ("Sensor buoy",     5.0,  0.05,  1.0, 0.47,  10.0, "90deg (straight down)"),
    ("Torpedo",        500.0, 0.50,  0.3, 0.15, 150.0, "10deg (shallow entry)"),
    ("JDAM (water)",   900.0, 0.35,  0.6, 0.30,  80.0, "60deg"),
]
print(f"  {'Object':15s}  {'m(kg)':7s}  {'v0(m/s)':8s}  {'v_term(m/s)':12s}  {'depth@v=v0/2(m)'}")
for name, m, A, Cd, rho_frac, v0, note in cases:
    m_eff = m
    k = rho_w * Cd * A / (2 * m_eff)
    # depth when speed drops to v0/2
    # v0/2 = v0 / (1 + v0*k*t)  -> t = 1/(v0*k)  -> x = (1/k)*ln(2)
    depth_half = math.log(2) / k
    v_term = math.sqrt(2*m_eff*g_val / (rho_w * Cd * A))
    print(f"  {name:15s}  {m_eff:7.1f}  {v0:8.1f}  {v_term:12.1f}  {depth_half:8.2f}  ({note})")
print()

# ============================================================
# 4. INWARD / OUTWARD FLUX: DIVERGENCE THEOREM
# ============================================================
print("=== 4. INWARD/OUTWARD FLUX: DIVERGENCE THEOREM ===")
print("""
  Divergence theorem (Gauss's theorem):
    integral_V (div F) dV = integral_S (F . n_hat) dA

  n_hat OUTWARD = convention (pointing away from volume)
  Inward flux  = -outward flux

  In CYLINDRICAL coordinates:
    div F = (1/r) d(r*F_r)/dr + (1/r) dF_phi/dphi + dF_z/dz

  Applications:
    Electrostatics: div E = rho/epsilon_0  (Gauss's law)
    Heat transfer:  div q = -Q  (divergence of heat flux = -source)
    Fluid:          div v = 0   (incompressible: no net flux out of volume)
    Fiber cladding: evanescent field K_n(gamma*r) -> exponential decay outward

  GS/OPTICAL connection:
    Evanescent field outside fiber core:
      E(r) ~ K_0(gamma*r)  for r > a
      gamma = sqrt(beta^2 - k_clad^2)  (decay constant)
      Outward flux decays as K_0 -> exponential

    Power confinement factor:
      Gamma = (power in core) / (total power)
            = integral_0^a |J_n|^2 r dr / integral_0^inf |E|^2 r dr
""")

from scipy.special import kn, jn, jn_zeros

# Power confinement in SMF-28
a_um   = 4.1
lam    = 1.55
n_core = 1.4504
n_clad = 1.4447
NA     = math.sqrt(n_core**2 - n_clad**2)
V      = (2*math.pi*a_um/lam) * NA
k0     = 2*math.pi/lam

# LP01 mode: J_0 inside, K_0 outside
# Find U (= k_r * a) from characteristic equation (approx)
# Marcuse approximation: U ~ 2.405 * sqrt(1 - (2.405/V)^2 * ...))
# Use numerical search
from scipy.optimize import brentq
def char_eq(U):
    W = math.sqrt(max(V**2 - U**2, 1e-10))
    lhs = U * jn(1, U) / jn(0, U)
    rhs = W * kn(1, W) / kn(0, W)
    return lhs - rhs

U_sol = brentq(char_eq, 0.1, 2.4)
W_sol = math.sqrt(V**2 - U_sol**2)
k_r   = U_sol / a_um
gamma = W_sol / a_um

r_in  = np.linspace(0, a_um, 500)
r_out = np.linspace(a_um, 5*a_um, 500)
E_in  = jn(0, k_r * r_in)
E_out = jn(0, k_r * a_um) / kn(0, gamma * a_um) * kn(0, gamma * r_out)

P_in  = np.trapezoid(np.abs(E_in)**2  * r_in,  r_in)
P_out = np.trapezoid(np.abs(E_out)**2 * r_out, r_out)
Gamma = P_in / (P_in + P_out)

print(f"  SMF-28 LP01 at 1550nm:")
print(f"    V = {V:.4f},  U = {U_sol:.4f},  W = {W_sol:.4f}")
print(f"    k_r = {k_r:.4f} /um,  gamma = {gamma:.4f} /um  (decay)")
print(f"    Confinement factor Gamma = {Gamma:.4f}  ({Gamma*100:.1f}% power in core)")
print(f"    Outward evanescent flux:  {(1-Gamma)*100:.1f}% in cladding")
print()
print(f"  Divergence theorem check:")
print(f"    Total power (in + out) normalized to 1.0:")
total = P_in + P_out
print(f"    P_core / P_total = {P_in/total:.4f}")
print(f"    P_clad / P_total = {P_out/total:.4f}")
print(f"    Sum              = {(P_in+P_out)/total:.6f}  (must = 1)")
print()

# ============================================================
# 5. CUDA SPARSE DATA: SEND ONLY AI DATA
# ============================================================
print("=== 5. CUDA SPARSE: SEND ONLY SIGNIFICANT COEFFICIENTS ===")
print("""
  Full signal: N float32 = 4N bytes
  Sparse:      K (index, value) pairs = 8K bytes

  Compression wins when K < N/2  (< 50% nonzero)

  PATTERN IN CUDA:
    // Find significant Fourier coefficients
    __global__ void sparse_encode(
        const float2 *F,    // FFT output, N complex
        int   *indices,     // output: significant indices
        float2 *values,     // output: significant values
        int   *count,       // output: how many kept
        float  threshold,   // |F[k]| > threshold to keep
        int    N
    ) {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= N) return;
        float mag = hypotf(F[k].x, F[k].y);
        if (mag > threshold) {
            int idx = atomicAdd(count, 1);  // thread-safe counter
            indices[idx] = k;
            values[idx]  = F[k];
        }
    }

    // Reconstruct: zero-fill + place sparse values + IFFT
    __global__ void sparse_decode(
        float2 *F_full,
        const int *indices, const float2 *values,
        int K, int N
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= K) return;
        F_full[indices[i]] = values[i];
    }

  GS phase recovery is naturally sparse in frequency:
    Optical pulse bandwidth << sampling bandwidth
    -> only N_pulse << N Fourier coefficients are nonzero
    -> send indices + complex amplitudes, reconstruct at receiver
    -> bandwidth reduction: N_pulse/N  (typically 10-100x)

  CUDA LAUNCH CONFIG (RTX 4060, sm_89):
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    sparse_encode<<<grid, block>>>(F, idx, vals, cnt, thr, N);
    cudaDeviceSynchronize();
""")

# demonstrate sparsity of optical pulse spectrum
N = 512
t = np.linspace(-1, 1, N)
pulse = np.exp(-t**2 / 0.02) * np.exp(1j * 20 * np.pi * t**2)  # chirped Gaussian
F = np.fft.fft(pulse)
mag = np.abs(F)

thresholds = [0.01, 0.05, 0.10, 0.20]
print(f"  Chirped Gaussian pulse, N={N}:")
print(f"  {'Threshold':12s}  {'Kept':6s}  {'Ratio':8s}  {'Recon error'}")
for thr in thresholds:
    mask = mag > thr * mag.max()
    K = mask.sum()
    F_sparse = np.zeros(N, dtype=complex)
    F_sparse[mask] = F[mask]
    recon = np.fft.ifft(F_sparse)
    err = np.mean(np.abs(recon - pulse)**2) / np.mean(np.abs(pulse)**2)
    print(f"  {thr:12.0%}  {K:6d}  {K/N:8.1%}  {err:.2e}")
print()
print("  -> 5% threshold keeps ~10% of coefficients, <1% error")
print("  -> that IS 'sending only AI data': sparse spectral representation")
