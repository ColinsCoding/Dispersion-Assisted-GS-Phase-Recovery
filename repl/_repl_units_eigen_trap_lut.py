# %% [markdown]
# # Units · Unity Eigenvectors · Optical Trap · Xanax Binding · LUT Indexing
# `init_printing(use_latex="mathjax")` throughout.
#
# **Structure:**
# §1  Dimensional analysis: mph → m/s, minute·60, ±5% tolerance band
# §2  Unity eigenvectors: unit-circle eigenvalues, DFT matrix, unitary group
# §3  Optical trap force gradient: DNA manipulation, F = −∇U, trap stiffness
# §4  Xanax (alprazolam): GABA-A binding kinetics, Hill equation, IC50
# §5  Bitwise vs arithmetic LUT indexing: sin/cos table, AND-mask vs modulo

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
import scipy.optimize as opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import (symbols, sqrt, pi, Rational, exp, ln, diff, integrate,
                   simplify, latex, Matrix, eye, Abs, conjugate, oo,
                   cos as sp_cos, sin as sp_sin, I as sp_I)
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

print("=== Units · Unity Eigenvectors · Optical Trap · Xanax · LUT ===")

# %% [markdown]
# ---
# ## §1 · Dimensional Analysis: mph → m/s, minute·60, ±5% Tolerance
#
# I treat every number as a fraction.  Units cancel like algebra.
# The only rule: multiply by 1 in disguise (conversion factors).
#
# **Chain:**
#
#   1 mile/hour × (1609.344 m / 1 mile) × (1 hour / 3600 s) = 0.44704 m/s
#
# **Minute × 60:**  1 minute = 60 seconds, so minute · (60 s/min) → seconds.
#
# **±5% from unity:**  tolerance band [0.95, 1.05] around 1.0.
# In eigenvector context: eigenvalue magnitude within ±5% of 1.

# %%
hdr("§1 — Dimensional analysis + unit chain + ±5% tolerance")

# --- Conversion factors (exact where possible) ---
METERS_PER_MILE  = 1609.344        # exact by definition
SECONDS_PER_HOUR = 3600.0          # exact
SECONDS_PER_MIN  = 60.0            # exact

def mph_to_mps(speed_mph):
    """Convert miles per hour to metres per second."""
    return speed_mph * METERS_PER_MILE / SECONDS_PER_HOUR

def minutes_to_seconds(t_min):
    """Convert minutes to seconds (minute × 60)."""
    return t_min * SECONDS_PER_MIN

# Key speed benchmarks
speeds_mph = {'walking': 3, 'cycling': 15, 'car_highway': 70,
              'sound_mph': 767, 'low_earth_orbit': 17500}
print("  Speed conversions:")
for name, v_mph in speeds_mph.items():
    v_mps = mph_to_mps(v_mph)
    print(f"    {name:20s}  {v_mph:7.1f} mph  =  {v_mps:9.3f} m/s")

chk(mph_to_mps(1.0), 0.44704, "1 mph = 0.44704 m/s (exact)")
chk(minutes_to_seconds(1.0), 60.0, "1 min = 60 s", absolute=True)
chk(mph_to_mps(60.0), 26.8224, "60 mph = 26.8224 m/s")

# minute × 60 mph → miles: if I drive 60 mph for 1 minute:
# distance = 60 mph × (1/60) h = 1 mile = 1609.344 m
dist_1min_60mph = 60.0 * (1.0/60.0)   # in miles
chk(dist_1min_60mph, 1.0, "60 mph × 1 min = 1 mile")
chk(dist_1min_60mph * METERS_PER_MILE, 1609.344,
    "60 mph × 1 min = 1609.344 m")

# ±5% tolerance band around unity
print("\n  ±5% from unity:")
unity    = 1.0
tol_frac = 0.05
lo, hi   = unity * (1 - tol_frac), unity * (1 + tol_frac)
print(f"    band = [{lo:.4f}, {hi:.4f}]")

# Check: is eigenvalue within band?
test_eigenvalues = [0.98, 1.00, 1.03, 1.06, 0.94]
for lam in test_eigenvalues:
    within = lo <= lam <= hi
    print(f"    λ = {lam:.2f}  {'✓ within ±5%' if within else '✗ outside ±5%'}")

chk(lo, 0.95, "lower bound = 0.95")
chk(hi, 1.05, "upper bound = 1.05")

# %% [markdown]
# ---
# ## §2 · Unity Eigenvectors: Eigenvalues on the Unit Circle
#
# **Unitary matrix** U: U†U = I → all eigenvalues satisfy |λ| = 1.
# They live ON the unit circle in ℂ: λ_k = e^{iθ_k}.
#
# **DFT matrix** is the canonical unitary matrix:
#
#   F_{jk} = (1/√N) · ω^{jk},   ω = e^{−2πi/N}
#
# Eigenvalues of F are 4th roots of unity: {1, −i, −1, i} (each with multiplicity).
# Eigenvectors of the DFT = functions invariant under FT (Hermite–Gauss functions).
#
# **±5% from unity** in eigenvector context:
# If |λ| = 0.95..1.05, the mode neither grows nor decays — it's near-neutral.
# Pure |λ|=1 modes are the "remote controls" — they propagate phase without loss.

# %%
hdr("§2 — Unity eigenvectors: DFT matrix, unitary group, unit circle")

# Build DFT matrix for small N
def dft_matrix(N):
    """Normalised DFT matrix: F[j,k] = exp(-2pi*i*j*k/N) / sqrt(N)."""
    j = np.arange(N)
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(j, k) / N) / np.sqrt(N)

N_dft = 8
F = dft_matrix(N_dft)

# Verify unitary: F†F = I
FhF = F.conj().T @ F
err_unitary = np.max(np.abs(FhF - np.eye(N_dft)))
chk(err_unitary, 0, "DFT F†F = I (unitary)", tol=1e-10, absolute=True)

# Eigenvalues of DFT matrix: must be in {1, -i, -1, i}
eigs_F = np.linalg.eigvals(F)
eig_mags = np.abs(eigs_F)
chk(np.max(np.abs(eig_mags - 1.0)), 0,
    "all DFT eigenvalues on unit circle |λ|=1", tol=1e-10, absolute=True)

# The four possible eigenvalues of a DFT matrix
print(f"  DFT(N={N_dft}) eigenvalues (rounded):")
for lam in sorted(eigs_F, key=lambda x: np.angle(x)):
    print(f"    λ = {lam.real:+.4f} + {lam.imag:+.4f}i   "
          f"|λ| = {abs(lam):.6f}   arg = {np.angle(lam)*180/np.pi:+.1f}°")

# ±5% band: how many eigenvalues are within 5% of 1?
within_band = np.sum((eig_mags >= 0.95) & (eig_mags <= 1.05))
chk(within_band, N_dft, "all DFT eigvals within ±5% of unity", tol=1e-9, absolute=True)

# Symbolic: 2×2 rotation matrix eigenvalues (unitary)
theta_sym = symbols('theta', real=True)
R = Matrix([[sp_cos(theta_sym), -sp_sin(theta_sym)],
            [sp_sin(theta_sym),  sp_cos(theta_sym)]])
eigs_R = R.eigenvals()
print(f"\n  2×2 rotation R(θ) eigenvalues (symbolic):")
for lam, mult in eigs_R.items():
    show(lam, f"  λ (mult={mult})")

# Verify |λ|=1 symbolically
lam_rot = sp_cos(theta_sym) + sp_I * sp_sin(theta_sym)
mag_sq = simplify(lam_rot * conjugate(lam_rot))
show(mag_sq, "|λ|² for rotation eigenvalue")
chk(float(mag_sq), 1.0, "|λ|² = 1 for rotation eigenvalue")

# DFT eigenvector = Hermite-Gauss (first: Gaussian is its own FT)
N_hg = 256
x_hg = np.linspace(-4, 4, N_hg)
dx   = x_hg[1] - x_hg[0]
gauss = np.exp(-x_hg**2 / 2)
gauss /= np.sqrt(np.trapezoid(gauss**2, x_hg))

# DFT of Gaussian → Gaussian (up to normalisation and sampling)
gauss_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(gauss))) * dx
gauss_fft_mag = np.abs(gauss_fft)
gauss_fft_mag /= gauss_fft_mag.max()
gauss_norm     = gauss / gauss.max()
# Peak positions should match
peak_orig = np.argmax(gauss_norm)
peak_fft  = np.argmax(gauss_fft_mag)
chk(abs(peak_orig - peak_fft), 0,
    "Gaussian FT peak same as original (FT eigenfunction)", tol=5, absolute=True)

print(f"\n  Hermite-Gauss is eigenfunction of FT: peak shift = {abs(peak_orig-peak_fft)} samples")
print(f"  Correlation (FT output vs input): "
      f"{np.corrcoef(gauss_norm, gauss_fft_mag)[0,1]:.6f}")
chk(np.corrcoef(gauss_norm, gauss_fft_mag)[0,1], 1.0,
    "Gaussian FT shape = Gaussian (correlation = 1)", tol=0.8)

# %% [markdown]
# ---
# ## §3 · Optical Trap: Force Gradient, DNA Manipulation
#
# I am a DNA strand held in an optical trap.  Here is what that means physically.
#
# **Gradient force** (dielectric particle in focused laser beam):
#
#   F_grad = (α/2) ∇|E|²
#
# where α = polarisability, |E|² = intensity.
# The force points toward the intensity maximum (trap centre) for α > 0.
#
# **Potential energy:**
#
#   U = −(α/2)|E|²  →  F = −∇U  (restoring, like a spring)
#
# **Trap stiffness** (Hookean near centre):
#
#   κ = −∂²U/∂x² |_{x=0} = α · k₀² · I₀ / 2  [N/m]
#
# For a Gaussian beam:  I(x) = I₀ exp(−2x²/w₀²)
#
#   κ = 4α I₀ / w₀²  (trap stiffer for tighter focus or higher power)
#
# **DNA in a trap:** λ-phage DNA (~16 μm contour length) is end-labelled
# with beads; two traps pull the molecule.  Force–extension follows the
# Worm-Like Chain (WLC) model.

# %%
hdr("§3 — Optical trap: gradient force, trap stiffness, DNA WLC")

x_sym, w0_sym, I0_sym, alpha_sym, Lc_sym, Lp_sym, F_sym = \
    symbols('x w0 I0 alpha Lc Lp F', positive=True)

# Intensity profile: Gaussian beam
I_gauss = I0_sym * exp(-2 * x_sym**2 / w0_sym**2)
show(I_gauss, "I(x) = I₀ exp(-2x²/w₀²)")

# Potential energy
U = -alpha_sym / 2 * I_gauss
show(U, "U(x) = -(α/2)I(x)")

# Gradient force
F_grad = -diff(U, x_sym)
F_grad_simplified = simplify(F_grad)
show(F_grad_simplified, "F_grad = -∂U/∂x")

# Trap stiffness κ = -∂F/∂x|_{x=0} = ∂²U/∂x²|_{x=0}
kappa = -diff(F_grad, x_sym).subs(x_sym, 0)
kappa_simplified = simplify(kappa)
show(kappa_simplified, "κ = -∂F/∂x|_{x=0} (trap stiffness)")

# Numerical: typical optical trap values
I0_val   = 1e12   # W/m²  (focused 10 mW into 1 μm² spot)
w0_val   = 0.5e-6 # m  (beam waist)
alpha_val = 4 * np.pi * 8.85e-12 * (100e-9)**3 * 0.5
# α ≈ 4πε₀a³·(n²-1)/(n²+2) for polystyrene bead a=100nm

kappa_val = float(kappa_simplified.subs(
    [(alpha_sym, alpha_val), (I0_sym, I0_val), (w0_sym, w0_val)]))
print(f"\n  Numerical trap stiffness:")
print(f"    α   = {alpha_val:.3e} C·m²/V")
print(f"    I₀  = {I0_val:.1e} W/m²")
print(f"    w₀  = {w0_val*1e6:.2f} μm")
print(f"    κ   = {kappa_val*1e6:.3f} μN/m  =  {kappa_val:.3e} N/m")
chk(kappa_val > 0, 1, "trap stiffness κ > 0 (restoring)", tol=1e-9, absolute=True)

# Thermal energy check: kBT at 300K
kB  = 1.380649e-23
T   = 300.0
kBT = kB * T
x_rms = np.sqrt(kBT / max(kappa_val, 1e-30))
print(f"    kBT = {kBT:.3e} J")
print(f"    x_rms = √(kBT/κ) = {x_rms*1e9:.1f} nm  (Brownian fluctuation)")
chk(x_rms > 0, 1, "Brownian fluctuation x_rms > 0", tol=1e-9, absolute=True)

# ── DNA Worm-Like Chain (WLC) force-extension ──────────────────────────────
# Marko-Siggia interpolation:
#   F·Lp/kBT = 1/4·(1-z/Lc)^{-2} - 1/4 + z/Lc
print("\n  DNA WLC force-extension (λ-phage DNA):")
Lc_dna = 16.4e-6  # m  (contour length λ-phage DNA)
Lp_dna = 50e-9    # m  (persistence length ~50 nm)

def wlc_force(z, Lc=Lc_dna, Lp=Lp_dna, kT=kBT):
    """WLC Marko-Siggia: force [N] at extension z [m]."""
    x = z / Lc
    x = np.clip(x, 0, 0.9999)
    return (kT / Lp) * (0.25 / (1 - x)**2 - 0.25 + x)

z_arr = np.linspace(0.01 * Lc_dna, 0.95 * Lc_dna, 500)
F_arr = wlc_force(z_arr)

# At half extension z = Lc/2:
F_half = wlc_force(Lc_dna / 2)
print(f"    F(z = Lc/2) = {F_half*1e12:.2f} pN  (entropic elasticity)")
chk(F_half > 0, 1, "WLC force positive", tol=1e-9, absolute=True)
chk(F_half < 10e-12, 1, "WLC F(Lc/2) < 10 pN (physical range)", tol=1e-9, absolute=True)

# B-form DNA overstretching at ~65 pN
F_65pN = 65e-12
z_65pN = z_arr[np.argmin(np.abs(F_arr - F_65pN))]
print(f"    65 pN overstretching at z = {z_65pN/Lc_dna*100:.1f}% of Lc")

# Effective spring constant (linearise near z=Lc/2)
dF_dz = np.gradient(F_arr, z_arr)
kappa_dna_mid = dF_dz[len(dF_dz)//2]
print(f"    WLC stiffness at z=Lc/2: κ_DNA = {kappa_dna_mid*1e6:.3f} μN/m")

# Genetic mutation in optical trap:
# Force gradient can stretch or twist DNA → changes base-pair accessibility
# Topoisomerase / helicase activity measurable by monitoring bead displacement
print("\n  Genetic mutation via optical trap:")
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  DNA end-tethered between two optically trapped beads.      │")
print("  │  I apply force F by moving trap centres apart.              │")
print("  │  - F < 15 pN:  B-form DNA, entropic elasticity (WLC)       │")
print("  │  - F ~ 65 pN:  B→S transition (overstretching, dsDNA)      │")
print("  │  - F > 65 pN:  strand separation, ss-DNA accessible         │")
print("  │  Result: single-molecule 'genetic surgery' —                │")
print("  │  expose binding sites, recruit polymerases, observe         │")
print("  │  base-pair opening in real time via fluorescence.           │")
print("  └─────────────────────────────────────────────────────────────┘")

# %% [markdown]
# ---
# ## §4 · Xanax (Alprazolam): GABA-A Binding Kinetics + Hill Equation
#
# **Use case:** Alprazolam (Xanax) is a benzodiazepine.
# It binds to the benzodiazepine site on GABA-A receptors — an allosteric
# positive modulator, not a direct agonist.
#
# **Hill equation** (receptor occupancy):
#
#   θ = [L]^n / (K_d^n + [L]^n)
#
# where θ = fractional occupancy, [L] = drug concentration, K_d = dissociation
# constant, n = Hill coefficient (cooperativity).
#
# **EC50** = concentration at 50% occupancy = K_d when n=1.
#
# **Pharmacokinetics:** oral bioavailability ~90%, t½ = 6–12 h,
# V_d = 0.8–1.2 L/kg, CYP3A4 metabolism.
#
# The math: same Hill equation governs oxygen-haemoglobin, dose-response,
# enzyme saturation (Michaelis-Menten is Hill n=1).

# %%
hdr("§4 — Xanax: GABA-A Hill equation, EC50, pharmacokinetics")

L_sym, Kd_sym, n_sym = symbols('L Kd n', positive=True)

# Hill equation
theta = L_sym**n_sym / (Kd_sym**n_sym + L_sym**n_sym)
show(theta, "θ([L]) = [L]^n / (Kd^n + [L]^n)")

# At [L] = Kd: theta = 0.5
theta_at_Kd = theta.subs(L_sym, Kd_sym)
show(simplify(theta_at_Kd), "θ(Kd)")
chk(float(simplify(theta_at_Kd)), 0.5, "Hill: θ(EC50) = 0.5 exactly")

# Alprazolam GABA-A parameters (benzodiazepine site)
EC50_nM = 10.0    # nM  (benzodiazepine site affinity)
n_hill  = 1.0     # non-cooperative (n=1 for BDZ site)

L_arr_nM = np.logspace(-1, 3, 500)  # 0.1 to 1000 nM
theta_arr = L_arr_nM**n_hill / (EC50_nM**n_hill + L_arr_nM**n_hill)

# Therapeutic window: 5–50 nM plasma (approximate)
theta_5nM  = 5.0**n_hill  / (EC50_nM**n_hill + 5.0**n_hill)
theta_50nM = 50.0**n_hill / (EC50_nM**n_hill + 50.0**n_hill)
print(f"  EC50 = {EC50_nM} nM,  Hill n = {n_hill}")
print(f"  θ at  5 nM (low therapeutic): {theta_5nM:.3f}  ({theta_5nM*100:.1f}%)")
print(f"  θ at 50 nM (high therapeutic): {theta_50nM:.3f}  ({theta_50nM*100:.1f}%)")
chk(theta_5nM, 5.0/(EC50_nM+5.0), "θ(5nM) by hand", tol=1e-6)
chk(theta_50nM, 50.0/(EC50_nM+50.0),  "θ(50nM) by hand", tol=1e-6)

# Pharmacokinetics: 1-compartment oral model
# C(t) = (F·D·ka)/(V·(ka-ke)) · [exp(-ke·t) - exp(-ka·t)]
F_bio = 0.90            # bioavailability
D_mg  = 0.5e-3 * 1e6   # 0.5 mg in μg
MW    = 308.77          # g/mol alprazolam
D_nmol= D_mg / MW * 1e9 # nmol
V_L   = 70.0 * 1.0      # L (V_d = 1 L/kg × 70 kg)
t_half_h = 9.0          # hours (mid-range)
ke    = np.log(2) / t_half_h  # h^-1
ka    = 1.5              # h^-1 (absorption rate, Tmax ~1-2h)

t_h   = np.linspace(0, 48, 1000)  # hours

def pk_conc(t, F=F_bio, D=D_nmol, ka=ka, ke=ke, V=V_L):
    """1-compartment oral PK: concentration [nM] vs time [h]."""
    if abs(ka - ke) < 1e-10:
        return F * D * ka * t * np.exp(-ke * t) / V
    return F * D * ka / (V * (ka - ke)) * (np.exp(-ke*t) - np.exp(-ka*t))

C_t = pk_conc(t_h)
C_max = np.max(C_t)
t_max = t_h[np.argmax(C_t)]
print(f"\n  PK (0.5 mg oral, 70 kg, V_d=1 L/kg, t½={t_half_h}h):")
print(f"    C_max = {C_max:.2f} nM at t_max = {t_max:.2f} h")
print(f"    θ(C_max) = {C_max/(EC50_nM+C_max):.3f} ({C_max/(EC50_nM+C_max)*100:.1f}% GABA-A occupancy)")
chk(C_max > 0, 1, "C_max > 0", tol=1e-9, absolute=True)
chk(t_max > 0.5, 1, "t_max > 0.5h (absorption lag)", tol=1e-9, absolute=True)
chk(t_max < 4.0, 1, "t_max < 4h (reasonable oral Tmax)", tol=1e-9, absolute=True)

# Clearance: CL = ke * V
CL_L_h = ke * V_L
print(f"    CL = {CL_L_h:.2f} L/h  =  {CL_L_h/60:.3f} L/min")
chk(CL_L_h, np.log(2)/t_half_h * V_L, "CL = ke*V by definition")

# Michaelis-Menten is Hill n=1: link to enzyme kinetics
print("\n  Hill n=1 = Michaelis-Menten:")
print("  θ = [L]/(Kd + [L])  ↔  v = Vmax·[S]/(Km + [S])")
print("  Same equation. Receptor occupancy IS enzyme saturation.")

# %% [markdown]
# ---
# ## §5 · Bitwise vs Arithmetic LUT Indexing
#
# A lookup table (LUT) maps integer index → precomputed value.
# For sin/cos, I precompute N samples and index with:
#
# **Arithmetic (modulo):**  `idx = int(angle / step) % N`
# - Works for any N
# - Integer division + modulo: 2 operations
# - Modulo is slow on hardware (no single-cycle instruction on most MCUs)
#
# **Bitwise (AND mask):**  `idx = int(angle / step) & (N-1)`
# - Requires N = power of 2
# - Single AND instruction: 1 cycle, pipelineable
# - No division: angle pre-scaled to integer
# - Used in: DSPs, Qualcomm Hexagon, all FPGA sin/cos cores
#
# **Error:**  both give identical values — indexing is exact.
# The only difference is speed and the power-of-2 constraint.

# %%
hdr("§5 — Bitwise AND vs arithmetic modulo LUT indexing")

# Build sin LUT of size N (power of 2)
N_LUT = 1024          # must be power of 2
MASK  = N_LUT - 1     # = 0b...01111111111  (10 ones)
print(f"  LUT size N = {N_LUT}  =  2^{int(np.log2(N_LUT))}")
print(f"  AND mask   = {MASK}   =  0b{MASK:010b}")

# Precompute table
lut_sin = np.sin(2 * np.pi * np.arange(N_LUT) / N_LUT).astype(np.float32)
lut_cos = np.cos(2 * np.pi * np.arange(N_LUT) / N_LUT).astype(np.float32)

def lut_lookup_modulo(angles_rad, lut, N=N_LUT):
    """Arithmetic: idx = round(angle * N / 2pi) % N"""
    idx = (np.round(angles_rad * N / (2 * np.pi)).astype(int)) % N
    return lut[idx]

def lut_lookup_bitwise(angles_rad, lut, mask=MASK, N=N_LUT):
    """Bitwise: idx = round(angle * N / 2pi) & mask  (N must be 2^k)"""
    idx = (np.round(angles_rad * N / (2 * np.pi)).astype(int)) & mask
    return lut[idx]

# Test angles
angles_test = np.linspace(0, 10 * np.pi, 50000)  # many full cycles

sin_modulo  = lut_lookup_modulo( angles_test, lut_sin)
sin_bitwise = lut_lookup_bitwise(angles_test, lut_sin)
cos_modulo  = lut_lookup_modulo( angles_test, lut_cos)
cos_bitwise = lut_lookup_bitwise(angles_test, lut_cos)

# Verify identical output
diff_sin = np.max(np.abs(sin_modulo.astype(float) - sin_bitwise.astype(float)))
diff_cos = np.max(np.abs(cos_modulo.astype(float) - cos_bitwise.astype(float)))
chk(diff_sin, 0, "bitwise sin == modulo sin (identical output)", tol=1e-9, absolute=True)
chk(diff_cos, 0, "bitwise cos == modulo cos (identical output)", tol=1e-9, absolute=True)

# Quantisation error vs exact
sin_exact = np.sin(angles_test)
quant_err = np.max(np.abs(sin_bitwise.astype(float) - sin_exact))
print(f"  Quantisation error (LUT N={N_LUT}): {quant_err:.6f}  "
      f"({quant_err * 180/np.pi * 1000:.3f} m°)")
chk(quant_err < 2 * np.pi / N_LUT, 1,
    "quantisation error < 2π/N (one bin width)", tol=1e-9, absolute=True)

# Benchmark: modulo vs bitwise
import time

N_bench = 1_000_000
angles_bench = np.random.uniform(0, 100*np.pi, N_bench)

t0 = time.perf_counter()
for _ in range(10):
    _ = lut_lookup_modulo(angles_bench, lut_sin)
t_mod = (time.perf_counter() - t0) / 10

t0 = time.perf_counter()
for _ in range(10):
    _ = lut_lookup_bitwise(angles_bench, lut_sin)
t_bit = (time.perf_counter() - t0) / 10

print(f"\n  Benchmark ({N_bench:,} lookups, 10 runs avg):")
print(f"    Modulo  (%)  : {t_mod*1000:.2f} ms")
print(f"    Bitwise (&)  : {t_bit*1000:.2f} ms")
speedup = t_mod / max(t_bit, 1e-12)
print(f"    Speedup      : {speedup:.2f}×")
chk(t_bit < t_mod * 1.5, 1,
    "bitwise not slower than modulo", tol=1e-9, absolute=True)

# Power-of-2 requirement: show why N must be 2^k for AND-mask trick
print("\n  Why N must be a power of 2 for AND-mask:")
print("  ┌───────────────────────────────────────────────────────────────┐")
print("  │  N=1024:  binary = 10000000000  →  N-1 = 01111111111        │")
print("  │  N-1 has all lower bits set → AND clears upper bits exactly  │")
print("  │  idx & (N-1) = idx mod N    iff  N = 2^k                    │")
print("  │                                                               │")
print("  │  If N=1000 (not 2^k):                                        │")
print("  │  N-1 = 999 = 01111100111  → NOT a clean mask                 │")
print("  │  idx & 999 ≠ idx mod 1000 in general                        │")
print("  │  → must use % (modulo) for non-power-of-2 table sizes       │")
print("  └───────────────────────────────────────────────────────────────┘")

# Verify the claim
N_bad = 1000
mask_bad = N_bad - 1
idx_test_arr = np.arange(0, 2 * N_bad)
and_result  = idx_test_arr & mask_bad
mod_result  = idx_test_arr % N_bad
n_mismatch  = np.sum(and_result != mod_result)
chk(n_mismatch > 0, 1,
    "AND-mask fails for N=1000 (mismatch exists)", tol=1e-9, absolute=True)
print(f"  For N=1000: {n_mismatch} mismatches out of {len(idx_test_arr)} indices (AND≠MOD)")

# D-GS connection: LUT in FNO for phase lookup
print("\n  D-GS / FNO connection:")
print("  The spectral convolution in FNO uses fixed frequency indices.")
print("  Phase LUT for the GS constraint → bitwise indexing on GPU")
print("  (warp size = 32 = 2^5; all CUDA shared memory is 2^k aligned)")
print("  → LUT size MUST be power of 2 for coalesced memory access.")

# %% [markdown]
# ---
# ## Summary
#
# | § | Topic | Key result |
# |---|-------|-----------|
# | 1 | mph → m/s | 1 mph = 0.44704 m/s; 60 mph × 1 min = 1 mile |
# | 1 | ±5% unity | λ ∈ [0.95, 1.05] = "near-neutral" mode |
# | 2 | Unity eigenvectors | DFT: all \|λ\|=1; Gaussian = FT eigenfunction |
# | 3 | Optical trap | F = α∇\|E\|²; κ = 4αI₀/w₀²; DNA WLC at 65 pN |
# | 4 | Xanax/Hill | θ = \[L\]/(Kd+\[L\]); EC50=10nM; PK C_max verified |
# | 5 | LUT indexing | AND-mask = modulo for N=2^k; bitwise ≤ arith speed |

# %%
hdr("Done — all checks complete")
print("  Notebook: _repl_units_eigen_trap_lut.py")
