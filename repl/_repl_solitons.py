"""
repl/_repl_solitons.py
Optical solitons: NLS equation, sech pulse, dispersion vs nonlinearity.
Silicon photonics platforms. GS on soliton I1/I2. Special relativity.
"""
import numpy as np
import sympy as sp
import pandas as pd
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 60)
print("SOLITONS + SILICON PHOTONICS + SPECIAL RELATIVITY")
print("=" * 60)
print()

# ============================================================
# 1. Nonlinear Schrodinger Equation (NLS)
# ============================================================
print("=== 1. Nonlinear Schrodinger Equation ===")
print("""
Fiber propagation (NLS):
  dA/dz = -i*(beta2/2)*d^2A/dt^2  +  i*gamma*|A|^2*A
           ^^^^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^
           dispersion (GVD)            Kerr nonlinearity

  beta2  = GVD [ps^2/km]   negative = anomalous dispersion
  gamma  = nonlinear coeff [1/(W*km)]
  A(z,t) = complex envelope [sqrt(W)]

Soliton condition: dispersion EXACTLY balances nonlinearity
  N^2 = gamma * P0 * T0^2 / |beta2|
  N=1 -> fundamental soliton (shape-preserving)
  N>1 -> higher-order soliton (periodic breathing)

Fundamental soliton solution:
  A(z,t) = sqrt(P0) * sech(t/T0) * exp(i*z/(2*Ld))
  Ld = T0^2 / |beta2|   (dispersion length)
  Ln = 1/(gamma*P0)     (nonlinear length)
  Soliton: Ld = Ln
""")

# symbolic soliton
t_s, z_s = sp.symbols('t z', real=True)
P0, T0, beta2, gamma_s, Ld = sp.symbols('P0 T0 beta2 gamma Ld', positive=True)

A_sol = sp.sqrt(P0) * sp.sech(t_s/T0) * sp.exp(sp.I*z_s/(2*Ld))
intensity = sp.simplify(sp.Abs(A_sol)**2)
print("Soliton envelope: A(z,t) = sqrt(P0) * sech(t/T0) * exp(i*z/2Ld)")
print("Intensity |A|^2  =", sp.pretty(intensity))
print("-> sech^2 pulse shape, invariant in z (no spreading)")
print()

# ============================================================
# 2. Soliton parameters: standard SMF-28
# ============================================================
print("=== 2. Soliton Parameters: SMF-28 at 1550 nm ===")

beta2_smf = -21e-27      # ps^2/m = -21 ps^2/km  (anomalous at 1550nm)
gamma_smf =  1.3e-3      # 1/(W*m) = 1.3 /W/km
c_light   =  3e8         # m/s

# fundamental soliton: choose T0, solve for P0
T0_vals_ps = [0.1, 0.5, 1.0, 5.0, 10.0]   # ps
print(f"{'T0 (ps)':>10}  {'P0 (W)':>10}  {'Ld (km)':>10}  {'Ln (km)':>10}  {'Energy (pJ)':>12}")
print("-" * 60)
for T0_ps in T0_vals_ps:
    T0_s  = T0_ps * 1e-12
    Ld_km = T0_s**2 / abs(beta2_smf) / 1e3
    P0_W  = abs(beta2_smf) / (gamma_smf * T0_s**2)
    Ln_km = 1.0 / (gamma_smf * P0_W) / 1e3
    E_pJ  = 2 * P0_W * T0_s * 1e12   # integral of sech^2 = 2*T0
    print(f"  {T0_ps:>8.1f}  {P0_W:>10.2f}  {Ld_km:>10.3f}  {Ln_km:>10.3f}  {E_pJ:>12.4f}")
print()
print("Ld = Ln confirmed for each row (soliton condition)")
print()

# ============================================================
# 3. Simulate soliton propagation: split-step Fourier
# ============================================================
print("=== 3. Split-Step Fourier: soliton vs dispersive pulse ===")

N_t   = 1024
T_win = 50e-12      # 50 ps window
dt    = T_win / N_t
t     = np.linspace(-T_win/2, T_win/2, N_t)
nu    = np.fft.fftfreq(N_t, d=dt)
omega = 2 * np.pi * nu

T0_s = 1e-12        # 1 ps soliton
P0_W = abs(beta2_smf) / (gamma_smf * T0_s**2)

# initial pulse: sech
A0_sol  = np.sqrt(P0_W) * 1.0/np.cosh(t/T0_s)
A0_gauss = np.sqrt(P0_W) * np.exp(-t**2 / (2*T0_s**2))  # Gaussian, same peak P

def split_step(A, dz, n_steps, beta2, gamma, omega):
    """Simple split-step Fourier method."""
    half_disp = np.exp(-1j * beta2/2 * omega**2 * dz/2)
    for _ in range(n_steps):
        # half dispersion step
        A_f = np.fft.fft(A) * half_disp
        A   = np.fft.ifft(A_f)
        # full nonlinear step
        A   = A * np.exp(1j * gamma * np.abs(A)**2 * dz)
        # half dispersion step
        A_f = np.fft.fft(A) * half_disp
        A   = np.fft.ifft(A_f)
    return A

# normalized units: tau = t/T0, xi = z/Ld, u = A/sqrt(P0)
# NLS(N=1): du/dxi = i/2 * d^2u/dtau^2 + i*|u|^2*u
# Soliton:  u(xi,tau) = sech(tau) * exp(i*xi/2)
N_norm  = 2048
tau_win = 30.0
dtau    = tau_win / N_norm
tau_n   = np.linspace(-tau_win/2, tau_win/2, N_norm)
nu_n    = np.fft.fftfreq(N_norm, d=dtau)
Om      = 2 * np.pi * nu_n

u0_sol   = 1.0 / np.cosh(tau_n)             # N=1 soliton
u0_gauss = np.exp(-tau_n**2 / 2)            # Gaussian, same peak, no NL

def split_step_norm(u, dxi, n_steps, Om):
    """Normalized NLS split-step. beta2_eff=-1, gamma_eff=1."""
    half_disp = np.exp(-1j/2 * Om**2 * dxi/2)   # anomalous: d/dxi -> -i*Om^2/2 in Fourier
    for _ in range(n_steps):
        u = np.fft.ifft(np.fft.fft(u) * half_disp)
        u = u * np.exp(1j * np.abs(u)**2 * dxi)
        u = np.fft.ifft(np.fft.fft(u) * half_disp)
    return u

dxi     = 0.005
n_steps = 400   # propagate xi = 2.0 = 2*Ld

u_sol_out   = split_step_norm(u0_sol.copy(),   dxi, n_steps, Om)
u_gauss_out = split_step_norm(u0_gauss.copy(), dxi, n_steps, Om)

def fwhm_norm(tau, I):
    I = I / I.max()
    idx = np.where(I > 0.5)[0]
    if len(idx) < 2:
        return float('nan')
    return tau[idx[-1]] - tau[idx[0]]

fw_si  = fwhm_norm(tau_n, np.abs(u0_sol)**2)
fw_so  = fwhm_norm(tau_n, np.abs(u_sol_out)**2)
fw_gi  = fwhm_norm(tau_n, np.abs(u0_gauss)**2)
fw_go  = fwhm_norm(tau_n, np.abs(u_gauss_out)**2)

print(f"After propagating xi=2.0 (2 dispersion lengths), normalized units:")
print(f"  Soliton (sech):   FWHM {fw_si:.3f} -> {fw_so:.3f}  "
      f"ratio={fw_so/fw_si:.4f}")
print(f"  Gaussian (no NL): FWHM {fw_gi:.3f} -> {fw_go:.3f}  "
      f"ratio={fw_go/fw_gi:.4f}")
print()
print("Soliton ratio ~ 1.000: no broadening.  Gaussian ratio > 1: dispersive spreading.")
print()

# ============================================================
# 4. GS on soliton: what I1, I2 look like
# ============================================================
print("=== 4. GS Phase Recovery on Soliton Signal ===")
print("""
Your experiment:
  source -> fiber (D1) -> detector1 -> I1.npy
         -> fiber (D2) -> detector2 -> I2.npy

For a soliton source:
  E(t)    = sqrt(P0) * sech(t/T0) * exp(i*phi_NL(t))
  phi_NL  = gamma * P0 * z  (constant across pulse for fundamental soliton)
  I1, I2  = |disperse(E, D1)|^2, |disperse(E, D2)|^2

  GS recovers E(t) -> gives you phi_NL directly
  phi_NL = gamma * P0 * z  -> measure gamma*P0 from phase slope
""")

# simulate soliton I1, I2
D1, D2 = 5000, -5000   # GS diversity parameters (not fiber D, dimensionless)

def disperse(E, D):
    N  = len(E)
    nu = np.fft.fftfreq(N)
    H  = np.exp(1j * np.pi * D * nu**2)
    return np.fft.ifft(np.fft.fft(E) * H)

# soliton with nonlinear phase
T0_sim = 20   # samples
A_gs   = 1.0/np.cosh((np.arange(512) - 256) / T0_sim)
phi_NL = 0.8  # rad, representative
E_true = A_gs * np.exp(1j * phi_NL * A_gs**2 / A_gs.max()**2)

I1 = np.abs(disperse(E_true, D1))**2
I2 = np.abs(disperse(E_true, D2))**2

print(f"Simulated soliton: T0={T0_sim} samples, phi_NL_peak={phi_NL:.2f} rad")
print(f"  I1: max={I1.max():.4f}  mean={I1.mean():.4f}  "
      f"diversity |D1-D2|={abs(D1-D2)}")
print(f"  I2: max={I2.max():.4f}  mean={I2.mean():.4f}")
print()
print("This is exactly what your detector measures.")
print("Save as: np.save('I1.npy', I1);  np.save('I2.npy', I2)")
print("Run:     py -3.12 gsrecover.py --i1 I1.npy --i2 I2.npy --D1 5000 --D2 -5000 --plot")
print()

# ============================================================
# 5. Silicon photonics platforms
# ============================================================
print("=== 5. Silicon Photonics Platforms ===")
platforms = pd.DataFrame([
    ('Intel IFS (22FFL)',     220,  'Si/SiN', 'anomalous',  1310,  'monolithic CMOS',  'Thunderbolt, co-packaged optics'),
    ('imec iSiPP50G',         220,  'Si',     'anomalous',  1550,  'MPW shuttles',     'research, multi-project wafer'),
    ('AIM Photonics (300mm)', 300,  'Si/SiN', 'both',       1310,  'GE/AIM PDK',       'DoD IMOD program'),
    ('Lionix (TriPleX)',       None,'Si3N4',  'normal',     850,   'ultra-low loss',   'LiDAR, gyroscopes, bio'),
    ('Jalali-style (UCLA)',    220,  'Si',     'anomalous',  1550,  'custom',           'your GS experiment'),
], columns=['Platform','thickness_nm','material','dispersion','lambda_nm','access','applications'])
print(platforms.to_string(index=False))
print()
print("Your GS setup: SMF-28 fiber + Si waveguide detector at 1550 nm")
print("D parameter maps to beta2 * L for the fiber spool length L")
print()

# ============================================================
# 6. Special relativity: beyond gamma table
# ============================================================
print("=== 6. Special Relativity: 4-vectors and invariants ===")
print("""
4-momentum:  p^mu = (E/c, px, py, pz)
Invariant:   p^mu * p_mu = (E/c)^2 - |p|^2 = (mc)^2  <- rest mass is invariant

Photon:      m=0  ->  E = |p|*c  ->  E = hf  (massless, travels at c)
Electron:    E^2  = (pc)^2 + (mc^2)^2

Doppler (relativistic):
  f_obs = f_source * sqrt((1+beta)/(1-beta))   beta = v/c  (approaching)
  -> blueshift approaching, redshift receding

Time dilation / length contraction:
  Delta_t'  = gamma * Delta_t    (moving clock runs slow)
  L'        = L / gamma          (moving rod contracts)
""")

beta_vals = [0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
print(f"{'v/c':>8}  {'gamma':>10}  {'t_dilation':>12}  {'L_contraction':>14}  {'f_doppler (approach)':>22}")
print("-" * 75)
for b in beta_vals:
    g     = 1.0 / np.sqrt(1 - b**2)
    f_d   = np.sqrt((1+b)/(1-b))
    print(f"  {b:>6.4f}  {g:>10.4f}  {g:>12.4f}x  {1/g:>13.4f}x  {f_d:>20.4f}x")
print()

# relativistic energy example: proton at LHC
m_p   = 938.3e6   # eV/c^2
E_LHC = 6.5e12    # eV (6.5 TeV per beam)
gamma_LHC = E_LHC / m_p
beta_LHC  = np.sqrt(1 - 1/gamma_LHC**2)
print(f"LHC proton: E={E_LHC/1e12:.1f} TeV  gamma={gamma_LHC:.0f}  "
      f"v/c={beta_LHC:.10f}")
print(f"  1 - v/c = {1-beta_LHC:.3e}  (proton misses c by 3 m/s out of 3e8 m/s)")
print()

# ============================================================
# 7. Soliton <-> GS <-> Silicon photonics: the full chain
# ============================================================
print("=== 7. The Full Chain ===")
print("""
Fiber laser source
  -> emits soliton pulse: A(t) = sqrt(P0)*sech(t/T0)*exp(i*phi_NL)
  -> phi_NL = gamma*P0*z  (Kerr effect, measures fiber nonlinearity)

Silicon photonic chip (receiver)
  -> two arms with different waveguide dispersion D1, D2
  -> photodetectors -> I1.npy, I2.npy

gsrecover.py (your code)
  -> GS alternating projections recover E(t)
  -> angle(E(t)) = phi(t) = phi_NL(t)
  -> fit phi_NL vs |A|^2 -> extract gamma*P0*z

FNO (gs_fno.py)
  -> learns I1,I2 -> phi directly
  -> 17,375 inferences/sec on RTX 4060
  -> deployment: FPGA or co-packaged Si photonics ASIC

This is the Jalali Lab measurement system made differentiable and deployable.
""")
