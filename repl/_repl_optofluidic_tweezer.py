"""
_repl_optofluidic_tweezer.py
Optical tweezers x,y cell control + 100K throughput microfluidics +
integrated photonics on chip + aperture microscopy + image/laser channels
Run: py -3.12 repl/_repl_optofluidic_tweezer.py
"""

import numpy as np
import sympy as sp
from scipy.special import j1       # Airy pattern
from scipy.linalg import solve

kB   = 1.380649e-23
T    = 310.0   # K, body temperature
c    = 2.998e8
h    = 6.626e-34
e_c  = 1.602e-19
eta_water = 1e-3  # Pa*s dynamic viscosity at 20C

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: OPTICAL TWEEZERS — gradient force, trapping stiffness
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: OPTICAL TWEEZERS  F = -k_trap * x")
print("=" * 65)

print("""
  Gradient force on dielectric sphere (Rayleigh regime, r << lambda):
    F_grad = (2*pi*n_m*r^3/c) * ((m^2-1)/(m^2+2)) * grad(I)
    m = n_p / n_m  (relative refractive index)

  Scattering force (pushes along beam):
    F_scat = (8*pi*(n_m/c)) * (k*r)^4 * r^2 * ((m^2-1)/(m^2+2))^2 * I

  Stable 3D trap: F_grad > F_scat -> need tight focus (high NA > 1.2)

  Near focus, intensity I(r) ~ I0 * exp(-2r^2/w^2):
    F_grad ~ -k_trap * r   (Hookean spring!)
    k_trap = (4*pi*n_m*r^3/w^4) * ((m^2-1)/(m^2+2)) * P_laser
""")

# Trap stiffness for a typical bead
n_m  = 1.33    # water
n_p  = 1.59    # polystyrene bead
m_ri = n_p / n_m
r_bead = 0.5e-6   # 1 um diameter
lam0   = 1064e-9  # typical IR tweezer wavelength
w0     = lam0 / (np.pi * 1.25)  # beam waist for NA=1.25 oil objective: w0 ~ lambda/pi/NA... rough
P_laser_W = 0.050  # 50 mW at sample

# Clausius-Mossotti factor
CM = (m_ri**2 - 1) / (m_ri**2 + 2)

# Gradient force constant (lateral, per unit displacement, approximate)
# k_trap ~ (128 * pi^5 * n_m * r^6 * CM^2) / (3 * lambda^4) * (1/w0^2) * P
k_approx = (128 * np.pi**5 * n_m * r_bead**6 * CM**2) / (3 * lam0**4)
k_approx_per_I = k_approx / w0**2   # stiffness per W/m^2

# More practical: empirical k_trap ~ 0.1-1 pN/um/100mW
k_trap_pN_um = 0.3   # pN/um per 100mW (typical polystyrene in water)
k_trap = k_trap_pN_um * 1e-12 / 1e-6 * (P_laser_W / 0.1)   # N/m

print(f"  Bead: r={r_bead*1e6:.1f} um polystyrene, n_p={n_p}, n_m={n_m}, m={m_ri:.3f}")
print(f"  CM factor: {CM:.4f}")
print(f"  Laser: P={P_laser_W*1000:.0f} mW at {lam0*1e9:.0f} nm, NA=1.25")
print(f"  k_trap ~ {k_trap*1e6:.2f} pN/um  (empirical, lateral)")

# Brownian motion: equipartition <x^2> = kB*T / k_trap
x_rms = np.sqrt(kB * T / k_trap)
print(f"  Brownian position noise: x_rms = sqrt(kBT/k) = {x_rms*1e9:.1f} nm")
print(f"  Trap depth: U_0 = k*w0^2/2 = {k_trap*(w0**2)/2 / (kB*T):.1f} kBT  (>10 kBT for stable trap)")

# Stokes drag on sphere
gamma_drag = 6 * np.pi * eta_water * r_bead   # N*s/m
tau_relax  = gamma_drag / k_trap               # s
f_corner   = k_trap / (2 * np.pi * gamma_drag) # Hz (corner frequency)
print(f"\n  Stokes drag: gamma = 6*pi*eta*r = {gamma_drag:.2e} N*s/m")
print(f"  Relaxation time: tau = gamma/k = {tau_relax*1e3:.2f} ms")
print(f"  Corner frequency: fc = k/(2*pi*gamma) = {f_corner:.1f} Hz")
print(f"  (Power spectrum of Brownian motion: S(f) = kBT/(pi^2*gamma*(fc^2+f^2)))")

# ── x,y position control via AOD ─────────────────────────────────────────────
print("\n--- AOD (Acousto-Optic Deflector) x,y beam steering ---")
print("""
  AOD deflects beam by angle theta = lambda * f_RF / v_acoustic
  For TeO2 crystal: v_acoustic ~ 660 m/s
  Typical RF: 60-120 MHz -> scan range ~ 30 mrad at objective back aperture
  At sample (100x, f=1.8mm): scan range = f * theta ~ 50 um

  Two AODs (x and y) in series -> 2D raster or arbitrary position
  Switching time: beam fills crystal, t_switch ~ D_beam / v_acoustic ~ 5 us
  -> 200,000 positions/second theoretically achievable
""")

v_ac     = 660.0    # m/s TeO2
lam_nm   = 1064e-9
D_beam   = 3e-3     # 3mm beam diameter in AOD
f_RF_Hz  = 80e6     # 80 MHz center
delta_f  = 40e6     # +/- 20 MHz scan range

theta_center = lam_nm * f_RF_Hz / v_ac
theta_range  = lam_nm * delta_f / v_ac
t_switch     = D_beam / v_ac
rate_max     = 1 / t_switch

# At objective (100x, NA=1.25, f~1.8mm for 100x with 180mm tube lens)
f_obj    = 1.8e-3
x_range  = f_obj * theta_range * 2   # total x range at sample

print(f"  AOD center angle:  {np.degrees(theta_center)*1000:.2f} mrad")
print(f"  Scan half-range:   +/- {np.degrees(theta_range)*1000:.2f} mrad")
print(f"  Sample scan range: +/- {x_range*1e6/2:.1f} um  (100x obj, f={f_obj*1e3:.1f}mm)")
print(f"  Switching time:    {t_switch*1e6:.1f} us")
print(f"  Max rate:          {rate_max/1000:.0f} K positions/s")

# PID feedback loop for cell tracking
print("\n--- PID position control (cell tracking feedback) ---")
print("""
  Error signal from image centroid: e(t) = x_target - x_cell(t)
  AOD frequency command: f_RF(t) = f_center + Kp*e + Ki*integral(e) + Kd*de/dt

  Cell dynamics (overdamped, Re << 1):
    gamma * dx/dt = F_trap + F_drag_flow + F_Brownian
    = -k_trap*x + F_ext + xi(t)    where <xi(t)*xi(t')> = 2*kBT*gamma*delta(t-t')
""")
dt    = 1e-3    # 1 ms control loop
N_sim = 500
Kp    = 1.5 * k_trap / gamma_drag
Ki    = 0.0
Kd    = 0.0

x_cell = np.zeros(N_sim)
x_target = 5e-6  # 5 um step
integral_e = 0.0
noise_amp  = np.sqrt(2 * kB * T * gamma_drag / dt)

np.random.seed(7)
for i in range(1, N_sim):
    e         = x_target - x_cell[i-1]
    F_control = -k_trap * (x_cell[i-1] - x_target)   # trap follows AOD command
    F_brown   = noise_amp * np.random.randn()
    dx        = (F_control + F_brown) / gamma_drag * dt
    x_cell[i] = x_cell[i-1] + dx

# Settling time: first time |x - x_target| < 5% of x_target
settled = np.where(np.abs(x_cell - x_target) < 0.05 * x_target)[0]
t_settle = settled[0] * dt * 1000 if len(settled) > 0 else np.inf
x_ss_nm  = np.std(x_cell[400:]) * 1e9

print(f"  Step: {x_target*1e6:.0f} um,  k_trap={k_trap*1e6:.2f} pN/um,  P={P_laser_W*1000:.0f}mW")
print(f"  Settling time (5%): {t_settle:.0f} ms")
print(f"  Steady-state noise (last 100ms): {x_ss_nm:.1f} nm rms")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: 100K CELLS — THROUGHPUT MICROFLUIDICS + FLOW CYTOMETRY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: 100,000 CELLS/HR THROUGHPUT DESIGN")
print("=" * 65)

print("""
  Constraint: detect each cell as it passes laser interrogation zone
  Flow cytometry: cells in single file, spaced >= 2*cell_diameter
  Channel: w x h cross-section, focused sample stream width << w

  Throughput N_dot = Q_sample / V_cell_volume * packing_fraction
  Detection rate requires: dwell_time >= t_min for SNR
  dwell_time = laser_spot_size / v_cell
""")

# Channel geometry
w_ch   = 50e-6    # um channel width
h_ch   = 30e-6    # um channel height
r_cell = 7.5e-6   # um typical lymphocyte radius

# Poiseuille flow: Q = (w*h^3/12*eta) * (delta_P / L) for rectangular channel (approx)
# More accurately for w>>h: Q = (w*h^3)/(12*eta) * dP/dL
dP_Pa  = 5000.0   # Pa applied pressure
L_ch   = 0.02     # m channel length
Q_total = (w_ch * h_ch**3) / (12 * eta_water) * (dP_Pa / L_ch)

# Sample stream: hydrodynamic focusing to width w_s
w_s     = 2 * r_cell * 1.5   # 1.5x cell diameter
Q_ratio = w_s / w_ch          # rough sheath ratio
Q_sample = Q_total * Q_ratio

# Cell throughput
V_cell_um3 = (4/3) * np.pi * r_cell**3
V_cell_m3  = V_cell_um3 * 1e-18
cells_per_m3 = 1e15  # ~1 million cells/mL = 1e12/L = 1e9/m3... wait
# 1 million cells/mL = 1e6/1e-6 m3 = 1e12 cells/m3
cells_per_mL = 1e6                       # 1 million cells/mL (dilute suspension)
Q_mL_s       = Q_sample * 1e6           # m^3/s -> mL/s
N_dot_per_s  = cells_per_mL * Q_mL_s    # cells/s
N_dot_per_hr = N_dot_per_s * 3600

# Dwell time
v_cell  = Q_total / (w_ch * h_ch)   # mean velocity
w_spot  = 2e-6   # 2 um laser spot
t_dwell = w_spot / v_cell

print(f"  Channel: {w_ch*1e6:.0f} x {h_ch*1e6:.0f} um,  dP={dP_Pa:.0f} Pa,  L={L_ch*100:.0f} cm")
print(f"  Q_total = {Q_total*1e9:.2f} nL/s = {Q_total*1e6*60:.3f} uL/min")
print(f"  Mean velocity: {v_cell*1e3:.1f} mm/s")
print(f"  Sample stream: {w_s*1e6:.1f} um wide  (~{Q_ratio*100:.1f}% of total flow)")
print(f"  Cell concentration: {cells_per_mL:.0e} cells/mL")
print(f"  Throughput: {N_dot_per_s:.0f} cells/s = {N_dot_per_hr/1000:.0f}K cells/hr")
print(f"  Dwell time per cell: {t_dwell*1e6:.1f} us  (laser spot {w_spot*1e6:.0f} um)")

# SNR per cell
P_excite  = 5e-3    # 5 mW at 488 nm
sigma_fl  = 1e-16   # cm^2, fluorophore absorption cross-section ~ 10^-16 cm^2
QY        = 0.9     # quantum yield
N_dye     = 1e5     # fluorophores per cell (CD4+ T cell stained)
E_photon  = h * c / 488e-9
phi_inc   = P_excite / (E_photon * (np.pi*(w_spot/2)**2))  # photons/m^2/s
phi_abs   = sigma_fl * 1e-4 * phi_inc   # absorbed photons/s per dye molecule (convert cm^2 to m^2)
phi_emit  = phi_abs * QY * N_dye        # emitted photons/s from cell
N_photons = phi_emit * t_dwell          # photons collected during dwell

# Collection efficiency
NA_obj  = 0.75
Omega   = 2*np.pi*(1 - np.cos(np.arcsin(NA_obj)))  # sr solid angle
coll_eff = Omega / (4*np.pi) * 0.7                  # 70% optics transmission
N_det    = N_photons * coll_eff
SNR_shot = N_det / np.sqrt(N_det)  # shot-noise limited

print(f"\n  Fluorescence SNR per cell:")
print(f"  P_excite={P_excite*1000:.0f}mW, {N_dye:.0e} dye molecules, t_dwell={t_dwell*1e6:.1f}us")
print(f"  Emitted photons: {phi_emit:.2e} /s -> {N_photons:.2e} during dwell")
print(f"  Collected (NA={NA_obj}): {N_det:.2e} photons")
print(f"  Shot-noise SNR: {SNR_shot:.1f}  (>10 for reliable gating)")

# Scale to 100K/hr
print(f"\n  To hit exactly 100K cells/hr ({100000/3600:.1f} cells/s):")
target_per_s   = 100_000 / 3600
# Q_sample needed (m^3/s): N_dot = cells_per_mL * Q_mL_s -> Q = N_dot / conc
Q_s_needed_m3s = (target_per_s / cells_per_mL) * 1e-6   # mL/s -> m^3/s
Q_s_needed_nLs = Q_s_needed_m3s * 1e9
# Sheath ratio same -> total Q
Q_tot_needed   = Q_s_needed_m3s / Q_ratio
v_needed       = Q_tot_needed / (w_ch * h_ch) * 1e3   # mm/s
dP_needed      = 12 * eta_water * v_needed*1e-3 * L_ch / h_ch**2
print(f"    Required Q_sample: {Q_s_needed_nLs:.3f} nL/s")
print(f"    Required v_mean:   {v_needed:.2f} mm/s")
print(f"    Required dP:       {dP_needed:.1f} Pa = {dP_needed/6895:.3f} psi  (very achievable)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: INTEGRATED PHOTONICS ON CHIP — waveguide tweezer + ring sensor
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: INTEGRATED PHOTONICS ON CHIP")
print("=" * 65)

print("""
  On-chip optical tweezer: evanescent field from Si3N4 waveguide traps cells
  above the chip surface. No bulky objective needed.

  Evanescent trapping force ~ overlap integral of field with cell volume
  Advantage: scalable (many waveguides), CMOS fab compatible
  Limitation: 1D trap along waveguide -- need fluidics for y control

  Ring resonator biosensor:
    Cell/molecule landing on ring -> delta_n -> resonance shift delta_lambda
    delta_lambda = lambda_res * (delta_n_eff / n_g)
    Sensitivity S = delta_lambda / delta_n  [nm per RIU]
""")

# Si3N4 waveguide evanescent field
n_core = 2.00   # Si3N4
n_clad = 1.33   # water cladding
lam_wg = 785e-9 # nm trapping/excitation
w_wg   = 800e-9 # nm waveguide width
h_wg   = 300e-9 # nm waveguide height

# Effective index (approximate slab formula for TE)
NA_wg   = np.sqrt(n_core**2 - n_clad**2)
V_wg    = 2*np.pi/lam_wg * h_wg/2 * NA_wg   # V-number

# Evanescent field penetration depth
k0      = 2*np.pi / lam_wg
kappa_ev = np.sqrt((n_clad * k0)**2 - (n_core * k0 * 0.85)**2)  # rough beta ~ 0.85*k0*n_core
if kappa_ev**2 < 0:
    kappa_ev = 1e6  # fallback
delta_ev = 1 / np.abs(kappa_ev)

# Simpler: delta = lambda / (4*pi * sqrt(n_eff^2 - n_clad^2))
n_eff_approx = 0.5*(n_core + n_clad)
delta_ev2 = lam_wg / (4*np.pi * np.sqrt(n_core**2 - n_clad**2))

print(f"  Si3N4 waveguide: {w_wg*1e9:.0f}x{h_wg*1e9:.0f} nm, n_core={n_core}, n_clad={n_clad}")
print(f"  Evanescent penetration depth: {delta_ev2*1e9:.1f} nm")
print(f"  Trapping range above surface: ~{delta_ev2*3*1e9:.0f} nm (3 decay lengths)")

# Ring resonator sensitivity
lam_res   = 1310e-9  # resonance wavelength
n_g_Si3N4 = 2.15     # group index
FSR_nm    = lam_res**2 / (n_g_Si3N4 * 200e-6) * 1e9  # nm, L=200 um ring
Q_factor  = 50000    # loaded Q
lw_pm     = lam_res*1e9 / Q_factor * 1e3   # linewidth in pm
S_RIU     = lam_res * 1e9 / n_g_Si3N4 * 0.1  # nm/RIU (rough, depends on confinement)
LOD_RIU   = lw_pm*1e-3 / (10 * S_RIU)   # limit of detection (1/10 linewidth)

print(f"\n  Ring resonator (Si3N4, L=200um):")
print(f"  FSR = {FSR_nm:.2f} nm,  Q = {Q_factor:.0e},  linewidth = {lw_pm:.2f} pm")
print(f"  Bulk sensitivity S ~ {S_RIU:.1f} nm/RIU")
print(f"  LOD ~ {LOD_RIU:.2e} RIU  (1/10 linewidth shift detectable)")
print(f"  Cell binding (delta_n~0.01): shift = {S_RIU*0.01:.3f} nm = {S_RIU*0.01*1e3:.1f} pm")

# On-chip flow cytometry: waveguide excitation + collection
print("\n  On-chip flow cytometry channels:")
print("  1. Excitation waveguide crosses microfluidic channel at 90 deg")
print("  2. Collection waveguide picks up forward scatter + fluorescence")
print("  3. No free-space optics -- full PIC + PDMS integration")
print()

channels = {
    "Excitation (488nm)":    (488,  "GaN or frequency-doubled Si3N4"),
    "Scatter (488nm, 0deg)": (488,  "forward scatter, large particles"),
    "FL green (530/30)":     (530,  "FITC, GFP -- viability"),
    "FL red (660/20)":       (660,  "PE-Cy5, PerCP -- surface markers"),
    "FL NIR (785nm exc)":    (785,  "Si APD on-chip, deep tissue"),
}
for ch, (lam_c, desc) in channels.items():
    E_ph_eV = h * c / (lam_c*1e-9) / e_c
    print(f"  {ch:28s}  {lam_c}nm  {E_ph_eV:.2f}eV  {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: APERTURE MICROSCOPY — NA, Abbe, NSOM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: APERTURE MICROSCOPY — NA, ABBE, NSOM")
print("=" * 65)

print("""
  Abbe diffraction limit (coherent):   d = lambda / (2*NA)
  Rayleigh criterion (incoherent):      d = 0.61*lambda / NA
  Sparrow criterion:                    d = 0.47*lambda / NA

  NA = n * sin(theta_max)    n=immersion index
  Depth of field:  DOF = lambda / (2*NA^2)  +  n/(M*NA) * pixel_size

  Near-field aperture (NSOM): probe with sub-lambda hole
  Resolution ~ aperture diameter a, independent of lambda if a < lambda/10
""")

objectives = {
    "4x air":          (0.10, 1.00, 4),
    "10x air":         (0.30, 1.00, 10),
    "40x air":         (0.65, 1.00, 40),
    "40x water":       (0.80, 1.33, 40),
    "60x oil":         (1.25, 1.515, 60),
    "100x oil":        (1.40, 1.515, 100),
    "100x TIRF oil":   (1.49, 1.515, 100),
}

lam_vis = 488e-9   # 488nm excitation

print(f"\n  At lambda={lam_vis*1e9:.0f} nm:")
print(f"  {'Objective':18s}  {'NA':>5}  {'Abbe(nm)':>9}  {'Rayleigh(nm)':>13}  {'DOF(um)':>8}")
print("-" * 60)
for name, (NA, n_imm, M) in objectives.items():
    d_abbe    = lam_vis / (2*NA) * 1e9
    d_ray     = 0.61*lam_vis / NA * 1e9
    DOF_um    = lam_vis / (2*NA**2) * 1e6
    print(f"  {name:18s}  {NA:>5.2f}  {d_abbe:>9.1f}  {d_ray:>13.1f}  {DOF_um:>8.3f}")

# ── Airy disk pattern ─────────────────────────────────────────────────────────
print("\n--- Airy disk intensity profile I(r) = [2*J1(x)/x]^2 ---")
print("  x = pi*r*NA / (lambda * M)  (normalized radius at image plane)")
NA_100x = 1.40
r_um = np.array([0, 50, 100, 150, 213, 300])   # nm at sample
print(f"  NA={NA_100x}, lambda={lam_vis*1e9:.0f}nm")
print(f"  {'r (nm)':>8}  {'I/I0':>10}  {'note'}")
for r_nm in r_um:
    r_m = r_nm * 1e-9
    x   = np.pi * r_m * NA_100x / lam_vis
    if x < 1e-6:
        I_norm = 1.0
    else:
        I_norm = (2*j1(x)/x)**2
    note = " <- first zero (Airy radius)" if abs(r_nm - 213) < 5 else ""
    print(f"  {r_nm:>8}  {I_norm:>10.6f}{note}")

airy_r_nm = 0.61 * lam_vis / NA_100x * 1e9
print(f"\n  Airy radius = 0.61*lambda/NA = {airy_r_nm:.1f} nm  (first zero of J1)")

# ── NSOM: near-field aperture ─────────────────────────────────────────────────
print("\n--- NSOM (Near-field Scanning Optical Microscopy) ---")
print("""
  Aperture probe: aluminum-coated fiber tip, hole diameter a ~ 50-100 nm
  Works in evanescent regime: z < a (tip-sample gap < aperture diameter)

  Resolution: ~ a  (NOT lambda-limited)
  Drawback: throughput ~ (a/lambda)^4  (very low for small apertures)
  Aperture transmission:
    T_ap ~ (a/lambda)^4  (Bethe theory for circular aperture)
""")
a_sizes = [20, 50, 100, 200, 500]   # nm aperture diameter
lam_ex  = 488.0   # nm
print(f"  lambda={lam_ex} nm excitation:")
print(f"  {'a (nm)':>8}  {'a/lambda':>10}  {'T_rel (a/lam)^4':>18}  {'Resolution':>12}")
for a_nm in a_sizes:
    ratio  = a_nm / lam_ex
    T_rel  = ratio**4
    print(f"  {a_nm:>8}  {ratio:>10.3f}  {T_rel:>18.3e}  {a_nm:>10} nm")

print("\n  50 nm aperture: T = (50/488)^4 = 1.1e-5  -> need bright source or SPAD detector")
print("  Alternative: tip-enhanced (TERS/TIP) -> field enhancement 10^4-10^6x via plasmons")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: IMAGE + LASER CHANNELS — PSF, Nyquist, pixel budget
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: IMAGE + LASER CHANNELS — PSF, NYQUIST, PIXEL BUDGET")
print("=" * 65)

print("""
  Confocal PSF (lateral):
    PSF_confocal(r) ~ [2*J1(u)/u]^4  (illumination x detection squared)
    FWHM_confocal  ~ 0.37*lambda/NA  (sqrt(2) narrower than widefield)

  Sampling theorem: pixel_size <= Abbe_limit / 2  (Nyquist)
  Typical: use 2.3x oversampling  (pixel = Abbe/2.3)

  Camera pixel budget:
    FOV = N_pixels * pixel_size_sample
    pixel_size_sample = camera_pixel / M_objective
""")

# Camera specs for flow cytometry imaging
cam_pixel_um = 6.5    # um physical pixel (sCMOS)
M_obj        = 40     # 40x objective
NA_img       = 0.65

pix_sample_um = cam_pixel_um / M_obj   # um at sample
pix_sample_nm = pix_sample_um * 1e3

d_abbe_nm = lam_vis / (2*NA_img) * 1e9
nyquist_nm = d_abbe_nm / 2

print(f"  40x objective, NA={NA_img}, lambda={lam_vis*1e9:.0f}nm, camera pixel={cam_pixel_um}um")
print(f"  Abbe limit:        {d_abbe_nm:.1f} nm")
print(f"  Nyquist pixel:     {nyquist_nm:.1f} nm  (need pixel < this)")
print(f"  Actual pixel:      {pix_sample_nm:.1f} nm at sample")
sampling_ratio = pix_sample_nm / nyquist_nm
print(f"  Sampling ratio:    {sampling_ratio:.1f}x  "
      f"({'UNDERSAMPLED -- aliasing!' if sampling_ratio > 1 else 'OK, Nyquist satisfied'})")

# Channel table: laser lines, filters, detectors
print("\n  Multicolor channel configuration (typical flow cytometer):")
channels_fc = [
    #  name,           exc_nm, em_band,  detector,      signal_use
    ("FSC",            488, "488/10",  "Si photodiode", "cell size"),
    ("SSC",            488, "488/10",  "Si photodiode", "granularity"),
    ("FITC/GFP",       488, "530/30",  "PMT",           "viability/transfection"),
    ("PE",             488, "575/26",  "PMT",           "surface markers"),
    ("PE-Cy5",         488, "660/20",  "PMT",           "CD4/CD8"),
    ("DAPI",           405, "450/50",  "PMT",           "DNA content / dead"),
    ("APC",            633, "660/20",  "PMT",           "high-sensitivity marker"),
    ("APC-Cy7",        633, "785/60",  "Si APD",        "dump channel / viability"),
]
print(f"  {'Channel':12s}  {'Exc':>5}  {'Em band':>10}  {'Detector':>14}  Use")
print("-" * 65)
for ch_name, exc, em, det, use in channels_fc:
    print(f"  {ch_name:12s}  {exc:>5}nm  {em:>10}  {det:>14}  {use}")

# Data rate calculation
print(f"\n  Data rate at 100K cells/hr with 8-channel 16-bit:")
N_channels = 8
bits       = 16
N_per_s    = 100000 / 3600
bytes_per_cell = N_channels * (bits // 8)
bytes_per_s    = N_per_s * bytes_per_cell
print(f"  Bytes/cell: {bytes_per_cell}")
print(f"  Data rate:  {bytes_per_s:.1f} bytes/s = {bytes_per_s/1000:.2f} kB/s  (trivial for modern DAQ)")
print(f"  Per 1hr run: {bytes_per_s*3600/1e6:.2f} MB  (fits in RAM)")

# If imaging (2D image per cell)
N_px_cell  = 64*64   # 64x64 pixel image per cell
img_bytes  = N_per_s * N_px_cell * 2   # 16-bit
print(f"\n  If imaging each cell (64x64 px, 16-bit):")
print(f"  Data rate:  {img_bytes/1e6:.2f} MB/s  (need fast SSD)")
print(f"  Per 1hr run: {img_bytes*3600/1e9:.2f} GB  (GPU-accelerated analysis needed)")

print("\nDone.")
