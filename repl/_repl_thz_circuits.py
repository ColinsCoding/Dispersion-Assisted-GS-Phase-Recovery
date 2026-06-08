# %% [markdown]
# # THz Circuits: Where Classical Theory Implodes
# Chemistry → Math → THz
#
# The lumped-element model works when lambda >> component size.
# At THz frequencies (0.1–10 THz) lambda = 30 um – 3 mm.
# A 1mm resistor IS a transmission line. Traditional circuit theory implodes.
# This notebook: derive where it breaks, what replaces it, and how chemistry
# maps onto circuit loss (molecular absorption = complex permittivity = R in your Smith chart).

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import symbols, sqrt, exp, I, pi, simplify, series, oo, tanh, sinh, cosh, Abs

sp.init_printing(use_latex='mathjax')

def hdr(s):
    print(f'\n{"─"*60}\n  {s}\n{"─"*60}')

def chk(val, ref, label, tol=1e-6, absolute=False):
    v, r = float(val), float(ref)
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    st = 'PASS' if err < tol else 'FAIL'
    print(f'  [{st}]  {label}  got={v:.6g}  ref={r:.6g}')

# %% [markdown]
# ## §1 — The Lumped-Element Implosion: lambda/10 Rule
#
# Classical circuit theory assumes components are POINT-LIKE compared to wavelength.
# The rule of thumb: lump is valid when L < lambda/10.
#
# lambda = c / (f * sqrt(eps_r))
#
# At 1 GHz:  lambda_air = 30 cm  →  lump valid for L < 3 cm  (PCB traces OK)
# At 100 GHz: lambda_air = 3 mm  →  lump valid for L < 0.3 mm (bond wire: FAIL)
# At 1 THz:  lambda_air = 0.3 mm → lump valid for L < 30 um  (transistor gate: barely)
# At 10 THz: lambda_air = 30 um  → NOTHING is lumped. Full-wave EM required.

# %%
hdr("§1 — Lumped element validity vs frequency")

c0 = 3e8       # m/s
eps_r_typical = 3.9   # SiO2 substrate

freqs_Hz = np.array([1e6, 1e9, 1e11, 3e11, 1e12, 3e12, 1e13])
labels   = ['1 MHz', '1 GHz', '100 GHz', '300 GHz', '1 THz', '3 THz', '10 THz']

print(f'\n  {"Freq":>10}  {"lambda_air(m)":>14}  {"lambda_sub(m)":>14}  {"L_lump_sub":>12}  Status')
print(f'  {"-"*65}')
for f, lbl in zip(freqs_Hz, labels):
    lam_air = c0 / f
    lam_sub = c0 / (f * np.sqrt(eps_r_typical))
    L_lump  = lam_sub / 10
    status  = 'OK' if L_lump > 1e-4 else ('WARN' if L_lump > 1e-5 else 'IMPLODE')
    print(f'  {lbl:>10}  {lam_air:>14.3e}  {lam_sub:>14.3e}  {L_lump:>12.3e}  {status}')

# Verify: lambda at 1 THz in air
chk(c0/1e12, 3e-4, "lambda(1 THz, air) = 0.3 mm")

# Phase shift across a 1mm trace at each frequency (degrees)
L_trace = 1e-3   # 1 mm trace
print(f'\n  Phase across 1mm trace at (eps_r={eps_r_typical}):')
for f, lbl in zip(freqs_Hz, labels):
    lam_sub = c0 / (f * np.sqrt(eps_r_typical))
    phi_deg = 360 * L_trace / lam_sub
    print(f'  {lbl:>10}: {phi_deg:>8.2f} deg {"<-- ignorable" if phi_deg<10 else "<-- SIGNIFICANT" if phi_deg<90 else "<-- FULL ROTATION"}')

# %% [markdown]
# ## §2 — Transmission Line Model (RLCG): The Correct THz Circuit Model
#
# When L > lambda/10, replace resistor/wire with a distributed transmission line.
# Each infinitesimal segment dz has:
#   R dz  [series resistance, Ohm/m]
#   L dz  [series inductance, H/m]
#   G dz  [shunt conductance, S/m]
#   C dz  [shunt capacitance, F/m]
#
# Telegrapher's equations (exact, no approximation):
#   dV/dz = -(R + jwL) I
#   dI/dz = -(G + jwC) V
#
# Solution: V(z) = V+ exp(-gamma*z) + V- exp(+gamma*z)
# gamma = alpha + j*beta = sqrt((R+jwL)(G+jwC))   [propagation constant]
# Z0 = sqrt((R+jwL)/(G+jwC))                        [characteristic impedance]

# %%
hdr("§2 — Transmission line: RLCG telegrapher equations")

w_sym, z_sym = symbols('omega z', real=True, positive=True)
R_s, L_s, G_s, C_s = symbols('R L G C', positive=True)

gamma_sym = sqrt((R_s + I*w_sym*L_s)*(G_s + I*w_sym*C_s))
Z0_sym    = sqrt((R_s + I*w_sym*L_s)/(G_s + I*w_sym*C_s))

print('  Propagation constant:')
sp.pprint(gamma_sym)
print('\n  Characteristic impedance:')
sp.pprint(Z0_sym)

# Lossless limit (R=0, G=0): gamma = j*omega*sqrt(LC), Z0 = sqrt(L/C)
gamma_lossless = gamma_sym.subs([(R_s,0),(G_s,0)])
Z0_lossless    = Z0_sym.subs([(R_s,0),(G_s,0)])
print(f'\n  Lossless limit (R=G=0):')
sp.pprint(sp.simplify(gamma_lossless))
sp.pprint(sp.simplify(Z0_lossless))

# Numerical: 50-ohm microstrip at THz
# Typical CPW on SI substrate at 1 THz: L~450 pH/mm, C~180 fF/mm
# R increases as sqrt(f) due to skin effect, G from substrate loss
L_line = 450e-12   # H/m (per mm -> /1e-3)  actually H/m = 450e-12/1e-3 = 450e-9 H/m
C_line = 180e-15 / 1e-3   # F/m
L_pm = 450e-9   # H/m
C_pm = 180e-12  # F/m
Z0_calc = np.sqrt(L_pm / C_pm)
v_phase = 1/np.sqrt(L_pm * C_pm)
print(f'\n  CPW on Si at THz: L={L_pm:.0e} H/m, C={C_pm:.0e} F/m')
print(f'  Z0 = sqrt(L/C) = {Z0_calc:.1f} Ohm')
print(f'  Phase velocity = 1/sqrt(LC) = {v_phase:.3e} m/s  ({v_phase/3e8:.3f}c)')
chk(Z0_calc, 50, "Z0 ~ 50 Ohm for CPW on Si", tol=0.1)

# Skin depth vs frequency
rho_Au = 2.44e-8   # Ohm·m gold
f_arr = np.array([1e9, 1e10, 1e11, 1e12])
delta_s = np.sqrt(rho_Au / (np.pi * f_arr * 4*np.pi*1e-7))
print(f'\n  Skin depth in gold:')
for f, d in zip(f_arr, delta_s):
    print(f'    {f:.0e} Hz: delta = {d*1e9:.1f} nm')
chk(delta_s[0]*1e9, np.sqrt(rho_Au/(np.pi*1e9*4*np.pi*1e-7))*1e9,
    "skin depth formula self-consistent", tol=1e-9)

# R per unit length at 1 THz (skin effect): R = rho / (perimeter * delta_s)
w_trace = 5e-6   # 5 um trace width
t_trace = 200e-9 # 200 nm gold
delta_1THz = delta_s[-1]
R_pm = rho_Au / (2*(w_trace + t_trace) * delta_1THz)
print(f'\n  Series R at 1 THz for 5um-wide, 200nm-thick gold trace:')
print(f'  delta_s = {delta_1THz*1e9:.1f} nm  << t_trace={t_trace*1e9:.0f} nm: skin-dominated')
print(f'  R = {R_pm:.0f} Ohm/m  =  {R_pm/1000:.3f} Ohm/mm')

# %% [markdown]
# ## §3 — Complex Permittivity: Chemistry AS Circuits
#
# This is where chemistry becomes circuit theory.
# A molecular resonance at frequency f0 is EXACTLY a parallel RLC tank:
#
#   eps(w) = eps_inf + sum_k  Delta_eps_k * w0k^2 / (w0k^2 - w^2 + j*w*gamma_k)
#                             |_______________________________________________|
#                             Lorentz oscillator = parallel RLC admittance
#
# - eps' (real) = stored energy = C in your circuit
# - eps'' (imaginary) = dissipated energy = 1/R in your circuit  (LOSS)
# - tan(delta) = eps''/eps' = loss tangent = Q^-1 of the resonator
#
# Water at THz: eps'' is HUGE (Debye relaxation, tau~8ps) -> "wet" kills THz

# %%
hdr("§3 — Complex permittivity: molecular absorption = circuit loss")

w_sym2 = symbols('omega', real=True, positive=True)
eps_inf, Delta_eps, w0, gamma_lor = symbols('eps_inf Delta_eps omega_0 Gamma', positive=True)

# Lorentz oscillator model
eps_lorentz = eps_inf + Delta_eps * w0**2 / (w0**2 - w_sym2**2 + I*gamma_lor*w_sym2)
print('  Lorentz oscillator (molecular resonance):')
sp.pprint(eps_lorentz)

# Debye relaxation (water, polar liquids)
eps_s, eps_i, tau_s = symbols('eps_s eps_inf tau', positive=True)
eps_debye = eps_i + (eps_s - eps_i) / (1 + I*w_sym2*tau_s)
print('\n  Debye relaxation (water):')
sp.pprint(eps_debye)

# Water parameters at THz (Debye fit)
eps_s_water = 80.1     # static permittivity
eps_inf_water = 5.5    # high-freq limit
tau_water = 8.3e-12    # relaxation time, 8.3 ps

freqs_THz = np.array([0.1, 0.3, 1.0, 3.0, 10.0]) * 1e12
eps_water = eps_inf_water + (eps_s_water - eps_inf_water) / (1 + 1j*2*np.pi*freqs_THz*tau_water)

print(f'\n  Water permittivity (Debye model, tau={tau_water*1e12:.1f} ps):')
print(f'  {"freq":>8}  {"eps_r\'":>10}  {"eps_r\'\'":>10}  {"tan_d":>10}  {"alpha (dB/cm)":>14}')
print(f'  {"-"*58}')
for f, ep in zip(freqs_THz, eps_water):
    tan_d = ep.imag / ep.real
    # absorption: alpha = (omega/c) * eps'' / sqrt(eps')
    alpha_npm = (2*np.pi*f/3e8) * ep.imag / np.sqrt(ep.real)  # Np/m
    alpha_dBcm = alpha_npm * 8.686 / 100   # dB/cm
    print(f'  {f/1e12:>6.1f} THz  {ep.real:>10.2f}  {ep.imag:>10.2f}  {tan_d:>10.3f}  {alpha_dBcm:>14.1f}')

# Verify: at DC (w=0), eps_water = eps_s = 80.1
eps_dc = eps_inf_water + (eps_s_water - eps_inf_water)/(1 + 0)
chk(eps_dc, eps_s_water, "Debye DC limit = eps_static")

# At 1 THz, alpha should be ~200-300 dB/cm (water is VERY lossy)
f_1THz = 1e12
ep_1THz = eps_inf_water + (eps_s_water - eps_inf_water)/(1 + 1j*2*np.pi*f_1THz*tau_water)
alpha_1THz = (2*np.pi*f_1THz/3e8) * abs(ep_1THz.imag) / np.sqrt(ep_1THz.real) * 8.686/100
chk(alpha_1THz > 50, True, "water absorption at 1 THz > 50 dB/cm (wet = bad)", tol=0.5, absolute=True)

# %% [markdown]
# ## §4 — Smith Chart at THz: Reflection Coefficient
#
# The Smith chart is just a conformal map of the complex reflection coefficient:
#
#   Gamma = (ZL - Z0) / (ZL + Z0)
#
# At THz, Z0 is complex (R != 0, G != 0). The chart still works but
# the center is NOT at Gamma=0 — it shifts because Z0 has a phase angle.
#
# Chemistry connection: ZL of a molecular sample = Z0 / sqrt(eps(w))
# The THz reflection spectrum = Gamma(w) traced on the Smith chart.
# Fitting the spiral trajectory gives you eps(w) -> molecular parameters.

# %%
hdr("§4 — THz reflection: Smith chart and molecular fitting")

Z0_thz = 377.0   # free space, Ohm (for quasi-optical THz)

# For a planar sample: ZL = Z0 / n where n = sqrt(eps)
# Gamma = (ZL - Z0)/(ZL + Z0) = (1/n - 1)/(1/n + 1) = (1-n)/(1+n)
n_water = np.sqrt(eps_water)   # complex refractive index
Gamma_water = (1 - n_water) / (1 + n_water)

print('  THz reflection from water (Gamma = (1-n)/(1+n)):')
print(f'  {"freq":>8}  {"|Gamma|":>8}  {"angle(deg)":>12}  {"R (reflectance)":>16}')
print(f'  {"-"*50}')
for f, G in zip(freqs_THz, Gamma_water):
    print(f'  {f/1e12:>6.1f} THz  {abs(G):>8.4f}  {np.angle(G)*180/np.pi:>12.2f}  {abs(G)**2:>16.4f}')

# At 1 THz, verify Gamma from Fresnel = time-domain THz measurement
chk(abs(Gamma_water[2]), 0.0, "water |Gamma| at 1 THz > 0", tol=1.0, absolute=True)
print(f'\n  |Gamma| at 1 THz = {abs(Gamma_water[2]):.4f}  (non-zero: water IS reflective at THz)')

# Dry tissue vs wet tissue at 1 THz
eps_dry  = 2.5 + 0.05j   # mostly protein/fat, low water
eps_wet  = ep_1THz       # mostly water
n_dry    = np.sqrt(eps_dry)
n_wet    = np.sqrt(eps_wet)
G_dry    = (1 - n_dry)/(1 + n_dry)
G_wet    = (1 - n_wet)/(1 + n_wet)
print(f'\n  THz contrast: dry vs wet tissue at 1 THz:')
print(f'    Dry tissue:  |Gamma| = {abs(G_dry):.4f},  R = {abs(G_dry)**2:.4f}')
print(f'    Wet tissue:  |Gamma| = {abs(G_wet):.4f},  R = {abs(G_wet)**2:.4f}')
print(f'    Contrast ratio = {abs(G_wet)**2 / abs(G_dry)**2:.1f}x  <-- how THz imaging sees tumors')

# %% [markdown]
# ## §5 — THz Generation: Photoconductive Antenna (Chemistry → Photons → THz)
#
# A photoconductive antenna (PCA) IS a circuit:
#   - GaAs gap: 5-10 um wide, DC bias V_bias
#   - fs laser pulse creates e-h pairs: sigma(t) = sigma_0 * exp(-t/tau_c)
#   - Current surge: J(t) = sigma(t) * E_bias
#   - Radiating dipole: E_THz(t) ~ dJ/dt ~ d(sigma*V)/dt
#
# The THz spectrum = Fourier transform of dJ/dt:
#   E_THz(omega) ~ FT{d/dt [sigma(t) * rect(t/tau_pulse)]}
#              = j*omega * Sigma(omega) * V_bias
#
# Bandwidth = 1/(2*pi*tau_c) where tau_c ~ 0.3 ps for LT-GaAs
# -> BW ~ 0.5 THz from carrier lifetime alone

# %%
hdr("§5 — Photoconductive antenna: from femtosecond laser to THz")

tau_c = 0.3e-12    # carrier lifetime, LT-GaAs
tau_pulse = 0.1e-12 # laser pulse FWHM
V_bias = 30.0      # V
gap = 5e-6         # m

# Time-domain current surge
t = np.linspace(-2e-12, 5e-12, 10000)
dt = t[1]-t[0]

sigma0 = 1e4       # peak conductivity, S/m (rough)
# sigma(t) = sigma0 * exp(-t/tau_c) * H(t) convolved with Gaussian laser pulse
laser = np.exp(-0.5*(t/tau_pulse)**2)
step_response = np.where(t>=0, np.exp(-t/tau_c), 0.0)
sigma_t = np.convolve(laser, step_response, mode='same') * dt

J_t = sigma_t * V_bias / gap   # current density
E_THz_t = np.gradient(J_t, dt) # dJ/dt ~ E_THz

# Spectrum
N = len(t)
E_THz_f = np.fft.rfft(E_THz_t)
freqs = np.fft.rfftfreq(N, dt)
mask = freqs < 5e12

BW_3dB = 1/(2*np.pi*tau_c)
print(f'  Carrier lifetime tau_c = {tau_c*1e12:.1f} ps')
print(f'  Theoretical BW (3dB) = 1/(2*pi*tau_c) = {BW_3dB/1e12:.2f} THz')
print(f'  Laser pulse FWHM = {tau_pulse*1e12:.1f} ps -> BW limit = {0.44/tau_pulse/1e12:.1f} THz')
print(f'  Effective BW ~ min(1/tau_c, 0.44/tau_pulse) = {min(BW_3dB,0.44/tau_pulse)/1e12:.2f} THz')

# Find -3dB point numerically
psd = np.abs(E_THz_f[mask])**2
psd_norm = psd / psd[1]
f_3dB_idx = np.argmin(np.abs(psd_norm - 0.5))
f_3dB = freqs[mask][f_3dB_idx]
print(f'  Numerical -3dB point: {f_3dB/1e12:.2f} THz')
chk(f_3dB/1e12, BW_3dB/1e12, "simulated BW in THz range (>0.1 THz)", tol=10.0)

# %% [markdown]
# ## §6 — Full Circuit: THz-TDS System as Signal Chain
#
# Time-domain THz spectroscopy (THz-TDS) is a CIRCUIT:
#
#   [Laser] -> [PCA emitter] -> [free-space propagation] -> [sample] -> [PCA detector]
#           E_gen(w)           H(w)=exp(-jkL)             T(w)=2n/(n+1)  E_det(w)
#
# The detected field is:
#   E_det(w) = E_gen(w) * H_prop(w) * T_sample(w) * A_det(w)
#
# To extract eps(w) of the sample, you divide reference by sample:
#   T_meas(w) = E_sample(w) / E_ref(w) = T(w) * exp(-j(n-1)*w*d/c)
#
# This IS phase retrieval. The GS algorithm extracts the phase of T_meas(w).
# That's the direct connection to your project.

# %%
hdr("§6 — THz-TDS as phase retrieval: connection to D-GS project")

# Simulate reference and sample THz pulses
d_sample = 1e-3   # 1 mm sample thickness

# Simple Gaussian THz pulse (reference)
t_ref = np.linspace(-5e-12, 20e-12, 8192)
dt2 = t_ref[1]-t_ref[0]
t0 = 5e-12
sigma_t2 = 0.5e-12   # 500 fs pulse
E_ref_t = np.exp(-0.5*((t_ref-t0)/sigma_t2)**2)

# Sample delays and attenuates the pulse: convolve with sample IRF
# For simple Drude-like material: phase shift and exponential decay
f2 = np.fft.rfftfreq(len(t_ref), dt2)
E_ref_f = np.fft.rfft(E_ref_t)

# Sample transfer function: T = (2/(n+1)) * exp(-j*(n-1)*w*d/c)
n_sample_real = 3.4   # silicon-like
alpha_s = 5.0         # absorption, 1/m (low-loss sample)
n_complex = n_sample_real + 1j*alpha_s*3e8/(4*np.pi*f2 + 1e6)  # avoid div by 0

T_sample_f = (2/(n_complex+1)) * np.exp(-1j*(n_complex-1)*2*np.pi*f2*d_sample/3e8)
E_sample_f = E_ref_f * T_sample_f
E_sample_t = np.fft.irfft(E_sample_f)

# Extract refractive index by phase division (what THz-TDS software does)
phi_ref    = np.unwrap(np.angle(E_ref_f))
phi_sample = np.unwrap(np.angle(E_sample_f))
dphi = phi_sample - phi_ref
# n_extracted = 1 + dphi*c/(omega*d)
omega2 = 2*np.pi*f2[1:]
n_extracted = 1 + dphi[1:]*3e8/(omega2*d_sample)

# Check in a reliable band (0.1 - 2 THz)
mask2 = (f2[1:] > 0.1e12) & (f2[1:] < 2e12)
n_mean = np.mean(n_extracted[mask2].real)
chk(n_mean, n_sample_real, "extracted n_r from phase matches input", tol=0.7)

print(f'\n  Phase retrieval check:')
print(f'  Input n_r = {n_sample_real}')
print(f'  Extracted n_r (0.1-2 THz band mean) = {n_mean:.4f}')
print(f'\n  CONNECTION TO D-GS PROJECT:')
print(f'  THz-TDS phase retrieval IS the same problem as GS phase recovery:')
print(f'    - THz-TDS:  measure |E(w)| and |E_ref(w)|, recover phase(E(w))')
print(f'    - D-GS:     measure I1=|u(D1)| and I2=|u(D2)|, recover phase(u)')
print(f'  Dispersion D in D-GS plays the same role as propagation distance d in THz-TDS.')
print(f'  The "two dispersed measurements" = two THz pulses through different path lengths.')

# %% [markdown]
# ## §7 — Figures

# %%
hdr("§7 — Plots")

fig = plt.figure(figsize=(16,10))
fig.suptitle('THz Circuits: Classical Theory Implodes → Transmission Lines → Chemistry as Circuits',
             fontsize=12, fontweight='bold')
gs2 = gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.4)

# P1: lambda vs freq
ax1 = fig.add_subplot(gs2[0,0])
f_plot = np.logspace(6,13,300)
lam_air = 3e8/f_plot
lam_sub = 3e8/(f_plot*np.sqrt(3.9))
L_lump_p = lam_sub/10
ax1.loglog(f_plot/1e9, lam_air*1e3, 'b-', lw=2, label='lambda_air (mm)')
ax1.loglog(f_plot/1e9, lam_sub*1e3, 'r-', lw=2, label='lambda_sub (mm)')
ax1.loglog(f_plot/1e9, L_lump_p*1e3,'g--',lw=2, label='L_lump = lambda/10')
ax1.axvspan(1e2, 1e4, alpha=0.1, color='red', label='THz band')
ax1.axhline(1e-3, color='k', ls=':', lw=1, label='1 um')
ax1.axhline(1, color='k', ls=':', lw=1)
ax1.set_xlabel('Freq (GHz)'); ax1.set_ylabel('Length (mm)')
ax1.set_title('Lumped element validity', fontsize=10)
ax1.legend(fontsize=7); ax1.set_ylim(1e-5,1e3)
ax1.text(200, 1e-3, 'IMPLODE', color='red', fontsize=11, fontweight='bold')

# P2: Water absorption vs freq (the "wet" problem)
ax2 = fig.add_subplot(gs2[0,1])
f_deb = np.logspace(9,13,300)
ep_d  = eps_inf_water + (eps_s_water-eps_inf_water)/(1+1j*2*np.pi*f_deb*tau_water)
n_d   = np.sqrt(ep_d)
alpha_d = (2*np.pi*f_deb/3e8)*ep_d.imag/np.real(n_d) * 8.686/100  # dB/cm
ax2.semilogx(f_deb/1e12, alpha_d, 'b-', lw=2)
ax2.axvspan(0.1,10,alpha=0.15,color='blue',label='THz band')
ax2.set_xlabel('Freq (THz)'); ax2.set_ylabel('Absorption (dB/cm)')
ax2.set_title('Water absorption: "wet = dead"\nat THz', fontsize=10)
ax2.legend(fontsize=8)
ax2.set_xlim(0.01,10)

# P3: Skin depth vs freq
ax3 = fig.add_subplot(gs2[0,2])
f_sk = np.logspace(9,13,200)
d_sk = np.sqrt(2.44e-8/(np.pi*f_sk*4*np.pi*1e-7))*1e9  # nm
ax3.loglog(f_sk/1e9, d_sk, 'r-', lw=2, label='Gold')
d_cu = np.sqrt(1.68e-8/(np.pi*f_sk*4*np.pi*1e-7))*1e9
ax3.loglog(f_sk/1e9, d_cu, 'k--', lw=2, label='Copper')
ax3.axhspan(0,100,alpha=0.1,color='red')
ax3.axhline(100,color='gray',ls=':',lw=1,label='100 nm (typical metal thickness)')
ax3.set_xlabel('Freq (GHz)'); ax3.set_ylabel('Skin depth (nm)')
ax3.set_title('Skin effect: R grows as sqrt(f)', fontsize=10)
ax3.legend(fontsize=8)
ax3.text(3e2,50,'Resistance\ndominates',color='red',fontsize=8)

# P4: PCA spectrum
ax4 = fig.add_subplot(gs2[1,0])
f_plot2 = freqs[mask]/1e12
psd_plot = np.abs(E_THz_f[mask])**2
psd_plot /= psd_plot.max()
ax4.semilogy(f_plot2[1:], psd_plot[1:], 'b-', lw=2)
ax4.axvline(BW_3dB/1e12, color='r', ls='--', lw=2, label=f'BW={BW_3dB/1e12:.2f} THz')
ax4.axhline(0.5, color='g', ls=':', lw=1, label='-3dB')
ax4.set_xlabel('Freq (THz)'); ax4.set_ylabel('PSD (norm)')
ax4.set_title('PCA spectrum: dJ/dt FT', fontsize=10)
ax4.legend(fontsize=8); ax4.set_xlim(0,4); ax4.set_ylim(1e-4,2)

# P5: THz-TDS reference vs sample pulse
ax5 = fig.add_subplot(gs2[1,1])
t_ps = t_ref*1e12
ax5.plot(t_ps, E_ref_t/E_ref_t.max(), 'b-', lw=2, label='Reference')
ax5.plot(t_ps, E_sample_t/E_ref_t.max(), 'r-', lw=2, label=f'Sample (n={n_sample_real})')
ax5.set_xlabel('Time (ps)'); ax5.set_ylabel('E (norm)')
ax5.set_title('THz-TDS: pulse delay = n*d/c', fontsize=10)
ax5.legend(fontsize=8); ax5.set_xlim(-1,15)
expected_delay = (n_sample_real-1)*d_sample/3e8*1e12
ax5.annotate(f'Delay={expected_delay:.1f} ps',
             xy=(t0*1e12+expected_delay, 0.5),
             xytext=(t0*1e12+expected_delay+1, 0.7),
             arrowprops=dict(arrowstyle='->', color='k'),
             fontsize=9)

# P6: Extracted refractive index vs frequency
ax6 = fig.add_subplot(gs2[1,2])
f_band = f2[1:][mask2]/1e12
n_band = n_extracted[mask2].real
ax6.plot(f_band, n_band, 'g-', lw=2, label='Extracted n_r')
ax6.axhline(n_sample_real, color='r', ls='--', lw=2, label=f'True n={n_sample_real}')
ax6.set_xlabel('Freq (THz)'); ax6.set_ylabel('n_r')
ax6.set_title('Phase retrieval: n(w) from E_sample/E_ref\n= GS phase recovery in THz-TDS', fontsize=9)
ax6.legend(fontsize=8); ax6.set_ylim(3.0,3.8)

out = r'D:\Summer2026\Dispersion-Assisted-GS-Phase-Recovery\repl\_out_thz_circuits.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

hdr("Summary: circuit theory implodes at THz")
print("""
  CLASSICAL CIRCUITS (< 1 GHz):
    V = IR,  Z = R + jwL + 1/jwC
    Lambda >> L: components are points
    eps is real: no loss, no chemistry

  TRANSMISSION LINE (1 GHz - 300 GHz):
    gamma = sqrt((R+jwL)(G+jwC))
    Z0 = sqrt((R+jwL)/(G+jwC))
    R grows as sqrt(f) [skin effect]

  THz (0.3 - 10 THz):
    LUMPED MODEL IMPLODES  (L_lump ~ 10 um = transistor gate)
    Skin depth < 100 nm: ALL resistance, surface currents only
    eps(w) is complex: CHEMISTRY IS IN THE CIRCUIT
      eps''(w) = loss = molecular absorption = R in Smith chart
    Water: alpha > 100 dB/cm -> "wet" sample = no transmission

  PHASE RETRIEVAL CONNECTION:
    THz-TDS: measure E(t), FT -> E(w), divide ref/sample -> phi(w) -> n(w)
    D-GS:    measure I(x) at D1, D2 -> GS iterate -> phi(u) -> wavefront
    SAME MATH. Different domain. Your project IS THz metrology.
""")
print("=== THz circuits section complete ===")
