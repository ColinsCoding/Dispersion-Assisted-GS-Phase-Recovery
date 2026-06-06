"""
_repl_superconductor_photonics.py
Superconductivity + JWST blackbody + commercial photonics + dB vision + laser tattoo + P=IV
Run: py -3.12 repl/_repl_superconductor_photonics.py
"""

import numpy as np
import sympy as sp
from scipy.special import k0 as K0_bessel
from scipy.integrate import quad

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SUPERCONDUCTIVITY — BCS + London equations
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: SUPERCONDUCTIVITY (BCS + London)")
print("=" * 65)

kB  = 1.380649e-23   # J/K
hbar = 1.0545718e-34 # J*s
h    = 6.62607e-34
e_c  = 1.60218e-19   # C
m_e  = 9.10938e-31   # kg
mu0  = 4*np.pi*1e-7
eps0 = 8.8542e-12

print("""
  BCS gap equation (weak coupling):
    2*Delta(0) = 3.528 * kB * Tc     (universal ratio)
    Delta(T)   ~ Delta(0) * sqrt(1 - T/Tc)   (near Tc)

  Physical meaning:
    Delta = binding energy of Cooper pair (2 electrons)
    Cooper pair size: xi_0 = hbar*vF / (pi*Delta)
    Pairs condense into macroscopic quantum state (wavefunction Psi)

  London equations:
    dJ_s/dt = (n_s * e^2 / m) * E      (1st: lossless acceleration)
    curl(J_s) = -(n_s * e^2 / m) * B   (2nd: Meissner effect)
    -> B decays as exp(-z/lambda_L)     (London penetration depth)
    lambda_L = sqrt(m / (mu0 * n_s * e^2))
""")

# BCS gap and penetration depth for key superconductors
# (Tc K, lambda_L nm, xi_0 nm, kappa=lambda/xi)
supers = {
    "Al  (type I)":   (1.19,  50,   1600,  "Type I"),
    "Pb  (type I)":   (7.19,  39,    83,   "Type I"),
    "Nb  (type II)":  (9.26,  39,    38,   "Type II soft"),
    "NbTi (wire)":    (9.8,   300,   4,    "Type II hard"),
    "YBCO (HTS)":     (93,    150,   1.5,  "Type II HTS"),
    "MgB2":           (39,    140,   5,    "Type II"),
    "BSCCO-2212":     (85,    200,   1,    "Type II HTS"),
}

print(f"  {'Material':18s}  {'Tc(K)':>7}  {'2*Delta(meV)':>12}  {'lambda_L(nm)':>12}  {'xi_0(nm)':>9}  {'kappa':>7}  {'type'}")
print("-" * 90)
for name, (Tc, lam_nm, xi_nm, stype) in supers.items():
    gap_meV = 3.528 * kB * Tc / e_c * 1000   # meV
    kappa   = lam_nm / xi_nm
    print(f"  {name:18s}  {Tc:>7.2f}  {gap_meV:>12.3f}  {lam_nm:>12}  {xi_nm:>9}  {kappa:>7.2f}  {stype}")

print("\n  kappa = lambda_L / xi_0:")
print("    kappa < 1/sqrt(2) -> Type I  (single Hc, complete expulsion)")
print("    kappa > 1/sqrt(2) -> Type II (Hc1 < H < Hc2: vortex lattice)")

# YBCO: compute London penetration depth from first principles
print("\n--- YBCO London penetration depth (first principles) ---")
n_s_YBCO = 1e27   # Cooper pair density m^-3 (order of magnitude)
lam_L = np.sqrt(m_e / (mu0 * n_s_YBCO * (2*e_c)**2))
print(f"  n_s = {n_s_YBCO:.0e} m^-3")
print(f"  lambda_L = sqrt(m / mu0 * n_s * (2e)^2) = {lam_L*1e9:.1f} nm")
print(f"  YBCO measured: ~150 nm  (our model: {lam_L*1e9:.0f} nm, correct order)")

# Josephson junction: V = hbar/(2e) * d(phi)/dt
print("\n--- Josephson junction ---")
print("  DC Josephson: I = Ic * sin(phi)  (supercurrent with no voltage)")
print("  AC Josephson: V = hbar/(2e) * dphi/dt")
print("  -> oscillation frequency: f = 2e*V / h  (Josephson frequency)")
f_per_V  = 2*e_c / h          # Hz/V  (Josephson constant K_J = 483597.9 GHz/V)
f_per_mV = f_per_V * 1e-3 * 1e-9  # GHz per mV
print(f"  f/V = 2e/h = K_J = {f_per_V*1e-9:.3f} GHz/V = {f_per_V*1e-9:.3f} GHz/V")
print(f"  At V=1 mV: f = {f_per_mV*1000:.3f} GHz  (microwave!)  -- basis of voltage standard")
print(f"  At V=1 uV: f = {f_per_V*1e-6*1e-6:.3f} MHz  (RF)")

# SNSPD (superconducting nanowire single-photon detector) -- JWST connection
print("\n--- SNSPD: Superconducting Nanowire Single-Photon Detector ---")
print("  Used in quantum photonics; NbN or WSi biased near Ic")
print("  Single photon breaks Cooper pairs -> hotspot -> resistance spike")
print("  Timing jitter < 30 ps, efficiency > 90%, dark count < 1/s")
E_photon_1550 = h * 3e8 / 1550e-9
E_gap_NbN     = 3.528 * kB * 16.0   # NbN Tc ~ 16 K
N_pairs_broken = E_photon_1550 / (2 * E_gap_NbN)
print(f"  Photon at 1550 nm: E = {E_photon_1550/e_c*1000:.2f} meV")
print(f"  NbN gap 2*Delta ~ {2*E_gap_NbN/e_c*1000:.2f} meV")
print(f"  Cooper pairs broken per photon: ~{N_pairs_broken:.0f}  -> easily detectable cascade")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BLACKBODY RADIATION — JWST + HAIR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: BLACKBODY RADIATION (Planck, Wien, JWST)")
print("=" * 65)

c_light = 2.998e8
h_pl    = 6.626e-34

def planck_lambda(lam, T):
    """Spectral radiance W/m^2/sr/m"""
    x = h_pl * c_light / (lam * kB * T)
    return (2*h_pl*c_light**2 / lam**5) / (np.exp(x) - 1)

def wien_peak(T):
    """Peak wavelength (Wien displacement) in meters"""
    return 2.898e-3 / T   # b = 2.898e-3 m*K

def stefan_boltzmann(T, eps=1.0):
    """Total power W/m^2"""
    sigma = 5.670e-8
    return eps * sigma * T**4

# Key thermal sources
sources = {
    "Sun surface":          5778,
    "Tungsten lamp filament":3400,
    "Incandescent bulb":    2900,
    "Human skin":           305,
    "Hair (scalp)":         307,
    "Room temperature":     293,
    "JWST mirror (50K)":    50,
    "CMB":                  2.725,
}

print(f"\n  {'Source':28s}  {'T(K)':>7}  {'peak lambda':>12}  {'P (W/m^2)':>12}  {'color'}")
print("-" * 75)
for name, T in sources.items():
    lam_p = wien_peak(T)
    P     = stefan_boltzmann(T)
    if   lam_p < 380e-9:   col = "UV"
    elif lam_p < 700e-9:   col = "visible"
    elif lam_p < 2500e-9:  col = "near-IR"
    elif lam_p < 25e-6:    col = "mid-IR"
    elif lam_p < 1e-3:     col = "far-IR"
    else:                  col = "microwave"
    print(f"  {name:28s}  {T:>7.1f}  {lam_p*1e9:>9.1f} nm  {P:>12.3e}  {col}")

# ── Human hair blackbody: how much IR does hair radiate? ─────────────────────
print("\n--- Human hair / scalp thermal radiation ---")
T_hair    = 307.0    # K, scalp surface
eps_hair  = 0.95     # skin/hair emissivity (close to 1)
A_scalp   = 0.06     # m^2 typical scalp area
P_scalp   = stefan_boltzmann(T_hair, eps_hair) * A_scalp
print(f"  T_scalp = {T_hair} K,  eps = {eps_hair},  A = {A_scalp} m^2")
print(f"  Peak emission: {wien_peak(T_hair)*1e6:.2f} um  (mid-IR, thermal cameras see this)")
print(f"  Total radiated power: {P_scalp:.2f} W  (that's why you lose heat from your head)")

# ── JWST: photon counting at 5 uW per detector ───────────────────────────────
print("\n--- James Webb Space Telescope photon budget ---")
print("  JWST bands: NIRCam (0.6-5um), MIRI (5-28um), NIRSpec, NIRISS")
print("  Primary mirror: 6.5m, 18 beryllium segments, T_mirror ~ 50 K")
print()

bands = {
    "NIRCam short (1.5um)": (1.5e-6,  3.0e-19),   # NEP W/sqrt(Hz)
    "NIRCam long  (4.4um)": (4.4e-6,  2.0e-19),
    "MIRI (10um)":          (10e-6,   5.0e-19),
    "MIRI (25um)":          (25e-6,   1.0e-18),
}

print(f"  {'Band':22s}  {'E_photon(eV)':>13}  {'NEP(W/rtHz)':>13}  {'photons/s at NEP':>18}")
for name, (lam_b, NEP) in bands.items():
    E_ph   = h_pl * c_light / lam_b
    E_ph_eV = E_ph / e_c
    phot_s = NEP / E_ph
    print(f"  {name:22s}  {E_ph_eV:>13.4f}  {NEP:>13.2e}  {phot_s:>18.3e}")

# Redshift of galaxy z=10: Lyman-alpha 121.6nm -> observed
z_gal = 10.0
lam_emit = 121.6e-9   # Lyman alpha
lam_obs  = lam_emit * (1 + z_gal)
print(f"\n  Galaxy at z={z_gal}: Lyman-alpha {lam_emit*1e9:.1f} nm -> observed {lam_obs*1e9:.1f} nm")
print(f"  Falls in {'NIRCam' if lam_obs < 5e-6 else 'MIRI'} band -- JWST can see first galaxies")

# Blackbody comparison: 5778K sun vs 50K JWST mirror contamination
lam_arr = np.linspace(0.5e-6, 30e-6, 1000)
B_sun   = planck_lambda(lam_arr, 5778) * (0.5e-6)**2   # scaled to same area
B_50K   = planck_lambda(lam_arr, 50)
peak_50K = lam_arr[np.argmax(B_50K)]
print(f"\n  JWST mirror at 50 K: peak emission {peak_50K*1e6:.1f} um")
print(f"  This is WHY JWST must be cold: mirror own emission would swamp faint galaxy signal")
print(f"  L2 orbit + sunshield keeps mirror at 50 K (5 layers of Kapton, each 50x colder)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: COMMERCIAL PHOTONICS — dB, NEP, responsivity, SNR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: COMMERCIAL PHOTONICS + dB IN VISION")
print("=" * 65)

print("""
  Detector figures of merit:
    Responsivity R = I_ph / P_opt  [A/W]
    Quantum efficiency QE = R * hnu / e  (photons converted to electrons)
    NEP = noise / R  [W/sqrt(Hz)]  (min detectable power in 1Hz BW)
    D* = sqrt(A*BW) / NEP  [cm*sqrt(Hz)/W]  (detectivity, area-normalized)
    SNR = P_signal / NEP / sqrt(BW)
""")

detectors = {
    "Si PIN (850nm)":       (0.55, 1e-14,  1e4,  "Si"),
    "InGaAs (1550nm)":      (0.90, 2e-14,  5e3,  "III-V"),
    "Ge APD (1310nm)":      (0.75, 5e-14,  50,   "Ge APD"),
    "HgCdTe MWIR (5um)":    (2.50, 1e-13,  1,    "II-VI cryo"),
    "SNSPD (1550nm)":       (0.92, 1e-19,  1e7,  "NbN cryo"),
    "Human eye (555nm)":    (0.10, 1e-16,  1,    "bio"),
}

print(f"  {'Detector':22s}  {'R(A/W)':>8}  {'NEP(W/rtHz)':>13}  {'D*(cmHz^0.5/W)':>16}  {'type'}")
print("-" * 80)
for name, (R, NEP, Dstar, dtype) in detectors.items():
    print(f"  {name:22s}  {R:>8.3f}  {NEP:>13.2e}  {Dstar:>16.2e}  {dtype}")

# ── dB in human vision ────────────────────────────────────────────────────────
print("\n--- dB dynamic range of human eye ---")
print("""
  Vision operates over ~120 dB of intensity range:
    Absolute threshold (rods, dark-adapted):  I_min ~ 10^-14 W/cm^2
    Damage threshold (bright sunlight):        I_max ~ 10^-2 W/cm^2
    Total range: 10 log10(I_max/I_min) = 10*12 = 120 dB

  Within a single scene (simultaneous contrast):
    Pupil adapts ~1mm (bright) to ~7mm (dark): area ratio 49x -> ~17 dB
    Neural adaptation (photoreceptor saturation): ~40 dB additional
    HDR display target: ~20 stops = 120 dB

  dB scale in photometry:
    dB = 10*log10(P2/P1)         power ratio
    dB = 20*log10(V2/V1)         amplitude ratio
    dBm = 10*log10(P_mW / 1mW)  absolute (telecom standard)
    dBW = 10*log10(P_W / 1W)
""")

# Dynamic range table
print(f"  {'Condition':28s}  {'I (W/cm^2)':>14}  {'dB re threshold':>17}")
conditions = [
    ("Absolute threshold (dark)",  1e-14),
    ("Full moon night",             1e-11),
    ("Indoor lighting",             1e-7),
    ("Overcast daylight",           1e-4),
    ("Direct sunlight",             1e-2),
    ("Retinal damage threshold",    1.0),
]
I_ref = 1e-14
for name, I in conditions:
    dB = 10*np.log10(I / I_ref)
    print(f"  {name:28s}  {I:>14.0e}  {dB:>17.1f} dB")

# ── Optical link budget (telecom fiber) ──────────────────────────────────────
print("\n--- Fiber link budget (100G DWDM, 80 km span) ---")
P_tx_dBm   = 0.0      # launch power 1 mW = 0 dBm
loss_fiber  = 0.2     # dB/km  (SMF-28 at 1550 nm)
span_km     = 80.0
EDFA_gain   = 20.0    # dB
n_spans     = 3
sensitivity_dBm = -28.0   # coherent receiver sensitivity

P_rx = P_tx_dBm - loss_fiber*span_km*n_spans + EDFA_gain*(n_spans-1)
margin = P_rx - sensitivity_dBm

print(f"  Tx power:       {P_tx_dBm:+.1f} dBm = {10**(P_tx_dBm/10):.1f} mW")
print(f"  Fiber loss:     {loss_fiber} dB/km x {span_km} km x {n_spans} spans = "
      f"{loss_fiber*span_km*n_spans:.1f} dB total")
print(f"  EDFA boost:     {EDFA_gain} dB x {n_spans-1} amplifiers = {EDFA_gain*(n_spans-1):.1f} dB")
print(f"  Rx power:       {P_rx:+.1f} dBm = {10**(P_rx/10)*1000:.2f} uW")
print(f"  Sensitivity:    {sensitivity_dBm:+.1f} dBm")
print(f"  Power margin:   {margin:+.1f} dB  ({'OK -- link closes' if margin > 0 else 'FAIL'})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: LASER TATTOO REMOVAL — selective photothermolysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: LASER TATTOO REMOVAL — selective photothermolysis")
print("=" * 65)

print("""
  Principle: Anderson & Parrish 1983 -- selective photothermolysis
    1. Choose wavelength absorbed by target (ink) but NOT surrounding tissue
    2. Pulse duration < thermal relaxation time of target
    3. Fluence high enough to destroy target, not surrounding

  Thermal relaxation time: tau_r = r^2 / (4*alpha_thermal)
    r = radius of ink particle or target vessel
    alpha_thermal = thermal diffusivity of tissue ~ 1.3e-7 m^2/s

  Ink particle size ~ 50-500 nm; hair follicle ~ 200 um diameter
""")

alpha_tissue = 1.3e-7   # m^2/s thermal diffusivity

targets = {
    "Tattoo ink particle (250nm)":  250e-9,
    "Tattoo cluster (10 um)":       10e-6,
    "Hair follicle (200 um)":       200e-6,
    "Blood vessel (100 um)":        100e-6,
    "Melanocyte (5 um)":            5e-6,
}

print(f"\n  {'Target':32s}  {'r':>10}  {'tau_r':>14}  {'ideal pulse'}")
print("-" * 70)
for name, r in targets.items():
    tau_r = r**2 / (4 * alpha_tissue)
    if tau_r < 1e-9:   pstr = f"{tau_r*1e12:.1f} ps"
    elif tau_r < 1e-6: pstr = f"{tau_r*1e9:.1f} ns"
    elif tau_r < 1e-3: pstr = f"{tau_r*1e6:.1f} us"
    else:              pstr = f"{tau_r*1e3:.1f} ms"
    if tau_r < 1e-8:   ideal = "Q-switch/mode-lock"
    elif tau_r < 1e-4: ideal = "Q-switched Nd:YAG"
    else:              ideal = "pulsed (ms) diode"
    print(f"  {name:32s}  {r*1e6:>7.3f} um  {pstr:>14}  {ideal}")

# ── Tattoo removal laser specs ────────────────────────────────────────────────
print("\n--- Q-switched Nd:YAG tattoo removal laser ---")
print("  Wavelength: 1064 nm (black ink) / 532 nm (red ink)")
print("  Pulse: 5-10 ns  (Q-switched; shorter than tau_r for ink cluster)")
print("  Fluence: 4-10 J/cm^2 per pulse")

E_pulse_J  = 0.5     # J per pulse
spot_cm    = 0.4     # cm diameter
A_spot_cm2 = np.pi * (spot_cm/2)**2
fluence    = E_pulse_J / A_spot_cm2
pulse_ns   = 8.0
P_peak_W   = E_pulse_J / (pulse_ns * 1e-9)
P_peak_MW  = P_peak_W / 1e6

print(f"\n  E_pulse    = {E_pulse_J*1000:.0f} mJ")
print(f"  Spot       = {spot_cm} cm diam,  A = {A_spot_cm2:.3f} cm^2")
print(f"  Fluence    = {fluence:.2f} J/cm^2  (threshold ~4 J/cm^2 for black ink)")
print(f"  Pulse      = {pulse_ns:.0f} ns")
print(f"  P_peak     = E/tau = {P_peak_W:.3e} W = {P_peak_MW:.1f} MW  (!!)")
print(f"  Irradiance = {fluence / (pulse_ns*1e-9) / 1e6:.0f} MW/cm^2  during pulse")

# Absorption: black ink (carbon) at 1064 nm
mu_a_ink   = 1e6    # m^-1 absorption coefficient of carbon black
mu_a_dermis = 3e3   # m^-1 dermis background at 1064 nm
selectivity = mu_a_ink / mu_a_dermis
print(f"\n  Absorption selectivity (ink/dermis at 1064nm): {selectivity:.0f}x")
print(f"  Ink penetration depth: {1/mu_a_ink*1e9:.0f} nm -- confined to particle")
print(f"  Dermis penetration:   {1/mu_a_dermis*1e6:.1f} mm -- beam passes through")

# ── Hair curl with laser? (wavelength for melanin) ────────────────────────────
print("\n--- Hair / melanin wavelength selectivity ---")
print("  Melanin absorbs broadly from UV to ~1000 nm (eumelanin)")
print("  Hair removal (alexandrite 755nm, diode 810nm, Nd:YAG 1064nm)")
print("  Hair follicle tau_r ~ 37 ms -> pulse 3-30 ms (long-pulse mode)")
print("  Darker hair (more melanin) absorbs more -> easier to remove")
print("  Blonde/white hair: low melanin -> need topical absorber or radiofrequency")

# Melanin absorption cross-section
lam_hair = np.array([400, 500, 600, 700, 755, 810, 1064])  # nm
# Eumelanin absorption: approximated as exponential falloff
# Eumelanin: mu_a ~ 6.6e6 * exp(-lam_nm/200) cm^-1 -> convert to m^-1
mu_a_melanin = 6.6e6 * np.exp(-lam_hair / 200) * 100   # m^-1
print(f"\n  {'lambda(nm)':>10}  {'mu_a (m^-1)':>14}  {'skin depth (um)':>16}")
for lam_h, mu_h in zip(lam_hair, mu_a_melanin):
    delta_um = 1/mu_h * 1e6 if mu_h > 0 else np.inf
    print(f"  {lam_h:>10}  {mu_h:>14.3e}  {delta_um:>16.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: P = I*V — laser driver, LED efficiency, club lighting
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: P = I*V — LASER DRIVER + LED + CLUB LIGHTING")
print("=" * 65)

print("""
  P = I * V  is the same relation whether you're:
    - biasing a laser diode (I_threshold + I_operating, V ~ 1.5-3V)
    - driving an LED (V_forward ~ 2-4V, I ~ 10-350 mA)
    - powering club speakers (I = P/V, big transformer)
    - Q-switch capacitor discharge (E = 0.5*C*V^2, I = V/R peak)
""")

# Laser diode I-V-P curves
print("--- Laser diode: threshold + slope efficiency ---")
print("  P_opt = eta_slope * (I - I_th)  for I > I_th")
print("  Wall-plug efficiency: WPE = P_opt / (I*V)")

lasers = {
    "1550nm DFB (telecom)":  (10e-3,  0.3,    1.5,  0.40),  # Ith, eta_d, V, WPE
    "808nm pump diode":      (200e-3, 1.2,    1.7,  0.65),
    "532nm DPSS (green)":    (5e-3,   0.05,   3.0,  0.10),  # frequency-doubled
    "405nm (Blu-ray)":       (30e-3,  0.60,   4.5,  0.35),
    "450nm (projector)":     (150e-3, 1.5,    4.0,  0.55),
}

I_op = 2.0   # operating at 2x threshold (normalized)
print(f"\n  At I = 2*I_th:")
print(f"  {'Laser':24s}  {'I_th(mA)':>9}  {'V_op(V)':>8}  {'P_opt(mW)':>10}  {'P_elec(W)':>10}  {'WPE':>6}")
print("-" * 80)
for name, (Ith, eta_d, V, WPE) in lasers.items():
    I_operating = 2 * Ith
    P_opt_mW    = eta_d * Ith * 1000   # mW  (I-Ith = Ith at I=2*Ith)
    P_elec_W    = I_operating * V
    print(f"  {name:24s}  {Ith*1000:>9.1f}  {V:>8.2f}  {P_opt_mW:>10.2f}  {P_elec_W:>10.4f}  {WPE:>6.0%}")

# ── LED: white light efficiency ───────────────────────────────────────────────
print("\n--- White LED efficiency ---")
print("  White LED: blue GaN chip + yellow YAG phosphor")
print("  V_forward ~ 3.0-3.5 V,  I ~ 350 mA (standard), 3A (high-power)")

leds = {
    "Standard 5mm (through-hole)":   (20e-3,   3.3,  0.25,  60),
    "Cree XP-L (350mA)":             (350e-3,  3.0,  0.70, 205),
    "Cree XP-L (3A high-power)":     (3.0,     3.2,  0.40, 138),
    "Seoul Semi 5630 (SMD)":         (150e-3,  3.1,  0.65, 175),
}

print(f"\n  {'LED':32s}  {'I(mA)':>7}  {'V(V)':>6}  {'P_in(W)':>8}  {'WPE':>6}  {'lm/W':>6}")
print("-" * 75)
for name, (I_led, V_led, WPE_led, lmpW) in leds.items():
    P_in = I_led * V_led
    print(f"  {name:32s}  {I_led*1000:>7.0f}  {V_led:>6.1f}  {P_in:>8.3f}  {WPE_led:>6.0%}  {lmpW:>6}")

print("\n  Sun: ~93 lm/W  (reference for efficiency comparison)")
print("  Incandescent: 10-15 lm/W")
print("  Best LED lab record: 303 lm/W (Cree 2023)")

# ── Club / venue lighting ─────────────────────────────────────────────────────
print("\n--- Club / venue power budget ---")
lights = {
    "Moving head spot (1200W)":  (1200, 240, 5.0),    # W, V, A
    "LED par 64 (100W)":         (100,  120, 0.83),
    "Laser show (10W green)":    (30,   120, 0.25),    # 10W optical, 30W electrical
    "Subwoofer amp (2000W RMS)": (2000, 240, 8.33),
    "DJ controller + mixer":     (150,  120, 1.25),
}

total_W = 0
print(f"\n  {'Device':30s}  {'P(W)':>7}  {'V(V)':>6}  {'I(A)':>7}")
print("-" * 55)
for name, (P, V, I) in lights.items():
    print(f"  {name:30s}  {P:>7}  {V:>6}  {I:>7.2f}")
    total_W += P

print(f"\n  Total venue load:  {total_W} W = {total_W/1000:.1f} kW")
print(f"  At $0.15/kWh, 6hr night: ${total_W/1000 * 6 * 0.15:.2f}")
print(f"  Circuit breakers needed: {total_W/240/20:.1f} x 20A/240V breakers")

# ── Laser safety: MPE and P=IV for safety interlock ─────────────────────────
print("\n--- Laser safety: MPE and safety interlock P=IV ---")
print("  Maximum Permissible Exposure (MPE) for eye at 1064nm, 10s:")
print("    MPE = 5e-3 J/cm^2   (ANSI Z136.1)")
print("  Nominal Ocular Hazard Distance (NOHD):")
print("    NOHD = sqrt(P_beam / (pi * MPE * BW)) where BW = bandwidth")

MPE_Jcm2 = 5e-3        # J/cm^2
BW_s      = 10.0        # s
P_laser_W = 1.0         # 1W CW laser

# NOHD: beyond this distance, beam expands enough that irradiance < MPE
# Assume Gaussian beam divergence theta ~ 1 mrad
theta_rad  = 1e-3
NOHD       = (1/theta_rad) * np.sqrt(P_laser_W / (np.pi * MPE_Jcm2/BW_s * 1e4))
print(f"\n  1W CW laser at 1064nm, theta={theta_rad*1000:.0f} mrad:")
print(f"  NOHD = {NOHD:.1f} m  (safety distance; beyond this -> safe for 10s exposure)")

print(f"\n  Safety interlock: current sensor monitors I in P=IV")
print(f"    If I > I_threshold -> kill relay -> shutter closes in < 1ms")
print(f"    Typical: I_trip = 1.1 * I_nominal  (10% over-current)")
print(f"    This is the 'deadman switch' in every Class 3B/4 laser system")

print("\nDone.")
