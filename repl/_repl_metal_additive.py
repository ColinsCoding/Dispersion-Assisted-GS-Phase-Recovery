"""
_repl_metal_additive.py
Laser powder bed fusion optics + pipe engineering (Barlow) + topology overhang + C header .h
Run: py -3.12 repl/_repl_metal_additive.py
"""

import numpy as np
import sympy as sp

kB     = 1.380649e-23
sigma_SB = 5.670e-8    # Stefan-Boltzmann

# -----------------------------------------------------------------------------
# SECTION 1: SBIR $275K BUDGET -- 3-PERSON TEAM
# -----------------------------------------------------------------------------
print("=" * 65)
print("SECTION 1: SBIR PHASE I $275K -- 3-PERSON TEAM BUDGET")
print("=" * 65)

print("""
  SBIR Phase I structure:
    Total award:   $275,000
    Period:        12 months
    Team:          3 people (PI + co-I + student/technician)

  Typical allocation (DoD/NIH guidelines):
    Personnel (incl. fringe):  ~60%
    Indirect (overhead):       ~25%  (university rate; startup < 10%)
    Equipment:                 ~10%
    Supplies/travel/other:     ~5%
""")

award      = 275_000
personnel_frac = 0.60
overhead_frac  = 0.25
equipment_frac = 0.10
other_frac     = 0.05

personnel = award * personnel_frac
overhead  = award * overhead_frac
equipment = award * equipment_frac
other     = award * other_frac

print(f"  {'Category':25s}  {'Amount':>10}  {'Per person (3)':>15}")
print("-" * 55)
print(f"  {'Personnel + fringe':25s}  ${personnel:>9,.0f}  ${personnel/3:>14,.0f}")
print(f"  {'Indirect / overhead':25s}  ${overhead:>9,.0f}  {'(institutional)':>15}")
print(f"  {'Equipment':25s}  ${equipment:>9,.0f}  {'(shared)':>15}")
print(f"  {'Supplies/travel/other':25s}  ${other:>9,.0f}  {'(shared)':>15}")
print(f"  {'TOTAL':25s}  ${award:>9,.0f}")

# Salary breakdown (pre-fringe, 12 months)
fringe_rate = 0.28   # typical fringe benefit rate
salaries_total = personnel / (1 + fringe_rate)
fringe_total   = personnel - salaries_total

print(f"\n  Salary pool (before fringe @ {fringe_rate*100:.0f}%): ${salaries_total:,.0f}")
print(f"  Fringe benefits:                         ${fringe_total:,.0f}")

roles = [
    ("PI (lead engineer)",       0.50, 130_000),  # 50% effort, $130K base salary
    ("Co-I (photonics/software)",0.35, 110_000),
    ("Graduate student",         1.00,  30_000),  # 100% effort, stipend
]
print(f"\n  {'Role':28s}  {'Effort':>7}  {'Base salary':>12}  {'SBIR salary':>12}  {'w/ fringe':>10}")
print("-" * 75)
total_sbir_sal = 0
for role, effort, base in roles:
    sbir_sal = effort * base
    with_fringe = sbir_sal * (1 + fringe_rate)
    total_sbir_sal += sbir_sal
    print(f"  {role:28s}  {effort:>6.0%}  ${base:>11,.0f}  ${sbir_sal:>11,.0f}  ${with_fringe:>9,.0f}")
print(f"  {'TOTAL':28s}         {'':>12}  ${total_sbir_sal:>11,.0f}  ${total_sbir_sal*(1+fringe_rate):>9,.0f}")

# Equipment shopping list at $27,500
print(f"\n  Equipment budget ${equipment:,.0f}:")
equip = [
    ("Raspberry Pi CM4 kit x2",           650),
    ("Dual-channel 1GS/s ADC board",     2500),
    ("Fiber patch panel + connectors",    800),
    ("Optical power meter (used)",        400),
    ("GPU server (RTX 4090)",           2800),
    ("SSD RAID array 100TB",            1200),
    ("Oscilloscope 500MHz",             1800),
    ("Lab supplies / solder / misc",     500),
    ("Conference travel (OFC/CLEO x2)", 4000),
    ("Cloud compute (AWS/GCP credits)", 2000),
    ("Subcontract: PCB fab",            1500),
    ("Remaining contingency",           equipment - 18150),
]
total_eq = 0
for item, cost in equip:
    total_eq += cost
    print(f"    ${cost:>6,}  {item}")
print(f"    ------")
print(f"    ${total_eq:>6,}  TOTAL  (budget: ${equipment:,.0f})")

print(f"\n  Phase II path: $1.75M (6x) -- same 3 people + 3 hires + hardware prototype")


# -----------------------------------------------------------------------------
# SECTION 2: LASER POWDER BED FUSION -- OPTICAL MELTING OF METAL
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SECTION 2: LASER POWDER BED FUSION (LPBF) -- OPTICAL PHYSICS")
print("=" * 65)

print("""
  LPBF (also: SLM, DMLS, DMLM):
    Gaussian laser beam scans over metal powder bed
    Beam melts powder -> solidifies -> layer done -> new powder -> repeat

  Key parameters:
    P     = laser power (W)
    v     = scan speed (mm/s)
    d     = hatch spacing (um)
    t     = layer thickness (um)
    E_vol = P / (v * d * t)   [J/mm^3]  -- volumetric energy density

  Melt pool modes:
    Conduction: E_vol < 50 J/mm^3  -> shallow bowl, stable
    Keyhole:    E_vol > 100 J/mm^3 -> deep vapor cavity, porosity risk
    Optimal:    60-80 J/mm^3 for most alloys
""")

# Gaussian beam parameters for LPBF
P_W    = 200.0    # W laser power
v_mms  = 800.0    # mm/s scan speed
d_um   = 100.0    # um hatch spacing
t_um   = 30.0     # um layer thickness

E_vol  = P_W / (v_mms * d_um*1e-3 * t_um*1e-3)   # J/mm^3
print(f"  P={P_W:.0f}W  v={v_mms:.0f}mm/s  d={d_um:.0f}um  t={t_um:.0f}um")
print(f"  E_vol = P/(v*d*t) = {E_vol:.1f} J/mm^3  "
      f"({'keyhole risk' if E_vol>100 else 'conduction' if E_vol<50 else 'optimal'})")

# Beam caustic: Gaussian beam propagation
lam_laser = 1070e-9   # nm, Yb fiber laser (most common LPBF)
w0_beam   = 35e-6     # m, beam waist radius at focus
z_R       = np.pi * w0_beam**2 / lam_laser   # Rayleigh range

print(f"\n  Laser: lambda={lam_laser*1e9:.0f}nm (Yb fiber), w0={w0_beam*1e6:.0f}um")
print(f"  Rayleigh range: z_R = pi*w0^2/lambda = {z_R*1e3:.2f} mm")
print(f"  Depth of focus (2*z_R) = {2*z_R*1e3:.2f} mm  (covers multiple powder layers)")

z_vals  = np.linspace(-3*z_R, 3*z_R, 100) * 1e3   # mm
w_z     = w0_beam * np.sqrt(1 + (z_vals*1e-3/z_R)**2) * 1e6  # um
I_peak  = 2 * P_W / (np.pi * w0_beam**2)  # W/m^2 at focus
I_z     = I_peak / (1 + (z_vals*1e-3/z_R)**2)

print(f"\n  Peak intensity at focus: {I_peak/1e9:.2f} GW/m^2 = {I_peak/1e4:.0f} W/cm^2")
print(f"  {'z (mm)':>8}  {'w(z) (um)':>10}  {'I (MW/cm^2)':>14}")
for z, w, I in zip(z_vals[::15], w_z[::15], I_z[::15]):
    print(f"  {z:>8.2f}  {w:>10.2f}  {I/1e10:>14.3f}")

# Melt pool thermal model (Rosenthal moving heat source)
print("\n--- Rosenthal melt pool model ---")
print("""
  Moving Gaussian source on semi-infinite solid:
  T(r,t) - T0 = (P / (2*pi*k_th*R)) * exp(-v*(R+x)/(2*alpha_th))

  R = sqrt(x^2 + y^2 + z^2)  (distance from beam center)
  k_th    = thermal conductivity (W/m/K)
  alpha_th = thermal diffusivity (m^2/s)
  v       = scan speed

  Melt pool length L ~ 2*alpha_th/v * ln(P/(2*pi*k_th*(Tm-T0)*w0))
""")

# Stainless steel 316L properties
k_th    = 15.0      # W/m/K
alpha_th = 3.9e-6   # m^2/s
Tm      = 1400.0    # C liquidus
T0      = 25.0      # C preheat
rho_mat = 7960.0    # kg/m^3
Cp_mat  = 500.0     # J/kg/K

# Melt pool dimensions (empirical scaling)
v_m_s   = v_mms * 1e-3   # m/s
E_lin   = P_W / v_m_s    # J/m linear energy

# Width from Gaussian beam: w_melt ~ w0 * sqrt(ln(2*P/(pi*w0^2*v*k_th*(Tm-T0))))
arg = 2*P_W / (np.pi * w0_beam**2 * v_m_s * k_th * (Tm - T0))
if arg > 1:
    w_melt_um = w0_beam * np.sqrt(np.log(arg)) * 1e6
else:
    w_melt_um = w0_beam * 1e6

depth_um  = w_melt_um * 0.5   # rough aspect ratio for conduction mode
remelted  = depth_um / t_um

print(f"  316L SS: k={k_th}W/mK, alpha={alpha_th:.1e}m^2/s, Tm={Tm}C")
print(f"  E_linear = P/v = {E_lin:.3f} J/m = {E_lin*1e-3:.3f} J/mm")
print(f"  Melt pool width:  ~{w_melt_um:.1f} um  (>{d_um:.0f}um hatch = overlapping, good)")
print(f"  Melt pool depth:  ~{depth_um:.1f} um")
print(f"  Remelted layers:  ~{remelted:.1f}  (>1 = good bonding between layers)")

# Cooling rate -> grain size
dT_dt = v_m_s * (Tm - T0) / (w_melt_um * 1e-6)   # K/s approximate
SDAS_um = 50 * (dT_dt * 1e-3)**(-0.33)  # secondary dendrite arm spacing
print(f"  Cooling rate:  ~{dT_dt:.2e} K/s")
print(f"  SDAS (grain size proxy): ~{SDAS_um:.2f} um  "
      f"({'fine grain, good strength' if SDAS_um < 5 else 'coarse'})")

# Material table for LPBF
print("\n  Common LPBF materials:")
print(f"  {'Material':20s}  {'P(W)':>6}  {'v(mm/s)':>8}  {'E_vol(J/mm3)':>13}  {'App'}")
print("-" * 65)
lpbf_params = [
    ("316L SS",          200,  800,  "structural, RogueGuard housing"),
    ("Ti-6Al-4V",        200,  700,  "aerospace, biomedical"),
    ("AlSi10Mg",         350, 1500,  "lightweight, thermal mgmt"),
    ("Inconel 718",      285,  960,  "high-temp, jet engines"),
    ("CuCrZr (copper)", 800,  500,  "heat exchangers, waveguides"),
    ("17-4 PH SS",       200,  800,  "high strength, tooling"),
]
for mat, P, v, app in lpbf_params:
    ev = P / (v * 0.1 * 0.03)   # typical d=100um, t=30um
    print(f"  {mat:20s}  {P:>6}  {v:>8}  {ev:>13.1f}  {app}")


# -----------------------------------------------------------------------------
# SECTION 3: PIPE ENGINEERING -- BARLOW'S FORMULA + LOWE'S STOCK
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SECTION 3: PIPE ENGINEERING -- BARLOW'S FORMULA + SCHEDULE 40/80")
print("=" * 65)

print("""
  Barlow's formula (hoop stress, thin-wall approximation):
    sigma_hoop = P * D / (2 * t)
    t_min      = P * D / (2 * SMYS * SF)

  P    = internal pressure (psi or Pa)
  D    = outside diameter
  t    = wall thickness
  SMYS = specified minimum yield strength
  SF   = safety factor (typically 0.72 for gas, 0.4 for water main)

  Schedule number: Sch = 1000 * P / S  (P in psi, S = allowable stress psi)
  Higher schedule -> thicker wall -> higher pressure rating
""")

# Pipe schedule table (ASME B36.10, steel pipe)
# (NPS, OD_in, sch40_wall_in, sch80_wall_in)
pipes = [
    (0.5,   0.840, 0.109, 0.147),
    (0.75,  1.050, 0.113, 0.154),
    (1.0,   1.315, 0.133, 0.179),
    (1.5,   1.900, 0.145, 0.200),
    (2.0,   2.375, 0.154, 0.218),
    (3.0,   3.500, 0.216, 0.300),
    (4.0,   4.500, 0.237, 0.337),
    (6.0,   6.625, 0.280, 0.432),
    (8.0,   8.625, 0.322, 0.500),
]

SMYS_steel = 35_000   # psi, A53 Grade B steel (common at Lowe's / Home Depot)
SMYS_pvc   = 2_000    # psi, Schedule 40 PVC
SF = 0.72

print(f"\n  Steel pipe A53 Gr-B (SMYS={SMYS_steel:,} psi), SF={SF}")
print(f"  {'NPS':>5}  {'OD(in)':>7}  {'t_40(in)':>9}  {'t_80(in)':>9}  "
      f"{'P_40(psi)':>10}  {'P_80(psi)':>10}  {'P_40(bar)':>10}")
print("-" * 75)
for nps, OD, t40, t80 in pipes:
    P40 = 2 * SMYS_steel * SF * t40 / OD
    P80 = 2 * SMYS_steel * SF * t80 / OD
    P40_bar = P40 / 14.5038
    print(f"  {nps:>5.2f}  {OD:>7.3f}  {t40:>9.3f}  {t80:>9.3f}  "
          f"{P40:>10.0f}  {P80:>10.0f}  {P40_bar:>10.1f}")

# PVC Schedule 40 -- what you actually buy at Lowe's
print(f"\n  PVC Sch 40 (SMYS={SMYS_pvc:,} psi, SF=0.5) -- Lowe's stock:")
pvc_pipes = [
    (0.5,  0.840, 0.109),
    (0.75, 1.050, 0.113),
    (1.0,  1.315, 0.133),
    (2.0,  2.375, 0.154),
    (3.0,  3.500, 0.216),
    (4.0,  4.500, 0.237),
]
print(f"  {'NPS':>5}  {'OD(in)':>7}  {'t(in)':>7}  {'P_max(psi)':>11}  {'P_max(bar)':>11}  {'App'}")
print("-" * 65)
for nps, OD, t in pvc_pipes:
    P_max_pvc = 2 * SMYS_pvc * 0.5 * t / OD
    P_bar = P_max_pvc / 14.5038
    app = "potable water" if nps <= 1 else "drain/vent" if nps >= 3 else "irrigation"
    print(f"  {nps:>5.2f}  {OD:>7.3f}  {t:>7.3f}  {P_max_pvc:>11.0f}  {P_bar:>11.2f}  {app}")

# UK equivalents (BS EN 10255, DN sizes)
print(f"\n  UK equivalents (BS EN 10255 / ISO 228):")
uk_pipes = [
    ("DN15 (1/2 in BSP)",  21.3, 2.6,  "domestic water"),
    ("DN20 (3/4 in BSP)",  26.9, 2.6,  "domestic water"),
    ("DN25 (1 in BSP)",    33.7, 3.2,  "mains cold water"),
    ("DN50 (2 in BSP)",    60.3, 3.6,  "commercial/industrial"),
    ("DN100 (4 in BSP)",  114.3, 4.0,  "main distribution"),
]
SMYS_uk = 235e6  # Pa, S235 steel (UK standard)
print(f"  {'Designation':22s}  {'OD(mm)':>7}  {'t(mm)':>6}  {'P_max(bar)':>11}  App")
print("-" * 60)
for name, OD_mm, t_mm, app in uk_pipes:
    P_Pa = 2 * SMYS_uk * 0.72 * t_mm*1e-3 / (OD_mm*1e-3)
    P_bar = P_Pa / 1e5
    print(f"  {name:22s}  {OD_mm:>7.1f}  {t_mm:>6.1f}  {P_bar:>11.1f}  {app}")


# -----------------------------------------------------------------------------
# SECTION 4: 3D DESIGN -- OVERHANG CONSTRAINTS (UP/DOWN TOPOLOGY)
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SECTION 4: 3D ADDITIVE DESIGN -- OVERHANG + TOPOLOGY")
print("=" * 65)

print("""
  Critical LPBF design rule: self-supporting angle
    Overhang angle > 45 deg from horizontal -> needs support structure
    Support adds cost, post-processing, surface roughness on down-skin

  "Think up and down":
    UP-SKIN surface:  laser hits fresh powder on top of solid -> good finish
    DOWN-SKIN surface: laser melts powder above unsupported void -> sags, rough

  Design rules for LPBF (no support needed):
    1. Overhang angle >= 45 deg from horizontal
    2. Bridging span < 1-2 mm (horizontal holes OK if small)
    3. Minimum feature size >= 3x layer thickness (>0.1mm for 30um layers)
    4. Thin walls >= 0.4mm (2-3 melt tracks wide)
    5. Holes in horizontal plane: use teardrop shape for down-skin quality
""")

# SIMP topology optimization (2D, simplified)
print("--- SIMP topology sketch (Solid Isotropic Material with Penalization) ---")
print("""
  Minimize compliance (maximize stiffness) subject to volume fraction V*:
    min_rho  C = U^T K U   (strain energy)
    s.t.     K(rho) U = F
             sum(rho_e * v_e) <= V* * V_total
             0 < rho_min <= rho_e <= 1

  SIMP interpolation (p=3 penalization):
    E(rho_e) = rho_e^p * E0   (drives density to 0 or 1)

  Sensitivity (gradient for steepest-descent):
    dC/d_rho_e = -p * rho_e^(p-1) * u_e^T k0 u_e

  For pipes: optimize wall profile to minimize material while meeting
  yield stress under internal pressure + bending loads.
""")

# LPBF lattice: gyroid infill (common for stiffness/weight)
print("--- Gyroid lattice (TPMS): the go-to LPBF infill ---")
print("""
  Gyroid surface: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = t
  t=0: minimal surface (zero mean curvature everywhere)
  t != 0: shell with controllable wall thickness

  Properties (at 30% volume fraction):
    Relative stiffness:  E/E0 ~ 0.3^2 = 9%  (Gibson-Ashby scaling: n=2)
    Relative strength:   sigma/sigma_0 ~ 0.3^1.5 = 16%
    No preferred direction (isotropic unlike honeycomb)
    Self-supporting: all surfaces > 45 deg from horizontal
    -> zero support structures needed
""")

# Gibson-Ashby scaling for lattice stiffness
rho_rel = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
n_bend  = 2.0    # bending-dominated lattice (open-cell foam)
n_str   = 1.5    # stretch-dominated (octet truss)
E0_SS   = 193e9  # Pa, 316L SS full density
sig0_SS = 500e6  # Pa, 316L SS yield

print(f"  316L SS lattice (E0={E0_SS/1e9:.0f}GPa, sigma_y={sig0_SS/1e6:.0f}MPa):")
print(f"  {'rho_rel':>8}  {'E_bend(GPa)':>12}  {'E_stretch(GPa)':>15}  {'sigma_y(MPa)':>13}")
for rho in rho_rel:
    E_bend = rho**n_bend * E0_SS / 1e9
    E_str  = rho**n_str  * E0_SS / 1e9
    sig_y  = rho**n_str  * sig0_SS / 1e6
    print(f"  {rho:>8.2f}  {E_bend:>12.2f}  {E_str:>15.2f}  {sig_y:>13.1f}")


# -----------------------------------------------------------------------------
# SECTION 5: C HEADER FILE -- get_used_to_it.h
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SECTION 5: C HEADER FILE STRUCTURE  get_used_to_it.h")
print("=" * 65)

header_content = r"""
/* get_used_to_it.h
 * RogueGuard firmware -- shared types and constants
 * SBIR Phase I: optical rogue wave monitor
 */

#ifndef GET_USED_TO_IT_H
#define GET_USED_TO_IT_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* -- Physical constants -------------------------------------------- */
#define C_LIGHT         2.99792458e8f   /* m/s */
#define H_PLANCK        6.62607015e-34  /* J*s */
#define KB_BOLTZMANN    1.380649e-23    /* J/K */
#define E_CHARGE        1.602176634e-19 /* C */

/* -- Dispersion GS parameters ----------------------------------------- */
#define GS_D_MIN        5000            /* |D| minimum for convergence */
#define GS_NITER_DEFAULT 50
#define GS_NITER_MAX    200
#define GS_SIGNAL_LEN   1024            /* must be power of 2 */

/* -- ADC / hardware --------------------------------------------------- */
#define ADC_SAMPLE_RATE_GSPS  1.0f      /* GS/s */
#define ADC_BITS              12
#define ADC_FULL_SCALE_MV     1000.0f
#define ADC_LSB_MV  (ADC_FULL_SCALE_MV / (1 << ADC_BITS))

/* -- Status codes (never use magic numbers) --------------------------- */
typedef enum {
    GS_OK            = 0,
    GS_ERR_D_TOO_SMALL = -1,   /* |D| < GS_D_MIN */
    GS_ERR_NULL_PTR  = -2,
    GS_ERR_NITER     = -3,     /* n_iter <= 0 */
    GS_ERR_CONVERGENCE = -4,   /* did not converge to threshold */
} gs_status_t;

/* -- Bitfield config (fits in 32 bits) -------------------------------- */
typedef struct {
    int16_t  D1;            /* dispersion parameter 1 */
    int16_t  D2;            /* dispersion parameter 2 */
    uint8_t  n_iter;        /* GS iterations */
    uint8_t  unit_amp : 1;  /* enforce unit amplitude */
    uint8_t  verbose  : 1;
    uint8_t  _pad     : 6;
} __attribute__((packed)) gs_config_t;

/* -- Rogue wave detection threshold ----------------------------------- */
/* Rogue wave: peak > 2x RMS of background (oceanographic definition) */
#define ROGUE_RATIO     2.0f
#define ROGUE_WINDOW_NS 10.0f           /* ns integration window */

/* -- Inline math helpers ----------------------------------------------- */
static inline float db_to_linear(float dB) { return powf(10.0f, dB / 10.0f); }
static inline float linear_to_db(float x)  { return 10.0f * log10f(x + 1e-30f); }
static inline float phred_to_prob(float Q)  { return powf(10.0f, -Q / 10.0f); }
static inline float prob_to_phred(float p)  { return -10.0f * log10f(p + 1e-30f); }

/* -- Function prototypes ----------------------------------------------- */
gs_status_t gs_retrieve_phase(
    const float *I1,       /* input intensity 1 (length GS_SIGNAL_LEN) */
    const float *I2,       /* input intensity 2 */
    float        D1,       /* dispersion 1 */
    float        D2,       /* dispersion 2 */
    int          n_iter,   /* GS iterations */
    float       *phi_out   /* output phase (length GS_SIGNAL_LEN) */
);

gs_status_t gs_config_validate(const gs_config_t *cfg);

void        gs_apply_dispersion(
    const float *E_re, const float *E_im,  /* complex field in */
    float        D,                         /* dispersion */
    float       *E_out_re, float *E_out_im  /* complex field out */
);

#endif /* GET_USED_TO_IT_H */
"""

print(header_content)

# Validate: C header rules
print("--- C header best practices (for the .h file) ---")
rules = [
    ("Include guard",       "#ifndef / #define / #endif  -- prevents double inclusion"),
    ("No definitions",      "Never define variables/functions in .h (only declarations)"),
    ("static inline OK",    "Short math helpers can be inline -- compiler inlines at callsite"),
    ("Packed structs",      "__attribute__((packed)) for hardware registers / bitfields"),
    ("Enum for status",     "Never use bare ints for error codes -- enum is self-documenting"),
    ("const pointers",      "Input arrays: const float* -- lets compiler optimize"),
    ("No global state",     "No extern globals in .h -- breaks with multiple translation units"),
    ("stdint.h types",      "uint8_t not unsigned char -- size-explicit on all platforms"),
]
print(f"  {'Rule':22s}  Detail")
print("-" * 65)
for rule, detail in rules:
    print(f"  {rule:22s}  {detail}")

print("\nDone.")
