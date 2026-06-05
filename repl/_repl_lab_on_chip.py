"""
repl/_repl_lab_on_chip.py
Lab-on-chip physics: microfluidics, gravity variation, cleanroom building,
vibration isolation. Civil engineering meets biotech.
Low Reynolds number, Poiseuille flow, capillary number, electroosmosis.
"""
import math
import numpy as np
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 62)
print("LAB-ON-CHIP  PHYSICS  (microfluidics + building + gravity)")
print("=" * 62)
print()

# ============================================================
# 1. GRAVITY: NORTHERN vs SOUTHERN, LATITUDE DEPENDENCE
# ============================================================
print("=== 1. GRAVITY: g(latitude) -- does location matter? ===")
print("""
  g varies with latitude due to:
    1. Earth's oblateness (equatorial bulge, r_equator > r_pole)
    2. Centrifugal acceleration from rotation
    3. Local geology (minor)

  Somigliana formula (WGS-84 ellipsoid):
    g(phi) = g_e * (1 + k*sin^2(phi)) / sqrt(1 - e^2*sin^2(phi))

  where:
    g_e = 9.7803253359  m/s^2  (equatorial)
    k   = 0.00193185265241
    e^2 = 0.00669437999014  (eccentricity^2)

  Simplified (Helmert 1901):
    g(phi) ~ 9.80620 - 0.02586*cos(2*phi) + 0.000059*cos^2(2*phi)

  NORTH vs SOUTH at same |latitude|:
    g(+phi) = g(-phi)  EXACTLY (symmetric about equator)
    -> hemisphere does NOT matter for gravity
    -> only |latitude| matters

  Altitude correction: g(h) ~ g(0) * (1 - 2h/R_earth)
    ~3 uGal per meter height  (1 Gal = 1 cm/s^2)
""")

g_e  = 9.7803253359
k    = 0.00193185265241
e2   = 0.00669437999014

locs = [
    ("Equator (Quito)",          0.0,     2850),
    ("Singapore",                1.3,       15),
    ("Sacramento (CSUS)",       38.6,       30),
    ("UCLA (Jalali lab)",        34.1,       95),
    ("Tokyo (Japan lab)",        35.7,       40),
    ("London UK",                51.5,       11),
    ("Helsinki Finland",         60.2,       26),
    ("McMurdo Station S.Pole",  -77.8,       79),
    ("North Pole",               90.0,        0),
]

print(f"  {'Location':28s}  {'lat':6s}  {'alt(m)':7s}  {'g (m/s^2)':12s}  {'delta_g (mm/s^2)'}")
g_sac = None
for name, lat, alt in locs:
    phi = math.radians(lat)
    g0  = g_e * (1 + k*math.sin(phi)**2) / math.sqrt(1 - e2*math.sin(phi)**2)
    g_alt = g0 * (1 - 2*alt/6371000)
    if "Sacramento" in name:
        g_sac = g_alt
    delta = (g_alt - 9.80665) * 1000  # mm/s^2 from standard g
    print(f"  {name:28s}  {lat:6.1f}  {alt:7.0f}  {g_alt:12.7f}  {delta:+.2f}")
print()
print(f"  LAB-ON-CHIP IMPACT:")
print(f"    Sedimentation velocity: v = d^2*(rho_p-rho_f)*g / (18*eta)")
print(f"    g difference equator vs pole: ~5.2 cm/s^2 = 0.5%")

# sedimentation for 1um bead in water
d_bead  = 1e-6    # m
rho_p   = 1050.0  # polystyrene bead kg/m3
rho_f   = 1000.0  # water
eta_w   = 1e-3    # Pa.s
g_equator = 9.780
g_pole    = 9.832

v_eq   = d_bead**2 * (rho_p - rho_f) * g_equator / (18 * eta_w)
v_pole = d_bead**2 * (rho_p - rho_f) * g_pole    / (18 * eta_w)
print(f"    1um bead in water:")
print(f"      v_sed equator: {v_eq*1e9:.3f} nm/s")
print(f"      v_sed pole:    {v_pole*1e9:.3f} nm/s")
print(f"      ratio:         {v_pole/v_eq:.4f}  (0.5% difference -> negligible)")
print(f"    Coriolis in 100um channel: omega*v ~ 7e-5 * 10e-6 = 7e-10 m/s^2")
print(f"    vs gravity: 9.8 m/s^2 -> ratio 7e-11 -> COMPLETELY NEGLIGIBLE")
print(f"    -> hemisphere does not matter for lab-on-chip")
print()

# ============================================================
# 2. MICROFLUIDICS: LOW REYNOLDS NUMBER PHYSICS
# ============================================================
print("=== 2. MICROFLUIDICS: LOW Re PHYSICS ===")
print("""
  Reynolds number:   Re = rho * v * L / eta
  Capillary number:  Ca = eta * v / gamma  (viscous vs surface tension)
  Peclet number:     Pe = v * L / D_diff   (advection vs diffusion)

  MICROFLUIDIC REGIME:
    Channel width L ~ 10-200 um
    Flow velocity v ~ 0.1-10 mm/s
    Water: rho=1000, eta=1e-3, gamma=0.072

  Re << 1  -> viscous dominates, NO turbulence, fully laminar
  Ca << 1  -> surface tension dominates flow shape (droplets)
  Pe >> 1  -> convection dominates, mixing requires special design

  POISEUILLE FLOW (rectangular channel, width w >> height h):
    Q  = (w * h^3 * dP/dz) / (12 * eta)    [m^3/s]
    v_max = (h^2 / (8*eta)) * (-dP/dz)      [m/s]
    v_avg = v_max * 2/3
    tau_wall = (h/2) * (-dP/dz)             [Pa] wall shear stress

  CAPILLARY PRESSURE (wettable channel):
    P_cap = 2*gamma*cos(theta) * (1/w + 1/h)
    Drives flow when dP_ext = 0  (passive pump)
""")

# numerical examples
cases = [
    ("Blood cell counting", 50e-6,  100e-6, 1e-3,   1e-3,  1000.0),
    ("DNA sorting",         10e-6,   20e-6, 1e-4,   1e-3,  1000.0),
    ("Droplet generator",  100e-6,  100e-6, 1e-3,   1e-3,  1000.0),
    ("Fiber cladding mode", 4.1e-6,  9e-6,  1e-6,   1.45e-3*1e-3, 2200.0),
]
eta_w = 1e-3
gamma_w = 0.072
D_diff = 1e-12  # m^2/s (small molecule in water)

print(f"  {'Application':22s}  {'h(um)':7s}  {'v(mm/s)':8s}  {'Re':8s}  {'Ca':10s}  {'Pe':8s}")
for name, h, w, v, eta, rho in cases:
    Re = rho * v * h / eta
    Ca = eta * v / gamma_w
    Pe = v * h / D_diff
    print(f"  {name:22s}  {h*1e6:7.1f}  {v*1e3:8.3f}  {Re:8.4f}  {Ca:10.4e}  {Pe:8.1f}")
print()

# Poiseuille flow pressure drop
print("  Poiseuille pressure drop for Q = 1 uL/min:")
Q = 1e-9 / 60   # m^3/s  (1 uL/min)
for h_um, w_um in [(10, 50), (50, 200), (100, 500)]:
    h = h_um * 1e-6
    w = w_um * 1e-6
    dP_per_m = 12 * eta_w * Q / (w * h**3)
    L_chip = 0.01  # 10mm channel
    dP_kPa = dP_per_m * L_chip / 1e3
    v_avg = Q / (w * h) * 1e3  # mm/s
    print(f"    h={h_um:3d}um w={w_um:3d}um: dP={dP_kPa:.2f} kPa/cm  v_avg={v_avg:.3f} mm/s")
print()

# ============================================================
# 3. CAPILLARY-DRIVEN FLOW: PASSIVE PUMP (no power needed)
# ============================================================
print("=== 3. CAPILLARY PUMP: PASSIVE LAB-ON-CHIP ===")
print("""
  Wicking velocity (Lucas-Washburn):
    dL/dt = (r * gamma * cos(theta)) / (4 * eta * L)
    L(t) = sqrt(r * gamma * cos(theta) * t / (2*eta))

  For PDMS (hydrophobic, theta~110 deg): needs surface treatment
  For glass/SiO2 (hydrophilic, theta~20 deg): passive wicking

  POINT-OF-CARE DIAGNOSTICS (lateral flow assay):
    Nitrocellulose membrane: pore r ~ 5-10 um
    Fills 40mm strip in ~10-15 min
    Cost: $0.10 per test strip
    COVID antigen test IS this: capillary + antibody lines
""")

gamma_val = 0.072  # N/m
eta_val   = 1e-3   # Pa.s

substrates = [
    ("Nitrocellulose",  5e-6,  20.0, "COVID/flu lateral flow"),
    ("Glass capillary", 50e-6, 20.0, "blood glucose meter"),
    ("PDMS (plasma)",   10e-6, 30.0, "microfluidic chip"),
    ("Paper (porous)",   2e-6, 40.0, "urine dipstick"),
]
print(f"  {'Substrate':22s}  {'r(um)':7s}  {'theta':6s}  {'L@60s (mm)':12s}  {'application'}")
for name, r, theta_deg, app in substrates:
    theta_r = math.radians(theta_deg)
    L_60s = math.sqrt(r * gamma_val * math.cos(theta_r) * 60.0 / (2*eta_val)) * 1e3
    print(f"  {name:22s}  {r*1e6:7.1f}  {theta_deg:6.1f}  {L_60s:12.2f}  {app}")
print()

# ============================================================
# 4. ELECTROOSMOTIC FLOW (no moving parts, voltage-driven)
# ============================================================
print("=== 4. ELECTROOSMOTIC FLOW (voltage-driven, no pump) ===")
print("""
  Helmholtz-Smoluchowski:
    v_eo = -(eps * zeta / eta) * E_field

  zeta = zeta potential of channel wall (mV)
  eps  = permittivity of fluid
  E    = applied electric field (V/m)

  Typical values (glass channel, 10mM buffer):
    zeta ~ -50 mV,  eps = 80*8.85e-12,  eta = 1e-3
    v_eo = (80*8.85e-12 * 0.050) / 1e-3 * E
         = 3.54e-8 * E  m^2/(V.s)
    For E = 100 V/cm = 10,000 V/m:  v_eo = 0.354 mm/s

  NO PRESSURE NEEDED: plug flow profile (uniform v across channel)
  vs Poiseuille: parabolic profile (bad for separation)

  ELECTROPHORESIS: charged molecules move in same field
    v_ep = mu_ep * E
    mu_ep ~ 1-10 * 1e-8 m^2/(V.s)  (proteins, DNA)
    Net velocity = v_eo + v_ep  (or v_eo - v_ep if opposite sign)

  CAPILLARY ELECTROPHORESIS (CE):
    Separate DNA/proteins by size in 10-60 min
    Resolution: baseline separation of 1 bp DNA fragments
    Same principle in microfluidic chip -> chip-CE in 60 seconds
""")

eps_f = 80 * 8.85e-12
zeta  = -50e-3
eta_v = 1e-3
E_fields = [100, 500, 1000]   # V/cm -> V/m

print(f"  Glass channel, zeta=-50mV, 10mM buffer:")
print(f"  {'E (V/cm)':10s}  {'v_eo (mm/s)':12s}  {'voltage for 10mm chip'}")
for E_vcm in E_fields:
    E_vm  = E_vcm * 100   # V/m
    v_eo  = abs(eps_f * zeta / eta_v) * E_vm * 1e3  # mm/s
    V_chip = E_vm * 0.010  # V for 10mm chip
    print(f"  {E_vcm:10.0f}  {v_eo:12.4f}  {V_chip:.1f} V")
print()

# ============================================================
# 5. BUILDING DESIGN: VIBRATION ISOLATION FOR SENSITIVE LAB
# ============================================================
print("=== 5. BUILDING DESIGN: VIBRATION ISOLATION ===")
print("""
  SENSITIVE LAB REQUIREMENTS:
    Optical tables:    < 1 um/s RMS vibration  (VC-A/B/C standard)
    Electron microscope: < 0.025 um/s (VC-D)
    Atom interferometer: < 0.001 um/s (VC-G/H)
    Lab-on-chip optics: typically VC-A or VC-B sufficient

  VIBRATION CRITERIA (BBN VC curves):
    VC-A:  50 um/s  generic labs
    VC-B:  25 um/s  microscopes, microelectronics
    VC-C:  12.5 um/s  e-beam, confocal microscopy
    VC-D:   6 um/s  electron microscopes
    VC-E:   3 um/s  nanofabrication
    VC-F:  1.5 um/s  next-gen lithography
    VC-G:  0.75 um/s  advanced research

  BUILDING-LEVEL MITIGATION:
    1. Ground floor or basement (lower ambient vibration)
    2. Concrete inertia slab (>200mm thick, isolated from structure)
    3. Air spring tables (Newport, TMC): f_n ~ 1-3 Hz (below floor 5-20 Hz)
    4. Seismic mass: heavy optical table on pneumatic legs
    5. Stay away from HVAC ductwork, elevator shafts, roads

  TRANSFER FUNCTION (single-degree-of-freedom isolator):
    T(f) = sqrt(1 + (2*zeta*f/f_n)^2) / sqrt((1-(f/f_n)^2)^2 + (2*zeta*f/f_n)^2)
    At f >> f_n: T ~ (f_n/f)^2  (isolation improves as f^-2)

  CIVIL ENGINEERING SPECS:
    Column spacing: <= 6m (minimize floor deflection)
    Floor slab: 300mm concrete, decoupled from walls
    Slab natural freq: target > 15 Hz (avoid resonance with lab equipment)
    Damping ratio: zeta ~ 0.02-0.05 for concrete
""")

# Vibration isolation transmissibility
f_n    = 2.0    # Hz air spring table
zeta_v = 0.02   # low damping
freqs  = np.array([1, 2, 5, 10, 20, 50, 100])

print(f"  Air spring table f_n={f_n}Hz, zeta={zeta_v}:")
print(f"  {'f (Hz)':8s}  {'T(f)':10s}  {'Isolation (dB)':14s}  {'description'}")
for f in freqs:
    r = f / f_n
    T = math.sqrt(1 + (2*zeta_v*r)**2) / math.sqrt((1-r**2)**2 + (2*zeta_v*r)**2)
    dB = 20*math.log10(T)
    desc = ("resonance" if abs(r-1)<0.2 else
            "isolating" if f > f_n*2 else "below resonance")
    print(f"  {f:8.0f}  {T:10.4f}  {dB:14.1f}  {desc}")
print()

# ============================================================
# 6. FIBER + LOC: THE INTEGRATED PLATFORM
# ============================================================
print("=== 6. FIBER OPTIC + LAB-ON-CHIP INTEGRATION ===")
print("""
  PROBLEM: lab-on-chip needs optical detection
    -> spectrometer is expensive and off-chip
    -> fiber waveguide built INTO chip solves this

  ARCHITECTURE (optofluidic chip):
    +-----------+    +--------------+    +-----------+
    | PDMS chip |    | SiO2 waveguide|    | Fiber out |
    | microfluidic <- evanescent -> to photodetector |
    | channels  |    | W ~ 2-4 um   |    | or camera |
    +-----------+    +--------------+    +-----------+

  DETECTION MODES:
    Absorption:    I = I0 * exp(-epsilon*c*L)  Beer-Lambert on-chip
    Fluorescence:  excite with 488nm fiber, collect 520nm emission
    Scattering:    count cells by Mie scattering (forward scatter)
    Phase:         GS recovery of phase shift from analyte

  CELL COUNTING (flow cytometry on chip):
    Channel width 20um, cell diameter 10-15um
    One cell at a time -> single-cell resolution
    v_cell ~ 1 mm/s -> 1000 cells/s throughput
    Cost: PDMS chip $5, fiber coupler $200 -> $205 vs $50K Coulter counter

  GS CONNECTION:
    Analyte changes refractive index -> phase shift delta_phi
    delta_phi = (2*pi/lambda) * delta_n * L_interaction
    GS recovers delta_phi from intensity measurement alone
    -> label-free, quantitative, real-time
    -> the SBIR product: fiber GS phase sensor on LOC chip
""")

# Phase shift from analyte
lam_nm    = 1550.0
n_water   = 1.333
dn_analyte = 1e-4   # RIU change from binding
L_um_vals = [100, 500, 1000, 5000]  # um interaction length

print(f"  Phase shift from delta_n = {dn_analyte:.0e} RIU (antibody binding):")
print(f"  {'L_int (um)':12s}  {'delta_phi (rad)':16s}  {'detectable?'}")
for L_um in L_um_vals:
    L_m = L_um * 1e-6
    dphi = (2*math.pi / (lam_nm*1e-9)) * dn_analyte * L_m
    detect = "YES (GS: > 0.01 rad)" if dphi > 0.01 else "marginal" if dphi > 0.001 else "need longer L"
    print(f"  {L_um:12.0f}  {dphi:16.5f}  {detect}")
print()
print(f"  -> at L=5mm interaction length: delta_phi = {(2*math.pi/(lam_nm*1e-9))*dn_analyte*5e-3:.4f} rad")
print(f"  -> GS minimum detectable phase: ~0.01 rad (from convergence tests)")
print(f"  -> binding detection YES at L >= 1mm (achievable in spiral waveguide)")
print()
print("""  SPIRAL WAVEGUIDE TRICK:
    Coil waveguide 20 turns * 500um diameter = 31.4 mm path in 1mm^2 footprint
    -> fit 5mm effective interaction length in tiny chip area
    -> same principle as fiber delay line coils in sensing
""")
