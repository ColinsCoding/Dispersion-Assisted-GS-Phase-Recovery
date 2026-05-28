"""
rogueguard_spice.py
==============================================================================
What IS the circuit?  Full SPICE netlist + differential equations for the
RogueGuard optical-to-electrical analog.

Sections
--------
A. Component values  -- match RLC group delay to D1=-600ps2, D2=-900ps2
B. SPICE netlist     -- LTspice/ngspice .cir file (ready to simulate)
C. Derivatives       -- all governing ODEs/PDEs annotated with regime truth
D. Lenz / law chart  -- which laws are EXACT (true), APPROXIMATE (partial),
                        or BROKEN (false) and how this changes with frequency

"Law favors [Lenz]:  dE = -dPhi/dt  -- TRUE for linear inductors,
                                       FALSE above self-resonance,
                                       CHANGING with time (frequency regime)"

Run: python rogueguard_spice.py
Outputs: rogueguard_circuit.cir   (open in LTspice / ngspice)
         rogueguard_derivatives.png
==============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# A. COMPONENT VALUES
# ==============================================================================
# Optical disperser transfer function:
#   H(omega) = exp(j * D * omega^2 / 2)      D in ps^2, omega in rad/ps
#
# Group delay:  tau_g(f) = D * 2*pi*f         f in GHz -> tau_g in ps
#
# Goal: find RLC values whose group delay MATCHES the disperser's
#       linear-in-frequency group delay over the signal band (0-50 GHz).
#
# Strategy: use a 2nd-order Bessel all-pass section (maximally flat group
# delay) and scale to match tau_g at the band edge f_max.
#
# For a 2nd-order all-pass:
#   H_AP(s) = (s^2 - w0/Q * s + w0^2) / (s^2 + w0/Q * s + w0^2)
#   Group delay at DC: tau_g(0) = 2Q/w0
#
# Disperser target tau_g at f=10 GHz:
#   D = -600 ps^2: tau_g = |D| * 2*pi * 10e9 / 1e12 = 37.7 ps
#   D = -900 ps^2: tau_g = |D| * 2*pi * 10e9 / 1e12 = 56.5 ps
#
# Match 2Q/w0 = tau_g(0) at DC (approximation valid for moderate Q):
#   Set Q = 0.866 (Bessel optimal), w0 = 2Q / tau_g(0)
#
# Implement as RLC ladder:
#   Series L, shunt C, termination R (standard bandpass prototype)
#   L = Q / (w0 * Z0)   C = 1 / (w0 * Q * Z0)   R = Z0 = 50 ohm
# ==============================================================================

Z0   = 50.0       # system impedance (ohms)
Q    = 0.866      # Bessel 2nd-order Q (maximally flat group delay)
f_GHz = np.linspace(0.01, 60, 3000)
omega = 2 * np.pi * f_GHz * 1e9   # rad/s

def disperser_H(omega_rad_s, D_ps2):
    """Optical disperser: H = exp(j*D*omega^2/2), D in ps^2."""
    D_s2 = D_ps2 * 1e-24    # ps^2 -> s^2
    return np.exp(1j * D_s2 * omega_rad_s**2 / 2.0)

def group_delay_s(H, omega_rad_s):
    """Group delay in seconds from H(omega)."""
    return -np.gradient(np.unwrap(np.angle(H)), omega_rad_s)

def allpass2_H(omega_rad_s, w0, Q):
    """2nd-order all-pass: H = (s^2 - w0/Q*s + w0^2)/(s^2 + w0/Q*s + w0^2)."""
    s  = 1j * omega_rad_s
    num = s**2 - (w0/Q)*s + w0**2
    den = s**2 + (w0/Q)*s + w0**2
    return num / den

def rlc_bandpass_H(omega_rad_s, R, L, C):
    """Series RLC, output across R: H = R/(R + jwL + 1/jwC)."""
    jw = 1j * omega_rad_s
    Z  = R + jw*L + 1.0/(jw*C + 1e-300)
    return R / Z

def compute_rlc_values(D_ps2, Z0=50.0, Q=0.866, f_design_GHz=10.0):
    """
    Find L, C, w0 for 2nd-order all-pass matching D at f_design_GHz.
    Returns dict with all values and derived quantities.
    """
    D_s2       = abs(D_ps2) * 1e-24          # s^2
    tau_target = D_s2 * 2*np.pi*f_design_GHz*1e9  # s  (group delay at f_design)

    w0   = 2*Q / tau_target                   # rad/s
    f0   = w0 / (2*np.pi) / 1e9              # GHz
    L    = Q / (w0 * Z0)                      # H
    C    = 1.0 / (w0 * Q * Z0)               # F

    return dict(D_ps2=D_ps2, tau_target_ps=tau_target*1e12,
                w0_rad_s=w0, f0_GHz=f0,
                L_nH=L*1e9, C_pF=C*1e12, R_ohm=Z0, Q=Q)

v1 = compute_rlc_values(-600.0)
v2 = compute_rlc_values(-900.0)

print("\n  RLC component values (matched to disperser group delay at 10 GHz)")
print("  " + "-"*60)
for label, v in [("D1 = -600 ps2", v1), ("D2 = -900 ps2", v2)]:
    print(f"\n  {label}")
    print(f"    Target tau_g at 10 GHz : {v['tau_target_ps']:.1f} ps")
    print(f"    Resonance f0           : {v['f0_GHz']:.3f} GHz")
    print(f"    L  (series inductor)   : {v['L_nH']:.4f} nH")
    print(f"    C  (shunt capacitor)   : {v['C_pF']:.4f} pF")
    print(f"    R  (termination)       : {v['R_ohm']:.1f} ohm")
    print(f"    Q                      : {v['Q']:.3f}")

# ==============================================================================
# B. SPICE NETLIST (.cir for LTspice / ngspice)
# ==============================================================================

def make_spice_netlist(v1, v2):
    """
    Generate LTspice-compatible SPICE netlist for the full RogueGuard
    optical bench electrical analog.

    Topology:
      V1 (laser) -> T1 (50:50 splitter = ideal transformer) ->
        arm1: L1-C1 (disperser D1=-600ps2) -> R_det1 (detector)
        arm2: L2-C2 (disperser D2=-900ps2) -> R_det2 (detector)

    SPICE node map:
      0    = GND
      1    = laser output (V_laser)
      2    = splitter port 1 (arm1 input)
      3    = splitter port 2 (arm2 input)
      4    = arm1 after disperser (det1 input)
      5    = arm2 after disperser (det2 input)
    """
    L1 = v1['L_nH'];  C1 = v1['C_pF']
    L2 = v2['L_nH'];  C2 = v2['C_pF']
    R  = v1['R_ohm']

    lines = [
        "* RogueGuard Optical Bench -- Electrical Circuit Analog",
        "* LTspice / ngspice SPICE netlist",
        "* Generated by rogueguard_spice.py (Jalali Lab phase-retrieval stack)",
        "*",
        "* Optical -> Electrical mapping:",
        "*   Laser source       -> V1 (AC voltage source, 1 Vpk)",
        "*   50/50 beam splitter-> T1 (ideal 1:1 transformer)",
        "*   Disperser D1=-600  -> L1 + C1 all-pass (Bessel 2nd order)",
        "*   Disperser D2=-900  -> L2 + C2 all-pass (Bessel 2nd order)",
        "*   InGaAs detector 1  -> R_det1 = 50 ohm (photodetector load)",
        "*   InGaAs detector 2  -> R_det2 = 50 ohm",
        "*   TD-GS DSP          -> .meas directives (post-processing)",
        "*",
        "* Governing law: V = L*dI/dt  (Faraday/Lenz)",
        "*                I = C*dV/dt  (Capacitor constitutive)",
        "*                V = I*R      (Ohm, linear regime only)",
        "* KVL: sum(V_k) = 0 [TRUE for lumped, FALSE above ~GHz self-resonance]",
        "* KCL: sum(I_k) = 0 [TRUE for lumped, FALSE when displacement current matters]",
        "*",
        ".title RogueGuard Optical-Electrical Analog",
        "",
        "* ── Source: CW laser at 1572nm modeled as AC source ────────────────",
        "V1  1 0  AC 1 SIN(0 1 1G 0 0)   ; 1 GHz carrier (normalized)",
        "",
        "* ── 50/50 beam splitter: ideal 1:1 transformer ─────────────────────",
        "* K-statement couples L_pri and L_sec (coupling coefficient = 1)",
        "L_pri  1 0  1uH",
        "L_sec1 2 0  1uH",
        "L_sec2 3 0  1uH",
        "K_split L_pri L_sec1  1.0   ; arm 1",
        "K_split2 L_pri L_sec2 1.0   ; arm 2",
        "",
        f"* ── Disperser D1 = -600 ps^2 (arm 1) ───────────────────────────────",
        f"* 2nd-order Bessel all-pass: L={L1:.4f} nH, C={C1:.4f} pF",
        f"* Matched group delay: {v1['tau_target_ps']:.1f} ps at 10 GHz",
        f"L1  2 4  {L1:.6f}nH",
        f"C1  4 0  {C1:.6f}pF",
        f"R_match1 4 0  {R:.1f}   ; impedance matching / Q control",
        "",
        f"* ── Disperser D2 = -900 ps^2 (arm 2) ───────────────────────────────",
        f"* L={L2:.4f} nH, C={C2:.4f} pF",
        f"L2  3 5  {L2:.6f}nH",
        f"C2  5 0  {C2:.6f}pF",
        f"R_match2 5 0  {R:.1f}",
        "",
        "* ── Detectors (50-ohm InGaAs PIN photodiodes) ───────────────────────",
        f"R_det1  4 0  {R:.1f}   ; detector arm 1 (I1 = V_det1^2 / R)",
        f"R_det2  5 0  {R:.1f}   ; detector arm 2 (I2 = V_det2^2 / R)",
        "",
        "* ── Analysis ─────────────────────────────────────────────────────────",
        ".ac dec 1000 1Meg 100G   ; AC sweep: 1 MHz to 100 GHz, 1000 pts/decade",
        "",
        "* ── Measurements ──────────────────────────────────────────────────────",
        ".meas AC tau_g1 PARAM '-deriv(phase(V(4)),frequency)'",
        ".meas AC tau_g2 PARAM '-deriv(phase(V(5)),frequency)'",
        ".meas AC mag1   PARAM 'mag(V(4))'",
        ".meas AC mag2   PARAM 'mag(V(5))'",
        "",
        "* ── SPICE options ─────────────────────────────────────────────────────",
        ".options NUMDGT=8 TRTOL=1 METHOD=Gear",
        ".backanno",
        ".end",
        "",
        "* ── How to run ─────────────────────────────────────────────────────────",
        "* LTspice: File -> Open -> select this .cir -> Run (Ctrl+R)",
        "* ngspice (terminal): ngspice rogueguard_circuit.cir",
        "*   then: plot v(4) v(5)",
        "* Export: File -> Export Data -> CSV for Python post-processing",
    ]
    return "\n".join(lines)

spice = make_spice_netlist(v1, v2)
with open('rogueguard_circuit.cir', 'w', encoding='utf-8') as f:
    f.write(spice)
print("\n  Saved: rogueguard_circuit.cir")

# ==============================================================================
# C. DIFFERENTIAL EQUATIONS AND LAW TRUTH TABLE
# ==============================================================================

# Compute responses for plotting
H_D1  = disperser_H(omega, -600.0)
H_D2  = disperser_H(omega,  -900.0)
H_AP1 = allpass2_H(omega, v1['w0_rad_s'], v1['Q'])
H_AP2 = allpass2_H(omega, v2['w0_rad_s'], v2['Q'])

gd_D1  = group_delay_s(H_D1,  omega) * 1e12   # ps
gd_D2  = group_delay_s(H_D2,  omega) * 1e12
gd_AP1 = group_delay_s(H_AP1, omega) * 1e12
gd_AP2 = group_delay_s(H_AP2, omega) * 1e12

# Phase
ph_D1  = np.unwrap(np.angle(H_D1))
ph_D2  = np.unwrap(np.angle(H_D2))
ph_AP1 = np.unwrap(np.angle(H_AP1))
ph_AP2 = np.unwrap(np.angle(H_AP2))

# ==============================================================================
# D. FIGURES
# ==============================================================================

fig = plt.figure(figsize=(18, 13))
fig.suptitle(
    'RogueGuard Circuit: What IS the Circuit?\n'
    'RLC analog of optical dispersers + Law truth table + Differential equations',
    fontsize=12, fontweight='bold'
)
gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

ax_gd   = fig.add_subplot(gs[0, 0])    # group delay comparison
ax_ph   = fig.add_subplot(gs[0, 1])    # phase comparison
ax_mag  = fig.add_subplot(gs[0, 2])    # magnitude
ax_law  = fig.add_subplot(gs[1, :2])   # law truth table
ax_eq   = fig.add_subplot(gs[1, 2])    # equations box
ax_circ = fig.add_subplot(gs[2, :])    # circuit schematic sketch

# ── Group delay ───────────────────────────────────────────────────────────────
ax_gd.plot(f_GHz, gd_D1,  'steelblue', lw=2.5, label='Optical D1=-600ps2 (exact)')
ax_gd.plot(f_GHz, gd_D2,  'tomato',    lw=2.5, label='Optical D2=-900ps2 (exact)')
ax_gd.plot(f_GHz, gd_AP1, 'steelblue', lw=1.5, ls='--', alpha=0.8,
           label='RLC all-pass D1 (approx)')
ax_gd.plot(f_GHz, gd_AP2, 'tomato',    lw=1.5, ls='--', alpha=0.8,
           label='RLC all-pass D2 (approx)')
ax_gd.axvline(10, color='gray', ls=':', lw=1, label='Design point 10 GHz')
ax_gd.set(xlabel='Frequency (GHz)', ylabel='Group delay (ps)',
          title='Group Delay Match\nOptical vs RLC all-pass')
ax_gd.set_xlim(0, 50); ax_gd.set_ylim(-50, 400)
ax_gd.legend(fontsize=6.5); ax_gd.grid(True, alpha=0.3)

# ── Phase ─────────────────────────────────────────────────────────────────────
ax_ph.plot(f_GHz, ph_D1,  'steelblue', lw=2.5, label='D1 optical (quadratic)')
ax_ph.plot(f_GHz, ph_D2,  'tomato',    lw=2.5, label='D2 optical (quadratic)')
ax_ph.plot(f_GHz, ph_AP1, 'steelblue', lw=1.5, ls='--', alpha=0.8,
           label='D1 RLC all-pass')
ax_ph.plot(f_GHz, ph_AP2, 'tomato',    lw=1.5, ls='--', alpha=0.8,
           label='D2 RLC all-pass')
ax_ph.set(xlabel='Frequency (GHz)', ylabel='Phase (rad)',
          title='Phase Response\nOptical phi=D*omega^2/2 vs RLC')
ax_ph.set_xlim(0, 50)
ax_ph.legend(fontsize=6.5); ax_ph.grid(True, alpha=0.3)

# ── Magnitude ─────────────────────────────────────────────────────────────────
ax_mag.plot(f_GHz, np.abs(H_D1),  'steelblue', lw=2.5, label='Optical (|H|=1 always)')
ax_mag.plot(f_GHz, np.abs(H_AP1), 'steelblue', lw=1.5, ls='--',
            label=f'RLC D1 (|H|~1 up to {v1["f0_GHz"]:.1f} GHz)')
ax_mag.plot(f_GHz, np.abs(H_AP2), 'tomato',    lw=1.5, ls='--',
            label=f'RLC D2 (|H|~1 up to {v2["f0_GHz"]:.1f} GHz)')
ax_mag.axvline(v1['f0_GHz'], color='steelblue', ls=':', lw=1)
ax_mag.axvline(v2['f0_GHz'], color='tomato',    ls=':', lw=1)
ax_mag.set(xlabel='Frequency (GHz)', ylabel='|H(f)|',
          title='Magnitude (optical = 1,\nRLC = 1 only below resonance)')
ax_mag.set_xlim(0, 50); ax_mag.set_ylim(-0.1, 1.3)
ax_mag.legend(fontsize=6.5); ax_mag.grid(True, alpha=0.3)

# ── Law truth table ───────────────────────────────────────────────────────────
ax_law.axis('off')
ax_law.set_title('Circuit Law Validity vs Frequency Regime (Lenz / Kirchhoff / Ohm / Maxwell)',
                 fontsize=9, fontweight='bold', pad=4)

LAW_TABLE = [
    # (Law,              DC/low-f,   RF (MHz-GHz),  Microwave,  Optical, Nonlinear)
    ("Ohm's  V=IR",      'EXACT',    'EXACT',        'APPROX',   'FALSE',  'FALSE'),
    ("KVL  sum(V)=0",    'EXACT',    'APPROX',       'FALSE',    'FALSE',  'FALSE'),
    ("KCL  sum(I)=0",    'EXACT',    'APPROX',       'FALSE',    'FALSE',  'FALSE'),
    ("Lenz  E=-dPhi/dt", 'EXACT',    'EXACT',        'EXACT',    'EXACT',  'EXACT'),
    ("Faraday (integral)",'EXACT',   'EXACT',        'EXACT',    'EXACT',  'EXACT'),
    ("Maxwell (PDE)",     'EXACT',   'EXACT',        'EXACT',    'EXACT',  'EXACT'),
    ("NLSE (fiber)",      'N/A',     'N/A',          'N/A',      'EXACT',  'EXACT'),
    ("Kramers-Kronig",    'EXACT',   'EXACT',        'EXACT',    'EXACT',  'APPROX'),
    ("Lumped L=L0",       'EXACT',   'APPROX',       'FALSE',    'FALSE',  'N/A'),
    ("Lumped C=C0",       'EXACT',   'APPROX',       'FALSE',    'FALSE',  'N/A'),
]
REGIMES = ['DC / kHz', 'RF\n(MHz-GHz)', 'Microwave\n(GHz)', 'Optical\n(THz)', 'Nonlinear']
COLOR_MAP = {
    'EXACT':  '#27ae60',
    'APPROX': '#f39c12',
    'FALSE':  '#e74c3c',
    'N/A':    '#95a5a6',
}

col_x = [0.0, 0.32, 0.46, 0.58, 0.70, 0.82]
# Header
for j, hdr in enumerate(['Law / Equation'] + REGIMES):
    ax_law.text(col_x[j], 0.97, hdr, fontsize=7.5, fontweight='bold',
                transform=ax_law.transAxes, va='top',
                color='#2c3e50')

for i, (law, *vals) in enumerate(LAW_TABLE):
    y = 0.91 - i * 0.088
    ax_law.text(col_x[0], y, law, fontsize=7, transform=ax_law.transAxes,
                va='top', fontfamily='monospace')
    for j, v in enumerate(vals):
        col = COLOR_MAP.get(v, '#95a5a6')
        ax_law.add_patch(plt.Rectangle(
            (col_x[j+1] - 0.005, y - 0.07), 0.115, 0.075,
            transform=ax_law.transAxes,
            facecolor=col, edgecolor='white', linewidth=0.5, alpha=0.85,
            zorder=2))
        ax_law.text(col_x[j+1] + 0.052, y - 0.035, v,
                    fontsize=6, transform=ax_law.transAxes,
                    ha='center', va='center', color='white',
                    fontweight='bold', zorder=3)

# Legend
for lbl, col in COLOR_MAP.items():
    pass
patches = [mpatches.Patch(color=COLOR_MAP[k], label=k)
           for k in ['EXACT', 'APPROX', 'FALSE', 'N/A']]
ax_law.legend(handles=patches, loc='lower right', fontsize=7,
              framealpha=0.9, ncol=4)

# Note on Lenz
ax_law.text(0.0, 0.04,
    "Lenz law (E = -dPhi/dt) is ALWAYS EXACT -- it IS Maxwell's curl E = -dB/dt.\n"
    "KVL/KCL are lumped approximations that FAIL when wavelength ~ circuit size.\n"
    "Ohm's law FAILS for semiconductors, superconductors, and nonlinear media.",
    fontsize=7, transform=ax_law.transAxes, va='bottom',
    color='#555', style='italic')

# ── Differential equations box ────────────────────────────────────────────────
ax_eq.axis('off')
ax_eq.set_title('Differential Equations\n("write derivatives")', fontsize=9,
                fontweight='bold', pad=4)

EQNS = [
    ("Resistor (Ohm):",        "V = I R",                     "#27ae60"),
    ("Inductor (Lenz/Faraday):","V = L dI/dt",                "#2980b9"),
    ("Capacitor:",             "I = C dV/dt",                  "#8e44ad"),
    ("RC low-pass ODE:",       "RC dV/dt + V = V_s",           "#e67e22"),
    ("",                       "(1st order, tau=RC)",           "#888"),
    ("RLC 2nd-order ODE:",     "L d2I/dt2 + R dI/dt + I/C",    "#c0392b"),
    ("",                       "  = dV_s/dt",                  "#c0392b"),
    ("",                       "(omega0=1/sqrt(LC), Q=...)",    "#888"),
    ("Telegrapher PDE:",       "d2V/dx2 = LC d2V/dt2",         "#16a085"),
    ("",                       "(wave, v=1/sqrt(LC))",          "#888"),
    ("NLSE (optical fiber):",  "dA/dz = -j*b2/2*d2A/dT2",     "#8e44ad"),
    ("",                       "       + j*gamma*|A|^2*A",     "#8e44ad"),
    ("Kramers-Kronig:",        "n(w) = 1 + c/pi * PV integral","#2c3e50"),
    ("",                       "(alpha<->phi, circuit R<->X)",  "#888"),
    ("WHEN KVL fails:",        "E_ind = -d/dt Int(B.dA)",      "#e74c3c"),
    ("",                       "(Faraday -- loop has EMF!)",    "#e74c3c"),
]

y0 = 0.98
dy = 0.062
for i, (lbl, eq, col) in enumerate(EQNS):
    y = y0 - i * dy
    if lbl:
        ax_eq.text(0.01, y, lbl, fontsize=6.5, transform=ax_eq.transAxes,
                   va='top', fontweight='bold', color=col)
    ax_eq.text(0.01, y - 0.028, eq, fontsize=6, transform=ax_eq.transAxes,
               va='top', color=col, fontfamily='monospace')

# ── Circuit schematic sketch ───────────────────────────────────────────────────
ax_circ.set_xlim(0, 20); ax_circ.set_ylim(0, 5)
ax_circ.axis('off')
ax_circ.set_title(
    f'SPICE Circuit Topology  --  D1={v1["D_ps2"]:.0f}ps2: '
    f'L={v1["L_nH"]:.3f}nH C={v1["C_pF"]:.3f}pF  |  '
    f'D2={v2["D_ps2"]:.0f}ps2: L={v2["L_nH"]:.3f}nH C={v2["C_pF"]:.3f}pF  '
    f'(Z0=50ohm, Q=0.866 Bessel)',
    fontsize=8.5, fontweight='bold', pad=3)

def wire(ax, x0, y0, x1, y1, col='k', lw=1.5):
    ax.plot([x0, x1], [y0, y1], color=col, lw=lw)

def node(ax, x, y, label, col='k', fs=7):
    ax.plot(x, y, 'o', color=col, ms=5, zorder=4)
    ax.text(x, y+0.2, label, ha='center', fontsize=fs, color=col)

def box(ax, x, y, w, h, label, sub, col):
    r = plt.Rectangle((x-w/2, y-h/2), w, h,
                       facecolor=col, edgecolor='k', linewidth=1.2, alpha=0.85, zorder=3)
    ax.add_patch(r)
    ax.text(x, y+0.08, label, ha='center', va='center', fontsize=7,
            fontweight='bold', color='white', zorder=4)
    ax.text(x, y-0.22, sub,  ha='center', va='center', fontsize=5.5,
            color='white', alpha=0.9, zorder=4)

# GND line
wire(ax_circ, 0, 1.0, 20, 1.0, col='#bbb', lw=1)
ax_circ.text(0.3, 0.8, 'GND', fontsize=6.5, color='#888')

# V1 source
box(ax_circ, 1.2, 2.5, 1.4, 1.0, 'V1', 'Laser\n1572nm', '#16a085')
wire(ax_circ, 1.2, 3.0, 1.2, 3.5)
wire(ax_circ, 1.2, 2.0, 1.2, 1.0)
node(ax_circ, 1.2, 3.5, 'net1', col='#16a085')
wire(ax_circ, 1.2, 3.5, 3.5, 3.5)

# Transformer / splitter
box(ax_circ, 4.0, 3.5, 1.0, 0.8, 'T1', '50:50\nsplit', '#2980b9')
wire(ax_circ, 3.5, 3.5, 3.5, 3.5)   # into T1
node(ax_circ, 3.5, 3.5, 'net1')
wire(ax_circ, 3.5, 3.5, 3.5, 3.5)

# Arm 1 (top)
wire(ax_circ, 4.5, 3.9, 6.5, 3.9)
node(ax_circ, 6.5, 3.9, 'arm1', col='#e67e22')
box(ax_circ, 7.5, 3.9, 1.6, 0.7, 'L1', f'{v1["L_nH"]:.3f} nH', '#e67e22')
wire(ax_circ, 8.3, 3.9, 9.5, 3.9)
node(ax_circ, 9.5, 3.9, 'mid1')
box(ax_circ, 10.5,3.9, 1.6, 0.7, 'C1', f'{v1["C_pF"]:.3f} pF', '#e67e22')
wire(ax_circ, 10.5,3.2, 10.5,1.0)    # C1 to GND
wire(ax_circ, 11.3,3.9, 12.5,3.9)
node(ax_circ, 12.5,3.9, 'det1')
box(ax_circ, 13.5,3.9, 1.4, 0.7, 'R_det1', '50 ohm', '#27ae60')
wire(ax_circ, 13.5,3.5, 13.5,1.0)    # det to GND

# Arm 2 (bottom)
wire(ax_circ, 4.5, 3.1, 6.5, 3.1)
node(ax_circ, 6.5, 3.1, 'arm2', col='#c0392b')
box(ax_circ, 7.5, 3.1, 1.6, 0.7, 'L2', f'{v2["L_nH"]:.3f} nH', '#c0392b')
wire(ax_circ, 8.3, 3.1, 9.5, 3.1)
node(ax_circ, 9.5, 3.1, 'mid2')
box(ax_circ, 10.5,3.1, 1.6, 0.7, 'C2', f'{v2["C_pF"]:.3f} pF', '#c0392b')
wire(ax_circ, 10.5,2.4, 10.5,1.0)
wire(ax_circ, 11.3,3.1, 12.5,3.1)
node(ax_circ, 12.5,3.1, 'det2')
box(ax_circ, 13.5,3.1, 1.4, 0.7, 'R_det2', '50 ohm', '#27ae60')
wire(ax_circ, 13.5,2.7, 13.5,1.0)

# DSP block
box(ax_circ, 16.5, 3.5, 2.2, 1.5, 'TD-GS DSP', 'reads V_det1\nV_det2\n->E(t)', '#9b59b6')
wire(ax_circ, 14.2, 3.9, 15.4, 3.9)
wire(ax_circ, 14.2, 3.1, 15.4, 3.1)
wire(ax_circ, 15.4, 3.9, 15.4, 3.5)
wire(ax_circ, 15.4, 3.1, 15.4, 3.5)
wire(ax_circ, 15.4, 3.5, 15.4, 3.5)
wire(ax_circ, 15.4, 3.5, 15.4, 3.5)
ax_circ.annotate('', xy=(15.4, 3.5), xytext=(14.8, 3.5),
                 arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

# CNN output
wire(ax_circ, 17.6, 3.5, 18.5, 3.5)
ax_circ.annotate('', xy=(18.5, 3.5), xytext=(17.9, 3.5),
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
ax_circ.text(18.6, 3.5, 'CNN\nP(rogue)', va='center', fontsize=6.5,
             color='#e74c3c', fontweight='bold')

# Lenz law annotation
ax_circ.text(7.5, 1.5,
    "V_L = L * dI/dt   [Lenz: ALWAYS TRUE]\n"
    "I_C = C * dV/dt   [Capacitor: ALWAYS TRUE]\n"
    "V   = I * R       [Ohm: TRUE only if linear]\n"
    "KVL: sum(V) = 0   [TRUE only if lumped + low freq]",
    fontsize=7, color='#2c3e50', va='top',
    bbox=dict(fc='lightyellow', ec='#ccc', pad=4))

fig.savefig('rogueguard_derivatives.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Saved: rogueguard_derivatives.png")

# ==============================================================================
# SUMMARY PRINTOUT
# ==============================================================================

print("\n" + "="*70)
print("  CIRCUIT SUMMARY")
print("="*70)
print("""
  What IS the circuit?
  --------------------
  The RogueGuard optical bench maps exactly to a SPICE circuit:

    V1 (laser) -> T1 (1:1 transformer = 50/50 splitter)
      Arm 1: L1={L1:.4f}nH + C1={C1:.4f}pF  (Bessel all-pass ~ D1=-600ps2)
      Arm 2: L2={L2:.4f}nH + C2={C2:.4f}pF  (Bessel all-pass ~ D2=-900ps2)
      Loads: R_det1 = R_det2 = 50 ohm (InGaAs detectors)
    DSP: reads V(det1), V(det2) -> TD-GS -> CNN -> P(rogue)

  Lenz law (law favors... Lenz):
  -----------------------------
  V_L = L * dI/dt  is ALWAYS EXACT -- it IS Faraday's law.
  KVL is only approximately true for lumped circuits.
  At optical frequencies (THz), KVL/KCL are completely invalid.
  Maxwell's equations are always exact.

  The law CHANGES WITH FREQUENCY:
    DC-kHz:     KVL exact,  Ohm exact,  L/C lumped exact
    MHz-GHz:    KVL approx, parasitic L/C start to matter
    GHz+:       KVL fails,  must use Maxwell (distributed)
    THz (optical): lumped elements meaningless, use NLSE/FDTD

  Derivatives (if the shoe fits):
  --------------------------------
  RC:     tau*dV/dt + V = V_s               (1st order ODE)
  RLC:    L*d2I/dt2 + R*dI/dt + I/C = ...   (2nd order ODE)
  Line:   d2V/dx2 = LC * d2V/dt2            (wave PDE)
  NLSE:   dA/dz = -j*b2/2*d2A/dT2 + jg*|A|^2*A  (nonlinear PDE)
  KK:     n(w) = 1 + c/pi * PV_integral     (integral equation)
""".format(
    L1=v1['L_nH'], C1=v1['C_pF'],
    L2=v2['L_nH'], C2=v2['C_pF'],
))
print("  Files:")
print("    rogueguard_circuit.cir     -> LTspice: open + Ctrl+R to simulate")
print("    rogueguard_derivatives.png -> law table + equations + schematic")
print("="*70)
