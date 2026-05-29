"""
element_cards.py
----------------
Hardware-realism element cards for photonics engineering.
Each card: full atomic properties + Griffiths EM/QM reference + PIC application.

Steel / metal aesthetic.  Dark background, chrome typography.

Run:  py -3.12 sim/element_cards.py
Out:  docs/element_cards.png   (3x4 grid of 12 cards)
      docs/element_radar.png   (radar comparison of normalised properties)
"""

import pathlib, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

_HERE = pathlib.Path(__file__).parent
_DOCS = _HERE.parent / "docs"
_DOCS.mkdir(parents=True, exist_ok=True)

# ── Element database ─────────────────────────────────────────────────────────
# Fields: symbol, name, Z, group, atomic_mass, std_state,
#         e_config, ox_states, electronegativity, atomic_radius_pm,
#         ionization_kJ, electron_affinity_kJ,
#         melting_C, boiling_C, density_gcc,
#         griffiths_ref, pic_role, wave_eqn
ELEMENTS = [
    dict(
        symbol="Si", name="Silicon",      Z=14,  group="IV-A",
        atomic_mass=28.085,  std_state="solid",
        e_config="[Ne] 3s² 3p²",
        ox_states="+4, +2, -4",          electronegativity=1.90,
        atomic_radius_pm=111,            ionization_kJ=786.5,
        electron_affinity_kJ=134,        melting_C=1414,
        boiling_C=3265,                  density_gcc=2.329,
        griffiths_ref="Griffiths QM §2.6\nFinite sq. well -> band gap",
        pic_role="SOI waveguide core\n220nm x 450nm wire",
        wave_eqn="E_g = 1.12 eV  [indirect]\nn_eff = 2.44 @ 1550nm",
        color="#4a90d9",
    ),
    dict(
        symbol="Ge", name="Germanium",    Z=32,  group="IV-A",
        atomic_mass=72.630,  std_state="solid",
        e_config="[Ar] 3d10 4s² 4p²",
        ox_states="+4, +2, -4",          electronegativity=2.01,
        atomic_radius_pm=125,            ionization_kJ=762.0,
        electron_affinity_kJ=119,        melting_C=938,
        boiling_C=2833,                  density_gcc=5.323,
        griffiths_ref="Griffiths QM §5.3\nIdentical particles,\nband theory",
        pic_role="SiGe photodetector\nIR absorption 1550nm",
        wave_eqn="E_g = 0.66 eV  [indirect]\nalpha(1550) >> Si",
        color="#7ec8a0",
    ),
    dict(
        symbol="In", name="Indium",       Z=49,  group="III-A",
        atomic_mass=114.818, std_state="solid",
        e_config="[Kr] 4d10 5s² 5p¹",
        ox_states="+3, +1",              electronegativity=1.78,
        atomic_radius_pm=167,            ionization_kJ=558.3,
        electron_affinity_kJ=28.9,       melting_C=156.6,
        boiling_C=2072,                  density_gcc=7.31,
        griffiths_ref="Griffiths EM §4.4\nPolarization,\ndielectric response",
        pic_role="InP laser substrate\n1310/1550nm III-V",
        wave_eqn="InP E_g=1.34eV  [direct]\nhigh mu_e mobility",
        color="#e8a838",
    ),
    dict(
        symbol="P",  name="Phosphorus",   Z=15,  group="V-A",
        atomic_mass=30.974,  std_state="solid",
        e_config="[Ne] 3s² 3p³",
        ox_states="+5, +3, -3",          electronegativity=2.19,
        atomic_radius_pm=98,             ionization_kJ=1011.8,
        electron_affinity_kJ=72,         melting_C=44,
        boiling_C=281,                   density_gcc=1.823,
        griffiths_ref="Griffiths QM §4.4\nSpin-orbit coupling\nin III-V lattice",
        pic_role="InP / GaP anion\nSpin-orbit splits VB",
        wave_eqn="SOC: H' = (e/2m²c²)\n(1/r)(dV/dr) L.S",
        color="#cc6666",
    ),
    dict(
        symbol="Ga", name="Gallium",      Z=31,  group="III-A",
        atomic_mass=69.723,  std_state="solid",
        e_config="[Ar] 3d10 4s² 4p¹",
        ox_states="+3",                  electronegativity=1.81,
        atomic_radius_pm=136,            ionization_kJ=578.8,
        electron_affinity_kJ=28.9,       melting_C=29.8,
        boiling_C=2229,                  density_gcc=5.91,
        griffiths_ref="Griffiths QM §4.1\nAngular momentum\nIII-V cation",
        pic_role="GaAs / GaN cation\nHEMT, LED, laser",
        wave_eqn="GaAs E_g=1.42eV [direct]\nGaN E_g=3.4eV [direct]",
        color="#b07af5",
    ),
    dict(
        symbol="As", name="Arsenic",      Z=33,  group="V-A",
        atomic_mass=74.922,  std_state="solid",
        e_config="[Ar] 3d10 4s² 4p³",
        ox_states="+5, +3, -3",          electronegativity=2.18,
        atomic_radius_pm=114,            ionization_kJ=947.0,
        electron_affinity_kJ=78,         melting_C=817,
        boiling_C=614,                   density_gcc=5.73,
        griffiths_ref="Griffiths QM §4.2\nHydrogen-like 3D\ndonor impurity",
        pic_role="GaAs / AlGaAs anion\nN-type dopant in Si",
        wave_eqn="E_donor ~ 13.6eV/eps_r²\n~ 0.05eV in GaAs",
        color="#f08080",
    ),
    dict(
        symbol="Li", name="Lithium",      Z=3,   group="I-A",
        atomic_mass=6.941,   std_state="solid",
        e_config="[He] 2s¹",
        ox_states="+1",                  electronegativity=0.98,
        atomic_radius_pm=167,            ionization_kJ=520.2,
        electron_affinity_kJ=59.6,       melting_C=180.5,
        boiling_C=1342,                  density_gcc=0.534,
        griffiths_ref="Griffiths EM §4.5\nPiezo / ferroelectric\npolarization",
        pic_role="LiNbO3 modulator\nPockels r33=30pm/V",
        wave_eqn="dn = -(1/2)n^3 r33 E\nVpi L = lambda/(2 n^3 r33)",
        color="#88ccee",
    ),
    dict(
        symbol="Nb", name="Niobium",      Z=41,  group="V-B",
        atomic_mass=92.906,  std_state="solid",
        e_config="[Kr] 4d4 5s¹",
        ox_states="+5, +3, +2",          electronegativity=1.60,
        atomic_radius_pm=146,            ionization_kJ=652.1,
        electron_affinity_kJ=86.1,       melting_C=2477,
        boiling_C=4744,                  density_gcc=8.57,
        griffiths_ref="Griffiths EM §7.3\nFaraday induction\nEO modulator drive",
        pic_role="LiNbO3 cation\nthin-film TFLN platform",
        wave_eqn="TFLN: Vpi < 1V\nBW > 100GHz possible",
        color="#ddaa44",
    ),
    dict(
        symbol="Er", name="Erbium",       Z=68,  group="lanthanide",
        atomic_mass=167.259, std_state="solid",
        e_config="[Xe] 4f12 6s²",
        ox_states="+3",                  electronegativity=1.24,
        atomic_radius_pm=176,            ionization_kJ=589.3,
        electron_affinity_kJ=50,         melting_C=1529,
        boiling_C=2868,                  density_gcc=9.07,
        griffiths_ref="Griffiths QM §9.3\nStimulated emission\ntime-dep. perturb.",
        pic_role="EDFA gain medium\n1530nm emission 4f->4f",
        wave_eqn="I4(13/2)->I4(15/2)\nE_photon = 0.80 eV",
        color="#ff9966",
    ),
    dict(
        symbol="Au", name="Gold",         Z=79,  group="I-B",
        atomic_mass=196.967, std_state="solid",
        e_config="[Xe] 4f14 5d10 6s¹",
        ox_states="+3, +1",              electronegativity=2.54,
        atomic_radius_pm=144,            ionization_kJ=890.1,
        electron_affinity_kJ=222.8,      melting_C=1064,
        boiling_C=2856,                  density_gcc=19.32,
        griffiths_ref="Griffiths EM §2.5\nConductors, surface\ncharge, E=0 inside",
        pic_role="Ohmic contacts\nbonding wire, SPP",
        wave_eqn="Surface plasmon:\nksp = k0*sqrt(eps_m/(1+eps_m))",
        color="#ffd700",
    ),
    dict(
        symbol="Fe", name="Iron",         Z=26,  group="VIII-B",
        atomic_mass=55.845,  std_state="solid",
        e_config="[Ar] 3d6 4s²",
        ox_states="+3, +2",              electronegativity=1.83,
        atomic_radius_pm=126,            ionization_kJ=762.5,
        electron_affinity_kJ=15.7,       melting_C=1538,
        boiling_C=2861,                  density_gcc=7.874,
        griffiths_ref="Griffiths EM §6.3\nMagnetic materials\nB = mu0(H+M)",
        pic_role="Chassis / shielding\nFaraday cage: E=0",
        wave_eqn="Skin depth delta\n= sqrt(2/(omega*mu*sigma))",
        color="#aaaaaa",
    ),
    dict(
        symbol="Cu", name="Copper",       Z=29,  group="I-B",
        atomic_mass=63.546,  std_state="solid",
        e_config="[Ar] 3d10 4s¹",
        ox_states="+2, +1",              electronegativity=1.90,
        atomic_radius_pm=128,            ionization_kJ=745.5,
        electron_affinity_kJ=118.4,      melting_C=1085,
        boiling_C=2562,                  density_gcc=8.96,
        griffiths_ref="Griffiths EM §7.1\nOhm's law J=sigma*E\nhighest sigma metal",
        pic_role="RF interconnect\ntransmission line",
        wave_eqn="sigma=5.96e7 S/m\ndelta(1GHz)=2.09 um",
        color="#b87333",
    ),
]

# ── Card layout ───────────────────────────────────────────────────────────────
CARD_BG     = "#111418"
CARD_BORDER = "#3a3f4a"
HEADER_BG   = "#1a1e26"
TEXT_MAIN   = "#dde3ee"
TEXT_DIM    = "#7a8399"
TEXT_ACCENT = "#00d4ff"
TEXT_WARN   = "#ff9944"
FONT_MONO   = "monospace"


def draw_card(ax, el: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Card background
    card = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle="round,pad=0.015",
                          linewidth=1.5, edgecolor=CARD_BORDER,
                          facecolor=CARD_BG, zorder=1)
    ax.add_patch(card)

    # Header band
    hdr = FancyBboxPatch((0.02, 0.78), 0.96, 0.20,
                         boxstyle="round,pad=0.01",
                         linewidth=0, facecolor=HEADER_BG, zorder=2)
    ax.add_patch(hdr)

    # Accent left strip
    strip = mpatches.Rectangle((0.02, 0.02), 0.025, 0.76,
                                facecolor=el["color"], zorder=3, alpha=0.85)
    ax.add_patch(strip)

    # Symbol — large
    ax.text(0.35, 0.905, el["symbol"],
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=el["color"], fontfamily=FONT_MONO, zorder=5)

    # Z number
    ax.text(0.08, 0.96, str(el["Z"]),
            ha="center", va="center", fontsize=7,
            color=TEXT_DIM, fontfamily=FONT_MONO, zorder=5)

    # Name + group
    ax.text(0.62, 0.915, el["name"],
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=TEXT_MAIN, zorder=5)
    ax.text(0.62, 0.875, f"Group {el['group']}   A={el['atomic_mass']:.3f} u",
            ha="center", va="center", fontsize=6.5,
            color=TEXT_DIM, fontfamily=FONT_MONO, zorder=5)

    # Properties block
    props = [
        ("State",   el["std_state"]),
        ("Config",  el["e_config"]),
        ("Ox.",     el["ox_states"]),
        ("EN",      f"{el['electronegativity']:.2f}  (Pauling)"),
        ("r_atom",  f"{el['atomic_radius_pm']} pm"),
        ("IE1",     f"{el['ionization_kJ']:.1f} kJ/mol"),
        ("EA",      f"{el['electron_affinity_kJ']:.1f} kJ/mol"),
        ("Tmelt",   f"{el['melting_C']} deg C"),
        ("Tboil",   f"{el['boiling_C']} deg C"),
        ("rho",     f"{el['density_gcc']:.3f} g/cm^3"),
    ]

    y0 = 0.76
    dy = 0.063
    x_label = 0.07
    x_val   = 0.38

    for label, val in props:
        ax.text(x_label, y0, label,
                ha="left", va="top", fontsize=5.8,
                color=TEXT_DIM, fontfamily=FONT_MONO, zorder=5)
        ax.text(x_val, y0, val,
                ha="left", va="top", fontsize=5.8,
                color=TEXT_MAIN, fontfamily=FONT_MONO, zorder=5)
        y0 -= dy

    # Divider
    ax.plot([0.06, 0.94], [0.115, 0.115], color=CARD_BORDER, lw=0.7, zorder=4)

    # Griffiths ref
    ax.text(0.07, 0.105,
            el["griffiths_ref"].replace("\n", "  "),
            ha="left", va="top", fontsize=5.2,
            color=TEXT_ACCENT, fontfamily=FONT_MONO, zorder=5)

    # Wave equation / physics
    ax.text(0.07, 0.065,
            el["wave_eqn"].replace("\n", "  "),
            ha="left", va="top", fontsize=5.0,
            color=TEXT_WARN, fontfamily=FONT_MONO, zorder=5)

    # PIC role
    ax.text(0.07, 0.032,
            "PIC: " + el["pic_role"].replace("\n", " | "),
            ha="left", va="top", fontsize=5.0,
            color="#88ff88", fontfamily=FONT_MONO, zorder=5)


# ── Main figure — 3 x 4 grid ─────────────────────────────────────────────────
NCOLS, NROWS = 4, 3
fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(NCOLS * 3.4, NROWS * 3.8),
                         facecolor="#080a0d")
fig.subplots_adjust(wspace=0.04, hspace=0.04,
                    left=0.01, right=0.99, top=0.97, bottom=0.01)

fig.suptitle(
    "PHOTONICS HARDWARE ELEMENTS  //  Griffiths EM+QM Reference Cards",
    color="#00d4ff", fontsize=11, fontfamily=FONT_MONO, y=0.995
)

for idx, (ax, el) in enumerate(zip(axes.flat, ELEMENTS)):
    draw_card(ax, el)

card_path = _DOCS / "element_cards.png"
plt.savefig(card_path, dpi=160, facecolor="#080a0d", bbox_inches="tight")
plt.close()
print(f"Saved {card_path}")


# ── Radar chart — compare 6 normalised properties across elements ─────────────
RADAR_PROPS = [
    ("EN",      "electronegativity",    5.0),
    ("IE1",     "ionization_kJ",        1100.0),
    ("EA",      "electron_affinity_kJ", 230.0),
    ("density", "density_gcc",          20.0),
    ("r_atom",  "atomic_radius_pm",     180.0),
    ("mass",    "atomic_mass",          200.0),
]
N_AX = len(RADAR_PROPS)
angles = np.linspace(0, 2 * np.pi, N_AX, endpoint=False).tolist()
angles += angles[:1]

fig2, ax2 = plt.subplots(figsize=(10, 10),
                         subplot_kw=dict(polar=True),
                         facecolor="#080a0d")
ax2.set_facecolor("#0d1017")
ax2.spines["polar"].set_color("#2a2f3a")
ax2.tick_params(colors="#5a6070", labelsize=7)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels([p[0] for p in RADAR_PROPS],
                    color=TEXT_ACCENT, fontsize=9, fontfamily=FONT_MONO)
ax2.set_yticklabels([])
ax2.set_ylim(0, 1.05)
ax2.yaxis.grid(True, color="#1e2330", linewidth=0.6)
ax2.xaxis.grid(True, color="#1e2330", linewidth=0.6)

for el in ELEMENTS:
    vals = [el[key] / norm for _, key, norm in RADAR_PROPS]
    vals += vals[:1]
    ax2.plot(angles, vals, lw=1.5, color=el["color"], alpha=0.85)
    ax2.fill(angles, vals, color=el["color"], alpha=0.06)
    # Label at position of max value
    max_i = int(np.argmax(vals[:-1]))
    ax2.annotate(el["symbol"],
                 xy=(angles[max_i], vals[max_i]),
                 color=el["color"], fontsize=7,
                 fontfamily=FONT_MONO, fontweight="bold")

ax2.set_title("Photonics Elements — Normalised Property Radar\n"
              "(EN / IE / EA / density / atomic radius / mass)",
              color=TEXT_ACCENT, fontsize=10, fontfamily=FONT_MONO, pad=20)

radar_path = _DOCS / "element_radar.png"
plt.savefig(radar_path, dpi=140, facecolor="#080a0d", bbox_inches="tight")
plt.close()
print(f"Saved {radar_path}")

# ── Print Griffiths wave equation summary ─────────────────────────────────────
print("\n---- Griffiths / Modern Physics Reference -------------------------")
print("Maxwell:  curl B = mu0*J + mu0*eps0*dE/dt  (EM wave source)")
print("Wave eq:  d^2E/dz^2 = mu*eps*d^2E/dt^2  ->  v=c/n, n=sqrt(eps_r)")
print("Schrod:   ih_bar dpsi/dt = -(h_bar^2/2m)laplacian(psi) + V*psi")
print("Dispers:  H(nu)=exp(i*pi*D*nu^2)  <->  free-particle phase in QM")
print("Griffiths EM ch4: dielectrics  -> LiNbO3, InP refractive index")
print("Griffiths EM ch7: Faraday      -> EO modulator bandwidth")
print("Griffiths QM ch2: sq. well     -> Si/Ge band gap engineering")
print("Griffiths QM ch9: perturbation -> Er3+ stimulated emission")
print("-------------------------------------------------------------------")
print(f"\nCards: {card_path}")
print(f"Radar: {radar_path}")
