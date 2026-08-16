"""Polymer refractive-index lookup -- the "periodic table" for microplastics.

The plastics industry already has its own indexing scheme, the resin
identification code (RIC, the number inside the recycling triangle stamped on
packaging, 1-7) -- the closest thing polymers have to atomic number. This
module is a lookup table keyed on that scheme plus each polymer's common
abbreviation, giving the two numbers month 2 of the microplastic-sensing
project actually needs: the real refractive index n (at the sodium D-line,
589.3 nm -- the standard single-wavelength reference point used across
materials science) and mass density (relevant to how a particle settles or
floats in the surrounding medium, independent of its optics).

Values here are representative literature ranges (Handbook of Plastics,
Elastomers, and Composites; refractiveindex.info-style tabulations), not a
single calibrated measurement of one batch of material. Per the project's own
uncertainty bookkeeping: this is MODEL uncertainty (imperfect knowledge of
which exact n applies to your sample), separate from MEASUREMENT uncertainty
(detector noise) and STATISTICAL uncertainty (finite samples) -- verify
against per-batch measured data before using these for anything beyond a
first-pass forward-model estimate.

NumPy + matplotlib only. Education / forward-model input, not a measurement.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── polymer lookup table ──────────────────────────────────────────────────
# n: refractive index at 589.3 nm (sodium D-line). density_g_cm3: bulk density.
POLYMERS = {
    "PET":  {"ric": 1, "name": "Polyethylene terephthalate",        "n": 1.575, "density_g_cm3": 1.380},
    "HDPE": {"ric": 2, "name": "High-density polyethylene",          "n": 1.540, "density_g_cm3": 0.955},
    "PVC":  {"ric": 3, "name": "Polyvinyl chloride",                 "n": 1.540, "density_g_cm3": 1.350},
    "LDPE": {"ric": 4, "name": "Low-density polyethylene",           "n": 1.515, "density_g_cm3": 0.925},
    "PP":   {"ric": 5, "name": "Polypropylene",                      "n": 1.490, "density_g_cm3": 0.905},
    "PS":   {"ric": 6, "name": "Polystyrene",                        "n": 1.590, "density_g_cm3": 1.050},
    "PC":   {"ric": 7, "name": "Polycarbonate",                      "n": 1.585, "density_g_cm3": 1.200},
    "PMMA": {"ric": 7, "name": "Poly(methyl methacrylate) / acrylic","n": 1.491, "density_g_cm3": 1.180},
    "PA6":  {"ric": 7, "name": "Polyamide 6 / nylon",                "n": 1.530, "density_g_cm3": 1.140},
    "ABS":  {"ric": 7, "name": "Acrylonitrile butadiene styrene",    "n": 1.540, "density_g_cm3": 1.060},
    "PLA":  {"ric": 7, "name": "Polylactic acid (bioplastic)",       "n": 1.460, "density_g_cm3": 1.245},
    "PTFE": {"ric": 7, "name": "Polytetrafluoroethylene / Teflon",   "n": 1.350, "density_g_cm3": 2.200},
}

# reference media the particles are typically suspended in
MEDIA = {
    "water":        {"name": "Pure water",             "n": 1.333, "density_g_cm3": 1.000},
    "blood_plasma": {"name": "Blood plasma (approx.)",  "n": 1.345, "density_g_cm3": 1.025},
    "seawater":     {"name": "Seawater (approx.)",      "n": 1.340, "density_g_cm3": 1.025},
}


def _lookup(table, key, label):
    key_norm = key.upper() if label == "polymer" else key.lower()
    if key_norm not in table:
        raise KeyError(f"unknown {label} '{key}'; available: {sorted(table)}")
    return table[key_norm]


def refractive_index(polymer):
    """n at 589.3 nm for a polymer abbreviation (e.g. 'PET', 'HDPE', 'PMMA')."""
    return _lookup(POLYMERS, polymer, "polymer")["n"]


def density(polymer):
    """Bulk density in g/cm^3 for a polymer abbreviation."""
    return _lookup(POLYMERS, polymer, "polymer")["density_g_cm3"]


def medium_refractive_index(medium):
    """n for a reference suspension medium ('water', 'blood_plasma', 'seawater')."""
    return _lookup(MEDIA, medium, "medium")["n"]


def list_polymers():
    """All known polymer abbreviations, sorted by RIC code then name."""
    return sorted(POLYMERS, key=lambda k: (POLYMERS[k]["ric"], k))


def by_ric(ric_code):
    """All polymer abbreviations sharing a given resin identification code (1-7)."""
    if ric_code not in range(1, 8):
        raise ValueError("ric_code must be 1-7")
    return [k for k in list_polymers() if POLYMERS[k]["ric"] == ric_code]


def optical_contrast(polymer, medium="water"):
    """Delta n = n_polymer - n_medium: the index mismatch that actually drives
    scattering strength (Fresnel/Mie contrast) -- a particle optically
    invisible in one medium (Delta n ~ 0) can be strongly scattering in
    another, independent of its own absolute n."""
    return refractive_index(polymer) - medium_refractive_index(medium)


def settling_sign(polymer, medium="water"):
    """+1 if the polymer sinks in the given medium (density > medium density),
    -1 if it floats, 0 if neutrally buoyant to within 1%. Optics tells you
    scattering strength; this tells you where in the water column to expect
    the particle -- the two are independent properties, easy to conflate."""
    ratio = density(polymer) / MEDIA[medium.lower()]["density_g_cm3"]
    if abs(ratio - 1.0) < 0.01:
        return 0
    return 1 if ratio > 1.0 else -1


def polymer_complex_index(polymer, kappa=0.0):
    """(n, kappa) pair for direct use with dgs.microplastic.physics.complex_index.
    kappa defaults to 0 -- bulk commodity plastics are close to non-absorbing
    in the visible band; a nonzero kappa (from dye, additive, or a different
    spectral band) must be supplied explicitly, not assumed."""
    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    return refractive_index(polymer), kappa


# ── visualization: a periodic-table-style grid ───────────────────────────
def plot_periodic_table(medium="water", ax=None):
    """Lay the polymers out as a grid indexed by RIC code (1-7), each tile
    showing abbreviation, n, and optical contrast against `medium` -- the
    same visual idiom as the periodic table, applied to the plastics
    industry's own 1-7 indexing scheme instead of atomic number."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    columns = {ric: by_ric(ric) for ric in range(1, 8)}
    max_rows = max(len(v) for v in columns.values())

    for col, ric in enumerate(range(1, 8)):
        for row, poly in enumerate(columns[ric]):
            n = refractive_index(poly)
            dn = optical_contrast(poly, medium)
            y = max_rows - row
            color = plt.cm.viridis((n - 1.33) / (1.60 - 1.33))
            ax.add_patch(plt.Rectangle((col, y), 0.95, 0.95, facecolor=color,
                                        edgecolor="black", lw=0.8))
            ax.text(col + 0.05, y + 0.80, f"RIC {ric}", fontsize=6, color="white")
            ax.text(col + 0.475, y + 0.5, poly, fontsize=11, ha="center",
                     va="center", color="white", weight="bold")
            ax.text(col + 0.475, y + 0.15, f"n={n:.3f}  Δn={dn:+.3f}",
                     fontsize=6.5, ha="center", va="center", color="white")

    ax.set_xlim(0, 7)
    ax.set_ylim(0, max_rows + 1)
    ax.set_xticks([c + 0.475 for c in range(7)])
    ax.set_xticklabels([f"RIC {r}" for r in range(1, 8)])
    ax.set_yticks([])
    ax.set_title(f"Microplastic polymer 'periodic table' (Δn relative to {medium})")
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    print("Polymers by resin identification code (the plastics 'periodic table'):")
    for ric in range(1, 8):
        polys = by_ric(ric)
        print(f"  RIC {ric}: {', '.join(polys)}")

    print(f"\nrefractive_index('PET')  = {refractive_index('PET')}")
    print(f"density('PET')           = {density('PET')} g/cm^3")
    print(f"optical_contrast('PET', 'water') = {optical_contrast('PET', 'water'):+.3f}")
    print(f"settling_sign('PP', 'water')     = {settling_sign('PP', 'water')}  (PP floats)")
    print(f"settling_sign('PET', 'water')    = {settling_sign('PET', 'water')}  (PET sinks)")

    n, kappa = polymer_complex_index("PS")
    print(f"\npolymer_complex_index('PS') = (n={n}, kappa={kappa})")
