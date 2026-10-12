"""Reproduce the main plots and save them to results/ (no window is opened).

    python scripts/make_plots.py

Figures 1-6 correspond to the figures of the original main.m with its default
values; the rest cover the regime comparison, wavelength dependence, diameter
sensitivity, the sensor chain and the block diagram. Model output and
synthetic data only.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarlett import physics as ph, synthetic as sy, visualization as viz  # noqa: E402


def main():
    out = ROOT / "results"
    figures = [
        ("fig1_2_seals_map", viz.plot_seals_map),
        ("fig3_irradiance_vs_angle", viz.plot_angular_pattern),
        ("fig4_polar_pattern", viz.plot_polar_pattern),
        ("fig5_irradiance_vs_wavelength", viz.plot_wavelength_response),
        ("fig6_phase_vs_wavelength", viz.plot_phase),
        ("regimes_vs_diameter", viz.plot_regimes),
        ("cross_section_vs_wavelength", viz.plot_cross_section_vs_wavelength),
        ("diameter_sensitivity", viz.plot_diameter_sensitivity),
        ("block_diagram", viz.plot_block_diagram),
        ("sensor_chain", lambda: viz.plot_sensor_chain(sy.generate_dataset(dls_samples=2000))),
    ]
    for name, make in figures:
        path = viz.save(make(), name, out)
        plt.close("all")
        print("wrote", path.relative_to(ROOT))

    p = ph.SEALS_DEFAULTS
    lam0 = 0.5 * (p["lam_min"] + p["lam_max"])
    print(f"\ndefault particle: dia = {p['dia']*1e6:.2f} um, size parameter x = "
          f"{ph.size_parameter(p['dia'], lam0, p['nmed']):.4f}, "
          f"Mie cross section = {ph.mie_cross_section(p['npar'], p['nmed'], p['dia'], lam0):.6e} m^2")


if __name__ == "__main__":
    main()
