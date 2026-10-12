"""Visualization. Every function returns a Matplotlib figure; nothing is shown
or saved unless you call save(). All plotted values are model output or
synthetic data, and every axis carries its unit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import estimator as es
from . import physics as ph
from .asic import ASIC_BLOCKS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def save(fig, name, results_dir=RESULTS_DIR):
    """Save a figure as results/<name>.png (folder created if needed)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def _seals(p):
    lam = np.linspace(p["lam_min"], p["lam_max"], p["n_wavelengths"])
    y, theta = ph.seals_wavelength_to_angle(
        p["groove_spacing"], p["grating_distance"], p["grating_angle"], p["dcorr"],
        p["pupil_diameter"], p["numerical_aperture"], lam)
    return lam, y, theta + p["measurement_angle"], 0.5 * (p["lam_min"] + p["lam_max"])


def plot_seals_map(p=ph.SEALS_DEFAULTS):
    """main.m figures 1 and 2: displacement and scattering angle vs wavelength."""
    lam, y, theta, _ = _seals(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(lam * 1e9, y * 1e3)
    axes[0].set(xlabel="wavelength (nm)", ylabel="beam displacement (mm)",
                title="SEALS: displacement vs wavelength")
    axes[1].plot(lam * 1e9, np.degrees(theta))
    axes[1].set(xlabel="wavelength (nm)", ylabel="scattering angle (deg)",
                title="SEALS: scattering angle vs wavelength")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_angular_pattern(p=ph.SEALS_DEFAULTS):
    """main.m figure 3: irradiance vs angle, with the polarization components."""
    _, _, theta, lam0 = _seals(p)
    mie = ph.mie_scattering(p["npar"], p["nmed"], p["dia"], lam0, theta, p["r"])
    rdg = ph.rayleigh_debye_intensity(p["dia"], lam0, p["nmed"], p["npar"], theta, p["r"])
    deg = np.degrees(theta)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogy(deg, mie.i_par, label="Mie, parallel-polarized input")
    ax.semilogy(deg, mie.i_perp, label="Mie, perpendicular-polarized input")
    ax.semilogy(deg, mie.i_unpolarized, "k", lw=1.6, label="Mie, unpolarized input")
    ax.semilogy(deg, rdg, "--", label="Rayleigh-Debye (not valid at this size)")
    ax.set(xlabel="scattering angle (deg)", ylabel="scattered irradiance (W/m$^2$), log scale",
           title=f"dia = {p['dia']*1e6:.2f} µm, x = {mie.x:.1f}, r = {p['r']} m, "
                 "$I_0$ = 1 W/m$^2$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_polar_pattern(p=ph.SEALS_DEFAULTS):
    """main.m figure 4: polar plot over the full circle (not only the SEALS range)."""
    lam0 = 0.5 * (p["lam_min"] + p["lam_max"])
    theta = np.linspace(0, 2 * np.pi, 1441)
    i = ph.mie_scattering(p["npar"], p["nmed"], p["dia"], lam0, theta, p["r"]).i_unpolarized
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="polar")
    ax.plot(theta, np.log10(i / i.min()))
    ax.set_title("Mie pattern: log$_{10}$(I / I$_{min}$), 0° = forward", fontsize=10)
    fig.tight_layout()
    return fig


def plot_wavelength_response(p=ph.SEALS_DEFAULTS):
    """main.m figure 5: detected spectrum, with the corrected and the legacy lineshape."""
    lam, _, theta, lam0 = _seals(p)
    i = ph.mie_scattering(p["npar"], p["nmed"], p["dia"], lam0, theta, p["r"]).i_unpolarized
    fixed = ph.laser_lineshape(lam, lam0, p["source_bandwidth"])
    legacy = ph.laser_lineshape(lam, lam0, p["source_bandwidth"], legacy=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(lam * 1e9, fixed, label="corrected")
    axes[0].plot(lam * 1e9, legacy, "--", label="legacy main.m")
    axes[0].set(xlabel="wavelength (nm)", ylabel="relative source power (peak = 1)",
                title="Source lineshape")
    axes[1].semilogy(lam * 1e9, i * fixed, label="corrected")
    axes[1].semilogy(lam * 1e9, i * legacy, "--", label="legacy main.m")
    axes[1].set(xlabel="wavelength (nm)", ylabel="irradiance × lineshape (W/m$^2$), log scale",
                title="Detected spectrum (Mie, unpolarized)")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_phase(p=ph.SEALS_DEFAULTS):
    """main.m figure 6: wrapped phase of the two field components."""
    lam, _, theta, lam0 = _seals(p)
    mie = ph.mie_scattering(p["npar"], p["nmed"], p["dia"], lam0, theta, p["r"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(lam * 1e9, np.degrees(mie.phase_par), label="parallel (p)")
    ax.plot(lam * 1e9, np.degrees(mie.phase_perp), label="perpendicular (s)")
    ax.set(xlabel="wavelength (nm)", ylabel="phase (deg), wrapped to [-180, 180]",
           title="Phase of the scattered Mie field")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_regimes(npar=1.39, nmed=1.0, lam=1590e-9, n_points=160):
    """Scattering efficiency vs diameter for Rayleigh, Rayleigh-Debye and Mie."""
    dia = np.logspace(np.log10(20e-9), np.log10(20e-6), n_points)
    area = np.pi * dia ** 2 / 4
    q_mie = np.array([ph.mie_cross_section(npar, nmed, d, lam) for d in dia]) / area
    q_ray = np.array([ph.rayleigh_cross_section(d, lam, nmed, npar) for d in dia]) / area
    q_rdg = np.array([ph.rayleigh_debye_cross_section(d, lam, nmed, npar, 2001) for d in dia]) / area
    x = ph.size_parameter(dia, lam, nmed)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.loglog(dia * 1e6, q_mie, "k", lw=1.8, label="Mie (exact for a sphere)")
    ax.loglog(dia * 1e6, q_ray, "--", label="Rayleigh")
    ax.loglog(dia * 1e6, q_rdg, ":", lw=2, label="Rayleigh-Debye")
    ax.axvline(dia[np.argmin(np.abs(x - 0.3))] * 1e6, color="gray", lw=0.8)
    ax.text(dia[np.argmin(np.abs(x - 0.3))] * 1e6, 3e-9, " x = 0.3", fontsize=8, color="gray")
    ax.set(xlabel="particle diameter (µm)", ylabel="scattering efficiency $Q_{sca}$ (dimensionless)",
           ylim=(1e-9, 1e2),
           title=f"Regimes at λ = {lam*1e9:.0f} nm, n$_{{par}}$ = {npar}, n$_{{med}}$ = {nmed}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_cross_section_vs_wavelength(diameters=(50e-9, 500e-9, 5e-6), npar=1.39, nmed=1.0):
    """Wavelength dependence of the cross section: lambda^-4 only for small particles."""
    lam = np.linspace(400e-9, 1700e-9, 261)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for d in diameters:
        sigma = np.array([ph.mie_cross_section(npar, nmed, d, l) for l in lam])
        slope = np.polyfit(np.log(lam), np.log(sigma), 1)[0]
        ax.loglog(lam * 1e9, sigma, label=f"dia = {d*1e9:.0f} nm, mean slope {slope:+.2f}")
    ax.set(xlabel="vacuum wavelength (nm)", ylabel="scattering cross section (m$^2$)",
           title="Wavelength dependence (Mie); Rayleigh predicts slope −4")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_diameter_sensitivity(p=ph.SEALS_DEFAULTS, rel_uncertainty=0.02):
    """dia ± uncertainty: how the predicted pattern moves when the diameter does."""
    _, _, theta, lam0 = _seals(p)
    deg = np.degrees(theta)
    pattern = lambda d: ph.mie_scattering(p["npar"], p["nmed"], d, lam0, theta, p["r"]).i_unpolarized
    nominal = pattern(p["dia"])
    lo, hi = pattern(p["dia"] * (1 - rel_uncertainty)), pattern(p["dia"] * (1 + rel_uncertainty))
    h = p["dia"] * 1e-4
    sensitivity = (pattern(p["dia"] + h) - pattern(p["dia"] - h)) / (2 * h) * p["dia"] / nominal
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    axes[0].semilogy(deg, nominal, "k", label="nominal diameter")
    axes[0].fill_between(deg, np.minimum(lo, hi), np.maximum(lo, hi), alpha=0.35,
                         label=f"dia ± {100*rel_uncertainty:g} %")
    axes[0].set(xlabel="scattering angle (deg)", ylabel="scattered irradiance (W/m$^2$), log scale",
                title="Predicted pattern for a range of diameters")
    axes[0].legend(fontsize=8)
    axes[1].plot(deg, sensitivity)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set(xlabel="scattering angle (deg)",
                ylabel="relative sensitivity (dia / I) · dI/d(dia) (dimensionless)",
                title="Where the signal is most sensitive to diameter")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_sensor_chain(dataset):
    """Synthetic scan through the sensor chain: optical power, voltage, ADC codes."""
    scan, dark = dataset["scan"], dataset["dark"]
    deg = np.degrees(dataset["scattering_angles_rad"])
    ok = dataset["usable_channels"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].semilogy(deg, scan["power_w"], "o-", ms=3)
    axes[0].set(xlabel="scattering angle (deg)", ylabel="optical power on detector (W), log scale",
                title="After the equalization filter (synthetic)")
    axes[1].errorbar(deg[ok], (scan["volts"] - dark["volts"])[ok] * 1e3,
                     yerr=scan["sigma_volts"][ok] * 1e3, fmt="o", ms=3)
    axes[1].set(xlabel="scattering angle (deg)", ylabel="TIA output, dark removed (mV)",
                title="Voltage with noise (synthetic)")
    axes[2].step(deg, scan["codes"], where="mid", label=f"signal, PGA gain ×{scan['gain']}")
    axes[2].step(deg, dark["codes"], where="mid", label="dark")
    axes[2].set(xlabel="scattering angle (deg)", ylabel="ADC code (counts)",
                title="Averaged ADC codes (synthetic)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvspan(-np.degrees(dataset["bench"].beam_stop_angle),
                   np.degrees(dataset["bench"].beam_stop_angle), color="gray", alpha=0.2)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_estimate(lut, estimate, truth_dia=None):
    """chi-square against candidate diameter, with the confidence interval."""
    best_n = lut.npar == estimate.npar
    dia = lut.dia[best_n]
    chi2 = estimate.chi2_grid[best_n]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.semilogy(dia * 1e6, chi2 / estimate.dof, lw=1)
    ax.axvspan(estimate.dia_ci[0] * 1e6, estimate.dia_ci[1] * 1e6, color="C1", alpha=0.5,
               label=f"{100*estimate.confidence_level:.0f} % interval")
    ax.axvline(estimate.dia * 1e6, color="C1", lw=1,
               label=f"estimate {estimate.dia*1e6:.4f} µm")
    if truth_dia is not None:
        ax.axvline(truth_dia * 1e6, color="k", ls=":", label=f"synthetic truth {truth_dia*1e6:.4f} µm")
    ax.set(xlabel="candidate diameter (µm)", ylabel="χ² per degree of freedom, log scale",
           title="Lookup-table fit of the angular pattern (synthetic)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_motion(counts, dt, max_lag, decay_rate_true=None):
    """Intensity autocorrelation g2(tau) and the fitted exponential."""
    g2 = es.intensity_autocorrelation(counts, max_lag)
    gamma, beta = es.fit_decay_rate(g2, dt)
    tau = dt * np.arange(max_lag + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(tau[1:], g2[1:] - 1, ".", ms=4, label="measured g2 − 1 (synthetic)")
    ax.plot(tau[1:], beta * np.exp(-2 * gamma * tau[1:]), label=f"fit, Γ = {gamma:.2f} s$^{{-1}}$")
    if decay_rate_true is not None:
        ax.plot(tau[1:], np.exp(-2 * decay_rate_true * tau[1:]), ":",
                label=f"truth, Γ = {decay_rate_true:.2f} s$^{{-1}}$")
    ax.set(xlabel="lag τ (s)", ylabel="g2(τ) − 1 (dimensionless)", xscale="log",
           title="Brownian motion: intensity autocorrelation")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_block_diagram():
    """Signal chain and proposed ASIC partition, drawn as boxes."""
    front = ["particle\nsample", "illumination\n(SEALS rainbow)", "Mie-scattered\noptical signal"]
    blocks = front + [name.replace(" ", "\n", 1) for name, *_ in ASIC_BLOCKS]
    kinds = ["optical"] * 3 + [kind for _, kind, *_ in ASIC_BLOCKS]
    colors = {"optical": "#f2d9a0", "analog": "#a9cfe8", "mixed": "#c6b6e6", "digital": "#b5e0b0"}
    per_row = 7
    fig, ax = plt.subplots(figsize=(14, 4.6))
    centres = []
    for i, (label, kind) in enumerate(zip(blocks, kinds)):
        row, col = divmod(i, per_row)
        cx = col if row == 0 else per_row - 1 - col        # second row runs right to left
        cy = -1.6 * row
        centres.append((cx, cy))
        ax.add_patch(plt.Rectangle((cx - 0.42, cy - 0.45), 0.84, 0.9, fc=colors[kind], ec="k"))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8)
    for (x0, y0), (x1, y1) in zip(centres[:-1], centres[1:]):
        if y0 == y1:
            sign = np.sign(x1 - x0)
            ax.annotate("", xy=(x1 - sign * 0.42, y1), xytext=(x0 + sign * 0.42, y0),
                        arrowprops=dict(arrowstyle="->"))
        else:
            ax.annotate("", xy=(x1, y1 + 0.45), xytext=(x0, y0 - 0.45),
                        arrowprops=dict(arrowstyle="->"))
    for kind, color in colors.items():
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, fc=color, ec="k", label=kind))
    ax.legend(loc="lower left", fontsize=8, ncol=4, frameon=False)
    ax.set(xlim=(-0.7, per_row - 0.3), ylim=(-2.9, 0.75))
    ax.set_title("Signal chain and PROPOSED ASIC partition (architecture proposal, behavioural model only)",
                 fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig
