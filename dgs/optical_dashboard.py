"""optical_dashboard.py -- a single-figure operator dashboard for a GS
phase-retrieval run, combining the measurement, the convergence, the
recovered phase, and (when laser parameters are given) an eye-safety panel.

Fills the "optical_dashboard" entry `dgs/ousd_alignment.py`'s
Human_Machine_Interfaces CTA table references but which did not exist in
`dgs/` as of this session (found and flagged in
`notebooks/phase_retrieval_connections.ipynb`'s Part 4). The motivating case
this repo keeps returning to (STEAM's ~440 ps shutter speed,
`dgs/sbir_portfolio.py`'s P2) is exactly why an operator needs a dashboard at
all: nothing at that timescale is human-watchable directly, only a
post-hoc summary of what the instrument measured and what GS recovered.

THE LASER-SAFETY PANEL IS ILLUSTRATIVE ONLY (see
`dgs/laser_safety_mpe.py`'s own `MODULE_SAFETY_DISCLAIMER`, surfaced verbatim
in this panel's output, not paraphrased away) -- a real "laser on goggles"
safety decision needs the actual ANSI Z136.1 standard and a certified Laser
Safety Officer, not this dashboard. A REAL LIMITATION, found and kept honest
rather than hidden: `laser_safety_mpe.py`'s illustrative model only covers
400-1050 nm (the retinal thermal-hazard region); this repo's own default
telecom wavelength (1550 nm, `dgs/gs_core.py` and most of this repo's STEAM
work) is OUTSIDE that range. The panel reports "outside covered range"
rather than silently producing a wrong number or crashing.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

from dgs import gs_core
from dgs import laser_safety_mpe as lsm


def _laser_safety_panel(ax, laser_power_W: Optional[float], wavelength_nm: float,
                         beam_diameter_um: Optional[float], exposure_s: float) -> Optional[Dict]:
    ax.axis("off")
    if laser_power_W is None or beam_diameter_um is None:
        ax.text(0.02, 0.5, "no laser-safety parameters provided\n"
                            "(pass laser_power_W and beam_diameter_um)",
                fontsize=9, va="center", family="monospace")
        return None
    if not (400.0 <= wavelength_nm <= 1050.0):
        ax.text(0.02, 0.5,
                f"wavelength_nm={wavelength_nm:.0f} is OUTSIDE the 400-1050nm range\n"
                f"dgs/laser_safety_mpe.py's illustrative model covers -- no\n"
                f"safety number computed. A real check needs the actual ANSI\n"
                f"Z136.1 standard's other wavelength bands.",
                fontsize=8.5, va="center", family="monospace", color="darkred")
        return None
    result = lsm.check_retinal_scan_exposure_illustrative(
        power_W=laser_power_W, beam_diameter_um=beam_diameter_um,
        wavelength_nm=wavelength_nm, exposure_s=exposure_s)
    verdict = "EXCEEDS illustrative MPE" if result["exceeds_mpe"] else "within illustrative MPE"
    color = "darkred" if result["exceeds_mpe"] else "darkgreen"
    text = (
        f"irradiance:      {result['irradiance_W_cm2']:.3e} W/cm^2\n"
        f"illustrative MPE: {result['mpe_J_cm2']:.3e} J/cm^2\n"
        f"margin factor:   {result['margin_factor']:.3e}\n"
        f"verdict:         {verdict}\n\n"
        f"{result['disclaimer']}"
    )
    ax.text(0.02, 0.5, text, fontsize=8.5, va="center", family="monospace", color=color)
    return result


def phase_retrieval_dashboard(I1: np.ndarray, I2: np.ndarray, D1: float, D2: float,
                               n_iter: int = 50, unit_amplitude: bool = True,
                               phi_true: Optional[np.ndarray] = None,
                               laser_power_W: Optional[float] = None,
                               wavelength_nm: float = 850.0,
                               beam_diameter_um: Optional[float] = None,
                               exposure_s: float = 1.0) -> Dict:
    """Run GS phase retrieval (dgs.gs_core.retrieve_phase_with_history) and
    build a 4-panel operator dashboard: (1) the two measured intensities,
    (2) convergence error vs iteration, (3) recovered phase (with ground
    truth overlay if given), (4) the laser-safety panel (see module
    docstring for what it does and does not cover).

    Returns dict: figure, phi, errors, E_history, laser_safety (None if no
    laser parameters given or wavelength outside the covered range).
    """
    phi, errors, E_history = gs_core.retrieve_phase_with_history(
        I1, I2, D1, D2, n_iter=n_iter, unit_amplitude=unit_amplitude)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    axs[0, 0].plot(I1, label="I1")
    axs[0, 0].plot(I2, label="I2")
    axs[0, 0].set_title("measured intensities"); axs[0, 0].legend(fontsize=8)
    axs[0, 0].set_xlabel("sample")

    axs[0, 1].semilogy(errors, "o-", ms=3)
    axs[0, 1].set_title("GS convergence"); axs[0, 1].set_xlabel("iteration")
    axs[0, 1].set_ylabel("self-consistency error")

    axs[1, 0].plot(phi, "b-", lw=1.2, label="recovered")
    if phi_true is not None:
        phi_true = np.asarray(phi_true, dtype=float)
        axs[1, 0].plot(phi_true, "k--", lw=1.0, label="true")
        rms_deg = float(np.degrees(np.sqrt(np.mean((phi - phi_true) ** 2))))
        axs[1, 0].set_title(f"recovered phase (RMS={rms_deg:.1f} deg)")
    else:
        axs[1, 0].set_title("recovered phase")
    axs[1, 0].legend(fontsize=8); axs[1, 0].set_xlabel("sample")

    laser_safety = _laser_safety_panel(axs[1, 1], laser_power_W, wavelength_nm,
                                        beam_diameter_um, exposure_s)
    axs[1, 1].set_title("laser-safety panel (illustrative only)")

    plt.tight_layout()
    return {"figure": fig, "phi": phi, "errors": errors, "E_history": E_history,
            "laser_safety": laser_safety}


if __name__ == "__main__":
    from dgs.gs_core import make_measurements

    m = make_measurements('QPSK', n_symbols=32, sps=8, D1=-5000, D2=-5750, snr_db=30, rng_seed=1)
    result = phase_retrieval_dashboard(
        m['I1'], m['I2'], m['D1'], m['D2'], n_iter=50, phi_true=m['phi_true'],
        laser_power_W=0.001, wavelength_nm=850.0, beam_diameter_um=50.0, exposure_s=1.0)
    print(f"final GS error: {result['errors'][-1]:.3e}")
    if result['laser_safety'] is not None:
        print(f"laser safety verdict: exceeds_mpe={result['laser_safety']['exceeds_mpe']}")
    result['figure'].savefig("optical_dashboard_demo.png", dpi=100)
    print("saved optical_dashboard_demo.png")

    print("\n--- same run, but with this repo's own default 1550nm (outside covered range) ---")
    result2 = phase_retrieval_dashboard(
        m['I1'], m['I2'], m['D1'], m['D2'], n_iter=50, phi_true=m['phi_true'],
        laser_power_W=0.001, wavelength_nm=1550.0, beam_diameter_um=50.0, exposure_s=1.0)
    print(f"laser_safety result at 1550nm: {result2['laser_safety']}  (None expected -- outside 400-1050nm)")
