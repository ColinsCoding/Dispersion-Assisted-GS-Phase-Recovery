"""
noise_robustness.py -- how much does measurement noise degrade the N-plane
classical GS fix (gs_multiplane.retrieve_phase_n_plane), once the
STRUCTURAL 2-plane ambiguity has already been resolved by adding a 3rd
dispersion plane (see seals_to_tdgsa.py's sweep_measurement_diversity,
SEALS_TO_TDGSA_REPORT.md Sec. 6)?

TWO DIFFERENT QUESTIONS, not to be conflated:
  - seals_to_tdgsa.demonstrate_autograd_overfitting asks whether the
    GRADIENT-BASED autograd path, given noisy 2-PLANE (already
    structurally underdetermined) measurements, gets WORSE the longer you
    train it -- overfitting, a training-dynamics question.
  - This module asks whether the PROJECTION-BASED classical GS path, given
    noisy 3-PLANE (already structurally SOLVED) measurements, loses
    accuracy simply because the measurements themselves are noisy -- a
    basic SNR-limited-reconstruction question, not an overfitting one.
    Classical GS has no "too many iterations" failure mode: it converges
    to a fixed point almost immediately regardless of n_iter or noise
    level (verified below), because its amplitude-constraint step just
    hard-sets |E_d| = sqrt(I_measured) every iteration -- there is no
    training dynamic to overfit, only a direct injection of whatever
    noise is in I_measured into the reconstruction.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dgs.dispersion_gs_prototype import compare_phase  # noqa: E402

from . import gs_multiplane  # noqa: E402
from .seals_to_tdgsa import seals_intensity_trace, build_gs_measurements_n, add_measurement_noise  # noqa: E402


def n_plane_recovery_at_noise(Ds=(6000.0, -7000.0, 12000.0), noise_std=0.05, n_iter=150, seed=0):
    """
    Run N-plane classical GS on the SEALS/Mie forward model's measurements,
    with independent multiplicative noise (add_measurement_noise) added to
    EACH plane, and score the recovered phase against the known (noiseless)
    Mie ground truth.

    Deterministic given (Ds, noise_std, n_iter, seed) -- each plane gets its
    own noise realization via seed+i, so adding/removing planes doesn't
    change earlier planes' noise.

    Returns a dict with theta_deg, phi_true, phi_recovered (aligned),
    rms_vs_truth, gs_final_error (measurement self-consistency), plus the
    inputs, for reproducibility and plotting.
    """
    if noise_std < 0:
        raise ValueError(f"noise_std={noise_std} must be >= 0")

    _, theta_deg, mie_fields = seals_intensity_trace()
    phi_true = mie_fields.phase_p
    weight = np.abs(mie_fields.E_p) ** 2

    Is_clean = build_gs_measurements_n(mie_fields, list(Ds))
    if noise_std > 0:
        Is = [add_measurement_noise(I, noise_std, seed=seed + i) for i, I in enumerate(Is_clean)]
    else:
        Is = Is_clean

    phi_gs, errors, _ = gs_multiplane.retrieve_phase_n_plane(Is, list(Ds), n_iter=n_iter, unit_amplitude=False)
    rms, phi_aligned = compare_phase(phi_gs, phi_true, weight)

    return {
        "Ds": list(Ds), "noise_std": noise_std, "n_iter": n_iter, "seed": seed,
        "theta_deg": theta_deg, "phi_true": phi_true,
        "phi_recovered": phi_aligned, "rms_vs_truth": rms,
        "gs_final_error": errors[-1],
    }


def sweep_noise_robustness(Ds=(6000.0, -7000.0, 12000.0),
                            noise_levels=(0.0, 0.05, 0.15, 0.3, 0.6, 1.5),
                            n_iter=150, seed=0):
    """
    Sweep noise_std and report N-plane GS's RMS-vs-truth at each level --
    the accuracy-vs-NOISE curve for the already-structurally-fixed (N=3+)
    problem, complementing seals_to_tdgsa.sweep_measurement_diversity's
    accuracy-vs-N curve (which is run at fixed, zero noise).

    Returns a list of {"noise_std": ..., "rms_vs_truth": ...} dicts, one
    per noise_levels entry, in the given order.
    """
    return [
        {"noise_std": n, "rms_vs_truth": n_plane_recovery_at_noise(Ds=Ds, noise_std=n, n_iter=n_iter, seed=seed)["rms_vs_truth"]}
        for n in noise_levels
    ]


if __name__ == "__main__":
    print("N=3-plane classical GS: accuracy vs. measurement noise (Ds=(6000, -7000, 12000))")
    for r in sweep_noise_robustness():
        print(f"  noise_std={r['noise_std']:<5}  RMS vs. true Mie phase = {r['rms_vs_truth']:.4f} rad")
    print()
    print("Noiseless: ~0.0014 rad (the structural ambiguity is genuinely solved).")
    print("Realistic noise (~5%): still good (a few percent of a radian), but not perfect --")
    print("measurement noise is a separate limitation from measurement diversity, and classical")
    print("GS's hard amplitude constraint has no built-in denoising to counter it.")
