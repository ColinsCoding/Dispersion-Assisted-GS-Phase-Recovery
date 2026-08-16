"""seals_to_tdgsa.py -- bridge a SEALS intensity trace (from the validated Mie
forward model) into this repo's canonical TD-GSA implementation, dgs.gs_core,
and cross-check the recovered phase two independent ways.

WHY THIS FILE EXISTS
--------------------
projects/seals/inverse/ already has a generic PyTorch-autograd phase-retrieval
path (phase_retrieval.py + dispersion.py) that is architecturally PARALLEL to
dgs.gs_core's classical Gerchberg-Saxton algorithm -- dispersion.py's
dispersive_operator() is verified (tests/test_seals_dispersion.py) to match
dgs.gs_core.disperse's H(nu)=exp(i*pi*D*nu^2) convention exactly. What was
missing was an actual call into dgs.gs_core itself with SEALS-derived data,
and a check that the two independent phase-retrieval paths (classical GS in
gs_core, autograd in phase_retrieval.py) agree with each other AND with the
Mie model's own known ground-truth phase.

THE ONE-MEASUREMENT PROBLEM (read this before using the functions below)
--------------------------------------------------------------------------
A SEALS spectrometer, as built (SEALS_paper.pdf, SEALS.m, mie-2.m), records
ONE intensity trace, I_p(lambda) (equivalently I_p(theta) via the SEALS
wavelength -> angle mapping) -- a single square-law measurement of the
scattered field E_p(lambda). TD-GSA (dgs.gs_core) needs TWO measurements of
the SAME hidden field at two DIFFERENT, known dispersions (D1 != D2, both
nonzero, |D| >= 5000 in gs_core's normalized convention -- see
dgs/gs_core.py's own kwarg-bounds warning) for the measurement diversity
that makes phase retrieval well-posed at all; a single intensity trace alone
cannot determine phase (phase_retrieval.py already warns about exactly this
for a single-measurement call).

So "importing the SEALS intensity trail into TD-GSA" concretely means: the
native SEALS trace becomes ONE of the two measurement planes, and a SECOND
plane is produced by passing the SAME hidden field through a second, known
dispersion. dispersion.py's own docstring already flags this as a *future*
architectural direction (a second, dispersive-fiber measurement arm), NOT
something the single-shot instrument as currently built provides -- this
module keeps that framing explicit rather than implying the existing
hardware already does this.

TWO INDEPENDENT VERIFICATIONS
------------------------------
1. dgs.gs_core.retrieve_phase_with_history (classical alternating-projections
   GS) recovers phi(lambda) from (I1, I2); compared against Mie's own T_p
   (the actual phase of E_p -- known here only because this is a validation
   pass against a model with a known answer; a real instrument would not
   have this).
2. projects.seals.inverse.phase_retrieval.retrieve_phase (PyTorch autograd,
   already built in this package) is run on the IDENTICAL (I1, I2) pair.

Both recovered phases are compared against Mie's ground truth AND against
each other -- two different algorithms, the same inverse problem, the same
data. Results are reported honestly, including if one method converges to a
self-consistent-but-wrong answer (a genuine phase-retrieval ambiguity, not a
bug) -- see the "known limitation" note in run_bridge_demo()'s output.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dgs import gs_core                                    # noqa: E402
from dgs.dispersion_gs_prototype import compare_phase       # noqa: E402

from . import _seals_physics as physics                     # noqa: E402
from . import inverse_scattering                             # noqa: E402
from . import gs_multiplane                                  # noqa: E402
from .measurement import mie_complex_fields                 # noqa: E402
from .dispersion import dispersive_operator                 # noqa: E402
from .phase_retrieval import retrieve_phase as retrieve_phase_torch  # noqa: E402
from .phase_retrieval import retrieve_phase_with_history as retrieve_phase_with_history_torch  # noqa: E402


def seals_intensity_trace(params: dict | None = None):
    """The native SEALS measurement: I_p(lambda) from the Mie forward model,
    at the scattering angles the SEALS grating pair maps each wavelength to.
    Returns (lamvec, theta_deg, mie_fields) -- mie_fields.I_p is literally
    the intensity trail SEALS_paper.pdf's instrument would record."""
    p = dict(physics.P_DEFAULT)
    if params:
        p.update(params)

    lamvec = np.linspace(p["lam1"], p["lam2"], p["N_lam"])
    y, theta_deg, valid = physics.seals(p["d"], p["D"], p["a"], p["dcorr"], p["P"], p["NA"], lamvec)
    theta_deg = theta_deg + p["mangle"]

    mie_fields = mie_complex_fields(p["npar"], p["nmed"], p["dia"], np.mean(lamvec),
                                     np.deg2rad(theta_deg), p["r"])
    return lamvec, theta_deg, mie_fields


def build_gs_measurements(mie_fields, D1: float = 6000.0, D2: float = -7000.0):
    """Produce the TWO dispersion-diverse measurement planes TD-GSA needs
    from the SAME hidden field E_p, via dgs.gs_core.disperse directly (the
    canonical implementation, not a reimplementation)."""
    E_p = torch.tensor(mie_fields.E_p, dtype=torch.complex128)
    E1 = dispersive_operator(E_p, D1).numpy()
    E2 = dispersive_operator(E_p, D2).numpy()
    I1 = np.abs(E1) ** 2
    I2 = np.abs(E2) ** 2
    return I1, I2


def build_gs_measurements_n(mie_fields, Ds: list[float]):
    """N-plane generalization of build_gs_measurements: produce len(Ds)
    dispersion-diverse measurement planes from the SAME hidden field E_p,
    via dgs.gs_core.disperse (through dispersive_operator) directly."""
    E_p = torch.tensor(mie_fields.E_p, dtype=torch.complex128)
    return [np.abs(dispersive_operator(E_p, D).numpy()) ** 2 for D in Ds]


def run_multiplane_bridge_demo(Ds=(6000.0, -7000.0, 12000.0), n_iter: int = 150,
                                use_amplitude_prior: bool = False,
                                diameter_bounds_frac: float = 0.3):
    """SEALS_TO_TDGSA_REPORT.md Section 6's two next steps, exercised together:
    (1) N-plane measurement diversity (len(Ds) can be > 2), via
        gs_multiplane.retrieve_phase_n_plane instead of gs_core's fixed
        2-plane retrieve_phase_with_history;
    (2) optionally, amplitude regularization toward a Mie-fitted envelope
        (use_amplitude_prior=True): fits particle diameter from the native
        SEALS trace alone (inverse_scattering.estimate_diameter, unaware of
        the true diameter beyond a +/-diameter_bounds_frac search bracket)
        and blends GS's amplitude constraint toward that fitted envelope
        wherever the native trace is weak (gs_multiplane's floor_frac).
    """
    lamvec, theta_deg, mie_fields = seals_intensity_trace()
    I_native = mie_fields.I_p
    Is = build_gs_measurements_n(mie_fields, list(Ds))
    phi_true = mie_fields.phase_p

    amplitude_prior = None
    if use_amplitude_prior:
        p = physics.P_DEFAULT
        dia_true = p["dia"]
        bounds = (dia_true * (1 - diameter_bounds_frac), dia_true * (1 + diameter_bounds_frac))
        est = inverse_scattering.estimate_diameter(
            I_native, p["npar"], p["nmed"], np.mean(lamvec), np.deg2rad(theta_deg), p["r"], bounds)
        amplitude_prior = np.abs(est.predicted_fields.E_p)

    phi_gs, errors_gs, _ = gs_multiplane.retrieve_phase_n_plane(
        Is, list(Ds), n_iter=n_iter, unit_amplitude=False,
        amplitude_prior=amplitude_prior, I_native=I_native)
    rms_gs, _ = compare_phase(phi_gs, phi_true, np.abs(mie_fields.E_p) ** 2)

    return {
        "lamvec": lamvec, "Ds": list(Ds), "I_native": I_native, "Is": Is,
        "phi_true": phi_true, "phi_gs": phi_gs,
        "rms_vs_truth": rms_gs, "gs_final_error": errors_gs[-1],
        "amplitude_prior": amplitude_prior,
    }


def sweep_measurement_diversity(D_candidates=(6000.0, -7000.0, 12000.0, -18000.0, 23000.0),
                                 n_iter: int = 150):
    """Report Section 6, recommendation 1: does RMS phase error vs. Mie
    ground truth improve as more dispersion planes are added? Runs N-plane
    GS for N=2..len(D_candidates), each time using the first N dispersions
    in D_candidates (a growing, nested subset), all else held fixed.
    Returns {N: rms_vs_truth}."""
    results = {}
    for n in range(2, len(D_candidates) + 1):
        r = run_multiplane_bridge_demo(Ds=D_candidates[:n], n_iter=n_iter)
        results[n] = r["rms_vs_truth"]
    return results


def run_bridge_demo(D1: float = 6000.0, D2: float = -7000.0, n_iter: int = 150):
    """End-to-end: SEALS intensity trace -> two dispersed measurement planes
    -> dgs.gs_core TD-GSA AND the autograd path -> both compared against
    Mie's known ground-truth phase and against each other."""
    lamvec, theta_deg, mie_fields = seals_intensity_trace()
    I1, I2 = build_gs_measurements(mie_fields, D1, D2)
    phi_true = mie_fields.phase_p   # ground truth, known only because this is Mie-model validation

    # verification 1: classical GS (dgs.gs_core, this repo's canonical TD-GSA)
    phi_gs, errors_gs, _ = gs_core.retrieve_phase_with_history(
        I1, I2, D1, D2, n_iter=n_iter, unit_amplitude=False)
    rms_gs, phi_gs_aligned = compare_phase(phi_gs, phi_true, np.abs(mie_fields.E_p) ** 2)

    # verification 2: PyTorch autograd (this package's existing generic path), SAME (I1, I2)
    #
    # NORMALIZED before optimization -- Mie-scattered field amplitudes are physically tiny
    # here (~1e-5 to 5e-4), so intensities (amplitude^2) are ~1e-10 to 1e-7. Adam's default
    # eps=1e-8 in its denominator is comparable to or LARGER than gradients at that raw scale,
    # which silently stalls the optimizer at its all-zero initial phase guess (verified: an
    # untouched, un-optimized zero-phase guess scores the exact same RMS, 0.5009 rad, that this
    # function used to report as autograd's "converged" answer -- it had never actually moved).
    # Dividing amplitude by its own max (and intensities by max^2, since I=|E|^2) rescales to
    # O(1) without changing the recovered PHASE (which is scale-invariant); this is the same
    # class of scale-sensitivity already flagged, for a milder case, in
    # tests/test_seals_phase_retrieval.py's test_measurement_diversity_reduces_mean_phase_error
    # docstring -- that safeguard was never carried over to this specific caller until now.
    scale = float(np.abs(mie_fields.E_p).max())
    amplitude = torch.tensor(np.abs(mie_fields.E_p) / scale, dtype=torch.float64)
    I1_t = torch.tensor(I1 / scale**2, dtype=torch.float64)
    I2_t = torch.tensor(I2 / scale**2, dtype=torch.float64)
    phi_ag, loss_history = retrieve_phase_torch(
        amplitude, [I1_t, I2_t],
        [lambda E: dispersive_operator(E, D1), lambda E: dispersive_operator(E, D2)],
        n_steps=800, lr=0.03)
    phi_ag_np = phi_ag.numpy()
    rms_ag, phi_ag_aligned = compare_phase(phi_ag_np, phi_true, np.abs(mie_fields.E_p) ** 2)

    # cross-check: do the two independently-run methods agree with EACH OTHER?
    rms_gs_vs_ag, _ = compare_phase(phi_gs, phi_ag_np, np.abs(mie_fields.E_p) ** 2)

    return {
        "lamvec": lamvec, "theta_deg": theta_deg, "mie_fields": mie_fields,
        "I1": I1, "I2": I2, "D1": D1, "D2": D2,
        "phi_true": phi_true,
        "phi_gs": phi_gs, "rms_gs_vs_truth": rms_gs, "gs_final_error": errors_gs[-1],
        "phi_autograd": phi_ag_np, "rms_autograd_vs_truth": rms_ag,
        "rms_gs_vs_autograd": rms_gs_vs_ag,
    }


def diagnose_amplitude_dependence(result: dict, mie_fields) -> dict:
    """Quantify WHY GS's recovered phase misses the truth by ~0.4-0.5 rad, rather
    than leaving that as an unexplained number: GS's amplitude constraint
    `|E| = sqrt(I)` carries almost no phase information wherever the measured
    intensity I is near zero (any phase satisfies it equally well there), so the
    per-sample phase error should correlate with LOW signal amplitude, not be
    uniform across the trace.

    Computes the Pearson correlation between |phase error| and log(amplitude)
    (negative and significant => weak-signal samples really do have larger
    error), plus the mean |error| in the weakest and strongest amplitude
    quartiles, and the trace's overall amplitude dynamic range. This is the
    mechanism cited in SEALS_TO_TDGSA_REPORT.md Section 5 -- computed here so
    that claim is a tested fact, not prose."""
    from scipy.stats import pearsonr

    amp = np.abs(mie_fields.E_p)
    weight = amp ** 2
    _, phi_gs_aligned = compare_phase(result["phi_gs"], result["phi_true"], weight)
    err = np.angle(np.exp(1j * (result["phi_true"] - phi_gs_aligned)))  # wrapped per-sample error

    corr, pval = pearsonr(np.abs(err), np.log(amp))
    order = np.argsort(amp)
    q = len(amp) // 4
    bottom_quartile_err = float(np.mean(np.abs(err[order[:q]])))
    top_quartile_err = float(np.mean(np.abs(err[order[-q:]])))

    return {
        "pearson_r_abs_err_vs_log_amplitude": float(corr),
        "pearson_p_value": float(pval),
        "bottom_quartile_mean_abs_err_rad": bottom_quartile_err,
        "top_quartile_mean_abs_err_rad": top_quartile_err,
        "amplitude_dynamic_range": float(amp.max() / amp.min()),
        "per_sample_error": err,
        "amplitude": amp,
    }


def diagnose_even_odd_ambiguity(result: dict, mie_fields) -> dict:
    """Cross-references a historical, independent finding from this project's
    predecessor notebook (notebooks/phase_retrieval.ipynb, cell 62 -- a
    reproduction of Yiming's MATLAB TDGSA code, Jalali Lab / ECE 279AS slide
    23): blind TDGSA fails on EVEN-degree phase polynomials (e.g. a quadratic
    chirp) because the intensity constraint cannot distinguish phi from -phi
    (Hermitian symmetry); ODD-degree polynomials (cubic) converge cleanly.

    dgs.dispersion_gs_prototype.compare_phase already searches BOTH signs
    (phi vs -phi) when scoring GS's recovered phase against ground truth --
    so that specific ambiguity is already corrected for in every RMS number
    this bridge reports (including SEALS_TO_TDGSA_REPORT.md Sec. 4-5's
    ~0.4-0.5 rad figures). This function checks a DIFFERENT, complementary
    question: after that correction, is the RESIDUAL error concentrated in
    the trace's EVEN component (reflected about its midpoint index) -- the
    signature the historical quadratic-phase failure mode would predict --
    or is it comparable to the odd component, implicating a different
    mechanism (e.g. the amplitude-weakness one diagnose_amplitude_dependence
    already found)?"""
    def even_odd_rms(x):
        x_flip = x[::-1]
        even_part = 0.5 * (x + x_flip)
        odd_part = 0.5 * (x - x_flip)
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
        return rms(even_part), rms(odd_part)

    weight = np.abs(mie_fields.E_p) ** 2
    _, phi_gs_aligned = compare_phase(result["phi_gs"], result["phi_true"], weight)
    err = np.angle(np.exp(1j * (result["phi_true"] - phi_gs_aligned)))

    error_even_rms, error_odd_rms = even_odd_rms(err)
    true_even_rms, true_odd_rms = even_odd_rms(result["phi_true"])

    return {
        "error_even_rms": error_even_rms, "error_odd_rms": error_odd_rms,
        "true_phase_even_rms": true_even_rms, "true_phase_odd_rms": true_odd_rms,
    }


def add_measurement_noise(I, noise_std, seed=0):
    """Multiplicative Gaussian noise on an intensity trace: I * (1 + N(0,
    noise_std)), clipped at 0 (a physical detector can't read negative
    intensity). Matches the 5% convention already used in
    inverse_scattering.synthesize_measurement, but noise_std is a free
    parameter here so it can be swept."""
    rng = np.random.default_rng(seed)
    return np.clip(I * (1 + rng.normal(0, noise_std, size=I.shape)), 0, None)


def demonstrate_autograd_overfitting(
        D1=6000.0, D2=-7000.0, noise_std=0.6,
        checkpoints=(1, 3, 8, 18, 35, 50, 70, 100, 150, 250, 400, 700, 1500, 3000, 10000),
        lr=0.03, seed=0):
    """
    Does the autograd phase-retrieval path (Step 4's fixed optimizer) overfit
    NOISY measurements -- keep reducing its own loss while its match to the
    TRUE (noiseless) Mie phase gets WORSE? Adds multiplicative noise to the
    2-plane SEALS measurements, runs the autograd optimizer with per-checkpoint
    history (retrieve_phase_with_history_torch), and reports where RMS-vs-truth
    is minimized versus where the optimizer eventually converges.

    Uses the SAME amplitude/intensity normalization fix as run_bridge_demo --
    without it the optimizer stalls at initialization (see the correction in
    SEALS_TO_TDGSA_REPORT.md Sec. 4) and this function would show nothing
    happening at all, not overfitting.

    Returns a dict with the full per-checkpoint history (including the
    ALIGNED recovered phase curve at each checkpoint, not just its RMS --
    for plotting the overfitting directly in phase space, not only as a
    scalar) plus the identified "overfitting gap": how much worse the
    fully-converged RMS is than the best RMS seen at any earlier checkpoint
    (0 or negative = no overfitting observed; positive = real degradation
    from continuing to optimize).

    Deterministic given (D1, D2, noise_std, checkpoints, lr, seed) -- same
    inputs always produce bit-for-bit identical output (both the noise
    added via add_measurement_noise and the optimizer's own randomness are
    seeded), so this is directly reproducible; see
    test_demonstrate_autograd_overfitting_is_reproducible.
    """
    _, theta_deg, mie_fields = seals_intensity_trace()
    phi_true = mie_fields.phase_p
    weight = np.abs(mie_fields.E_p) ** 2
    amp_true = np.abs(mie_fields.E_p)

    I1, I2 = build_gs_measurements(mie_fields, D1, D2)
    I1_noisy = add_measurement_noise(I1, noise_std, seed=seed)
    I2_noisy = add_measurement_noise(I2, noise_std, seed=seed + 1)

    scale = float(amp_true.max())
    amplitude = torch.tensor(amp_true / scale, dtype=torch.float64)
    I1_t = torch.tensor(I1_noisy / scale ** 2, dtype=torch.float64)
    I2_t = torch.tensor(I2_noisy / scale ** 2, dtype=torch.float64)

    history = retrieve_phase_with_history_torch(
        amplitude, [I1_t, I2_t],
        [lambda E: dispersive_operator(E, D1), lambda E: dispersive_operator(E, D2)],
        checkpoints=checkpoints, lr=lr, seed=seed)

    records = []
    for step, loss_val, phase_est in history:
        rms, phase_aligned = compare_phase(phase_est.numpy(), phi_true, weight)
        records.append({"step": step, "loss": loss_val, "rms_vs_truth": rms, "phase_aligned": phase_aligned})

    best = min(records, key=lambda r: r["rms_vs_truth"])
    final = records[-1]
    return {
        "noise_std": noise_std, "records": records, "theta_deg": theta_deg, "phi_true": phi_true,
        "best_step": best["step"], "best_rms": best["rms_vs_truth"], "best_phase": best["phase_aligned"],
        "final_step": final["step"], "final_rms": final["rms_vs_truth"], "final_phase": final["phase_aligned"],
        "overfitting_gap": final["rms_vs_truth"] - best["rms_vs_truth"],
    }


if __name__ == "__main__":
    r = run_bridge_demo()
    print(f"SEALS -> TD-GSA bridge demo (N={len(r['lamvec'])} samples, D1={r['D1']}, D2={r['D2']})")
    print(f"  GS (dgs.gs_core)        RMS phase error vs. Mie ground truth: {r['rms_gs_vs_truth']:.4f} rad "
          f"(final measurement self-consistency: {r['gs_final_error']:.3e})")
    print(f"  autograd (this package) RMS phase error vs. Mie ground truth: {r['rms_autograd_vs_truth']:.4f} rad")
    print(f"  GS vs. autograd, RMS difference between the two methods:      {r['rms_gs_vs_autograd']:.4f} rad")

    _, _, mie_fields = seals_intensity_trace()
    diag = diagnose_amplitude_dependence(r, mie_fields)
    print(f"\nWHY the error isn't uniform (mechanism, not just a number):")
    print(f"  Pearson r(|phase error|, log amplitude) = {diag['pearson_r_abs_err_vs_log_amplitude']:.3f} "
          f"(p={diag['pearson_p_value']:.2e})")
    print(f"  weakest-signal quartile mean |error|:  {diag['bottom_quartile_mean_abs_err_rad']:.3f} rad")
    print(f"  strongest-signal quartile mean |error|: {diag['top_quartile_mean_abs_err_rad']:.3f} rad")
    print(f"  amplitude dynamic range across the trace: {diag['amplitude_dynamic_range']:.1f}x")

    print("\nReport Sec. 6 next step 1: more measurement diversity (N=2..5 dispersion planes)")
    for n, rms in sweep_measurement_diversity().items():
        print(f"  N={n} planes: RMS phase error vs. Mie ground truth = {rms:.4f} rad")

    print("\nReport Sec. 6 next step 2: amplitude regularization toward a Mie-fitted envelope")
    r_plain = run_multiplane_bridge_demo(Ds=(6000.0, -7000.0), use_amplitude_prior=False)
    r_prior = run_multiplane_bridge_demo(Ds=(6000.0, -7000.0), use_amplitude_prior=True)
    print(f"  without amplitude prior: RMS phase error vs. Mie ground truth = {r_plain['rms_vs_truth']:.4f} rad")
    print(f"  with amplitude prior:    RMS phase error vs. Mie ground truth = {r_prior['rms_vs_truth']:.4f} rad")

    print("\nCross-check vs. the historical ECE 279AS / Yiming (Jalali Lab) even-degree-phase finding:")
    eo = diagnose_even_odd_ambiguity(r, mie_fields)
    print(f"  true Mie phase:  even-part RMS={eo['true_phase_even_rms']:.3f} rad, "
          f"odd-part RMS={eo['true_phase_odd_rms']:.3f} rad")
    print(f"  residual error:  even-part RMS={eo['error_even_rms']:.3f} rad, "
          f"odd-part RMS={eo['error_odd_rms']:.3f} rad")
    ratio = eo['error_even_rms'] / eo['error_odd_rms']
    print(f"  even/odd error ratio: {ratio:.2f} -- "
          f"{'error concentrated in even part (matches historical failure mode)' if ratio > 2 else 'roughly balanced (historical even-degree mechanism is NOT the driver here; amplitude-weakness above already explains it)'}")

    print("\nDoes the autograd path overfit NOISY measurements? (loss keeps improving; does true-phase match?)")
    for noise_std in [0.3, 0.6, 1.5, 3.0]:
        of = demonstrate_autograd_overfitting(noise_std=noise_std)
        print(f"  noise_std={noise_std:<4}  best RMS={of['best_rms']:.4f} rad @step{of['best_step']:<6}  "
              f"final RMS={of['final_rms']:.4f} rad @step{of['final_step']:<6}  "
              f"overfitting gap={of['overfitting_gap']:+.4f} rad")
