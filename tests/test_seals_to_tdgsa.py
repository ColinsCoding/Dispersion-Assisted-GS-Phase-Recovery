"""Tests for projects.seals.inverse.seals_to_tdgsa -- the SEALS-intensity-trace
to dgs.gs_core TD-GSA bridge, verified two independent ways (classical GS and
PyTorch autograd), per the project's own docstring."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from dgs.dispersion_gs_prototype import compare_phase
from projects.seals.inverse.seals_to_tdgsa import (
    seals_intensity_trace, build_gs_measurements, build_gs_measurements_n, run_bridge_demo,
    diagnose_amplitude_dependence, run_multiplane_bridge_demo, sweep_measurement_diversity,
    diagnose_even_odd_ambiguity, add_measurement_noise, demonstrate_autograd_overfitting,
)


def test_seals_intensity_trace_matches_native_mie_shape():
    lamvec, theta_deg, mie_fields = seals_intensity_trace()
    assert lamvec.shape == theta_deg.shape == mie_fields.I_p.shape
    assert np.all(mie_fields.I_p >= 0)   # intensity is non-negative
    assert np.all(np.isfinite(mie_fields.phase_p))


def test_build_gs_measurements_are_nonnegative_and_energy_bounded():
    _, _, mie_fields = seals_intensity_trace()
    I1, I2 = build_gs_measurements(mie_fields, D1=6000.0, D2=-7000.0)
    assert I1.shape == I2.shape == mie_fields.I_p.shape
    assert np.all(I1 >= -1e-9) and np.all(I2 >= -1e-9)
    # dispersion is phase-only (all-pass): total energy is conserved to numerical precision
    E0_energy = (np.abs(mie_fields.E_p) ** 2).sum()
    assert abs(I1.sum() - E0_energy) < 1e-6 * E0_energy
    assert abs(I2.sum() - E0_energy) < 1e-6 * E0_energy


def test_bridge_demo_runs_and_gs_fits_its_own_measurements():
    """The classical GS path must converge to a solution that is internally
    self-consistent with (I1, I2) -- that is a real, checkable claim
    regardless of whether it recovers the TRUE phase (see the next test)."""
    r = run_bridge_demo(D1=6000.0, D2=-7000.0, n_iter=150)
    assert r["gs_final_error"] < 1e-10, \
        "GS should fit its own measurements to near-zero residual (self-consistency), whatever phase it lands on"


def test_bridge_demo_known_limitation_blind_gs_does_not_cleanly_recover_mie_phase():
    """HONEST regression test, not a success claim: two-measurement blind
    phase retrieval (both the classical GS in dgs.gs_core and the autograd
    path in this package) does NOT cleanly recover the true Mie scattering
    phase for this varying-amplitude signal -- GS lands around 0.4-0.5 rad
    RMS error, autograd (properly normalized -- see
    test_autograd_optimizer_actually_moves_from_its_initial_guess below)
    does somewhat better at ~0.2-0.4 rad but still not near-zero, not the
    near-zero error achieved elsewhere in this repo on unit-amplitude (e.g.
    QPSK-like) signals. This is a genuine phase-retrieval ambiguity for
    varying-amplitude fields with only 2 measurement planes, not a bug in
    this bridge -- see SEALS_TO_TDGSA_REPORT.md for the full discussion
    and recommended next steps (more measurement diversity, or the
    model-based inverse_scattering.py approach already in this package).
    If this test starts failing because the error dropped dramatically,
    that's genuinely interesting and the report should be updated to match.
    """
    r = run_bridge_demo(D1=6000.0, D2=-7000.0, n_iter=150)
    assert 0.2 < r["rms_gs_vs_truth"] < 0.8, \
        f"expected the known ~0.4-0.5 rad ambiguity, got {r['rms_gs_vs_truth']:.4f} rad -- update the report if this changed"
    assert 0.1 < r["rms_autograd_vs_truth"] < 0.6, \
        f"expected autograd to also miss the true phase but do somewhat better than GS, got {r['rms_autograd_vs_truth']:.4f} rad -- update the report if this changed"
    # the two independent methods should also not fully agree with EACH OTHER --
    # if they converged to the identical wrong answer that would suggest a shared
    # bug in how (I1, I2) are built, not two independently-arrived-at solutions
    assert r["rms_gs_vs_autograd"] > 0.05


def test_autograd_optimizer_actually_moves_from_its_initial_guess():
    """Regression test for a real bug found and fixed in run_bridge_demo:
    Mie-scattered field amplitudes are physically tiny (~1e-5 to 5e-4), so
    without normalization the raw intensity-matching loss Adam minimizes is
    ~1e-16-scale -- small enough that Adam's default eps=1e-8 silently
    stalled every update, and the "converged" autograd phase was literally
    an untouched all-zero initial guess (which happened to score ~0.50 rad,
    coincidentally close to GS's real answer, making the bug invisible in
    the existing RMS-range assertions above). Fixed by normalizing
    amplitude/intensities to O(1) before optimization (phase is
    scale-invariant, so this doesn't change the problem being solved).
    This test directly checks the optimizer MOVED, rather than trusting the
    final RMS to be different enough from a stuck-at-init value by chance."""
    _, _, mie_fields = seals_intensity_trace()
    phi_true_zero_guess_rms, _ = compare_phase(
        np.zeros_like(mie_fields.phase_p), mie_fields.phase_p, np.abs(mie_fields.E_p) ** 2)

    r = run_bridge_demo(D1=6000.0, D2=-7000.0, n_iter=150)
    assert abs(r["rms_autograd_vs_truth"] - phi_true_zero_guess_rms) > 0.05, \
        "autograd's result should differ meaningfully from an untouched all-zero initial " \
        f"phase guess (RMS {phi_true_zero_guess_rms:.4f} rad) -- if it doesn't, the optimizer " \
        "may have silently stalled again (check amplitude/intensity normalization scale)"


def test_amplitude_dependence_explains_the_mechanism():
    """This is TD-GSA/GS working as designed, not a bug: the algorithm fits
    the MEASUREMENTS (proven by test_bridge_demo_runs_and_gs_fits_its_own_measurements's
    near-zero residual), not necessarily the true field -- those coincide only
    where the inverse problem is well-posed. GS's amplitude constraint
    |E|=sqrt(I) carries almost no phase information wherever I is near zero,
    so phase error should be concentrated in the WEAK-signal part of the
    trace, not spread uniformly. Verified directly (not left as an
    unquantified report claim): a significant negative correlation between
    |phase error| and log(amplitude), and a large gap between the
    weakest/strongest amplitude quartiles' mean error."""
    r = run_bridge_demo(D1=6000.0, D2=-7000.0, n_iter=150)
    _, _, mie_fields = seals_intensity_trace()
    diag = diagnose_amplitude_dependence(r, mie_fields)

    assert diag["pearson_r_abs_err_vs_log_amplitude"] < -0.3, \
        "expected a clear negative correlation: weaker signal -> larger phase error"
    assert diag["pearson_p_value"] < 1e-6, "the correlation should be statistically significant, not noise"
    assert diag["bottom_quartile_mean_abs_err_rad"] > diag["top_quartile_mean_abs_err_rad"], \
        "the weakest-signal quartile should have WORSE error than the strongest-signal quartile"
    assert diag["bottom_quartile_mean_abs_err_rad"] > 2 * diag["top_quartile_mean_abs_err_rad"], \
        "expected a substantial (>2x) gap, matching the mechanism, not a marginal difference"
    assert diag["amplitude_dynamic_range"] > 5, \
        "the mechanism requires a real amplitude spread across the trace -- confirm it's actually present"


# ── Report Sec. 6 next steps: N-plane diversity + amplitude-prior regularization ──

def test_build_gs_measurements_n_matches_2plane_for_two_dispersions():
    """N-plane builder with exactly 2 dispersions must reproduce the
    original 2-plane build_gs_measurements exactly -- pure regression check
    that generalizing to N planes didn't change the D1/D2 case."""
    _, _, mie_fields = seals_intensity_trace()
    I1_ref, I2_ref = build_gs_measurements(mie_fields, D1=6000.0, D2=-7000.0)
    I1_n, I2_n = build_gs_measurements_n(mie_fields, [6000.0, -7000.0])
    np.testing.assert_allclose(I1_n, I1_ref)
    np.testing.assert_allclose(I2_n, I2_ref)


def test_more_measurement_diversity_dramatically_improves_recovery():
    """Report Sec. 6 recommendation 1, verified as a fact rather than left as
    a suggestion: with only 2 dispersion planes, blind GS is stuck at the
    known ~0.4-0.5 rad ambiguity (see the 2-plane test above). Adding a 3rd,
    independent dispersion plane to the SAME hidden Mie field resolves the
    amplitude-constraint ambiguity almost completely -- this is the general
    fix the report predicted, now measured directly."""
    r2 = run_multiplane_bridge_demo(Ds=(6000.0, -7000.0), n_iter=150)
    r3 = run_multiplane_bridge_demo(Ds=(6000.0, -7000.0, 12000.0), n_iter=150)

    assert 0.2 < r2["rms_vs_truth"] < 0.8, \
        f"2-plane case should reproduce the known ambiguity, got {r2['rms_vs_truth']:.4f} rad"
    assert r3["rms_vs_truth"] < 0.05, \
        f"3rd measurement plane should nearly resolve the ambiguity, got {r3['rms_vs_truth']:.4f} rad " \
        "-- if this regressed, re-check gs_multiplane's per-plane loop order"
    assert r3["rms_vs_truth"] < r2["rms_vs_truth"] / 5


def test_sweep_measurement_diversity_is_monotonically_better_past_two_planes():
    results = sweep_measurement_diversity(n_iter=150)
    assert set(results) == {2, 3, 4, 5}
    assert results[2] > 0.2                      # the known 2-plane ambiguity
    for n in (3, 4, 5):
        assert results[n] < 0.05                 # 3+ planes: essentially resolved


def test_amplitude_prior_regularization_honest_null_result():
    """HONEST regression test, not a success claim (matching this project's
    own testing standard -- see the "known limitation" test above): blending
    the amplitude constraint toward a Mie-fitted envelope, as implemented in
    gs_multiplane (once per iteration, in the undispersed domain), does NOT
    meaningfully improve 2-plane recovery. Diagnosed mechanism: each
    iteration's very next per-plane projection re-imposes |E_d|=sqrt(I_j)
    exactly in the DISPERSED domain, which overwrites whatever the prior
    contributed in the undispersed domain before that iteration completes --
    the hard per-plane constraints dominate. This is why Sec. 6's *other*
    recommendation (more measurement diversity) is the one that actually
    fixes the problem (see test_more_measurement_diversity_... above); this
    test exists so a future "fix" to the prior-blending mechanism has a
    baseline to compare against, and so this module doesn't silently claim
    a win it didn't earn."""
    r_plain = run_multiplane_bridge_demo(Ds=(6000.0, -7000.0), n_iter=150, use_amplitude_prior=False)
    r_prior = run_multiplane_bridge_demo(Ds=(6000.0, -7000.0), n_iter=150, use_amplitude_prior=True)

    assert abs(r_prior["rms_vs_truth"] - r_plain["rms_vs_truth"]) < 0.1, \
        "expected the amplitude prior to make little difference under hard per-plane " \
        f"constraints (got plain={r_plain['rms_vs_truth']:.4f}, prior={r_prior['rms_vs_truth']:.4f}) " \
        "-- if this changed a lot, the mechanism diagnosis in this test's docstring needs updating"
    assert r_prior["amplitude_prior"] is not None and np.all(np.isfinite(r_prior["amplitude_prior"]))


def test_even_odd_ambiguity_is_not_the_driver_here():
    """Cross-check against the historical ECE 279AS / Yiming (Jalali Lab)
    finding (notebooks/phase_retrieval.ipynb cell 62): blind TDGSA fails on
    EVEN-degree phase polynomials because the intensity constraint can't
    distinguish phi from -phi. dgs.dispersion_gs_prototype.compare_phase
    already searches both signs when scoring GS's phase (so that ambiguity
    is already corrected for in every RMS number this bridge reports) --
    this test checks whether the RESIDUAL error is nonetheless concentrated
    in the trace's even component, which the historical finding would
    predict. HONEST result: it is NOT -- error is roughly balanced between
    even and odd parts, so this specific historical mechanism does not
    explain the SEALS/Mie 2-plane residual (the amplitude-weakness
    mechanism in test_amplitude_dependence_explains_the_mechanism does)."""
    r = run_bridge_demo(D1=6000.0, D2=-7000.0, n_iter=150)
    _, _, mie_fields = seals_intensity_trace()
    eo = diagnose_even_odd_ambiguity(r, mie_fields)

    ratio = eo["error_even_rms"] / eo["error_odd_rms"]
    assert 0.5 < ratio < 2.0, \
        f"expected the error roughly balanced between even/odd parts (ratio={ratio:.2f}) -- " \
        "if this became lopsided, the historical even-degree mechanism may actually be at play " \
        "and this test + the report should be updated"


# ── Does the autograd path overfit noisy measurements? ──

def test_add_measurement_noise_is_nonnegative_and_matches_shape():
    I = np.array([1.0, 2.0, 0.0, 5.0])
    I_noisy = add_measurement_noise(I, noise_std=0.5, seed=0)
    assert I_noisy.shape == I.shape
    assert np.all(I_noisy >= 0)


def test_no_overfitting_at_realistic_low_noise():
    """At the ~5% multiplicative noise level used elsewhere in this package
    (inverse_scattering.synthesize_measurement's convention), the autograd
    optimizer should NOT overfit: RMS-vs-truth should keep improving (or
    plateau) as it converges, not get measurably worse. This is the
    contrast case for the noise sweep below -- overfitting is a real but
    noise-level-dependent phenomenon, not something that always happens."""
    of = demonstrate_autograd_overfitting(noise_std=0.05, seed=1)
    assert of["overfitting_gap"] < 0.02, \
        f"expected negligible overfitting at low noise, got a {of['overfitting_gap']:.4f} rad gap"


def test_overfitting_gap_grows_with_noise():
    """The actual demonstration requested: does the autograd path overfit
    noisy measurements (keep reducing its own loss while its match to the
    TRUE, noiseless Mie phase gets WORSE)? Real, measured effect -- at
    moderate-to-high noise, the RMS-vs-truth at an EARLY checkpoint beats
    the fully-converged RMS, and the gap between them grows with noise
    level. Modest in absolute size (a few hundredths of a radian at 60%
    noise) because the underlying 2-plane problem is already badly
    underdetermined (~0.5 rad baseline error) -- overfitting is a real,
    secondary effect on top of that, not the dominant one."""
    gaps = []
    for noise_std in [0.3, 0.6, 1.5, 3.0]:
        of = demonstrate_autograd_overfitting(noise_std=noise_std, seed=2)
        gaps.append(of["overfitting_gap"])

    assert all(g > 0 for g in gaps), \
        f"expected a positive overfitting gap (final RMS worse than best RMS) at every noise level, got {gaps}"
    assert gaps == sorted(gaps), \
        f"expected the overfitting gap to grow monotonically with noise level, got {gaps}"
    assert gaps[-1] > 3 * gaps[0], \
        f"expected the gap at the highest noise level to be substantially larger than at the lowest, got {gaps}"


def test_demonstrate_autograd_overfitting_is_reproducible():
    """Same (D1, D2, noise_std, checkpoints, lr, seed) must give bit-for-bit
    identical output -- the noise (add_measurement_noise) and the optimizer
    (torch.manual_seed inside retrieve_phase_with_history) are both seeded,
    so nothing here should be allowed to vary run-to-run. This is what makes
    the overfitting demonstration a reproducible block, not just an
    illustrative one-off."""
    of1 = demonstrate_autograd_overfitting(noise_std=1.5, seed=7)
    of2 = demonstrate_autograd_overfitting(noise_std=1.5, seed=7)

    assert of1["best_rms"] == of2["best_rms"]
    assert of1["final_rms"] == of2["final_rms"]
    assert of1["overfitting_gap"] == of2["overfitting_gap"]
    np.testing.assert_array_equal(of1["best_phase"], of2["best_phase"])
    np.testing.assert_array_equal(of1["final_phase"], of2["final_phase"])

    # a DIFFERENT seed should generally give a different noise realization/result --
    # confirms the seed is actually controlling something, not silently ignored
    of3 = demonstrate_autograd_overfitting(noise_std=1.5, seed=8)
    assert of3["final_rms"] != of1["final_rms"]


def test_demonstrate_autograd_overfitting_returns_plottable_phase_curves():
    """best_phase/final_phase/theta_deg/phi_true must be usable directly for
    a 'true vs. best-checkpoint vs. overfit' phase-space plot -- the actual
    visual demonstration, not just the RMS scalars."""
    of = demonstrate_autograd_overfitting(noise_std=1.5, seed=0)
    N = len(of["theta_deg"])
    assert of["phi_true"].shape == (N,)
    assert of["best_phase"].shape == (N,)
    assert of["final_phase"].shape == (N,)
    assert np.all(np.isfinite(of["best_phase"])) and np.all(np.isfinite(of["final_phase"]))
    # the final (overfit) phase should be farther from truth than the best checkpoint's,
    # by the SAME weighted metric overfitting_gap itself is defined from -- not a separately
    # invented unweighted metric, which (checked) does not always agree sample-by-sample,
    # since a handful of high-weight (strong-signal) samples can dominate the weighted score
    # while many low-weight (weak-signal, noisier) samples dominate an unweighted average
    assert of["final_rms"] > of["best_rms"]