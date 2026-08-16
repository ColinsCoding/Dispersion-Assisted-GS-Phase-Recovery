"""Proposed extension: label-free optical viability/morphology classification
of insect sperm, for two REAL agricultural/entomological techniques --
Sterile Insect Technique (SIT) and honeybee instrumental insemination --
using this repo's existing STEAM imaging + Bayesian classifier stack
(dgs/steam_imaging.py, dgs/cell_morphology.py, the same GNB pattern as
dgs/sbir_portfolio.py's P5 CTC classifier).

WHAT IS REAL (checked against the literature, not assumed):
  - Sterile Insect Technique (SIT): mass-rear and sterilize (historically
    via irradiation) male insects, release them to mate with wild females
    whose offspring don't survive. Founding paper: Knipling, E.F. (1955)
    "Possibilities of insect control or eradication through the use of
    sexually sterile males," J. Econ. Entomol. 48(4), 459-462. A real,
    large-scale, ongoing IAEA/FAO Insect Pest Control program (used against
    Mediterranean fruit fly, New World screwworm, and mosquito disease
    vectors) -- not a fringe or speculative technique.
  - Honeybee instrumental insemination: real, established since the 1940s.
    Mackensen, O. & Roberts, W.C. (1948) "A manual for the artificial
    insemination of queen bees," USDA. Widely used today in bee-breeding
    programs for disease resistance and temperament selection -- directly
    relevant to pollinator conservation, a real economic and ecological
    concern (colony collapse disorder).
  - Flow-cytometric DNA-content sperm SEX sorting (used in livestock, e.g.
    dairy cattle "sexed semen"): real technique, but REQUIRES a fluorescent
    DNA-binding dye (Hoechst 33342) to detect the ~3-4% DNA content
    difference between X- and Y-bearing sperm. Garner, D.L. & Seidel, G.E.
    (2008) "History of commercializing sexed semen for cattle," Theriogenology.

WHAT IS PROPOSED, NOT DEMONSTRATED (this module's actual contribution):
  Label-free phase-contrast VIABILITY/MORPHOLOGY classification (motile vs.
  non-motile, normal vs. abnormal head/flagellum shape) via this repo's
  existing STEAM + GS phase recovery + Bayesian classifier pipeline. This is
  explicitly NOT DNA-content sex sorting -- STEAM's phase-contrast imaging
  measures refractive index (membrane integrity, mitochondrial density
  along the flagellar sheath), not DNA content, and has no fluorescent
  channel in this repo's forward model. Nor is it DNA SEQUENCING --
  "sequencing" a genome requires a separate wet-lab technique (PCR/NGS),
  which imaging cannot substitute for; this module characterizes cell
  morphology/motility only. Both distinctions matter and should not be
  blurred in any proposal-facing text built on top of this module.
"""
import numpy as np

from dgs.steam_imaging import time_stretch_pulse
from dgs.cell_morphology import extract_cell_features, segment_mask


def flagellar_phase_signature(t, motile, beat_frequency_hz=20.0, membrane_delta_n=2e-4,
                              flagellum_length_um=40.0, wavelength_nm=1550.0):
    """Model the phase signal a moving (motile) vs. static (non-motile)
    sperm flagellum imprints on a probe pulse, via the SAME
    delta_phi = 2*pi*delta_n*L/lambda relation dgs/sbir_portfolio.py's P3
    (CRISPR) proposal uses for refractive-index-based phase sensing --
    reused here, not re-derived, since it's the same physics (a local
    refractive index perturbation accumulating phase over a path length).
    Motile flagella beat at a characteristic frequency (real insect sperm
    beat frequencies are on the order of 10-40 Hz depending on species);
    non-motile flagella contribute a static (non-oscillating) phase offset.

    Parameters
    ----------
    t                  : float array, seconds -- observation time axis
    motile             : bool -- whether to include flagellar beat oscillation
    beat_frequency_hz  : float > 0 -- flagellar beat frequency
    membrane_delta_n   : float > 0 -- refractive index modulation amplitude
                         from the beating flagellar membrane/mitochondrial sheath
    flagellum_length_um: float > 0 -- path length the probe traverses
    wavelength_nm      : float > 0

    Returns
    -------
    phi(t) : float array, radians
    """
    t = np.asarray(t, float)
    if beat_frequency_hz <= 0:
        raise ValueError("beat_frequency_hz must be positive")
    if membrane_delta_n <= 0:
        raise ValueError("membrane_delta_n must be positive")
    if flagellum_length_um <= 0:
        raise ValueError("flagellum_length_um must be positive")
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive")

    L_m = flagellum_length_um * 1e-6
    lambda_m = wavelength_nm * 1e-9
    delta_phi_static = 2 * np.pi * membrane_delta_n * L_m / lambda_m

    if motile:
        return delta_phi_static * (1 + 0.5 * np.sin(2 * np.pi * beat_frequency_hz * t))
    return np.full_like(t, delta_phi_static)


def simulate_sperm_phase_image(N, motile, rng_seed=0, D1=-5000.0, D2=-5750.0,
                               beat_frequency_hz=20.0):
    """Forward-model a STEAM phase-contrast measurement of one sperm cell's
    flagellar signal, then recover phase via this repo's existing GS engine
    -- reusing dgs.steam_imaging.time_stretch_pulse and dgs.gs_core exactly
    as dgs/sbir_portfolio.py's P2/P3/P5 proposals already do for other cell
    types, not a new forward model.

    Returns
    -------
    dict with keys: t, phi_true, phi_recovered, I1, I2
    """
    from dgs.gs_core import disperse, retrieve_phase

    rng = np.random.default_rng(rng_seed)
    t = np.linspace(0, 0.2, N)   # 200 ms observation window
    phi_true = flagellar_phase_signature(t, motile, beat_frequency_hz=beat_frequency_hz)
    E = np.exp(1j * phi_true)

    I1 = np.abs(disperse(E, D1)) ** 2
    I2 = np.abs(disperse(E, D2)) ** 2
    noise_floor = np.mean(I1) * 10 ** (-25.0 / 10)
    I1 = np.maximum(I1 + rng.normal(0, np.sqrt(noise_floor), N), 0)
    I2 = np.maximum(I2 + rng.normal(0, np.sqrt(noise_floor), N), 0)

    phi_recovered, _ = retrieve_phase(I1, I2, D1, D2, n_iter=50, unit_amplitude=True)
    return {"t": t, "phi_true": phi_true, "phi_recovered": phi_recovered, "I1": I1, "I2": I2}


def viability_features_from_phase(phi_recovered, beat_frequency_hz=20.0, sample_rate_hz=None, t=None):
    """Extract viability-classification features from a recovered phase
    trace: oscillation power at the expected beat frequency (motile sperm
    should show a clear spectral peak there; non-motile should not) plus
    the same morphology_entropy measure dgs/cell_morphology.py already
    defines (reused directly, not reimplemented) as a general
    heterogeneity feature.

    Returns
    -------
    dict: beat_band_power (float), morphology_entropy (float)
    """
    from dgs.cell_morphology import shannon_entropy

    phi_recovered = np.asarray(phi_recovered, float)
    N = len(phi_recovered)
    if t is not None:
        dt = float(np.mean(np.diff(t)))
        fs = 1.0 / dt
    elif sample_rate_hz is not None:
        fs = sample_rate_hz
    else:
        raise ValueError("must provide either t or sample_rate_hz")

    spectrum = np.abs(np.fft.rfft(phi_recovered - np.mean(phi_recovered))) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    band = (freqs > beat_frequency_hz * 0.7) & (freqs < beat_frequency_hz * 1.3)
    beat_band_power = float(np.sum(spectrum[band])) if np.any(band) else 0.0

    return {
        "beat_band_power": beat_band_power,
        "morphology_entropy": shannon_entropy(phi_recovered),
    }


def train_viability_classifier(n_per_class=150, beat_frequency_hz=20.0, rng_seed=0):
    """Gaussian Naive Bayes viable/non-viable classifier -- exact same
    pattern as dgs/sbir_portfolio.py's P5 CTC classifier (priors, per-class
    Gaussian likelihoods on engineered features), applied to the beat-band-
    power + morphology-entropy features above instead of {I_max, phi_mean,
    phi_std, morphology_entropy}.

    Returns
    -------
    dict: priors, means, stds (per class), and a classify(feat) closure
    """
    rng_states = np.random.default_rng(rng_seed)
    classes = ["viable", "non_viable"]
    feats_by_class = {c: [] for c in classes}

    for i in range(2 * n_per_class):
        motile = (i % 2 == 0)
        seed = int(rng_states.integers(0, 2**31 - 1))
        sim = simulate_sperm_phase_image(64, motile, rng_seed=seed, beat_frequency_hz=beat_frequency_hz)
        feat = viability_features_from_phase(sim["phi_recovered"], beat_frequency_hz=beat_frequency_hz, t=sim["t"])
        cls = "viable" if motile else "non_viable"
        feats_by_class[cls].append([feat["beat_band_power"], feat["morphology_entropy"]])

    feats_by_class = {c: np.array(v) for c, v in feats_by_class.items()}
    priors = {c: 0.5 for c in classes}
    means = {c: feats_by_class[c].mean(axis=0) for c in classes}
    stds = {c: feats_by_class[c].std(axis=0) + 1e-10 for c in classes}

    def classify(feat_vec):
        log_posts = {}
        for c in classes:
            log_prior = np.log(priors[c])
            log_like = -0.5 * np.sum(((feat_vec - means[c]) / stds[c]) ** 2) - np.sum(np.log(stds[c]))
            log_posts[c] = log_prior + log_like
        return max(log_posts, key=log_posts.get)

    return {"priors": priors, "means": means, "stds": stds, "classes": classes, "classify": classify}


def sit_throughput_economics(cells_per_hour, viability_sort_accuracy, cost_per_cell_usd=0.001):
    """Real-world framing: SIT and instrumental-insemination programs care
    about SORTING THROUGHPUT and ACCURACY, not raw imaging speed -- this
    connects the accuracy number from the classifier above to what actually
    matters operationally (rejecting non-viable sperm before use, at scale).
    Cost figures are illustrative placeholders (explicitly not sourced from
    a real SIT program budget) -- flagged for verification before citing in
    an actual proposal, same honesty standard as the rest of this repo."""
    if cells_per_hour <= 0:
        raise ValueError("cells_per_hour must be positive")
    if not (0.0 <= viability_sort_accuracy <= 1.0):
        raise ValueError("viability_sort_accuracy must be in [0, 1]")
    if cost_per_cell_usd < 0:
        raise ValueError("cost_per_cell_usd must be non-negative")

    correctly_sorted_per_hour = cells_per_hour * viability_sort_accuracy
    daily_cost_usd = cells_per_hour * 24 * cost_per_cell_usd
    return {
        "cells_per_hour": cells_per_hour,
        "correctly_sorted_per_hour": correctly_sorted_per_hour,
        "daily_cost_usd": daily_cost_usd,
        "note": "cost_per_cell_usd is an illustrative placeholder, NOT sourced from a real "
                "SIT/apiculture program budget -- verify before use in an actual proposal.",
    }


if __name__ == "__main__":
    print("=== Label-free insect sperm viability classification (proposed extension) ===\n")

    clf = train_viability_classifier(n_per_class=100, beat_frequency_hz=20.0, rng_seed=1)

    correct = 0
    n_test = 100
    rng = np.random.default_rng(99)
    for i in range(n_test):
        motile = (i % 2 == 0)
        seed = int(rng.integers(0, 2**31 - 1))
        sim = simulate_sperm_phase_image(64, motile, rng_seed=seed)
        feat = viability_features_from_phase(sim["phi_recovered"], t=sim["t"])
        pred = clf["classify"]([feat["beat_band_power"], feat["morphology_entropy"]])
        true_label = "viable" if motile else "non_viable"
        if pred == true_label:
            correct += 1

    accuracy = correct / n_test
    print(f"Viability classifier accuracy on {n_test} held-out synthetic samples: {accuracy:.1%}")

    econ = sit_throughput_economics(cells_per_hour=36_000_000, viability_sort_accuracy=accuracy)
    print(f"\nAt STEAM's real 36 Mfps frame rate ({econ['cells_per_hour']:,} cells/hour):")
    print(f"  correctly sorted/hour: {econ['correctly_sorted_per_hour']:,.0f}")
    print(f"  illustrative daily cost: ${econ['daily_cost_usd']:,.0f}  ({econ['note']})")

    print("\nReminder: this classifies VIABILITY/MOTILITY from phase-contrast imaging,")
    print("NOT DNA-content sex sorting (needs fluorescence) and NOT DNA sequencing")
    print("(needs a separate wet-lab technique). See module docstring.")
