"""Test dgs/insect_reproductive_optics.py: label-free viability/motility
classification of insect sperm via this repo's existing STEAM + GS +
Bayesian-classifier stack (real techniques cited: SIT, honeybee
instrumental insemination; explicitly NOT DNA-content sex sorting or
sequencing -- see module docstring)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import insect_reproductive_optics as iro

# 1. flagellar_phase_signature: motile oscillates around the same static
#    baseline that non-motile sits flat at
t = np.linspace(0, 0.2, 128)
phi_motile = iro.flagellar_phase_signature(t, motile=True, beat_frequency_hz=20.0)
phi_static = iro.flagellar_phase_signature(t, motile=False, beat_frequency_hz=20.0)
assert np.std(phi_motile) > 10 * np.std(phi_static)   # motile clearly oscillates, static doesn't
assert abs(np.mean(phi_motile) - np.mean(phi_static)) / np.mean(phi_static) < 0.05  # same baseline

# 2. Input validation on flagellar_phase_signature
for bad_call in [
    lambda: iro.flagellar_phase_signature(t, True, beat_frequency_hz=0),
    lambda: iro.flagellar_phase_signature(t, True, membrane_delta_n=-1),
    lambda: iro.flagellar_phase_signature(t, True, flagellum_length_um=0),
    lambda: iro.flagellar_phase_signature(t, True, wavelength_nm=-1),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

# 3. simulate_sperm_phase_image: shapes correct, GS recovery correlates
#    with the true phase signal (reuses the repo's own tested GS engine)
sim = iro.simulate_sperm_phase_image(64, motile=True, rng_seed=1)
assert sim["t"].shape == (64,)
assert sim["phi_true"].shape == (64,)
assert sim["phi_recovered"].shape == (64,)
corr = float(np.corrcoef(sim["phi_true"], sim["phi_recovered"])[0, 1])
assert corr > 0.3, f"expected GS to recover a phase correlated with truth, got corr={corr:.4f}"

# 4. viability_features_from_phase: motile samples should show
#    substantially higher beat-band spectral power than non-motile ones,
#    on average over several trials (the actual discriminating feature)
motile_powers, static_powers = [], []
for seed in range(20):
    sim_m = iro.simulate_sperm_phase_image(64, motile=True, rng_seed=seed)
    sim_s = iro.simulate_sperm_phase_image(64, motile=False, rng_seed=seed)
    motile_powers.append(iro.viability_features_from_phase(sim_m["phi_recovered"], t=sim_m["t"])["beat_band_power"])
    static_powers.append(iro.viability_features_from_phase(sim_s["phi_recovered"], t=sim_s["t"])["beat_band_power"])
assert np.mean(motile_powers) > np.mean(static_powers), (
    f"expected motile beat-band power ({np.mean(motile_powers):.4f}) > "
    f"non-motile ({np.mean(static_powers):.4f})")

# 5. viability_features_from_phase requires either t or sample_rate_hz
try:
    iro.viability_features_from_phase(sim_m["phi_recovered"])
    assert False, "should have raised ValueError"
except ValueError:
    pass

# 6. train_viability_classifier: accuracy clearly above chance (50%) on
#    held-out synthetic data -- not asserting a high bar, just "it works"
clf = iro.train_viability_classifier(n_per_class=60, rng_seed=2)
correct = 0
n_test = 60
rng = np.random.default_rng(123)
for i in range(n_test):
    motile = (i % 2 == 0)
    seed = int(rng.integers(0, 2**31 - 1))
    sim_t = iro.simulate_sperm_phase_image(64, motile, rng_seed=seed)
    feat = iro.viability_features_from_phase(sim_t["phi_recovered"], t=sim_t["t"])
    pred = clf["classify"]([feat["beat_band_power"], feat["morphology_entropy"]])
    true_label = "viable" if motile else "non_viable"
    if pred == true_label:
        correct += 1
accuracy = correct / n_test
assert accuracy > 0.6, f"expected classifier accuracy clearly above chance, got {accuracy:.1%}"

# 7. sit_throughput_economics: formulas and input validation
econ = iro.sit_throughput_economics(cells_per_hour=1000, viability_sort_accuracy=0.8, cost_per_cell_usd=0.01)
assert abs(econ["correctly_sorted_per_hour"] - 800) < 1e-9
assert abs(econ["daily_cost_usd"] - 1000 * 24 * 0.01) < 1e-9

for bad_call in [
    lambda: iro.sit_throughput_economics(cells_per_hour=0, viability_sort_accuracy=0.8),
    lambda: iro.sit_throughput_economics(cells_per_hour=100, viability_sort_accuracy=1.5),
    lambda: iro.sit_throughput_economics(cells_per_hour=100, viability_sort_accuracy=-0.1),
    lambda: iro.sit_throughput_economics(cells_per_hour=100, viability_sort_accuracy=0.5, cost_per_cell_usd=-1),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.insect_reproductive_optics tests passed")
