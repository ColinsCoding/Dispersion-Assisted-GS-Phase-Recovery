"""ASIC behavioural model: fixed-point estimator against the floating-point
reference, word-length checks, and bit-exact agreement with the C code."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import scan_volts
from scarlett import asic
from scarlett import estimator as es
from scarlett import synthetic as sy

C_DIR = Path(__file__).resolve().parents[1] / "c"


def integer_measurement(truth, bench, sensor, lut, seed):
    scan, dark, volts, sigma = scan_volts(truth, bench, sensor, seed)
    memory = asic.build_calibration_memory(lut, dark["code_sums"], scan["gain"], sensor)
    meas = asic.subtract_dark(scan["code_sums"], memory)
    counts_per_volt = scan["gain"] * sensor.samples_per_channel / sensor.lsb
    sigma_counts = float(np.sqrt(np.mean(sigma[lut.mask] ** 2)) * counts_per_volt)
    return memory, meas, sigma_counts, volts, sigma


def test_calibration_memory_holds_unit_energy_q15_templates(bench, sensor, lut):
    memory, *_ = integer_measurement(sy.SampleTruth(), bench, sensor, lut, seed=1)
    assert memory.templates_q15.dtype == np.int16
    assert memory.templates_q15.shape == (lut.dia.size, int(lut.mask.sum()))
    energy = np.sum(memory.templates_q15.astype(np.int64) ** 2, axis=1)
    np.testing.assert_allclose(np.sqrt(energy), asic.Q15, rtol=2e-4)


@pytest.mark.parametrize("dia", [5.0e-6, 4.871e-6, 5.207e-6])
def test_fixed_point_estimate_agrees_with_reference(dia, bench, sensor, lut):
    truth = sy.SampleTruth(dia=dia)
    memory, meas, sigma_counts, volts, sigma = integer_measurement(truth, bench, sensor, lut, seed=4)
    fixed = asic.fixed_point_estimate(meas, memory.templates_q15, sigma_counts)
    reference = es.estimate_particle(volts, sigma, lut, bench, sensor=None)
    step = 20e-9
    assert abs(lut.dia[fixed["best_index"]] - reference.dia) <= step + 1e-15
    assert abs(lut.dia[fixed["best_index"]] - dia) <= step
    assert lut.dia[fixed["ci_lo_index"]] - step / 2 <= dia <= lut.dia[fixed["ci_hi_index"]] + step / 2
    best = fixed["best_index"]
    count = asic.particle_count_from_correlation(int(fixed["corr"][best]), fixed["template_energy"][best],
                                                 memory.norm_counts[best])
    assert count == pytest.approx(20.0, rel=0.03)
    assert not fixed["ambiguous"]


def test_fixed_point_equals_unweighted_least_squares(bench, sensor, lut):
    """The correlator bank is exactly least squares with equal weights."""
    memory, meas, sigma_counts, *_ = integer_measurement(sy.SampleTruth(dia=5.1e-6), bench, sensor, lut, 6)
    fixed = asic.fixed_point_estimate(meas, memory.templates_q15, sigma_counts)
    t = memory.templates_q15.astype(float)
    v = meas.astype(float)
    scale = (t @ v) / np.sum(t * t, axis=1)
    chi2 = np.sum((v - scale[:, None] * t) ** 2, axis=1)          # direct residuals, no shortcut
    assert int(np.argmin(chi2)) == fixed["best_index"]
    assert fixed["chi2_min"] == pytest.approx(chi2.min(), rel=1e-6)
    np.testing.assert_array_equal(fixed["accepted"], chi2 - chi2.min() <= 3.841458820694124 * sigma_counts ** 2)


def test_assuming_exact_unit_energy_would_distort_the_confidence_set(bench, sensor, lut):
    """Why the stored template energy is used: 16-bit rounding shifts chi-square
    by a sizeable fraction of the 95 % threshold (about 40 % in this case)."""
    memory, meas, sigma_counts, *_ = integer_measurement(sy.SampleTruth(dia=5.1e-6), bench, sensor, lut, 6)
    fixed = asic.fixed_point_estimate(meas, memory.templates_q15, sigma_counts)
    best = fixed["best_index"]
    naive_chi2 = fixed["energy"] - (int(fixed["corr"][best]) / asic.Q15) ** 2
    assert abs(naive_chi2 - fixed["chi2_min"]) > 0.2 * 3.84 * sigma_counts ** 2


def test_word_length_and_type_checks(bench, sensor, lut):
    memory, meas, sigma_counts, *_ = integer_measurement(sy.SampleTruth(), bench, sensor, lut, seed=1)
    assert np.max(np.abs(meas)) < 2 ** 19                     # fits the proposed 20-bit word
    with pytest.raises(OverflowError, match="measurement"):
        asic.fixed_point_estimate(meas * 100, memory.templates_q15, sigma_counts)
    with pytest.raises(TypeError):
        asic.fixed_point_estimate(meas.astype(float), memory.templates_q15, sigma_counts)
    with pytest.raises(ValueError, match="positive"):
        asic.fixed_point_estimate(meas, memory.templates_q15, 0.0)
    with pytest.raises(ValueError, match="no template"):
        asic.fixed_point_estimate(-np.abs(meas), np.abs(memory.templates_q15), sigma_counts)
    with pytest.raises(TypeError, match="integers"):
        asic.subtract_dark(np.ones(bench.n_channels), memory)


def test_integer_autocorrelation_matches_floating_point(bench):
    counts, _ = sy.generate_dls_trace(sy.SampleTruth(), bench, 4000, 5e-3, rng=np.random.default_rng(0))
    acf = asic.integer_autocorrelation(counts, 50)
    assert all(isinstance(a, int) for a in acf)
    assert acf[0] == int(np.sum(counts.astype(object) ** 2))
    g2 = asic.g2_from_integer_autocorrelation(acf, counts)
    np.testing.assert_allclose(g2, es.intensity_autocorrelation(counts, 50), rtol=1e-12)
    with pytest.raises(TypeError):
        asic.integer_autocorrelation(counts.astype(float), 5)


def test_block_list_covers_the_required_architecture():
    names = " | ".join(name for name, *_ in asic.ASIC_BLOCKS)
    for required in ("photodiode", "transimpedance", "programmable gain", "anti-alias", "ADC",
                     "autocorrelation", "calibration memory", "digital estimator", "RF telemetry"):
        assert required in names


# ------------------------------------------------------------ C implementation
@pytest.fixture(scope="module")
def c_binary(tmp_path_factory):
    compiler = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("no C compiler on PATH")
    exe = tmp_path_factory.mktemp("c") / "scarlett_estimator"
    subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(C_DIR / "main.c"), str(C_DIR / "scarlett_estimator.c"), "-o", str(exe), "-lm"],
                   check=True, capture_output=True, text=True)
    return exe if exe.exists() else exe.with_suffix(".exe")


@pytest.mark.parametrize("dia,seed", [(5.0e-6, 1), (4.871e-6, 2), (5.207e-6, 3)])
def test_c_estimator_is_bit_exact_with_python(c_binary, tmp_path, dia, seed, bench, sensor, lut):
    memory, meas, sigma_counts, *_ = integer_measurement(sy.SampleTruth(dia=dia), bench, sensor, lut, seed)
    vectors = tmp_path / "estimate.txt"
    asic.write_estimate_vectors(vectors, meas, memory.templates_q15, sigma_counts)
    run = subprocess.run([str(c_binary), "estimate", str(vectors)], check=True,
                         capture_output=True, text=True)
    *integers, chi2_text = run.stdout.split()
    best, lo, hi, n_accepted, ambiguous, corr, energy = (int(v) for v in integers)
    chi2_min = float(chi2_text)
    expected = asic.fixed_point_estimate(meas, memory.templates_q15, sigma_counts)
    assert best == expected["best_index"]
    assert (lo, hi) == (expected["ci_lo_index"], expected["ci_hi_index"])
    assert n_accepted == int(expected["accepted"].sum())
    assert bool(ambiguous) == expected["ambiguous"]
    assert corr == int(expected["corr"][best]) and energy == expected["energy"]
    assert chi2_min == pytest.approx(expected["chi2_min"], rel=1e-12, abs=1e-6)


def test_c_autocorrelator_is_bit_exact_with_python(c_binary, tmp_path, bench):
    counts, _ = sy.generate_dls_trace(sy.SampleTruth(), bench, 3000, 5e-3, rng=np.random.default_rng(5))
    vectors = tmp_path / "acf.txt"
    asic.write_acf_vectors(vectors, counts, 40)
    run = subprocess.run([str(c_binary), "acf", str(vectors)], check=True, capture_output=True, text=True)
    assert [int(v) for v in run.stdout.split()] == list(asic.integer_autocorrelation(counts, 40))


def test_c_program_reports_errors(c_binary, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("3 1 1.0 3.84\n1 2\n")                      # file ends early
    run = subprocess.run([str(c_binary), "estimate", str(bad)], capture_output=True, text=True)
    assert run.returncode != 0 and "ERROR" in run.stderr
    run = subprocess.run([str(c_binary), "estimate", str(tmp_path / "missing.txt")],
                         capture_output=True, text=True)
    assert run.returncode != 0 and "cannot open" in run.stderr
