"""Smoke tests for dgs/real_data_loader.py."""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dgs.gs_core import make_measurements
from dgs.real_data_loader import (
    load_mat_measurement, diagnose_measurement, convergence_sanity_sweep,
    default_candidates, load_or_synthesize,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_load_mat_measurement_reads_known_schema():
    path = SAMPLE_DIR / "QPSK_56GSas_D600_D1200.mat"
    if not path.exists():
        print("SKIP: sample_data not present")
        return
    data = load_mat_measurement(path)
    assert data["I1"].ndim == 1
    assert data["I2"].ndim == 1
    assert len(data["I1"]) == len(data["I2"])
    assert data["fs_GSas"] == 56.0
    assert data["D1_raw"] == -600.0
    assert data["D2_raw"] == -1200.0
    assert "Synthetic" in data["description"]
    print("PASS: load_mat_measurement reads all expected fields")


def test_diagnose_measurement_passes_known_good_data():
    """The core correctness check: given the TRUE generating D as a candidate,
    diagnose_measurement must report recoverable=True, not just always fail."""
    data = make_measurements(modulation="QPSK", n_symbols=64, sps=8,
                              D1=-5000.0, D2=-5750.0, snr_db=25.0, rng_seed=0)
    report = diagnose_measurement(data["I1"], data["I2"],
                                   D_ratio=data["D2"] / data["D1"], D1_candidates=[5000.0])
    assert report["recoverable"] is True
    assert report["sweep_result"]["converged"] is True
    print("PASS: diagnose_measurement correctly passes known-good data")


def test_diagnose_measurement_flags_length_mismatch():
    report = diagnose_measurement(np.ones(10), np.ones(20), run_convergence_sweep=False)
    assert report["recoverable"] is False
    assert any("FAIL" in m for m in report["messages"])
    print("PASS: diagnose_measurement flags mismatched lengths")


def test_diagnose_measurement_flags_nan():
    I1 = np.ones(100)
    I2 = np.ones(100)
    I2[5] = np.nan
    report = diagnose_measurement(I1, I2, run_convergence_sweep=False)
    assert report["recoverable"] is False
    print("PASS: diagnose_measurement flags NaN")


def test_convergence_sweep_finds_true_D_when_given_as_candidate():
    data = make_measurements(modulation="QPSK", n_symbols=64, sps=8,
                              D1=-5000.0, D2=-5750.0, snr_db=25.0, rng_seed=0)
    ratio = data["D2"] / data["D1"]
    result = convergence_sanity_sweep(data["I1"], data["I2"], D1_candidates=[5000.0], D_ratio=ratio)
    assert result["converged"] is True
    assert result["best_final_error"] < 0.1
    print(f"PASS: convergence_sanity_sweep finds true D (final_error={result['best_final_error']:.4f})")


def test_convergence_sweep_narrow_basin_finding():
    """Regression test for this module's key empirical finding: a candidate
    within 4% of the true D should NOT converge under this signal model."""
    data = make_measurements(modulation="QPSK", n_symbols=64, sps=8,
                              D1=-5000.0, D2=-5750.0, snr_db=25.0, rng_seed=0)
    ratio = data["D2"] / data["D1"]
    result = convergence_sanity_sweep(data["I1"], data["I2"], D1_candidates=[4800.0, 5200.0], D_ratio=ratio)
    assert result["converged"] is False, (
        "Expected candidates 4% off the true D to fail -- if this now passes, the narrow-basin "
        "finding documented in real_data_loader.py's module docstring may no longer hold and "
        "that documentation should be revisited."
    )
    print(f"PASS: +-4%-off candidates correctly fail to converge (narrow-basin finding holds)")


def test_default_candidates_includes_raw_value_scaled():
    candidates = default_candidates(D_raw=-600.0)
    assert -600.0 in candidates or 600.0 in candidates
    assert -6000.0 in candidates or 6000.0 in candidates
    print("PASS: default_candidates includes scaled versions of the raw hint")


def test_load_or_synthesize_falls_back_loudly():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data, report = load_or_synthesize(path=None, modulation="QPSK", rng_seed=0)
    assert data["source_path"] == "SYNTHETIC (dgs.gs_core.make_measurements)"
    assert len(caught) >= 1
    assert "SYNTHETIC" in str(caught[0].message)
    print("PASS: load_or_synthesize falls back to synthetic data with an explicit warning")


def test_real_sample_data_correctly_flagged_not_recoverable():
    """Regression test for this module's key real-data finding: the shipped
    sample_data/*.mat files should NOT be reported as recoverable under any
    of the default candidates, given the investigation in the module docstring."""
    path = SAMPLE_DIR / "QPSK_56GSas_D600_D1200.mat"
    if not path.exists():
        print("SKIP: sample_data not present")
        return
    data = load_mat_measurement(path)
    report = diagnose_measurement(data["I1"], data["I2"],
                                   D_ratio=data["D2_raw"] / data["D1_raw"], D_raw=data["D1_raw"])
    assert report["recoverable"] is False, (
        "sample_data/QPSK_56GSas_D600_D1200.mat unexpectedly converged -- if this now passes, "
        "the data-generation mismatch documented in real_data_loader.py's module docstring may "
        "have been resolved and that documentation should be updated, not this test silently "
        "left describing a stale finding."
    )
    print("PASS: real sample_data file correctly flagged as not recoverable under default candidates")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")
