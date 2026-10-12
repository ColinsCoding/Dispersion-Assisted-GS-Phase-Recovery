"""Tests for the perf_counter stage timer and the end-to-end CLI outputs."""

import json
import time

import pytest

from crack_photonics import cli
from crack_photonics.timing import (
    REQUIRED_TIMING_KEYS,
    STAGE_NAMES,
    TOTAL_KEY,
    StageTimer,
    format_report,
)

EXPECTED_KEYS = {
    "image_loading_s",
    "preprocessing_s",
    "symmetry_analysis_s",
    "edge_detection_s",
    "metric_extraction_s",
    "total_runtime_s",
}


# -------------------------------------------------------------- timing keys
def test_required_keys_constant_matches_specification():
    assert set(REQUIRED_TIMING_KEYS) == EXPECTED_KEYS


def test_report_has_all_required_keys_even_if_no_stage_ran():
    report = StageTimer().report()
    assert EXPECTED_KEYS <= set(report)
    for name in STAGE_NAMES:
        assert report[f"{name}_s"] == 0.0


def test_report_values_are_non_negative_floats():
    timer = StageTimer()
    for name in STAGE_NAMES:
        with timer.stage(name):
            sum(range(1000))
    report = timer.report()
    for key in EXPECTED_KEYS | {"untimed_overhead_s"}:
        assert isinstance(report[key], float)
        assert report[key] >= 0.0
    assert report["clock"] == "time.perf_counter"


# ------------------------------------------------------------ timer behaviour
def test_stage_measures_elapsed_time():
    timer = StageTimer()
    with timer.stage("preprocessing"):
        time.sleep(0.05)
    timer.stop()
    report = timer.report()
    assert 0.04 <= report["preprocessing_s"] < 1.0
    assert report[TOTAL_KEY] >= report["preprocessing_s"]


def test_total_is_at_least_sum_of_stages():
    timer = StageTimer()
    for name in STAGE_NAMES:
        with timer.stage(name):
            time.sleep(0.002)
    time.sleep(0.005)  # untimed work between stages
    timer.stop()
    report = timer.report()
    stage_sum = sum(report[f"{name}_s"] for name in STAGE_NAMES)
    assert report[TOTAL_KEY] >= stage_sum
    assert report["untimed_overhead_s"] == pytest.approx(report[TOTAL_KEY] - stage_sum, abs=1e-9)


def test_repeated_stage_accumulates():
    timer = StageTimer()
    with timer.stage("edge_detection"):
        time.sleep(0.01)
    first = timer.report()["edge_detection_s"]
    with timer.stage("edge_detection"):
        time.sleep(0.01)
    assert timer.report()["edge_detection_s"] > first


def test_stop_freezes_total():
    timer = StageTimer()
    timer.stop()
    frozen = timer.report()[TOTAL_KEY]
    time.sleep(0.01)
    assert timer.report()[TOTAL_KEY] == frozen


def test_exception_inside_stage_propagates_and_time_is_kept():
    timer = StageTimer()
    with pytest.raises(RuntimeError, match="boom"):
        with timer.stage("metric_extraction"):
            time.sleep(0.005)
            raise RuntimeError("boom")
    assert timer.report()["metric_extraction_s"] > 0.0


def test_stage_after_stop_is_an_error():
    timer = StageTimer()
    timer.stop()
    with pytest.raises(RuntimeError, match="already stopped"):
        with timer.stage("preprocessing"):
            pass


def test_empty_stage_name_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        with StageTimer().stage(""):
            pass


def test_format_report_lists_every_stage():
    text = format_report(StageTimer().report())
    for name in STAGE_NAMES:
        assert name in text
    assert "total_runtime" in text


def test_format_report_rejects_incomplete_report():
    with pytest.raises(KeyError, match="missing required keys"):
        format_report({"total_runtime_s": 1.0})


# ------------------------------------------------------- end-to-end CLI run
def test_cli_headless_run_writes_all_outputs(tmp_path, capsys):
    input_path = tmp_path / "data" / "example.png"
    output_dir = tmp_path / "results" / "example"

    status = cli.main([
        "--input", str(input_path),
        "--output", str(output_dir),
        "--headless", "--make-example",
    ])
    assert status == 0

    for name in cli.OUTPUT_FILES:
        path = output_dir / name
        assert path.is_file(), f"missing output {name}"
        assert path.stat().st_size > 0

    timing = json.loads((output_dir / "timing.json").read_text())
    assert EXPECTED_KEYS <= set(timing)
    assert all(timing[key] > 0.0 for key in EXPECTED_KEYS)
    assert timing["total_runtime_s"] >= sum(timing[f"{name}_s"] for name in STAGE_NAMES)

    metrics = json.loads((output_dir / "metrics.json").read_text())
    crack = metrics["crack"]
    for key in ("crack_area_percent", "crack_length_px", "mean_width_px",
                "orientation_deg", "connected_components", "confidence"):
        assert key in crack
    assert set(metrics["symmetry"]) == {"horizontal", "vertical"}

    printed = capsys.readouterr().out
    assert "total_runtime" in printed


def test_cli_missing_input_reports_error_and_status_2(tmp_path, capsys):
    status = cli.main([
        "--input", str(tmp_path / "nope.png"),
        "--output", str(tmp_path / "out"),
        "--headless",
    ])
    assert status == 2
    assert "ERROR: Input image not found" in capsys.readouterr().err
    assert not (tmp_path / "out" / "metrics.json").exists()


def test_cli_results_are_reproducible(tmp_path):
    """Two runs on the same seed give byte-identical metrics."""
    outputs = []
    for run_name in ("a", "b"):
        input_path = tmp_path / run_name / "example.png"
        output_dir = tmp_path / run_name / "out"
        assert cli.main(["--input", str(input_path), "--output", str(output_dir),
                         "--headless", "--make-example", "--seed", "0"]) == 0
        metrics = json.loads((output_dir / "metrics.json").read_text())
        metrics.pop("input")  # the only field that depends on the folder name
        outputs.append((metrics, (output_dir / "crack_mask.png").read_bytes()))
    assert outputs[0] == outputs[1]
