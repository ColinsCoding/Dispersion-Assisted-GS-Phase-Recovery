import shutil
import pytest
from dgs.voltage_divider_vhdl import (
    voltage_divider_output, run_vhdl_voltage_divider, cross_validate_voltage_divider,
    VOLTAGE_DIVIDER_VHDL_TEMPLATE,
)

GHDL_AVAILABLE = shutil.which("ghdl") is not None
requires_ghdl = pytest.mark.skipif(not GHDL_AVAILABLE, reason="ghdl not found on PATH")


def test_voltage_divider_output_basic():
    assert voltage_divider_output(5.0, 1000.0, 2000.0) == pytest.approx(3.333333, abs=1e-5)


def test_voltage_divider_output_equal_resistors_halves_voltage():
    assert voltage_divider_output(12.0, 100.0, 100.0) == pytest.approx(6.0)


def test_voltage_divider_output_zero_r1_passes_full_voltage():
    assert voltage_divider_output(5.0, 0.0, 1000.0) == pytest.approx(5.0)


def test_voltage_divider_output_rejects_negative_resistance():
    with pytest.raises(ValueError):
        voltage_divider_output(5.0, -100.0, 1000.0)


def test_voltage_divider_output_rejects_zero_total_resistance():
    with pytest.raises(ValueError):
        voltage_divider_output(5.0, 0.0, 0.0)


def test_vhdl_template_formats_without_error():
    source = VOLTAGE_DIVIDER_VHDL_TEMPLATE.format(v_in=5.0, r1=1000.0, r2=2000.0)
    assert "entity voltage_divider_tb" in source
    assert "5.0" in source and "1000.0" in source and "2000.0" in source


@requires_ghdl
def test_run_vhdl_voltage_divider_matches_python():
    python_result = voltage_divider_output(5.0, 1000.0, 2000.0)
    vhdl_result = run_vhdl_voltage_divider(5.0, 1000.0, 2000.0)
    assert vhdl_result == pytest.approx(python_result, abs=1e-4)


@requires_ghdl
def test_cross_validate_voltage_divider_agrees():
    result = cross_validate_voltage_divider(3.3, 470.0, 1000.0)
    assert result["agree"] is True
    assert result["diff"] < 1e-4


@requires_ghdl
def test_run_vhdl_voltage_divider_bad_ghdl_path_raises():
    with pytest.raises(FileNotFoundError):
        run_vhdl_voltage_divider(5.0, 1000.0, 2000.0, ghdl_path="definitely_not_a_real_binary")
