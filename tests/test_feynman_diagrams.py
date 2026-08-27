import numpy as np
import pytest
from dgs.feynman_diagrams import (
    PROCESSES, energy_conservation_check, diagram_order,
    phase_matching_condition, detector_response,
    measurement_vs_memory_table, feynman_sympy_5,
)


def test_processes_registry_has_expected_entries():
    for name in ["SHG", "SFG", "DFG", "THG", "FWM", "SRS"]:
        assert name in PROCESSES
        assert "order" in PROCESSES[name]
        assert "legs" in PROCESSES[name]


def test_diagram_order_chi2_vs_chi3():
    assert diagram_order("SHG") == 2
    assert diagram_order("FWM") == 3
    assert diagram_order("THG") == 3


def test_energy_conservation_shg_balanced():
    res = energy_conservation_check("SHG", {"omega": 1.5})
    assert res["conserved"] is True
    assert abs(res["energy_change"]) < 1e-9


def test_energy_conservation_fwm_balanced():
    res = energy_conservation_check("FWM", {"omega_p": 1.0, "omega_s": 1.3, "omega_i": 0.7})
    assert res["conserved"] is True


def test_energy_conservation_fwm_unbalanced_raises_flag():
    # signal+idler don't sum to 2*pump -- physically invalid combo
    res = energy_conservation_check("FWM", {"omega_p": 1.0, "omega_s": 1.5, "omega_i": 0.7})
    assert res["conserved"] is False
    assert abs(res["energy_change"]) > 1e-9


def test_energy_conservation_unknown_process_raises():
    with pytest.raises(ValueError):
        energy_conservation_check("NOT_A_PROCESS", {"omega": 1.0})


def test_energy_conservation_thg():
    res = energy_conservation_check("THG", {"omega": 0.8})
    assert res["conserved"] is True


def test_energy_conservation_srs():
    res = energy_conservation_check(
        "SRS", {"omega_p": 1.0, "omega_s": 1.0})  # degenerate case, no Stokes shift
    assert res["conserved"] is True


def test_phase_matching_condition_is_equation():
    import sympy as sp
    eq = phase_matching_condition("FWM")
    assert isinstance(eq, sp.Eq)


def test_phase_matching_condition_leg_count_matches_process():
    eq = phase_matching_condition("SHG")
    n_legs = len(PROCESSES["SHG"]["legs"])
    free_k_symbols = [s for s in eq.free_symbols if str(s).startswith('k')]
    assert len(free_k_symbols) == n_legs


def test_detector_response_intensity_is_modulus_squared():
    E = [3 + 4j]
    dr = detector_response(E)
    assert dr["intensity_measured"][0] == pytest.approx(25.0)


def test_detector_response_phase_matches_angle():
    E = [1j]  # pure imaginary -> phase = pi/2
    dr = detector_response(E)
    assert dr["phase_discarded"][0] == pytest.approx(np.pi / 2)


def test_detector_response_zero_field():
    dr = detector_response([0 + 0j])
    assert dr["intensity_measured"][0] == 0.0


def test_measurement_vs_memory_table_structure():
    table = measurement_vs_memory_table()
    assert "Photodetector" in table
    assert "GS algorithm" in table
    assert "common_lesson" in table
    for key in ["Photodetector", "FWM vertex", "GS algorithm"]:
        assert "changes" in table[key]
        assert "measured/remembered" in table[key]


def test_feynman_sympy_5_count_and_type():
    import sympy as sp
    eqs = feynman_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
