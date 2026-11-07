"""Tests for dgs.circuit_techniques_ch4 (Nilsson & Riedel Ch. 4 on an R-2R DAC)."""
import numpy as np
import pytest
import sympy as sp

from dgs import circuit_techniques_ch4 as c
from dgs import circuit_terminology as ct

LSB = 3.3 / 256


def test_ideal_ladder_is_exact_for_all_codes():
    assert np.allclose(c.transfer_curve(), c.ideal_output(np.arange(256)), atol=1e-12)


def test_thevenin_resistance_is_R_for_every_code_and_norton_agrees():
    for code in (0, 1, 77, 128, 255):
        els, n, out = c.r2r_netlist(code)
        v, r, i_n = c.thevenin(els, n, out)
        assert np.isclose(r, 10e3)
        assert np.isclose(v, c.ideal_output(code))
        assert np.isclose(i_n, v / r, atol=1e-15)


def test_superposition_sums_to_full_output():
    for code in (0b10110011, 255, 1, 0b01010101):
        parts = c.superposition_bits(code)
        assert np.isclose(sum(parts), c.dac_output(code))
        for k, p in enumerate(parts):
            if (code >> k) & 1:
                assert np.isclose(p, 3.3 * 2 ** k / 256)


def test_source_transformation_and_max_power():
    v, r = 2.0, 10e3
    i_n, r_n = c.thevenin_to_norton(v, r)
    assert c.norton_to_thevenin(i_n, r_n) == (v, r)
    RL = np.logspace(2, 6, 2001)
    P = c.load_power(v, r, RL)
    assert np.isclose(RL[np.argmax(P)], r, rtol=0.01)
    assert np.isclose(P.max(), v ** 2 / (4 * r), rtol=1e-4)


def test_node_and_mesh_methods_agree_on_a_2bit_ladder():
    # 2-bit ladder, code 0b11 (both legs to 1 V): series R=1, legs 2R=2, terminator 2
    els = [("Vr", "ref", "0", "V", 1), ("R0", "n1", "ref", "R", 2), ("R1", "n2", "ref", "R", 2),
           ("Rs", "n1", "n2", "R", 1), ("Rt", "n1", "0", "R", 2)]
    _, cur_n, _ = ct.node_voltage_solution(els)
    _, cur_m = ct.mesh_current_solution(els)
    for k in cur_n:
        assert sp.simplify(cur_n[k] - cur_m[k]) == 0
    # tie to the numeric MNA ladder: the terminator branch carries V(n1) / 2R
    e2, n2, _ = c.r2r_netlist(3, n_bits=2, R=1.0, v_ref=1.0)
    v_n1 = c.node_voltage(e2, n2, 1)
    assert any(np.isclose(abs(float(i)), v_n1 / 2) for i in cur_n.values())
    assert np.isclose(c.dac_output(3, n_bits=2, R=1.0, v_ref=1.0), 0.75)


def test_buffer_dependent_source():
    v, ideal = c.buffered_output(200, gain=1e5)
    assert abs(v - ideal) < 0.01 * LSB
    v10, _ = c.buffered_output(200, gain=10, r_out=50, r_load=1e3)
    beta = 1e3 / 1050
    assert np.isclose(v10, ideal * 10 * beta / (1 + 10 * beta))


def test_tolerance_monte_carlo_trends():
    d1, i1, m1 = c.monte_carlo_tolerance(0.001, 60)
    d5, i5, m5 = c.monte_carlo_tolerance(0.05, 60)
    assert np.median(d1) < 0.2 and m1 == 1.0
    assert np.median(d5) > 1 and m5 < 0.6


def test_ratiometric_tempco_and_johnson_noise():
    common = c.tempco_errors(8, 100, 50)
    assert abs(c.dac_output(200, errors=common) - c.dac_output(200)) < 1e-12
    assert np.isclose(c.johnson_noise_rms(10e3, 1e6), np.sqrt(4 * c.K_B * 300 * 1e4 * 1e6))
    assert c.johnson_noise_rms(10e3, 1e6) < 1e-3 * LSB


def test_bench_function_generator_and_scope():
    assert np.isclose(c.fgen_into_load(1.0, 50.0), 1.0)
    assert np.isclose(c.fgen_into_load(1.0, 1e6), 2.0, rtol=1e-4)
    assert np.isclose(c.scope_loading_error(10e3) * 256, 2.53, rtol=0.01)
    f3 = 1 / (2 * np.pi * 50 * 15e-12)
    assert np.isclose(c.scope_frequency_response([f3])[0], 1 / np.sqrt(2), rtol=1e-3)


@pytest.mark.parametrize("call", [
    lambda: c.r2r_netlist(256),
    lambda: c.r2r_netlist(1, n_bits=0),
    lambda: c.r2r_netlist(1, errors=[0.0]),
    lambda: c.monte_carlo_tolerance(0.7),
    lambda: c.fgen_into_load(1.0, 0.0),
    lambda: c.johnson_noise_rms(1e3, 0.0),
    lambda: c.buffered_output(1, gain=0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
