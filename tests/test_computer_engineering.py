"""Tests for dgs/computer_engineering.py"""
import numpy as np
from dgs.computer_engineering import (
    carrier_concentrations, diode_iv, diode_small_signal,
    diode_or_gate, majority_gate, minority_gate,
    half_adder, full_adder, ripple_carry_adder,
    transmission_line, pcb_trace_impedance, ground_bounce,
    wave_equation_pcb_sympy, SI_NI,
)


def test_mass_action_ntype():
    r = carrier_concentrations(N_D=1e16)
    assert r["type"] == "n-type"
    np.testing.assert_allclose(r["n_times_p"], SI_NI**2, rtol=1e-6)
    assert r["n_cm3"] > r["p_cm3"]   # majority > minority


def test_mass_action_ptype():
    r = carrier_concentrations(N_A=1e17)
    assert r["type"] == "p-type"
    np.testing.assert_allclose(r["n_times_p"], SI_NI**2, rtol=1e-6)
    assert r["p_cm3"] > r["n_cm3"]


def test_diode_iv_forward():
    I = diode_iv(0.6, I_s=1e-12)
    assert I > 0.01   # significant forward current


def test_diode_iv_reverse():
    I = diode_iv(-5.0, I_s=1e-12)
    np.testing.assert_allclose(I, -1e-12, rtol=0.01)


def test_diode_small_signal():
    r = diode_small_signal(0.65)
    assert r["r_d_ohm"] < 100   # forward-biased: low dynamic resistance
    assert r["gd_S"] > 0


def test_diode_or_gate():
    # truth table: OR logic
    combos = [(0,0,0), (0,1,1), (1,0,1), (1,1,1)]
    for A, B, expected in combos:
        r = diode_or_gate(A, B)
        assert r["logic_out"] == expected, f"OR({A},{B}) expected {expected}"


def test_majority_gate():
    assert majority_gate(0,0,0) == 0
    assert majority_gate(1,0,0) == 0
    assert majority_gate(1,1,0) == 1
    assert majority_gate(1,1,1) == 1


def test_minority_gate_complement():
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                assert minority_gate(a,b,c) == 1 - majority_gate(a,b,c)


def test_half_adder():
    cases = [(0,0,0,0),(0,1,1,0),(1,0,1,0),(1,1,0,1)]
    for A, B, S, C in cases:
        r = half_adder(A, B)
        assert r["S"] == S and r["Cout"] == C, f"HA({A},{B}) wrong"


def test_full_adder():
    # all 8 combinations
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                r = full_adder(a, b, c)
                total = a + b + c
                assert r["S"] == total % 2
                assert r["Cout"] == total // 2


def test_ripple_carry_adder():
    # 6 + 7 = 13 in 4 bits
    r = ripple_carry_adder([0,1,1,0], [0,1,1,1])
    assert r["sum_int"] == 13
    assert r["carry_out"] == 0

    # 15 + 1 = 16 -> sum_bits=0000, carry_out=1
    r2 = ripple_carry_adder([1,1,1,1], [0,0,0,1])
    assert r2["sum_int"] == 16


def test_transmission_line_matched():
    r = transmission_line(Z0=50, Z_L=50)
    np.testing.assert_allclose(r["Gamma"], 0.0, atol=1e-10)
    assert bool(r["matched"]) is True


def test_transmission_line_open():
    r = transmission_line(Z0=50, Z_L=1e9)
    np.testing.assert_allclose(r["Gamma"], 1.0, atol=0.001)
    assert bool(r["matched"]) is False


def test_pcb_trace_impedance():
    r = pcb_trace_impedance(w_mm=1.0, h_mm=1.6, er=4.5)
    assert 20 < r["Z0_ohm"] < 150   # physical range


def test_ground_bounce():
    gb = ground_bounce(n_outputs=8, I_per_output_A=0.02, L_pkg_nH=3, t_rise_ns=0.5)
    assert gb["V_bounce_mV"] > 0
    assert gb["C_decoupling_nF"] > 0


def test_wave_equation_sympy():
    import sympy as sp
    r = wave_equation_pcb_sympy()
    assert sp.simplify(r["residual_at_dispersion"]) == 0


if __name__ == "__main__":
    import sys
    print("Running computer_engineering tests...")
    fns = [v for k,v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
