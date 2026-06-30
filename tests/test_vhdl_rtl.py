"""Tests for dgs/vhdl_rtl.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.vhdl_rtl import (
    std_logic_resolve, std_logic_and, std_logic_or, std_logic_not,
    slv_to_int, int_to_slv,
    build_full_adder_block, DFlipFlop, Register, ALU,
    build_traffic_light_fsm, build_sequence_detector_fsm,
    rtl_pipeline_sympy, work_energy_sympy,
)
import sympy as sp


def test_std_logic_resolution():
    assert std_logic_resolve("0", "1") == "X"
    assert std_logic_resolve("1", "Z") == "1"
    assert std_logic_resolve("0", "Z") == "0"
    assert std_logic_resolve("Z", "Z") == "Z"


def test_std_logic_gates():
    assert std_logic_and("X", "0") == "0"   # dominating 0
    assert std_logic_and("1", "1") == "1"
    assert std_logic_or("1", "X") == "1"    # dominating 1
    assert std_logic_not("0") == "1"
    assert std_logic_not("1") == "0"
    assert std_logic_not("X") == "X"


def test_slv_roundtrip():
    assert slv_to_int("1010") == 10
    assert slv_to_int("1010", signed=True) == -6
    assert int_to_slv(-6, 4) == "1010"
    assert int_to_slv(10, 8) == "00001010"


def test_full_adder_all_cases():
    fa = build_full_adder_block()
    for a in (0, 1):
        for b in (0, 1):
            for cin in (0, 1):
                out = fa.evaluate(A=a, B=b, Cin=cin)
                expected_sum = (a + b + cin) % 2
                expected_cout = (a + b + cin) // 2
                assert out["Sum"] == expected_sum
                assert out["Cout"] == expected_cout


def test_dff_basic():
    dff = DFlipFlop(init=0)
    trace = dff.simulate([1, 0, 1, 1])
    assert trace == [1, 0, 1, 1]


def test_dff_synchronous_reset():
    dff = DFlipFlop(init=0)
    trace = dff.simulate([1, 1, 1], rst_seq=[0, 1, 0])
    assert trace[1] == 0   # reset takes effect on cycle 2


def test_register_load_and_reset():
    reg = Register(8)
    reg.load(200)
    assert reg.read() == 200
    reg.load(0, rst=1)
    assert reg.read() == 0


def test_alu_add():
    alu = ALU(8)
    res, flags = alu.execute(10, 3, 0b000)
    assert res == 13
    assert flags["Z"] == 0


def test_alu_sub_negative():
    alu = ALU(8)
    res, flags = alu.execute(3, 5, 0b001)
    assert flags["N"] == 1   # result is negative


def test_alu_and():
    alu = ALU(8)
    res, _ = alu.execute(0b11001100, 0b10101010, 0b010)
    assert res == (0b11001100 & 0b10101010)


def test_traffic_light_cycle():
    fsm = build_traffic_light_fsm()
    # Three transitions should cycle green->yellow->red->green
    out = fsm.run([1, 1, 1])
    assert out == ["yellow", "red", "green"]


def test_sequence_detector_101():
    det = build_sequence_detector_fsm()
    det.reset()
    out = det.run([1, 0, 1])
    assert out[-1] == 1, "Should detect '101'"


def test_rtl_pipeline_symbolic():
    pipe = rtl_pipeline_sympy(3)
    # Latency should be 3 * T_clock
    assert pipe["n_stages"] == 3
    T = pipe["T_clock"]
    lat = pipe["latency"]
    assert sp.simplify(lat - 3 * T) == 0


def test_work_energy_symbols():
    we = work_energy_sympy()
    # KE = (1/2)*m*v^2
    m, v = sp.symbols("m v", positive=True)
    assert sp.simplify(we["KE"] - sp.Rational(1, 2) * m * v**2) == 0
    # Power = F*v
    F, v2 = sp.symbols("F v", positive=True)
    assert "power_inst" in we
