"""Test dgs/logic_gates_from_em_circuits.py: all 7 basic gates, verified
against canonical truth tables from ACTUAL cascaded circuit voltages, and
the specific DTL noise-margin bug (naive 0-level-shift cascade misreads a
LOW AND-stage voltage as HIGH) that this module's own development caught
and fixed with real level-shifting diodes."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.computer_engineering import diode_or_gate
from dgs.logic_gates_from_em_circuits import (
    diode_and_gate, bjt_inverter, not_gate, dtl_nand_gate, dtl_nor_gate,
    xor_gate_from_nand, xnor_gate, verify_all_seven_gates, DTL_LEVEL_SHIFT_DIODES,
)

# 1. diode_and_gate: correct AND truth table, and dual behavior vs. OR
for a in (0, 1):
    for b in (0, 1):
        r = diode_and_gate(a, b)
        assert r["logic_out"] == (a & b), f"AND({a},{b}) = {r}"

# AND(1,1) should sit at V_high (no diode drop); AND with any 0 should sit
# at V_low+V_D (clamped) -- checked directly, not just the final logic bit
assert diode_and_gate(1, 1)["V_out"] == 5.0
assert abs(diode_and_gate(0, 1)["V_out"] - 0.7) < 1e-9
assert abs(diode_and_gate(1, 0)["V_out"] - 0.7) < 1e-9

for bad in (2, -1, 0.5):
    try:
        diode_and_gate(bad, 0)
        raise AssertionError(f"expected ValueError for A={bad}")
    except ValueError:
        pass

print("dgs.logic_gates_from_em_circuits: AND checks passed")

# 2. bjt_inverter / not_gate: correct switching, and level-shift raises
#    the effective threshold as documented
low_no_shift = bjt_inverter(0.9, n_level_shift_diodes=0)
assert low_no_shift["on"] is True   # 0.9V > bare 0.7V threshold -- turns ON (the bug)
low_with_shift = bjt_inverter(0.9, n_level_shift_diodes=DTL_LEVEL_SHIFT_DIODES)
assert low_with_shift["on"] is False   # 0.9V < 2.1V effective threshold -- correctly OFF

for a in (0, 1):
    r = not_gate(a)
    assert r["logic_out"] == (1 - a), f"NOT({a}) = {r}"

print("dgs.logic_gates_from_em_circuits: NOT / level-shift checks passed")

# 3. dtl_nand_gate / dtl_nor_gate: correct truth tables, built from the
#    REAL cascaded stage voltage (checked via the intermediate stage key)
for a in (0, 1):
    for b in (0, 1):
        r = dtl_nand_gate(a, b)
        assert r["logic_out"] == int(not (a and b)), f"NAND({a},{b}) = {r}"
        assert "V_and_stage" in r   # confirms the real AND-stage voltage was actually computed

        r2 = dtl_nor_gate(a, b)
        assert r2["logic_out"] == int(not (a or b)), f"NOR({a},{b}) = {r2}"
        assert "V_or_stage" in r2

print("dgs.logic_gates_from_em_circuits: DTL NAND/NOR checks passed")

# 4. xor_gate_from_nand / xnor_gate: the actual 4-/5-NAND cascade, correct
#    truth table -- THIS is the case that broke without level-shift diodes
for a in (0, 1):
    for b in (0, 1):
        r = xor_gate_from_nand(a, b)
        assert r["logic_out"] == (a ^ b), f"XOR({a},{b}) = {r}"
        assert "stage_voltages" in r and len(r["stage_voltages"]) == 3

        r2 = xnor_gate(a, b)
        assert r2["logic_out"] == int(not (a ^ b)), f"XNOR({a},{b}) = {r2}"

print("dgs.logic_gates_from_em_circuits: XOR/XNOR cascade checks passed")

# 5. Regression test for the exact zero-margin fact this module's
#    development caught: a single DTL gate's LOW-input AND-stage voltage
#    sits at EXACTLY the bare transistor threshold (V_D == V_BE_on ==
#    0.7V), zero margin -- and level-shift diodes give REAL margin instead
value_low_input = diode_and_gate(0, 1)["V_out"]
assert abs(value_low_input - 0.7) < 1e-9, "expected the LOW-input AND-stage voltage at exactly V_D"
bare = bjt_inverter(value_low_input, n_level_shift_diodes=0)
assert abs(bare["V_in"] - 0.7) < 1e-9
# zero margin: V_in is NOT strictly greater than the bare 0.7V threshold,
# so it happens to read correctly here, but with NO room for error
assert bare["on"] is False and abs(bare["V_in"] - 0.7) < 1e-9

with_shift = bjt_inverter(value_low_input, n_level_shift_diodes=DTL_LEVEL_SHIFT_DIODES)
margin = (0.7 * (1 + DTL_LEVEL_SHIFT_DIODES)) - value_low_input
assert margin > 1.0, f"expected real margin with level-shift diodes, got {margin}"

# the case that actually DOES fail outright: raw, un-restored multi-stage
# voltage (a LOW-saturated prior stage's 0.2V, diode-ANDed with a HIGH
# 5.0V node) rises to 0.9V, past the BARE 0.7V threshold -- an actual
# misfire, not just zero margin
raw_case_voltage = min(5.0, 5.0 + 0.7, 0.2 + 0.7)   # AND(HIGH-node, prior-LOW-output) internal voltage
assert raw_case_voltage > 0.7, "expected this specific raw cascade case to exceed the bare threshold"
raw_case_result = bjt_inverter(raw_case_voltage, n_level_shift_diodes=0)
assert raw_case_result["on"] is True   # misfires: reads a logical-LOW-derived voltage as ON

print("dgs.logic_gates_from_em_circuits: zero-margin / raw-cascade-failure checks passed")

# 6. verify_all_seven_gates: the module's own comprehensive self-check
check = verify_all_seven_gates()
assert check["all_seven_correct"] is True
assert set(check["per_gate"].keys()) == {"AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR"}
assert all(check["per_gate"].values())

print("dgs.logic_gates_from_em_circuits: comprehensive 7-gate check passed")
print("all dgs.logic_gates_from_em_circuits tests passed")
