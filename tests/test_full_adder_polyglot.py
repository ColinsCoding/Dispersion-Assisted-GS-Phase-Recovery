"""Test dgs/full_adder_polyglot.py: the full adder already defined in
dgs.computer_engineering.full_adder, cross-validated EXHAUSTIVELY (all 8
input combinations) against a compiled C binary and a GHDL-simulated VHDL
entity. Requires gcc on PATH and ghdl (winget install ghdl.ghdl.ucrt64.mcode)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tempfile
from dgs.full_adder_polyglot import (
    ALL_INPUTS, run_python, compile_c, run_c, run_vhdl,
    cross_validate_languages, GHDL_DEFAULT,
)

# 1. run_python: matches the known full-adder truth table exactly (hand
#    derived: S is odd parity of A+B+Cin, Cout is majority of A,B,Cin)
expected_truth_table = {
    (0, 0, 0): (0, 0), (0, 0, 1): (1, 0), (0, 1, 0): (1, 0), (0, 1, 1): (0, 1),
    (1, 0, 0): (1, 0), (1, 0, 1): (0, 1), (1, 1, 0): (0, 1), (1, 1, 1): (1, 1),
}
py_table = run_python()
assert py_table == expected_truth_table, f"got {py_table}"
assert set(py_table.keys()) == set(ALL_INPUTS)

# 2. compile_c / run_c: matches the same truth table independently
with tempfile.TemporaryDirectory() as tmp:
    exe_c = compile_c(tmp)
    c_table = run_c(exe_c)
assert c_table == expected_truth_table, f"C table: {c_table}"

# 3. run_vhdl: matches the same truth table independently -- the actual
#    HDL simulation, not a stand-in
with tempfile.TemporaryDirectory() as tmp:
    vhdl_table = run_vhdl(tmp)
assert vhdl_table == expected_truth_table, f"VHDL table: {vhdl_table}"

# 4. cross_validate_languages: full three-language run, zero mismatches,
#    all 8 combinations present in every language's table
with tempfile.TemporaryDirectory() as tmp:
    report = cross_validate_languages(tmp)
assert report["all_agree"] is True
assert report["mismatches"] == []
for lang in ("python", "c", "vhdl"):
    assert set(report["results"][lang].keys()) == set(ALL_INPUTS), f"{lang} missing combinations"

# 5. run_vhdl: a bad ghdl_path must raise a clear error, not a cryptic one
try:
    with tempfile.TemporaryDirectory() as tmp:
        run_vhdl(tmp, ghdl_path=r"C:\nonexistent\ghdl.exe")
    raise AssertionError("expected RuntimeError for a missing ghdl executable")
except RuntimeError as e:
    assert "ghdl not found" in str(e)

# 6. cross_validate_languages: run_c_lang=False / run_vhdl_lang=False must
#    skip cleanly and still agree (Python-only)
with tempfile.TemporaryDirectory() as tmp:
    py_only = cross_validate_languages(tmp, run_c_lang=False, run_vhdl_lang=False)
assert py_only["all_agree"] is True
assert set(py_only["results"].keys()) == {"python"}

# 7. sanity: Cout really is the majority function of (A, B, Cin) for
#    every combination -- an independent identity check, not just
#    "matches the hardcoded table"
for a, b, cin in ALL_INPUTS:
    s, cout = expected_truth_table[(a, b, cin)]
    majority = 1 if (a + b + cin) >= 2 else 0
    assert cout == majority, f"Cout({a},{b},{cin})={cout} != majority={majority}"
    parity = (a + b + cin) % 2
    assert s == parity, f"S({a},{b},{cin})={s} != odd-parity={parity}"

print("all dgs.full_adder_polyglot tests passed")
