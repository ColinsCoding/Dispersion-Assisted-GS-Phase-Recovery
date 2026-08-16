"""Test dgs/expr_filter_sympy_torch.py's stdin/stdout math-expression
filter: parsing, symbolic simplification, and the SymPy-vs-Torch
derivative cross-check that is the actual point of using both libraries.
Requires py -3.12 (torch is py-3.12 only in this repo, not 3.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.expr_filter_sympy_torch import (
    parse_input_lines, evaluate_expression, format_result, run_filter, DEFAULT_VALUE,
)

# 1. parse_input_lines: numeric assignments recognized, expressions passed through
parsed = parse_input_lines(["x = 2", "x**2 + 1", "", "# a comment", "y == 3"])
assert parsed[0] == ("x = 2", ("x", 2.0))
assert parsed[1] == ("x**2 + 1", None)
assert len(parsed) == 3, "blank line and comment should be skipped"
assert parsed[2] == ("y == 3", None), "'==' must not be misparsed as an assignment"

# 2. evaluate_expression: a known RC-circuit value, V0*(1-exp(-t/(R*C)))
#    at t = one time constant (t=R*C) should give V0*(1-1/e)
result = evaluate_expression("V0*(1-exp(-t/(R*C)))",
                              {"R": 1000.0, "C": 1e-6, "V0": 5.0, "t": 1e-3})
import math
expected = 5.0 * (1 - math.exp(-1))
assert abs(result["value"] - expected) < 1e-9, f"expected {expected}, got {result['value']}"

# 3. evaluate_expression: SymPy and Torch derivatives must match for every
#    free symbol -- this is the actual point of the module
for name in result["free_symbols"]:
    assert result["grad_match"][name] is True, (
        f"SymPy/Torch derivative mismatch for {name}: "
        f"sympy={result['sympy_grad'][name]} torch={result['torch_grad'][name]}")

# 4. evaluate_expression: unassigned symbols default to DEFAULT_VALUE, not
#    silently to something else
result2 = evaluate_expression("x**2 + y", {})
assert result2["substitution_values"]["x"] == DEFAULT_VALUE
assert result2["substitution_values"]["y"] == DEFAULT_VALUE
assert abs(result2["value"] - (DEFAULT_VALUE**2 + DEFAULT_VALUE)) < 1e-9

# 5. evaluate_expression: an expression with no free symbols evaluates directly
result3 = evaluate_expression("2 + 3*4", {})
assert result3["value"] == 14.0
assert result3["free_symbols"] == []

# 6. evaluate_expression: a genuinely bad expression returns an error dict,
#    not a raised exception -- a text-editor filter must not crash the pipe
result_bad = evaluate_expression("x +* 2", {})
assert "error" in result_bad

# 7. format_result: must not raise on either a normal or an error result
assert "ERROR" in format_result(result_bad)
assert "simplified" in format_result(result)

# 8. run_filter: a full multi-line block, assignments applying to later
#    (not earlier) expressions
output = run_filter(["a = 3", "a**2", "a = 10", "a**2"])
assert "value:      9" in output, "first a**2 should use a=3 (=9), not a later assignment"
assert "value:      100" in output, "second a**2 should use the updated a=10 (=100)"

print("all dgs.expr_filter_sympy_torch tests passed")
