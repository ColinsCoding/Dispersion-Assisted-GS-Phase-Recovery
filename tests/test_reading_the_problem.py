"""Test dgs/reading_the_problem.py's four worked examples: each Problem's
extracted givens/find/constraints must actually produce the correct
result when run, and the constraints (especially the strict-inequality
edge cases) must be honored exactly, not approximately."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.reading_the_problem import (
    Problem, TWO_SUM, PROJECTILE_RANGE, TEMPS_ABOVE_FREEZING,
    OHMS_LAW_SAFETY_FILTER, ALL_PROBLEMS,
)

# 1. Two Sum: known answer, and the constraint "cannot reuse the same
#    element twice" must actually hold (i != j)
i, j = TWO_SUM.run()
assert i != j, "two_sum must not reuse the same index twice"
assert TWO_SUM.givens["nums"][i] + TWO_SUM.givens["nums"][j] == TWO_SUM.givens["target"]
assert (i, j) == (0, 1)

# 2. Projectile range: known closed-form value, R = v0^2*sin(2*theta)/g
import math
v0, angle, g = PROJECTILE_RANGE.givens["v0_ms"], PROJECTILE_RANGE.givens["angle_deg"], PROJECTILE_RANGE.givens["g_ms2"]
expected_R = v0 ** 2 * math.sin(2 * math.radians(angle)) / g
result_R = PROJECTILE_RANGE.run()
assert abs(result_R - expected_R) < 1e-9
assert abs(result_R - 35.348) < 0.01

# 3. Temperatures above freezing: exactly 3 of the 6 given temperatures
#    are strictly above 32F -- and the "at exactly 32F" edge case (not
#    counted) is the actual point of this example, checked explicitly
result_count = TEMPS_ABOVE_FREEZING.run()
assert result_count == 3
temps = TEMPS_ABOVE_FREEZING.givens["temps_f"]
assert 32.0 in temps, "the test data must include the exactly-32F edge case"
celsius_at_32F = (32.0 - 32) * 5.0 / 9.0
assert celsius_at_32F == 0.0, "32F must convert to exactly 0C for the edge case to be meaningful"

# 4. Ohm's law safety filter: only the reading that STRICTLY exceeds
#    0.5A is flagged -- the reading at EXACTLY 0.5A (5V/10ohm) must NOT
#    be included, the actual point of this example's constraint
result_indices = OHMS_LAW_SAFETY_FILTER.run()
voltages = OHMS_LAW_SAFETY_FILTER.givens["voltages_V"]
R = OHMS_LAW_SAFETY_FILTER.givens["resistance_ohm"]
limit = OHMS_LAW_SAFETY_FILTER.givens["max_current_A"]
assert result_indices == [3]
exactly_at_limit_idx = voltages.index(5.0)
assert voltages[exactly_at_limit_idx] / R == limit, "5V/10ohm must equal the limit exactly for this edge case to be meaningful"
assert exactly_at_limit_idx not in result_indices, "a reading exactly AT the limit must not be flagged (constraint says STRICTLY exceeds)"

# 5. Every problem in ALL_PROBLEMS must actually run without error and
#    have non-empty givens/find/constraints (the structure itself must be
#    complete, not just individually-correct answers)
assert len(ALL_PROBLEMS) == 4
for p in ALL_PROBLEMS:
    assert isinstance(p, Problem)
    assert p.raw_text.strip() != ""
    assert len(p.givens) > 0
    assert p.find.strip() != ""
    assert len(p.constraints) > 0
    p.run()  # must not raise

print("all dgs.reading_the_problem tests passed")
