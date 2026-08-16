"""reading_the_problem.py -- intro-to-programming pedagogy: READING a
problem statement and extracting GIVENS / FIND / CONSTRAINTS is its own
explicit step, done BEFORE any code is written, not skipped past on the
way to syntax. Four worked examples, mixing CS and physics problems
deliberately, to show the same methodology transfers across domains --
this repo's usual throughline (Griffiths math <-> ML foundations, Maxwell
<-> circuits, etc.), here applied one level down, to the skill of
reading a problem in the first place.

Each Problem below carries its own GIVENS/FIND/CONSTRAINTS as actual data
(not just prose in a comment) so the extraction step is a real, inspectable
artifact -- something a beginner can print and check against the raw
problem text -- separate from the solve() function that comes after it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any


@dataclass
class Problem:
    """The explicit READ-THE-PROBLEM artifact: raw text, then what was
    extracted from it, kept separate from the eventual code."""
    name: str
    raw_text: str
    givens: Dict[str, Any]
    find: str
    constraints: List[str]
    solve: Callable[..., Any]

    def run(self) -> Any:
        """Call solve() with the extracted givens as keyword arguments --
        the actual translation from 'what was read' to 'what runs'."""
        return self.solve(**self.givens)


# ── Example 1 (CS): Two Sum -- a classic intro-programming problem ──────────

TWO_SUM = Problem(
    name="two_sum",
    raw_text=(
        "Given a list of integers `nums` and an integer `target`, return "
        "the indices of the two numbers that add up to `target`. Assume "
        "exactly one solution exists, and you may not use the same "
        "element twice."
    ),
    givens={"nums": [2, 7, 11, 15], "target": 9},
    find="the pair of INDICES (not values) whose nums sum to target",
    constraints=["exactly one solution exists", "cannot reuse the same element twice"],
    solve=lambda nums, target: next(
        (i, j) for i in range(len(nums)) for j in range(i + 1, len(nums))
        if nums[i] + nums[j] == target
    ),
)


# ── Example 2 (Physics): projectile range -- a classic intro word problem ───

def _projectile_range(v0_ms: float, angle_deg: float, g_ms2: float = 9.8) -> float:
    import math
    theta = math.radians(angle_deg)
    return v0_ms ** 2 * math.sin(2 * theta) / g_ms2


PROJECTILE_RANGE = Problem(
    name="projectile_range",
    raw_text=(
        "A ball is launched at 20 m/s at an angle of 30 degrees above "
        "horizontal, on flat ground, with g=9.8 m/s^2. Find the horizontal "
        "range (how far it travels before landing)."
    ),
    givens={"v0_ms": 20.0, "angle_deg": 30.0, "g_ms2": 9.8},
    find="the horizontal range R (a single number, in meters)",
    constraints=["flat ground (launch height = landing height)", "no air resistance"],
    solve=_projectile_range,
)


# ── Example 3 (mixed CS+physics): temperatures above freezing ───────────────

def _count_above_freezing(temps_f: List[float]) -> int:
    celsius = [(t - 32) * 5.0 / 9.0 for t in temps_f]
    return sum(1 for c in celsius if c > 0.0)


TEMPS_ABOVE_FREEZING = Problem(
    name="temps_above_freezing",
    raw_text=(
        "Given a list of daily temperatures in Fahrenheit, count how many "
        "days were above freezing (0 Celsius)."
    ),
    givens={"temps_f": [28.0, 33.0, 32.0, 45.0, 10.0, 50.5]},
    find="a COUNT (single integer), not the converted list itself",
    constraints=["freezing is 0C, i.e. exactly 32F -- strictly ABOVE, not at or above"],
    solve=_count_above_freezing,
)


# ── Example 4 (mixed CS+physics): Ohm's law safety filter ───────────────────

def _overcurrent_indices(voltages_V: List[float], resistance_ohm: float,
                          max_current_A: float) -> List[int]:
    currents = [v / resistance_ohm for v in voltages_V]
    return [i for i, I in enumerate(currents) if I > max_current_A]


OHMS_LAW_SAFETY_FILTER = Problem(
    name="ohms_law_safety_filter",
    raw_text=(
        "A fixed resistor of 10 ohms is tested at several supply voltages. "
        "Given the list of voltages and a maximum safe current of 0.5 A, "
        "find the INDICES of the voltage readings that would exceed the "
        "safe current limit (V=IR)."
    ),
    givens={"voltages_V": [2.0, 4.0, 5.0, 6.0, 3.0], "resistance_ohm": 10.0,
            "max_current_A": 0.5},
    find="the INDICES (not the voltages or currents themselves) that violate the limit",
    constraints=["resistance is fixed across all readings", "strictly EXCEEDS the limit, not equals it"],
    solve=_overcurrent_indices,
)


ALL_PROBLEMS = [TWO_SUM, PROJECTILE_RANGE, TEMPS_ABOVE_FREEZING, OHMS_LAW_SAFETY_FILTER]


def print_reading_breakdown(problem: Problem) -> None:
    """Print the explicit read-the-problem breakdown BEFORE running
    solve() -- the actual pedagogical point: this step is visible and
    separate, not skipped."""
    print(f"=== {problem.name} ===")
    print(f"RAW TEXT: {problem.raw_text}")
    print(f"GIVENS:      {problem.givens}")
    print(f"FIND:        {problem.find}")
    print(f"CONSTRAINTS: {problem.constraints}")
    result = problem.run()
    print(f"RESULT:      {result}\n")


if __name__ == "__main__":
    for p in ALL_PROBLEMS:
        print_reading_breakdown(p)
