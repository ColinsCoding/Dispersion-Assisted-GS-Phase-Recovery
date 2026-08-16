"""Build notebooks/reading_the_problem.ipynb

Intro-to-programming pedagogy: reading a problem statement and extracting
GIVENS/FIND/CONSTRAINTS is its own explicit step, done before any code,
demonstrated across four worked examples that deliberately mix CS and
physics problems -- showing the methodology transfers across domains.

Lighter template than this repo's research notebooks (this is intro
teaching content, not a research derivation): Theory -> Four Worked
Examples (read -> extract -> code -> verify, each) -> Why This Matters.

Engine: dgs/reading_the_problem.py (this session).
"""
import json, pathlib

cells = []

def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in src.splitlines()]})
def code(src): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s+"\n" for s in src.splitlines()]})

# ── Title ─────────────────────────────────────────────────────────────────────
md("""# Reading the Problem: The Step Before Writing Any Code

The most common intro-programming mistake isn't a syntax error -- it's
starting to type before actually reading the problem. This notebook makes
that step EXPLICIT and inspectable: for each problem, GIVENS / FIND /
CONSTRAINTS are extracted and printed BEFORE any code runs. Four examples,
deliberately mixing CS and physics problems, to show the same methodology
works regardless of subject. Engine: `dgs/reading_the_problem.py`.
""")

code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path('..').resolve()))

from dgs.reading_the_problem import (
    TWO_SUM, PROJECTILE_RANGE, TEMPS_ABOVE_FREEZING, OHMS_LAW_SAFETY_FILTER,
    print_reading_breakdown,
)
print("Setup complete.")
""")

# ── Example 1 ─────────────────────────────────────────────────────────────────
md("""## Example 1 (CS): Two Sum

Read the raw text FIRST. Notice it asks for INDICES, not the two numbers
themselves -- a detail easy to miss if you skim straight to coding.
""")

code("""print(TWO_SUM.raw_text)
""")

code("""print("GIVENS:     ", TWO_SUM.givens)
print("FIND:       ", TWO_SUM.find)
print("CONSTRAINTS:", TWO_SUM.constraints)
""")

code("""result = TWO_SUM.run()
print("RESULT:", result)
nums = TWO_SUM.givens['nums']
print(f"Check: nums[{result[0]}] + nums[{result[1]}] = {nums[result[0]]} + {nums[result[1]]} = {nums[result[0]]+nums[result[1]]}")
""")

# ── Example 2 ─────────────────────────────────────────────────────────────────
md("""## Example 2 (Physics): Projectile Range

A physics word problem has the SAME structure: given values, a target
quantity, and implicit assumptions (flat ground, no air resistance) that
are easy to gloss over but change the formula if violated.
""")

code("""print(PROJECTILE_RANGE.raw_text)
print()
print("GIVENS:     ", PROJECTILE_RANGE.givens)
print("FIND:       ", PROJECTILE_RANGE.find)
print("CONSTRAINTS:", PROJECTILE_RANGE.constraints)
print()
print("RESULT:", PROJECTILE_RANGE.run(), "meters")
""")

# ── Example 3 ─────────────────────────────────────────────────────────────────
md("""## Example 3 (Mixed CS + Physics): Temperatures Above Freezing

A unit-conversion trap hides in this one: "above freezing (0 Celsius)"
means the F->C conversion happens FIRST, and "above" is strict -- exactly
32F (0C) does NOT count. Reading the problem catches this before it
becomes an off-by-one-style bug.
""")

code("""print(TEMPS_ABOVE_FREEZING.raw_text)
print()
print("GIVENS:     ", TEMPS_ABOVE_FREEZING.givens)
print("FIND:       ", TEMPS_ABOVE_FREEZING.find)
print("CONSTRAINTS:", TEMPS_ABOVE_FREEZING.constraints)
print()
temps = TEMPS_ABOVE_FREEZING.givens['temps_f']
celsius = [(t-32)*5/9 for t in temps]
for f, c in zip(temps, celsius):
    print(f"  {f:5.1f} F -> {c:6.2f} C  {'(above freezing)' if c>0 else '(NOT above freezing)'}")
print()
print("RESULT:", TEMPS_ABOVE_FREEZING.run())
""")

# ── Example 4 ─────────────────────────────────────────────────────────────────
md("""## Example 4 (Mixed CS + Physics): Ohm's Law Safety Filter

Same trap as example 3, different domain: "exceed the safe current limit"
is strict. One of the five readings sits EXACTLY at the limit
(5V / 10ohm = 0.5A) -- reading the problem is what tells you that reading
should NOT be flagged.
""")

code("""print(OHMS_LAW_SAFETY_FILTER.raw_text)
print()
print("GIVENS:     ", OHMS_LAW_SAFETY_FILTER.givens)
print("FIND:       ", OHMS_LAW_SAFETY_FILTER.find)
print("CONSTRAINTS:", OHMS_LAW_SAFETY_FILTER.constraints)
print()
V = OHMS_LAW_SAFETY_FILTER.givens['voltages_V']
R = OHMS_LAW_SAFETY_FILTER.givens['resistance_ohm']
limit = OHMS_LAW_SAFETY_FILTER.givens['max_current_A']
for i, v in enumerate(V):
    I = v/R
    flag = "EXCEEDS limit" if I > limit else ("at the limit exactly -- NOT flagged" if I == limit else "safe")
    print(f"  index {i}: V={v}V -> I={I:.2f}A  ({flag})")
print()
print("RESULT (flagged indices):", OHMS_LAW_SAFETY_FILTER.run())
""")

# ── Why this matters ──────────────────────────────────────────────────────────
md("""## Why This Matters

Every one of these four problems has a detail that only shows up if you
actually read the CONSTRAINTS before coding: Two Sum wants indices, not
values; the projectile problem assumes flat ground; the temperature
problem's "above" is strict; the Ohm's law problem's "exceed" is strict.
None of these are hard to fix once noticed -- but if you start writing
code from the FIND alone, skipping the constraints, each one becomes a
bug you discover later instead of a detail you handled up front.

This is the same discipline this repo's own research notebooks follow at
a much larger scale: every one of them states a theory section, derives
it, and VERIFIES the result against something independent before trusting
it -- "read the problem" is that same habit, at the very first step.
""")

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
out = pathlib.Path("notebooks/reading_the_problem.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written: {out}  ({len(cells)} cells)")
