"""Tests for dgs/phys106_frq.py: answers match hand-worked values, the grader catches mistakes."""
import pytest
import sympy as sp

from dgs import phys106_frq as F

# Worked by hand (calculator, hc = 1240 eV nm, hbar c = 197.3 eV nm, mc^2 = 0.511 MeV).
HAND = {
    ("time_dilation", "a"): "3.203", ("time_dilation", "b"): "8.33e-8 s", ("time_dilation", "c"): "23.7 m",
    ("velocity_addition", "a"): "0.9155 c",
    ("relativistic_energy", "a"): "1.533", ("relativistic_energy", "b"): "0.758 c",
    ("relativistic_energy", "c"): "1090 MeV/c",
    ("photoelectric", "a"): "4.00 eV", ("photoelectric", "b"): "1.70 eV",
    ("photoelectric", "c"): "1.70 V", ("photoelectric", "d"): "539 nm",
    ("de_broglie", "a"): "150 eV", ("de_broglie", "b"): "0.100 nm",
    ("hydrogen", "a"): "2.55 eV", ("hydrogen", "b"): "486 nm", ("hydrogen", "c"): "3.40 eV",
    ("particle_in_box", "a"): "1.50 eV", ("particle_in_box", "b"): "275 nm", ("particle_in_box", "c"): "0.609",
    ("tunneling", "a"): "0.138 nm", ("tunneling", "b"): "3.0e-3", ("tunneling", "c"): "3.0e-3",
    ("uncertainty", "a"): "5.8e5 m/s",
    ("half_life", "a"): "0.772", ("half_life", "b"): "26.6 days", ("half_life", "c"): "1.0e-6 s^-1",
}


def test_every_numeric_part_has_a_hand_check():
    numeric = {(p.pid, q.label) for p in F.PROBLEMS.values() for q in p.parts if q.numeric}
    assert numeric == set(HAND)


@pytest.mark.parametrize("key", sorted(HAND))
def test_answer_matches_hand_value(key):
    res = F.grade(*key, HAND[key])
    assert res["correct"], (key, F.PROBLEMS[key[0]].part(key[1]).answer, res)


def test_every_part_has_a_solution_and_conceptual_parts_have_key_ideas():
    for p in F.PROBLEMS.values():
        assert p.setup and p.parts
        for q in p.parts:
            assert q.solution, (p.pid, q.label)
            assert q.numeric or q.key_ideas, (p.pid, q.label)


@pytest.mark.parametrize("pid, label, wrong, phrase", [
    ("velocity_addition", "a", "1.3 c", "Galilean"),
    ("photoelectric", "a", "1.28e-21 eV", "hc/lambda"),
    ("photoelectric", "b", "6.3 eV", "subtract"),
    ("time_dilation", "b", "8.12e-9 s", "SLOW"),
    ("relativistic_energy", "b", "1.03 c", "Newtonian"),
    ("de_broglie", "b", "8.27 nm", "photons only"),
    ("particle_in_box", "b", "825 nm", "DIFFERENCE"),
    ("tunneling", "b", "0.055", "-2w/eta"),
    ("half_life", "a", "0.813", "exponential"),
])
def test_common_mistakes_get_specific_feedback(pid, label, wrong, phrase):
    res = F.grade(pid, label, wrong)
    assert not res["correct"] and phrase in res["feedback"], res


def test_units_are_converted_and_checked():
    assert F.grade("photoelectric", "d", "5.39e-7 m")["correct"]          # nm vs m
    assert F.grade("time_dilation", "b", "83.3 ns")["correct"]
    assert F.grade("time_dilation", "c", "0.0237 km")["correct"]
    assert F.grade("photoelectric", "b", "2.72e-19 J")["correct"]          # eV vs J
    assert "Wrong units" in F.grade("photoelectric", "b", "1.70 nm")["feedback"]
    assert "unknown unit" in F.grade("photoelectric", "b", "1.7 bananas")["feedback"]
    assert F.grade("photoelectric", "b", "")["feedback"] == "No answer yet."


def test_partial_credit_and_conceptual():
    close = F.grade("photoelectric", "b", "1.80 eV")                       # ~6% off
    assert not close["correct"] and close["points"] == 1
    assert F.grade("photoelectric", "e", "anything")["points"] == 0
    score, best, rows = F.grade_problem("hydrogen", {"a": "2.55 eV", "b": "486 nm", "c": "3.4 eV"})
    assert score == best == 6 and len(rows) == 3


def test_quantity_parser():
    assert float(F.quantity("3")) == 3.0
    assert F.quantity("2 x 10^-3 s") == F.quantity("2e-3 s") == F.quantity("2×10^-3 s")
    with pytest.raises(ValueError):
        F.quantity("eV 3")


def test_practice_exam():
    ex = F.practice_exam(3, seed=1)
    assert len(ex) == 3 and ex == F.practice_exam(3, seed=1)
    assert all(p.chapter == 40 for p in F.practice_exam(2, chapters=[40]))
    with pytest.raises(ValueError):
        F.practice_exam(5, chapters=[40])
    with pytest.raises(KeyError):
        F.grade("nope", "a", "1")
