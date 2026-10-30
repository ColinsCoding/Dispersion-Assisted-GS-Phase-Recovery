"""Tests for the C89 FRQ bot: answers are right, grading catches wrong ones, the timer bites."""
import math

import pytest
import sympy as sp

from dgs import frqbot as B
from dgs.eventsim import ToolUnavailable


@pytest.fixture(scope="module")
def questions():
    try:
        return B.bank()
    except ToolUnavailable:
        pytest.skip("gcc not installed")


def test_bank_covers_every_subject(questions):
    subjects = {s for s, _ in questions}
    assert subjects == set(B.SUBJECTS) and len(questions) == 16
    for _, prompt in questions:
        B.model_answer(prompt)                      # every question has a model answer


def test_model_answers_score_full_marks(questions):
    results, (score, asked) = B.drill(B.model_answer, seed=11)
    assert asked == 16 and score == 16, [r for r in results if r["verdict"] != "CORRECT"]


def test_wrong_answers_score_zero(questions):
    _, (score, asked) = B.drill(lambda p: "I am not sure", seed=2)
    assert score == 0 and asked == 16


def test_subject_filter_and_count(questions):
    results, (_, asked) = B.drill(B.model_answer, subject="econ", n=2, seed=5)
    assert asked == 2 and all(r["subject"] == "econ" for r in results)
    with pytest.raises(ValueError):
        B.drill(B.model_answer, subject="history")


def test_partial_keywords_explain_what_is_missing(questions):
    results, _ = B.drill(lambda p: "price goes up" if "Demand increases" in p else "x", subject="econ", seed=1)
    r = next(r for r in results if "Demand increases" in r["prompt"])
    assert r["verdict"] == "NOT YET" and "[quantity]" in r["detail"]


def test_timer_scores_late_answers_zero(questions):
    results, (score, _) = B.drill(B.model_answer, subject="comptia", n=1, limit=1, seed=1, delay_s=2.2)
    assert results[0]["verdict"] == "TIME" and score == 0


def test_numeric_answers_independently():
    P = sp.Symbol("P")
    assert sp.solve(sp.Eq(100 - 2 * P, 20 + 2 * P), P) == [20]               # econ equilibrium
    assert (100 - 2 * 15) - (20 + 2 * 15) == 20                              # ceiling shortage
    q = math.sqrt(0.16); assert 2 * (1 - q) * q == pytest.approx(0.48)       # Hardy-Weinberg
    assert -math.log10(1.0e-3) == pytest.approx(3.0)                         # pH
    assert 36.0 / 18.02 == pytest.approx(2.0, rel=0.01)                      # moles
    assert math.log10(50 / 2.5e-3) == pytest.approx(4.30, abs=0.01)          # OD
    assert 2 ** (32 - 26) - 2 == 62                                          # /26 hosts
    assert 7 // 2 * 2 + 7 % 2 == 7                                           # C integer arithmetic
