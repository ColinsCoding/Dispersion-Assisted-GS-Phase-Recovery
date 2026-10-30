"""Tests for dgs/modern_physics_vocab.py: completeness, dimensional consistency, decisions."""
import pytest

from dgs import modern_physics_vocab as V


@pytest.mark.parametrize("term", [t for t in V.VOCAB if t.equation], ids=lambda t: t.term)
def test_every_equation_is_dimensionally_consistent(term):
    assert V.check_dimensions(term)


def test_entries_complete_and_unique():
    names = [t.term.lower() for t in V.VOCAB]
    assert len(names) == len(set(names)), "duplicate terms"
    for t in V.VOCAB:
        assert t.chapter in V.CHAPTERS
        assert t.english and t.science and t.decide, t.term
    for ch in (36, 37, 38, 39, 40, 41, 42):
        assert len(V.by_chapter(ch)) >= 5, ch


@pytest.mark.parametrize("eq, units", [
    ("Eq(E, h*lam)", {"E": "J", "h": "J*s", "lam": "m"}),                 # the Midterm 1 mistake
    ("Eq(Ptun, exp(-2*w))", {"Ptun": "1", "w": "m"}),                      # units inside exp
    ("Eq(v, Ef + B)", {"v": "m/s", "Ef": "V/m", "B": "T"}),               # adding E and B
    ("Eq(L, L0*gamma - v)", {"L": "m", "L0": "m", "gamma": "1", "v": "m/s"}),
    ("Eq(K, m*v)", {"K": "J", "m": "kg", "v": "m/s"}),                     # mass symbol m vs meter
])
def test_dimension_errors_are_caught(eq, units):
    with pytest.raises(ValueError):
        V.check_dimensions(V.Term("planted", 0, "", "", "", eq, units))


def test_missing_units_reported():
    with pytest.raises(ValueError):
        V.check_dimensions(V.Term("x", 0, "", "", "", "Eq(E, h*f)", {"E": "J", "h": "J*s"}))


@pytest.mark.parametrize("question, expected_given", [
    ("A photon has a wavelength of 178 nm. Find its energy in eV.", "photon wavelength"),
    ("Light hits a metal and the stopping potential is 5.7 V. Find the work function.", "light on a metal, stopping potential"),
    ("Two eruptions are simultaneous on Earth. Are they simultaneous for the moving observer?", "events simultaneous in one frame"),
    ("A rock moves toward the ship; find the velocity of the rock relative to the ship.", "two velocities in different frames"),
    ("Use the energy diagram to find the emitted wavelengths of the atom.", "an atom's energy levels or a diagram"),
    ("A charge passes undeflected through crossed electric and magnetic fields.", "a charge in crossed E and B fields"),
    ("A radioactive sample has a half-life of 5 days. How much remains?", "half-life and elapsed time"),
    ("An electron with energy 2 eV meets a barrier 0.5 nm wide. Find the tunneling probability.", "E below a barrier of width w"),
])
def test_decide_routes_midterm_style_questions(question, expected_given):
    ranked = V.decide(question)
    assert ranked and ranked[0][1]["given"] == expected_given, [r["given"] for _, r in ranked]


def test_lookup_flashcards_markdown():
    assert V.lookup("half-life")[0].chapter == 42
    with pytest.raises(KeyError):
        V.lookup("flux capacitor")
    with pytest.raises(ValueError):
        V.by_chapter(99)
    cards = V.flashcards(chapter=40, n=5, seed=1)
    assert len(cards) == 5 and all(c.chapter == 40 for c in cards)
    assert V.flashcards(n=5, seed=1) == V.flashcards(n=5, seed=1)
    md = V.to_markdown()
    assert "## Ch. 40" in md and "Decisions" in md and md.count("|---|") >= 8
