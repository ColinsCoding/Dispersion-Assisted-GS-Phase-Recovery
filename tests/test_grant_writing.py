import os
import json
import tempfile
import pytest
from dgs.grant_writing import (
    ProposalSection, ProposalOutline,
    GRANT_WRITING_RULES, probability_of_funding,
)


def test_section_word_count():
    s = ProposalSection("Test", "hello world foo bar", word_limit=10)
    assert s.word_count() == 4


def test_section_within_limit():
    s = ProposalSection("Test", "a b c", word_limit=5)
    assert s.is_within_limit() is True


def test_section_over_limit():
    s = ProposalSection("Test", "a b c d e f", word_limit=3)
    assert s.is_within_limit() is False


def test_section_format_contains_title():
    s = ProposalSection("Significance", "some content", word_limit=100)
    fmt = s.format()
    assert "SIGNIFICANCE" in fmt


def test_outline_sbir_has_five_sections():
    outline = ProposalOutline.sbir_phase1_gs_receiver()
    assert len(outline.sections) == 5


def test_outline_sections_nonempty():
    outline = ProposalOutline.sbir_phase1_gs_receiver()
    for key, section in outline.sections.items():
        assert section.word_count() > 50, f"Section '{key}' too short"


def test_outline_save_and_reload_json():
    outline = ProposalOutline.sbir_phase1_gs_receiver()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        outline.save_json(path)
        outline2 = ProposalOutline.load_json(path)
        assert len(outline2.sections) == len(outline.sections)
        assert outline2.title == outline.title
    finally:
        os.unlink(path)


def test_outline_save_text():
    outline = ProposalOutline.sbir_phase1_gs_receiver()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        path = f.name
    try:
        outline.save(path)
        assert os.path.getsize(path) > 100
    finally:
        os.unlink(path)


def test_grant_rules_count():
    assert len(GRANT_WRITING_RULES) == 7


def test_grant_rules_have_title_and_detail():
    for rule, detail in GRANT_WRITING_RULES:
        assert len(rule) > 5
        assert len(detail) > 20


def test_probability_of_funding_sbir():
    r = probability_of_funding(0.15, 1)
    assert r["p_at_least_one_award"] == pytest.approx(0.15)


def test_probability_of_funding_increases():
    r1 = probability_of_funding(0.15, 1)
    r10 = probability_of_funding(0.15, 10)
    assert r10["p_at_least_one_award"] > r1["p_at_least_one_award"]


def test_probability_of_funding_expected_submissions():
    r = probability_of_funding(0.20, 5)
    assert r["expected_submissions_to_win"] == pytest.approx(5.0)
