"""Tests for dgs/lean_magic.py: cell assembly, error mapping, sorry detection, real compile."""
import pytest

from dgs import lean_magic as LM


def session(**cells):
    s = LM.LeanSession()
    for name, src in cells.items():
        s.add(name, src)
    return s


def test_assemble_line_map():
    s = session(header="import Mathlib.Tactic.Ring\n", a="theorem a : 1 = 1 := rfl\n\n", b="theorem b : 2 = 2 := rfl")
    text, spans = s.assemble()
    assert text.count("\n") == 3
    assert spans == [(1, 1, "header"), (2, 2, "a"), (3, 3, "b")]
    assert s.locate(spans, 3) == ("b", 1)
    assert s.theorems() == ["a", "b"]


def test_rerun_replaces_cell():
    s = session(a="theorem a : 1 = 1 := rfl")
    s.add("a", "theorem a2 : 1 = 1 := rfl")
    assert s.theorems() == ["a2"] and len(s.cells) == 1


def test_parse_maps_messages_to_cells():
    s = session(header="import X\nimport Y\n", proof="theorem t : 1 = 2 := by\n  norm_num\n")
    _, spans = s.assemble()
    out = ("/tmp/NotebookCells.lean:4:2: error: unsolved goals\n⊢ False\n"
           "/tmp/NotebookCells.lean:1:0: warning(lint.unused): something")
    msgs = s.parse(out, spans)
    assert (msgs[0].cell, msgs[0].line, msgs[0].severity) == ("proof", 2, "error")
    assert "⊢ False" in msgs[0].text
    assert (msgs[1].cell, msgs[1].severity) == ("header", "warning")


def test_bad_name_rejected():
    with pytest.raises(ValueError):
        LM.LeanSession().add("has space", "x")


def test_unverified_without_cells():
    assert LM.LeanSession().check().status == "UNVERIFIED"


requires_lean = pytest.mark.skipif(LM.find_lake() is None or LM.find_project() is None,
                                   reason="lake or a Mathlib Lake project not available")


@requires_lean
def test_real_compile_good_bad_and_sorry():
    s = session(header="import Mathlib.Tactic.NormNum\n",
                good="theorem good : (2 : ℕ) + 2 = 4 := by norm_num\n",
                bad="theorem bad : (2 : ℕ) + 2 = 5 := by\n  norm_num\n",
                lazy="theorem lazy : (1 : ℕ) = 1 := by\n  sorry\n")
    res = s.check(timeout=1800)
    assert res.status == "FAILED"
    cells = {m.cell for m in res.messages}
    assert "bad" in cells and "lazy" in cells and "good" not in cells
    s.cells.pop("bad")
    res2 = s.check(timeout=1800)
    assert res2.status == "FAILED" and res2.returncode == 0       # exit 0, but sorry is not proof
    s.cells.pop("lazy")
    assert s.check(timeout=1800).status == "VERIFIED"
