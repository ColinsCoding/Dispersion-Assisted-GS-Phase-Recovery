"""Tests for the C89 + SQLite lab tracker: bounds, clearance logic, eyewear OD vs laser_eyewear.py."""
import sqlite3

import pytest

from dgs import labtrack as T
from dgs import laser_eyewear as LS
from dgs.eventsim import ToolUnavailable


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    try:
        return T.fresh_db(tmp_path_factory.mktemp("lab") / "lab.db")
    except ToolUnavailable:
        pytest.skip("gcc not installed")


def test_report_sections(db):
    out = T.report(db, "2026-10-29")
    for head in ("standards by country", "Journal club", "Bench clearance", "Eyewear check"):
        assert head in out
    assert "https://doi.org/10.1364/AO.21.002758" in out and "IEC 60825-1" in out


def test_clearance_logic(db):
    out = T.report(db, "2026-10-29")
    lines = {l.split("  ")[0].strip(): l for l in out.splitlines() if "CLEARED" in l}
    assert lines["Avery (student)"].endswith("CLEARED")
    assert lines["Blake (grad)"].endswith("CLEARED")                 # renewal counts (MAX expires)
    assert "site-specific" in lines["Casey (staff)"]
    assert "laser safety" in lines["Devon (student)"]
    later = T.report(db, "2027-09-05")                               # Avery's laser training lapsed
    assert any(l.startswith("Avery") and "NOT CLEARED: laser safety" in l for l in later.splitlines())


def test_eyewear_od_matches_python_laser_safety(db):
    rows = {r["label"]: r for r in T.eyewear_rows(T.report(db, "2026-10-29"))}
    mpe = LS.mpe_visible(LS.BLINK_S)
    for label, P, d_mm in [("OD2 green goggles", 0.005, 1.0), ("OD4 red goggles", 0.020, 0.8)]:
        E = LS.irradiance(P, 0.0, d_mm / 10, 1e-3)
        assert rows[label]["E_W_cm2"] == pytest.approx(E, rel=1e-3)
        assert rows[label]["need_od"] == pytest.approx(LS.required_od(E, mpe), abs=0.01)
    assert rows["OD2 green goggles"]["verdict"].startswith("FAIL: optical density")
    assert rows["OD4 red goggles"]["verdict"] == "OK"
    assert "rated band" in rows["OD2 green goggles (wrong use)"]["verdict"]
    assert rows["OD5 IR goggles"]["verdict"].startswith("UNCHECKED")


@pytest.mark.parametrize("sql", [
    "INSERT INTO laser (name, wavelength_nm, power_w, beam_diameter_mm, hazard_class) VALUES ('uv', 100, 1, 1, '4')",
    "INSERT INTO laser (name, wavelength_nm, power_w, beam_diameter_mm, hazard_class) VALUES ('neg', 532, -1, 1, '2')",
    "INSERT INTO laser (name, wavelength_nm, power_w, beam_diameter_mm, hazard_class) VALUES ('cls', 532, 1, 1, '5')",
    "INSERT INTO eyewear (label, laser_id, od, lo_nm, hi_nm) VALUES ('x', 1, 11, 500, 560)",
    "INSERT INTO eyewear (label, laser_id, od, lo_nm, hi_nm) VALUES ('x', 1, 3, 560, 500)",
    "INSERT INTO eyewear (label, laser_id, od, lo_nm, hi_nm) VALUES ('x', 99, 3, 500, 560)",
    "INSERT INTO standard (country, body, document, scope, url) VALUES ('X', 'Y', 'Z', 's', 'http://insecure')",
    "INSERT INTO journal_club (held_on, presenter_id, title, doi_url, topic) VALUES ('2026-12-01', 1, 't', 'https://example.com', 'computing')",
    "INSERT INTO training (member_id, topic, completed, expires) VALUES (1, 'laser safety', '2026-10-01', '2026-09-01')",
    "INSERT INTO training (member_id, topic, completed, expires) VALUES (1, 'laser safety', 'Oct 1', '2027-10-01')",
    "INSERT INTO member (name, role) VALUES ('Zed', 'intern')",
])
def test_bounds_reject_bad_rows(db, sql):
    con = sqlite3.connect(db)
    con.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(sql)
    con.close()


def test_c_prepared_statement_bounds_and_injection(db):
    rc, _, err = T.run(db, "add-laser", "uv", 100, 1, 1, "4")
    assert rc == 3 and "CHECK constraint failed" in err
    evil = "x'); DROP TABLE laser; --"
    rc, out, _ = T.run(db, "add-laser", evil, 532, 0.001, 1, "2")
    assert rc == 0
    con = sqlite3.connect(db)
    assert con.execute("SELECT name FROM laser WHERE name = ?", (evil,)).fetchone()[0] == evil
    con.close()
