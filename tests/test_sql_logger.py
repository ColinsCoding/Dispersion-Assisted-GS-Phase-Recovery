import json
import pytest
from dgs.sql_logger import ExperimentLogger, demo_sql_queries


def make_db():
    return ExperimentLogger(":memory:")


def test_insert_and_count():
    db = make_db()
    db.insert("gs_tsdft", {"D": -5000}, result=0.01)
    assert db.count() == 1
    db.close()


def test_count_by_module():
    db = make_db()
    db.insert("gs_tsdft", {"D": -5000}, result=0.01)
    db.insert("beat_freq", {"f1": 440}, result=3.0)
    assert db.count("gs_tsdft") == 1
    assert db.count("beat_freq") == 1
    db.close()


def test_all_runs_returns_list():
    db = make_db()
    db.insert("test", {"x": 1}, result=0.5)
    runs = db.all_runs()
    assert len(runs) == 1
    assert runs[0]["module"] == "test"
    db.close()


def test_all_runs_filter_by_module():
    db = make_db()
    db.insert("A", {}, result=1.0)
    db.insert("B", {}, result=2.0)
    runs = db.all_runs("A")
    assert len(runs) == 1
    assert runs[0]["module"] == "A"
    db.close()


def test_params_json_roundtrip():
    db = make_db()
    params = {"D_ps2": -5000, "n_iter": 60, "signal": "BPSK"}
    db.insert("gs_tsdft", params, result=0.001)
    runs = db.all_runs()
    recovered = json.loads(runs[0]["params_json"])
    assert recovered == params
    db.close()


def test_best_n_minimize():
    db = make_db()
    for err in [0.5, 0.1, 0.9, 0.2, 0.05]:
        db.insert("test", {}, result=err)
    best = db.best_n("test", minimize=True, n=2)
    assert best[0]["result"] <= best[1]["result"]
    assert best[0]["result"] == pytest.approx(0.05)
    db.close()


def test_best_n_maximize():
    db = make_db()
    for val in [10, 30, 20, 50, 40]:
        db.insert("snr", {}, result=float(val))
    best = db.best_n("snr", minimize=False, n=1)
    assert best[0]["result"] == pytest.approx(50.0)
    db.close()


def test_summary_aggregates():
    db = make_db()
    for v in [1.0, 2.0, 3.0]:
        db.insert("mod_a", {}, result=v)
    db.insert("mod_b", {}, result=9.0)
    summary = db.summary()
    mod_a = next(r for r in summary if r["module"] == "mod_a")
    assert mod_a["n_runs"] == 3
    assert mod_a["mean_result"] == pytest.approx(2.0)
    db.close()


def test_raw_query():
    db = make_db()
    db.insert("gs_tsdft", {"D": -5000}, result=0.001)
    db.insert("gs_tsdft", {"D": -500}, result=0.9)
    rows = db.query("SELECT * FROM runs WHERE result < 0.01")
    assert len(rows) == 1
    db.close()


def test_delete_run():
    db = make_db()
    rid = db.insert("test", {}, result=1.0)
    assert db.count() == 1
    db.delete_run(rid)
    assert db.count() == 0
    db.close()


def test_context_manager():
    with ExperimentLogger(":memory:") as db:
        db.insert("test", {}, result=42.0)
        assert db.count() == 1


def test_notes_stored():
    db = make_db()
    db.insert("test", {}, result=1.0, notes="first attempt")
    runs = db.all_runs()
    assert runs[0]["notes"] == "first attempt"
    db.close()


def test_demo_runs_without_error():
    db = make_db()
    demo_sql_queries(db)
    assert db.count() > 0
    db.close()
