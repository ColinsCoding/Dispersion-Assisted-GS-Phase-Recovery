"""Tests for dgs/queueing.py (Erlang C) and the C++ / CUDA event simulators in dgs/eventsim.py."""
import math

import pytest

from dgs import eventsim as ES
from dgs import queueing as QU


# --- exact queueing formulas --------------------------------------------------------------
def test_mm1_closed_forms():
    lam, mu = 0.7, 1.0
    m = QU.mmc_metrics(lam, mu, 1)
    assert QU.erlang_c(lam, mu, 1) == pytest.approx(0.7)              # P(wait) = rho for M/M/1
    assert m["Wq"] == pytest.approx(0.7 / (mu - lam))
    assert m["W"] == pytest.approx(1 / (mu - lam))
    assert m["L"] == pytest.approx(0.7 / 0.3)                          # rho / (1 - rho)


def test_erlang_c_textbook_value():
    # a = 2 Erlangs, c = 3: C = (8/6 * 3) / (1 + 2 + 2 + 8/6 * 3) = 4/9
    assert QU.erlang_c(2.0, 1.0, 3) == pytest.approx(4 / 9)


def test_littles_law_and_monotone_staffing():
    lam, mu = 12.0, 4 / 3
    prev = math.inf
    for c in range(10, 16):
        m = QU.mmc_metrics(lam, mu, c)
        assert m["L"] == pytest.approx(lam * m["W"]) and m["Lq"] == pytest.approx(lam * m["Wq"])
        assert m["Wq"] < prev
        prev = m["Wq"]
    c = QU.min_servers(lam, mu, 10 / 60)
    assert QU.mmc_metrics(lam, mu, c)["Wq"] <= 10 / 60 < QU.mmc_metrics(lam, mu, c - 1)["Wq"]


def test_python_reference_simulator():
    wq, pw = QU.simulate_mmc(2.0, 1.0, 3, 100_000, warmup=5_000, seed=3)
    m = QU.mmc_metrics(2.0, 1.0, 3)
    assert wq == pytest.approx(m["Wq"], rel=0.06)
    assert pw == pytest.approx(m["P_wait"], rel=0.04)


@pytest.mark.parametrize("call", [
    lambda: QU.mmc_metrics(2.0, 1.0, 2),          # rho = 1: unstable
    lambda: QU.mmc_metrics(-1.0, 1.0, 2),
    lambda: QU.mmc_metrics(1.0, 1.0, 1.5),
    lambda: QU.min_servers(1.0, 1.0, 0),
    lambda: QU.simulate_mmc(1.0, 2.0, 1, 0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()


# --- C++ event-list simulator (g++) -------------------------------------------------------
def _cpu_or_skip():
    try:
        ES.build_cpu()
    except ES.ToolUnavailable as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("lam, mu, c", [(0.7, 1.0, 1), (2.0, 1.0, 3), (12.0, 4 / 3, 11)])
def test_cpu_matches_erlang_c(lam, mu, c):
    _cpu_or_skip()
    r = ES.run_cpu(lam, mu, c, 400_000, warmup=20_000, seed=11)
    m = QU.mmc_metrics(lam, mu, c)
    assert r["mean_wait"] == pytest.approx(m["Wq"], rel=0.06)
    assert r["p_wait"] == pytest.approx(m["P_wait"], rel=0.03)
    assert r["utilization"] == pytest.approx(m["rho"], rel=0.02)
    assert r["mean_in_system"] == pytest.approx(r["arrival_rate"] * r["mean_system"], rel=0.02)  # Little
    assert r["events"] == 2 * (400_000 + 20_000)


def test_cpu_is_reproducible_and_traces():
    _cpu_or_skip()
    a = ES.run_cpu(0.9, 1.0, 1, 5_000, warmup=0, seed=5, trace_events=50)
    b = ES.run_cpu(0.9, 1.0, 1, 5_000, warmup=0, seed=5)
    assert a["mean_wait"] == b["mean_wait"]
    tr = a["trace"]
    assert tr.shape == (50, 5)
    assert (tr[1:, 0] >= tr[:-1, 0]).all()               # the clock never runs backward
    assert (tr[:, 3] <= 1).all()                         # never more busy servers than c


def test_cpu_rejects_unstable():
    _cpu_or_skip()
    with pytest.raises(ValueError):
        ES.run_cpu(2.0, 1.0, 2, 100)


# --- CUDA replications (nvcc) -------------------------------------------------------------
def test_gpu_replications_match_erlang_c():
    try:
        r = ES.run_gpu(2.0, 1.0, 3, 20_000, reps=1024, warmup=2_000, seed=9)
    except ES.ToolUnavailable as e:
        pytest.skip(str(e))
    m = QU.mmc_metrics(2.0, 1.0, 3)
    mean, half = ES.confidence_interval(r["wq"])
    assert abs(mean - m["Wq"]) < 4 * half / 1.96 + 0.005   # within ~4 standard errors
    assert r["p_wait"].mean() == pytest.approx(m["P_wait"], rel=0.01)
    assert len(r["wq"]) == 1024 and r["kernel_ms"] > 0
