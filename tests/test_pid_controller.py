import pytest
import tempfile
import os
import sympy as sp
from dgs.pid_controller import (
    PIDController, simulate_first_order, step_response_metrics,
    pid_sympy_5, ScienceProblemSolver,
)


# ── PIDController ────────────────────────────────────────────────────

def test_pid_init_stores_gains():
    pid = PIDController(Kp=2.0, Ki=0.5, Kd=0.1, dt=0.01)
    assert pid.Kp == 2.0
    assert pid.Ki == 0.5
    assert pid.Kd == 0.1


def test_pid_invalid_dt():
    with pytest.raises(ValueError):
        PIDController(Kp=1.0, Ki=0.0, Kd=0.0, dt=0)


def test_pid_step_zero_error_returns_zero():
    pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01)
    assert pid.step(0.0) == pytest.approx(0.0)


def test_pid_proportional_only():
    pid = PIDController(Kp=3.0, Ki=0.0, Kd=0.0, dt=0.01)
    u = pid.step(2.0)
    assert u == pytest.approx(6.0)


def test_pid_reset_clears_state():
    pid = PIDController(Kp=1.0, Ki=1.0, Kd=0.0, dt=0.01)
    pid.step(1.0)
    pid.reset()
    assert pid.integral == pytest.approx(0.0)


def test_pid_clamp():
    pid = PIDController(Kp=100.0, Ki=0.0, Kd=0.0, dt=0.01, u_max=5.0)
    u = pid.step(1.0)
    assert u <= 5.0


# ── simulate_first_order ─────────────────────────────────────────────

def test_step_response_reaches_setpoint():
    pid = PIDController(Kp=1.5, Ki=0.5, Kd=0.1, dt=0.005)
    r = simulate_first_order(pid, setpoint=1.0, tau_plant=1.0, t_end=20.0)
    assert r["y"][-1] == pytest.approx(1.0, abs=0.05)


def test_step_response_zero_setpoint():
    pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01)
    r = simulate_first_order(pid, setpoint=0.0, t_end=2.0)
    assert r["y"][-1] == pytest.approx(0.0, abs=1e-6)


def test_step_response_arrays_same_length():
    pid = PIDController(Kp=2.0, Ki=0.5, Kd=0.1, dt=0.01)
    r = simulate_first_order(pid, setpoint=1.0, t_end=5.0)
    assert len(r["t"]) == len(r["y"]) == len(r["u"]) == len(r["error"])


# ── step_response_metrics ────────────────────────────────────────────

def test_metrics_overshoot_nonnegative():
    pid = PIDController(Kp=5.0, Ki=1.0, Kd=0.2, dt=0.01)
    r = simulate_first_order(pid, setpoint=1.0, tau_plant=1.0, t_end=10.0)
    m = step_response_metrics(r)
    assert m["overshoot_pct"] >= 0.0


def test_metrics_settling_time_positive():
    pid = PIDController(Kp=3.0, Ki=1.0, Kd=0.1, dt=0.01)
    r = simulate_first_order(pid, setpoint=2.0, tau_plant=1.0, t_end=15.0)
    m = step_response_metrics(r)
    assert m["settling_time_s"] >= 0.0


# ── pid_sympy_5 ──────────────────────────────────────────────────────

def test_pid_sympy_5_count_type():
    eqs = pid_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)


# ── ScienceProblemSolver ─────────────────────────────────────────────

def test_solver_beat_freq():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        solver = ScienceProblemSolver(path)
        r = solver.solve("beat_freq", f1=440, f2=443)
        assert r["f_beat_hz"] == pytest.approx(3.0)
    finally:
        os.unlink(path)


def test_solver_persists_to_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        solver = ScienceProblemSolver(path)
        solver.solve("beat_freq", f1=100, f2=102)
        # reload from disk
        solver2 = ScienceProblemSolver(path)
        assert len(solver2.history()) == 1
    finally:
        os.unlink(path)


def test_solver_unknown_problem_raises():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        solver = ScienceProblemSolver(path)
        with pytest.raises(ValueError):
            solver.solve("gravity_gun")
    finally:
        os.unlink(path)


def test_solver_clear():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        solver = ScienceProblemSolver(path)
        solver.solve("beat_freq", f1=440, f2=443)
        solver.clear()
        assert solver.history() == []
    finally:
        os.unlink(path)
