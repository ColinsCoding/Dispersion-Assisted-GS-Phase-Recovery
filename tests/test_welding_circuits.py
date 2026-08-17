import numpy as np
import sympy as sp
import pytest
from dgs.welding_circuits import (
    arc_resistance, arc_power, rl_transient_symbolic, rl_transient_current,
    time_to_steady_state, transformer_secondary_impedance,
    open_circuit_to_arc_voltage_drop, heat_input,
)


def test_arc_resistance_ohms_law():
    R = arc_resistance(V_arc=22.0, I_weld=150.0)
    assert R == pytest.approx(22.0 / 150.0)


def test_arc_resistance_rejects_nonpositive():
    with pytest.raises(ValueError):
        arc_resistance(0.0, 150.0)
    with pytest.raises(ValueError):
        arc_resistance(22.0, -1.0)


def test_arc_power_matches_VI():
    assert arc_power(22.0, 150.0) == pytest.approx(3300.0)


def test_arc_power_rejects_nonpositive():
    with pytest.raises(ValueError):
        arc_power(-1.0, 150.0)


def test_rl_transient_symbolic_is_equation_and_matches_known_form():
    ode, solution = rl_transient_symbolic()
    assert isinstance(ode, sp.Eq)
    assert isinstance(solution, sp.Eq)
    t, R, L, V = sp.symbols('t R L V', positive=True)
    expected = V / R - V * sp.exp(-R * t / L) / R
    assert sp.simplify(solution.rhs - expected) == 0


def test_rl_transient_current_starts_at_zero_and_approaches_steady_state():
    t = np.array([0.0, 1.0, 100.0])
    i = rl_transient_current(t, V=10.0, R=2.0, L=0.1)
    assert i[0] == pytest.approx(0.0)
    assert i[-1] == pytest.approx(10.0 / 2.0, rel=1e-6)
    assert i[0] < i[1] < i[2]


def test_rl_transient_current_rejects_negative_time():
    with pytest.raises(ValueError):
        rl_transient_current(-1.0, V=10.0, R=2.0, L=0.1)


def test_rl_transient_current_rejects_nonpositive_params():
    with pytest.raises(ValueError):
        rl_transient_current(1.0, V=0.0, R=2.0, L=0.1)
    with pytest.raises(ValueError):
        rl_transient_current(1.0, V=10.0, R=0.0, L=0.1)
    with pytest.raises(ValueError):
        rl_transient_current(1.0, V=10.0, R=2.0, L=0.0)


def test_time_to_steady_state_matches_current_curve():
    R, L, tol = 0.05, 0.0008, 0.02
    t_settle = time_to_steady_state(R, L, tolerance=tol)
    i_at_settle = rl_transient_current(t_settle, V=1.0, R=R, L=L)
    i_final = 1.0 / R
    assert i_at_settle / i_final == pytest.approx(1.0 - tol, rel=1e-6)


def test_time_to_steady_state_rejects_bad_tolerance():
    with pytest.raises(ValueError):
        time_to_steady_state(0.05, 0.0008, tolerance=0.0)
    with pytest.raises(ValueError):
        time_to_steady_state(0.05, 0.0008, tolerance=1.0)


def test_transformer_secondary_impedance_matches_R_plus_jwL():
    R, L, omega = 0.05, 0.0008, 2 * np.pi * 60.0
    Z = transformer_secondary_impedance(R, L, omega)
    expected = complex(R, omega * L)
    assert Z == pytest.approx(expected)


def test_transformer_secondary_impedance_rejects_nonpositive():
    with pytest.raises(ValueError):
        transformer_secondary_impedance(-1.0, 0.0008, 377.0)


def test_open_circuit_to_arc_voltage_drop_is_less_than_open_circuit():
    V_arc = open_circuit_to_arc_voltage_drop(V_open_circuit=70.0, I_weld=150.0, R_secondary=0.05)
    assert V_arc == pytest.approx(70.0 - 150.0 * 0.05)
    assert V_arc < 70.0


def test_open_circuit_to_arc_voltage_drop_rejects_negative_result():
    with pytest.raises(ValueError):
        open_circuit_to_arc_voltage_drop(V_open_circuit=5.0, I_weld=150.0, R_secondary=0.05)


def test_heat_input_matches_formula():
    HI = heat_input(V_arc=22.0, I_weld=150.0, travel_speed_mm_s=3.0, efficiency=0.8)
    assert HI == pytest.approx(0.8 * 22.0 * 150.0 / 3.0)


def test_heat_input_rejects_bad_efficiency():
    with pytest.raises(ValueError):
        heat_input(22.0, 150.0, 3.0, efficiency=0.0)
    with pytest.raises(ValueError):
        heat_input(22.0, 150.0, 3.0, efficiency=1.5)
