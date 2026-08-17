import numpy as np
import pytest
from dgs.lut_gain_scheduled_pid import (
    quantize_error_level, regime_lut, GainScheduledPID, simulate_gain_scheduled,
)
from dgs.pid import first_order_plant


def test_quantize_error_level_bounds():
    assert quantize_error_level(0.0, error_max=5.0, n_bits=2) == 0
    assert quantize_error_level(5.0, error_max=5.0, n_bits=2) == 3
    assert quantize_error_level(100.0, error_max=5.0, n_bits=2) == 3  # clipped


def test_quantize_error_level_uses_magnitude():
    assert quantize_error_level(-4.9, error_max=5.0, n_bits=2) == quantize_error_level(
        4.9, error_max=5.0, n_bits=2)


def test_quantize_error_level_rejects_bad_input():
    with pytest.raises(ValueError):
        quantize_error_level(1.0, error_max=0.0, n_bits=2)
    with pytest.raises(ValueError):
        quantize_error_level(1.0, error_max=5.0, n_bits=0)


def test_regime_lut_matches_hand_checked_truth_table():
    # n_bits=2, threshold=2: levels 0,1 -> 0 (gentle); levels 2,3 -> 1 (aggressive)
    lut_bits = regime_lut(n_bits=2, threshold_level=2)
    assert lut_bits == (0, 0, 1, 1)


def test_regime_lut_threshold_zero_is_always_aggressive():
    lut_bits = regime_lut(n_bits=2, threshold_level=0)
    assert lut_bits == (1, 1, 1, 1)


def test_regime_lut_rejects_out_of_range_threshold():
    with pytest.raises(ValueError):
        regime_lut(n_bits=2, threshold_level=4)
    with pytest.raises(ValueError):
        regime_lut(n_bits=2, threshold_level=-1)


def test_gain_scheduled_pid_rejects_bad_error_max():
    with pytest.raises(ValueError):
        GainScheduledPID((0.3, 0.05, 0.02), (1.2, 0.15, 0.05), error_max=0.0)


def test_gain_scheduled_pid_uses_aggressive_on_large_initial_error():
    controller = GainScheduledPID((0.3, 0.05, 0.02), (1.2, 0.15, 0.05),
                                   error_max=5.0, n_bits=2, setpoint=0.0, dt=1.0)
    controller.update(measurement=5.0)   # error = 0 - 5.0 = -5.0, |error|=error_max -> top level
    assert controller.regime_history[0] is True


def test_gain_scheduled_pid_uses_gentle_near_setpoint():
    controller = GainScheduledPID((0.3, 0.05, 0.02), (1.2, 0.15, 0.05),
                                   error_max=5.0, n_bits=2, setpoint=0.0, dt=1.0)
    controller.update(measurement=0.05)   # tiny error -> low level -> gentle
    assert controller.regime_history[0] is False


def test_simulate_gain_scheduled_shapes():
    controller = GainScheduledPID((0.3, 0.05, 0.02), (1.2, 0.15, 0.05),
                                   error_max=5.0, n_bits=2, setpoint=0.0, dt=1.0)
    plant = first_order_plant(tau=8.0, gain=1.0, dt=1.0, y0=5.0)
    result = simulate_gain_scheduled(controller, plant, n_steps=30)
    assert result["y"].shape == (30,)
    assert result["u"].shape == (30,)
    assert result["regime_history"].shape == (30,)


def test_simulate_gain_scheduled_converges_toward_setpoint():
    # NOTE: simulate_gain_scheduled mirrors dgs.pid.simulate's convention where
    # the loop's tracking variable starts at 0.0 regardless of the plant's y0
    # (the plant's own initial condition only appears from the first
    # plant(u) call onward) -- so y[0] is always 0.0, not a meaningful
    # "before" value to compare against. Compare the plant's actual early
    # excursion (after y0=5.0 first appears) against the final value instead.
    controller = GainScheduledPID((0.3, 0.05, 0.02), (1.2, 0.15, 0.05),
                                   error_max=5.0, n_bits=2, setpoint=0.0, dt=1.0)
    plant = first_order_plant(tau=8.0, gain=1.0, dt=1.0, y0=5.0)
    result = simulate_gain_scheduled(controller, plant, n_steps=40)
    assert abs(result["y"][-1]) < abs(result["y"][1])
    assert abs(result["y"][-1]) < 0.5


def test_simulate_gain_scheduled_rejects_bad_n_steps():
    controller = GainScheduledPID((0.3, 0.05, 0.02), (1.2, 0.15, 0.05), error_max=5.0)
    plant = first_order_plant(tau=8.0)
    with pytest.raises(ValueError):
        simulate_gain_scheduled(controller, plant, n_steps=0)
