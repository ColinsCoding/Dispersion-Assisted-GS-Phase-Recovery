"""Tests for dgs.calc2_primer -- change-of-base logs, exponentials, related
rates, and second derivatives (concavity + the circuits analog)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from dgs import calc2_primer as c2


def test_change_of_base_matches_known_values():
    assert c2.change_of_base(8, 2) == pytest.approx(3.0)
    assert c2.change_of_base(1000, 10) == pytest.approx(3.0)
    assert c2.change_of_base(np.e, np.e) == pytest.approx(1.0)


def test_change_of_base_matches_numpy_log2_and_log10():
    xs = [1.0, 2.0, 8.0, 100.0, 1e6]
    for x in xs:
        assert c2.change_of_base(x, 2) == pytest.approx(np.log2(x))
        assert c2.change_of_base(x, 10) == pytest.approx(np.log10(x))


def test_change_of_base_rejects_invalid_input():
    with pytest.raises(ValueError, match="positive"):
        c2.change_of_base(-1, 2)
    with pytest.raises(ValueError, match="positive"):
        c2.change_of_base(0, 2)
    with pytest.raises(ValueError, match="base"):
        c2.change_of_base(8, 1)
    with pytest.raises(ValueError, match="base"):
        c2.change_of_base(8, -2)


def test_bits_of_information_matches_a_byte():
    assert c2.bits_of_information(256) == pytest.approx(8.0)
    assert c2.bits_of_information(2) == pytest.approx(1.0)
    assert c2.bits_of_information(1) == pytest.approx(0.0)


def test_binary_search_steps_known_cases():
    assert c2.binary_search_steps(1) == 0
    assert c2.binary_search_steps(2) == 1
    assert c2.binary_search_steps(1024) == 10
    assert c2.binary_search_steps(1_000_000) == 20


def test_exponential_growth_and_decay():
    # dA/dt = k*A solution sanity: A(0) = A0
    assert c2.exponential(0.0, 5.0, 0.3) == pytest.approx(5.0)
    # decay: k<0 should decrease
    assert c2.exponential(10.0, 5.0, -0.1) < 5.0
    # growth: k>0 should increase
    assert c2.exponential(10.0, 5.0, 0.1) > 5.0


def test_half_life_round_trip():
    """A(t_half) should be exactly A0/2 for the derived rate constant, for
    several different half-lives."""
    for half_life in [1.0, 5730.0, 0.001]:
        k = c2.half_life_to_rate(half_life)
        assert c2.exponential(half_life, 1.0, k) == pytest.approx(0.5, rel=1e-6)


def test_half_life_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        c2.half_life_to_rate(0.0)
    with pytest.raises(ValueError, match="positive"):
        c2.half_life_to_rate(-5.0)


def test_ladder_related_rate_classic_3_4_5_triangle():
    """Textbook example: 6-8-10 ladder (2x the 3-4-5 triangle), base
    sliding out at 2 ft/s -> top slides down at 1.5 ft/s (known answer)."""
    dy_dt = c2.ladder_related_rate(x=6, y=8, L=10, dx_dt=2)
    assert dy_dt == pytest.approx(-1.5)


def test_ladder_related_rate_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="x\\^2\\+y\\^2"):
        c2.ladder_related_rate(x=1, y=1, L=10, dx_dt=1)   # doesn't satisfy x^2+y^2=L^2
    with pytest.raises(ValueError, match="positive"):
        c2.ladder_related_rate(x=6, y=8, L=0, dx_dt=1)
    with pytest.raises(ValueError, match="flat"):
        c2.ladder_related_rate(x=10, y=0, L=10, dx_dt=1)


def test_concavity_known_functions():
    assert c2.concavity(lambda x: x ** 2, 0.0) == 1     # concave up everywhere
    assert c2.concavity(lambda x: -x ** 2, 0.0) == -1    # concave down everywhere
    assert c2.concavity(lambda x: 3 * x + 1, 0.0) == 0   # linear: zero second derivative


def test_current_and_its_rate_matches_known_charge_function():
    """Q(t) = t^2 [C]  ->  I(t) = 2t [A]  ->  dI/dt = 2 [A/s] (constant),
    the circuits analog of position=t^2 -> velocity=2t -> acceleration=2."""
    for t in [1.0, 3.0, 10.0]:
        I, dI_dt = c2.current_and_its_rate(lambda tt: tt ** 2, t)
        assert I == pytest.approx(2 * t, rel=1e-4)
        assert dI_dt == pytest.approx(2.0, rel=1e-3)
