import numpy as np
import pytest
from dgs.login_security_probability import (
    single_attempt_success_prob, prob_crack_within_n_attempts,
    expected_attempts_to_crack, prob_crack_with_lockout,
    prob_crack_varying_attempts, product_rule_two_factors,
    leibniz_product_rule_n_factors, attempt_probability_product_rule_demo,
    login_sympy_5,
)


def test_single_attempt_prob_pin():
    p = single_attempt_success_prob(10000)
    assert p == pytest.approx(1e-4)


def test_single_attempt_prob_invalid():
    with pytest.raises(ValueError):
        single_attempt_success_prob(-1)


def test_prob_crack_zero_attempts():
    assert prob_crack_within_n_attempts(0.1, 0) == pytest.approx(0.0)


def test_prob_crack_certain_with_full_keyspace():
    p = single_attempt_success_prob(10)
    # exhausting the entire keyspace should give P close to 1
    prob = prob_crack_within_n_attempts(p, 100)
    assert prob > 0.9999  # 10^100 / 10 keyspace: 1 - (0.9)^100 ~ 0.99997


def test_prob_crack_increases_with_n():
    p = 0.01
    probs = [prob_crack_within_n_attempts(p, n) for n in [1, 10, 100]]
    assert probs[0] < probs[1] < probs[2]


def test_prob_crack_invalid_p_raises():
    with pytest.raises(ValueError):
        prob_crack_within_n_attempts(1.5, 10)


def test_expected_attempts_geometric():
    assert expected_attempts_to_crack(0.01) == pytest.approx(100.0)


def test_prob_crack_with_lockout_increases_wall_clock():
    p = 1e-4
    res = prob_crack_with_lockout(p, n_attempts=10000, lockout_after=5,
                                   lockout_duration_s=900, attempt_interval_s=1)
    # with lockout, more than 10000 seconds should pass
    assert res["wall_clock_s"] > 10000


def test_prob_crack_varying_constant_p_matches_formula():
    p = 0.01
    n = 5
    res = prob_crack_varying_attempts([p] * n)
    expected = 1 - (1 - p) ** n
    assert res["prob_at_least_one_success"] == pytest.approx(expected)


def test_prob_crack_varying_invalid_prob_raises():
    with pytest.raises(ValueError):
        prob_crack_varying_attempts([0.5, 1.5])


def test_prob_all_fail_plus_at_least_one_equals_one():
    res = prob_crack_varying_attempts([0.1, 0.2, 0.3])
    assert res["prob_all_fail"] + res["prob_at_least_one_success"] == pytest.approx(1.0)


def test_product_rule_two_factors_is_equation():
    import sympy as sp
    eq = product_rule_two_factors()
    assert isinstance(eq, sp.Eq)


def test_leibniz_product_rule_n2_matches_product_rule():
    import sympy as sp
    res = leibniz_product_rule_n_factors(2)
    x = sp.Symbol('x')
    f1 = sp.Function('f1')(x)
    f2 = sp.Function('f2')(x)
    expected = sp.expand(sp.diff(f1 * f2, x))
    assert sp.simplify(res["derivative_expanded"] - expected) == 0


def test_leibniz_product_rule_3_has_3_terms():
    import sympy as sp
    res = leibniz_product_rule_n_factors(3)
    # The expanded derivative should be a sum of 3 terms
    expr = res["derivative_expanded"]
    assert len(expr.as_ordered_terms()) == 3


def test_attempt_probability_product_rule_demo_structure():
    res = attempt_probability_product_rule_demo(2)
    assert "P_all_fail" in res
    assert "dP_dx_via_product_rule" in res


def test_login_sympy_5_count():
    eqs = login_sympy_5()
    assert len(eqs) == 5
