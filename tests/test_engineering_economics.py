"""Tests for dgs/engineering_economics.py"""
import math
import pytest

from dgs.engineering_economics import (
    compound_amount_factor, present_worth_factor,
    uniform_series_compound_amount, sinking_fund_factor,
    uniform_series_present_worth, capital_recovery_factor,
    gradient_present_worth, gradient_to_uniform_series,
    npv, irr, profitability_index,
)


class TestSinglePaymentFactors:
    def test_matches_textbook_table_FP_8_5(self):
        # standard engineering-economics table: (F/P, 8%, 5) = 1.4693
        F = compound_amount_factor(1.0, 0.08, 5)
        assert F == pytest.approx(1.4693, abs=1e-4)

    def test_PF_is_exact_inverse_of_FP(self):
        P = 10000.0
        F = compound_amount_factor(P, 0.08, 5)
        assert present_worth_factor(F, 0.08, 5) == pytest.approx(P, rel=1e-9)

    def test_zero_periods_is_identity(self):
        assert compound_amount_factor(500.0, 0.1, 0) == pytest.approx(500.0)


class TestUniformSeriesFactors:
    def test_matches_textbook_table_PA_8_5(self):
        # (P/A, 8%, 5) = 3.9927
        P = uniform_series_present_worth(1.0, 0.08, 5)
        assert P == pytest.approx(3.9927, abs=1e-4)

    def test_AF_is_exact_inverse_of_FA(self):
        A = 2000.0
        F = uniform_series_compound_amount(A, 0.08, 5)
        assert sinking_fund_factor(F, 0.08, 5) == pytest.approx(A, rel=1e-9)

    def test_AP_is_exact_inverse_of_PA(self):
        P = 7985.42
        A = capital_recovery_factor(P, 0.08, 5)
        assert uniform_series_present_worth(A, 0.08, 5) == pytest.approx(P, rel=1e-4)

    def test_zero_interest_rate_is_linear(self):
        # at i=0, F/A and P/A both degenerate to simple multiplication by n
        assert uniform_series_compound_amount(100.0, 0.0, 4) == pytest.approx(400.0)
        assert uniform_series_present_worth(100.0, 0.0, 4) == pytest.approx(400.0)

    def test_FA_equals_PA_compounded_forward(self):
        # F/A and P/A must be consistent: F = P*(1+i)^n
        A, i, n = 1500.0, 0.06, 8
        F = uniform_series_compound_amount(A, i, n)
        P = uniform_series_present_worth(A, i, n)
        assert F == pytest.approx(P * (1 + i) ** n, rel=1e-9)


class TestGradientFactors:
    def test_matches_textbook_table_AG_8_5(self):
        # (A/G, 8%, 5) = 1.8465
        A = gradient_to_uniform_series(1.0, 0.08, 5)
        assert A == pytest.approx(1.8465, abs=1e-4)

    def test_AG_equals_PG_via_AP(self):
        # A/G should equal (P/G) converted to an annuity via A/P -- the same
        # relationship used to derive it, checked as an independent identity
        G, i, n = 500.0, 0.08, 5
        P_grad = gradient_present_worth(G, i, n)
        A_from_P = capital_recovery_factor(P_grad, i, n)
        assert gradient_to_uniform_series(G, i, n) == pytest.approx(A_from_P, rel=1e-9)

    def test_zero_interest_gradient(self):
        # at i=0: P/G = G*n(n-1)/2 (sum of an arithmetic series), verified directly
        G, n = 100.0, 6
        expected = G * sum(range(n))   # 0+1+2+3+4+5 = 15 -> 1500
        assert gradient_present_worth(G, 0.0, n) == pytest.approx(expected)


class TestNPVandIRR:
    def test_npv_zero_rate_is_simple_sum(self):
        cfs = [-1000, 400, 400, 400]
        assert npv(cfs, 0.0) == pytest.approx(200.0)

    def test_irr_zeros_npv_exactly(self):
        # sub-cent precision on a $275k-scale cash flow -- far tighter than
        # any real financial use needs, and tight enough to catch the
        # rate-vs-dollar unit-conflation bug this replaced
        cfs = [-275000, 80000, 90000, 95000, 100000, 85000]
        r = irr(cfs)
        assert npv(cfs, r) == pytest.approx(0.0, abs=1e-4)

    def test_irr_known_case(self):
        # simple case: invest 1000, get back 1100 in one year -> IRR = 10% exactly
        cfs = [-1000, 1100]
        assert irr(cfs) == pytest.approx(0.10, abs=1e-8)

    def test_irr_raises_without_sign_change(self):
        # all-positive cash flow: NPV never crosses zero for any positive rate
        with pytest.raises(ValueError):
            irr([100, 100, 100], lo=0.0, hi=1.0)

    def test_higher_rate_lowers_npv_for_front_loaded_investment(self):
        cfs = [-1000, 300, 300, 300, 300, 300]
        assert npv(cfs, 0.05) > npv(cfs, 0.15)


class TestProfitabilityIndex:
    def test_pi_greater_than_one_matches_positive_npv(self):
        cfs = [-275000, 80000, 90000, 95000, 100000, 85000]
        rate = 0.10
        pi = profitability_index(cfs, rate)
        n = npv(cfs, rate)
        assert (pi > 1) == (n > 0)

    def test_pi_exactly_one_at_breakeven_rate(self):
        # at the IRR, NPV=0, so PV(benefits) should exactly equal PV(costs) -> PI=1
        cfs = [-1000, 1100]
        r = irr(cfs)
        assert profitability_index(cfs, r) == pytest.approx(1.0, abs=1e-6)

    def test_pi_requires_a_cost(self):
        with pytest.raises(ValueError):
            profitability_index([100, 200, 300], 0.1)


class TestBounds:
    def test_negative_periods_rejected(self):
        with pytest.raises(ValueError):
            compound_amount_factor(100.0, 0.1, -1)

    def test_rate_below_negative_one_rejected(self):
        with pytest.raises(ValueError):
            compound_amount_factor(100.0, -1.5, 5)

    def test_zero_periods_sinking_fund_rejected(self):
        with pytest.raises(ValueError):
            sinking_fund_factor(1000.0, 0.05, 0)
