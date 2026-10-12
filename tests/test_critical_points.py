"""Tests for dgs/critical_points.py"""
import numpy as np
import pytest
import sympy as sp

from dgs.critical_points import (
    classify_2var_second_derivative_test, classify_by_eigenvalues,
    find_critical_points_2d, two_layer_linear_network_loss,
    autograd_hessian_at_point, finite_difference_hessian_at_point,
    symbolic_hessian_two_layer_linear,
)


class Test2VarSecondDerivativeTest:
    def test_bowl_is_min(self):
        kind, D = classify_2var_second_derivative_test(fxx=2.0, fxy=0.0, fyy=2.0)
        assert kind == "min" and D > 0

    def test_inverted_bowl_is_max(self):
        kind, D = classify_2var_second_derivative_test(fxx=-2.0, fxy=0.0, fyy=-2.0)
        assert kind == "max" and D > 0

    def test_saddle(self):
        kind, D = classify_2var_second_derivative_test(fxx=2.0, fxy=0.0, fyy=-2.0)
        assert kind == "saddle" and D < 0

    def test_inconclusive_at_D_zero(self):
        kind, D = classify_2var_second_derivative_test(fxx=1.0, fxy=1.0, fyy=1.0)
        assert kind == "inconclusive" and D == pytest.approx(0.0)


class TestEigenvalueClassification:
    def test_positive_definite_is_min(self):
        kind, eig = classify_by_eigenvalues([[2, 0], [0, 3]])
        assert kind == "min"

    def test_negative_definite_is_max(self):
        kind, eig = classify_by_eigenvalues([[-2, 0], [0, -3]])
        assert kind == "max"

    def test_indefinite_is_saddle(self):
        kind, eig = classify_by_eigenvalues([[1, 0], [0, -1]])
        assert kind == "saddle"

    def test_zero_eigenvalue_is_degenerate(self):
        kind, eig = classify_by_eigenvalues([[0, 0], [0, 1]])
        assert kind == "degenerate"

    def test_rejects_asymmetric_matrix(self):
        with pytest.raises(ValueError):
            classify_by_eigenvalues([[1, 2], [0, 1]])


class TestFindCriticalPoints2D:
    def test_bowl_agrees_both_methods(self):
        x, y = sp.symbols("x y", real=True)
        results = find_critical_points_2d(x**2 + y**2, x, y)
        assert len(results) == 1
        r = results[0]
        assert r["point"] == pytest.approx((0.0, 0.0))
        assert r["kind_2var_test"] == "min"
        assert r["kind_eigenvalue_test"] == "min"
        assert r["agree"]

    def test_saddle_agrees_both_methods(self):
        x, y = sp.symbols("x y", real=True)
        results = find_critical_points_2d(x**2 - y**2, x, y)
        assert len(results) == 1
        assert results[0]["kind_2var_test"] == "saddle"
        assert results[0]["kind_eigenvalue_test"] == "saddle"
        assert results[0]["agree"]

    def test_shifted_bowl_finds_correct_point(self):
        x, y = sp.symbols("x y", real=True)
        results = find_critical_points_2d((x - 3)**2 + (y + 2)**2, x, y)
        assert len(results) == 1
        assert results[0]["point"] == pytest.approx((3.0, -2.0))
        assert results[0]["kind_eigenvalue_test"] == "min"


class TestTwoLayerLinearNetworkSaddle:
    """The well-known 'dead weights' saddle at the origin of a minimal
    2-layer linear network's loss surface (Baldi & Hornik 1989) --
    verified three independent ways, not assumed."""

    def setup_data(self):
        rng = np.random.default_rng(0)
        x_data = rng.uniform(-3, 3, 200)
        y_data = x_data.copy()
        return x_data, y_data

    def test_origin_is_a_genuine_critical_point(self):
        import torch
        x_data, y_data = self.setup_data()
        w1 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        w2 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        loss = two_layer_linear_network_loss(x_data, y_data, w1, w2)
        g1, g2 = torch.autograd.grad(loss, [w1, w2])
        assert abs(float(g1)) < 1e-9
        assert abs(float(g2)) < 1e-9

    def test_three_methods_agree_on_the_hessian(self):
        x_data, y_data = self.setup_data()
        H_autograd = autograd_hessian_at_point(x_data, y_data, 0.0, 0.0)
        H_findiff = finite_difference_hessian_at_point(x_data, y_data, 0.0, 0.0)
        C = float(np.mean(x_data ** 2))
        H_symbolic, grad_symbolic = symbolic_hessian_two_layer_linear(C)

        assert np.max(np.abs(H_autograd - H_findiff)) < 1e-4
        assert np.max(np.abs(H_autograd - H_symbolic)) < 1e-6
        assert grad_symbolic == [0.0, 0.0]

    def test_origin_is_classified_as_saddle(self):
        x_data, y_data = self.setup_data()
        H = autograd_hessian_at_point(x_data, y_data, 0.0, 0.0)
        kind, eigvals = classify_by_eigenvalues(H)
        assert kind == "saddle"
        assert eigvals[0] < 0 < eigvals[1]

    def test_eigenvalues_match_analytic_prediction(self):
        # eigenvalues of [[0,-2C],[-2C,0]] are exactly +-2C, C=E[x^2]
        x_data, y_data = self.setup_data()
        C = float(np.mean(x_data ** 2))
        H = autograd_hessian_at_point(x_data, y_data, 0.0, 0.0)
        kind, eigvals = classify_by_eigenvalues(H)
        assert np.allclose(sorted(eigvals), sorted([-2 * C, 2 * C]), rtol=1e-6)

    def test_global_minimum_curve_is_degenerate_not_a_clean_min(self):
        # at w1*w2=1 (e.g. w1=w2=1) the network fits the target exactly -> loss=0.
        # NOT a clean isolated minimum, though: w1*w2=1 is an entire CURVE of
        # equivalent global minima (any (t, 1/t) also gives loss=0), so the
        # Hessian is rank-deficient (one zero eigenvalue, flat along the curve) --
        # verified directly rather than assumed to be a simple "min" like the
        # textbook bowl example, a genuinely different (and correct) answer
        x_data, y_data = self.setup_data()
        H = autograd_hessian_at_point(x_data, y_data, 1.0, 1.0)
        kind, eigvals = classify_by_eigenvalues(H)
        assert kind == "degenerate"
        assert min(eigvals) == pytest.approx(0.0, abs=1e-6)
        assert max(eigvals) > 0   # curved (rising loss) in the direction transverse to the minima curve
