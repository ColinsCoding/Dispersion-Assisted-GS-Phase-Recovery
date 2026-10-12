"""Tests for dgs/ehrenfest_theorem.py"""
import numpy as np
import sympy as sp
import pytest

from dgs.ehrenfest_theorem import (
    integration_by_parts_identity, derive_ehrenfest_position_theorem,
    split_step_evolve, gaussian_wavepacket, harmonic_oscillator_potential,
)


class TestIntegrationByPartsIdentity:
    def test_holds_for_sin_exp(self):
        x = sp.symbols('x', real=True)
        result = integration_by_parts_identity(sp.sin(x), sp.exp(x), x, 0, 1)
        assert result["identity_holds"]

    def test_holds_for_polynomial_pair(self):
        x = sp.symbols('x', real=True)
        result = integration_by_parts_identity(x**2, sp.sin(x), x, 0, sp.pi)
        assert result["identity_holds"]

    def test_boundary_term_is_zero_for_periodic_functions_on_full_period(self):
        # a case where the boundary term vanishes -- checked, not assumed
        x = sp.symbols('x', real=True)
        result = integration_by_parts_identity(sp.sin(x), sp.sin(x), x, 0, 2*sp.pi)
        assert sp.simplify(result["boundary_term"]) == 0


class TestSymbolicDerivation:
    def test_potential_terms_cancel(self):
        result = derive_ehrenfest_position_theorem()
        assert result["potential_terms_cancel"]


class TestHarmonicOscillatorNumeric:
    """Ehrenfest's theorem is EXACT for a harmonic oscillator -- <x>(t)
    and <p>(t) must follow the classical trajectory exactly, for ANY
    initial wavepacket. Verified via split-step Fourier time evolution,
    not assumed."""

    def setup_evolution(self, x0=2.0, p0=1.5, sigma=1.0, n_steps=400):
        m, omega, hbar = 1.0, 1.0, 1.0
        N, L = 2048, 40.0
        x_grid = np.linspace(-L/2, L/2, N, endpoint=False)
        dt = 0.01
        psi0 = gaussian_wavepacket(x_grid, x0, p0, sigma, hbar)
        x_exp, p_exp = split_step_evolve(psi0, x_grid, dt, n_steps, m, omega, hbar)
        t = np.arange(n_steps + 1) * dt
        return x_exp, p_exp, t, m, omega, p0, x0

    def test_position_expectation_matches_classical_trajectory(self):
        x_exp, p_exp, t, m, omega, p0, x0 = self.setup_evolution()
        x_classical = x0 * np.cos(omega * t) + (p0 / (m * omega)) * np.sin(omega * t)
        assert np.max(np.abs(x_exp - x_classical)) < 1e-3

    def test_momentum_expectation_matches_classical_trajectory(self):
        x_exp, p_exp, t, m, omega, p0, x0 = self.setup_evolution()
        p_classical = -m * omega * x0 * np.sin(omega * t) + p0 * np.cos(omega * t)
        assert np.max(np.abs(p_exp - p_classical)) < 1e-3

    def test_dxdt_equals_p_over_m(self):
        x_exp, p_exp, t, m, omega, p0, x0 = self.setup_evolution()
        dt = t[1] - t[0]
        dxdt = np.gradient(x_exp, dt)
        interior = slice(5, -5)
        assert np.max(np.abs(dxdt[interior] - p_exp[interior] / m)) < 1e-2

    def test_dpdt_equals_negative_harmonic_force(self):
        x_exp, p_exp, t, m, omega, p0, x0 = self.setup_evolution()
        dt = t[1] - t[0]
        dpdt = np.gradient(p_exp, dt)
        interior = slice(5, -5)
        assert np.max(np.abs(dpdt[interior] - (-m * omega**2 * x_exp[interior]))) < 1e-2

    def test_holds_for_a_different_wavepacket_too(self):
        # not just one tuned example -- a wider, differently-placed, faster packet
        x_exp, p_exp, t, m, omega, p0, x0 = self.setup_evolution(x0=-3.0, p0=-2.0, sigma=1.8, n_steps=300)
        x_classical = x0 * np.cos(omega * t) + (p0 / (m * omega)) * np.sin(omega * t)
        assert np.max(np.abs(x_exp - x_classical)) < 1e-3


class TestGaussianWavepacket:
    def test_is_normalized(self):
        x = np.linspace(-30, 30, 4096, endpoint=False)
        dx = x[1] - x[0]
        psi = gaussian_wavepacket(x, x0=1.0, p0=0.5, sigma=1.0)
        norm = np.sum(np.abs(psi)**2) * dx
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_mean_position_matches_x0(self):
        x = np.linspace(-30, 30, 4096, endpoint=False)
        dx = x[1] - x[0]
        psi = gaussian_wavepacket(x, x0=2.5, p0=0.0, sigma=1.0)
        x_mean = np.sum(x * np.abs(psi)**2) * dx
        assert x_mean == pytest.approx(2.5, abs=1e-3)


class TestHarmonicPotential:
    def test_is_quadratic_and_symmetric(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        V = harmonic_oscillator_potential(x, m=1.0, omega=2.0)
        assert V[2] == 0.0
        assert np.allclose(V, V[::-1])
