"""Tests for dgs/pm_sideband_bessel.py"""
import numpy as np
import pytest

from dgs.pm_sideband_bessel import (
    phase_modulated_field, jacobi_anger_sidebands, sideband_powers,
    reconstruct_from_sidebands, sideband_energy_conservation,
    carrier_suppression_beta, modulation_index_from_voltage,
)
from dgs.lnbo3 import vpi


class TestJacobiAngerReconstruction:
    def test_reconstruction_matches_direct_formula(self):
        t = np.linspace(0, 5 / 10e9, 5000)
        E_direct = phase_modulated_field(t, E0=1.0, beta=1.5, fm=10e9)
        E_recon = reconstruct_from_sidebands(t, E0=1.0, beta=1.5, fm=10e9, n_max=30)
        assert np.max(np.abs(E_direct - E_recon)) < 1e-10

    def test_reconstruction_needs_enough_terms(self):
        # too few sideband terms should NOT match to high precision
        t = np.linspace(0, 5 / 10e9, 5000)
        E_direct = phase_modulated_field(t, E0=1.0, beta=5.0, fm=10e9)
        E_recon_few = reconstruct_from_sidebands(t, E0=1.0, beta=5.0, fm=10e9, n_max=2)
        assert np.max(np.abs(E_direct - E_recon_few)) > 1e-3


class TestSidebandProperties:
    def test_symmetric_beta_sidebands_are_real(self):
        # for real beta, J_n(beta) are real-valued
        ns, c = jacobi_anger_sidebands(1.5, n_max=5)
        assert np.all(np.isreal(c))

    def test_odd_order_antisymmetry(self):
        # J_{-n}(beta) = (-1)^n J_n(beta)
        ns, c = jacobi_anger_sidebands(2.0, n_max=6)
        for n in range(1, 6):
            j_n = c[ns == n][0]
            j_neg_n = c[ns == -n][0]
            assert abs(j_neg_n - (-1) ** n * j_n) < 1e-12

    def test_powers_are_squares_of_amplitudes(self):
        ns, c = jacobi_anger_sidebands(3.0, n_max=8)
        _, p = sideband_powers(3.0, n_max=8)
        np.testing.assert_allclose(p, c ** 2)


class TestEnergyConservation:
    @pytest.mark.parametrize("beta", [0.1, 1.0, 2.405, 5.0, 10.0, 20.0])
    def test_sideband_powers_sum_to_one(self, beta):
        total = sideband_energy_conservation(beta, n_max=100)
        assert abs(total - 1.0) < 1e-9


class TestCarrierSuppression:
    def test_matches_textbook_value(self):
        beta0 = carrier_suppression_beta()
        assert abs(beta0 - 2.4048) < 1e-3

    def test_carrier_power_vanishes_there(self):
        beta0 = carrier_suppression_beta()
        ns, p = sideband_powers(beta0, n_max=1)
        carrier_power = p[ns == 0][0]
        assert carrier_power < 1e-20

    def test_carrier_power_nonzero_elsewhere(self):
        ns, p = sideband_powers(1.0, n_max=1)
        carrier_power = p[ns == 0][0]
        assert carrier_power > 0.5   # most power still in carrier at small beta


class TestModulationIndexFromVoltage:
    def test_zero_voltage_gives_zero_beta(self):
        assert modulation_index_from_voltage(0.0, 5.0) == 0.0

    def test_v_pi_drive_gives_beta_pi(self):
        V_pi = 3.3
        beta = modulation_index_from_voltage(V_pi, V_pi)
        assert abs(beta - np.pi) < 1e-12

    def test_ties_to_real_lnbo3_device(self):
        # reuse dgs.lnbo3.vpi directly -- a real device parameter, not invented
        V_pi = vpi(L_mm=50.0)
        assert V_pi > 0
        beta0 = carrier_suppression_beta()
        V_rf = V_pi * beta0 / np.pi
        beta_recovered = modulation_index_from_voltage(V_rf, V_pi)
        assert abs(beta_recovered - beta0) < 1e-9
