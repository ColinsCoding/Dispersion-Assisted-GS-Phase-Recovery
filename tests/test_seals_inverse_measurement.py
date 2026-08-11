"""Tests for projects.seals.inverse.measurement: the |E|^2 detector boundary
and the Mie complex-field reconstruction."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from projects.seals.inverse import measurement
from projects.seals.inverse import _seals_physics as physics


def test_intensity_measurement_nonnegative_numpy():
    rng = np.random.RandomState(0)
    E = rng.normal(size=50) + 1j * rng.normal(size=50)
    I = measurement.intensity_measurement(E)
    assert np.all(I >= 0)
    assert np.allclose(I, np.abs(E) ** 2)


def test_intensity_measurement_nonnegative_torch():
    torch.manual_seed(0)
    E = torch.randn(50, dtype=torch.complex128)
    I = measurement.intensity_measurement(E)
    assert torch.all(I >= 0)
    assert torch.allclose(I, E.abs() ** 2)


def test_global_phase_invariance():
    """|E|^2 == |E * exp(i*const)|^2 -- the core reason phase retrieval needs
    more than one raw intensity trace (see phase_retrieval.py)."""
    torch.manual_seed(1)
    E = torch.randn(64, dtype=torch.complex128)
    for const_phase in (0.3, 1.7, -2.4, np.pi):
        E_shifted = E * torch.exp(1j * torch.tensor(const_phase, dtype=torch.float64))
        I_a = measurement.intensity_measurement(E)
        I_b = measurement.intensity_measurement(E_shifted)
        assert torch.allclose(I_a, I_b, atol=1e-12)


def test_mie_complex_fields_reconstructs_validated_outputs():
    """E_p, E_s reconstructed via sqrt(I)*exp(i*phase) must reproduce the
    validated I_p/I_s/T_p/T_s exactly -- this is an algebraic identity, not
    a new physics claim, and this test locks that in."""
    p = physics.P_DEFAULT
    lam0 = 0.5 * (p['lam1'] + p['lam2'])
    theta_rad = np.deg2rad(np.linspace(-20, 40, 30))

    sigma_s, I_p, I_s, an, bn, T_p, T_s = physics.mie(
        p['npar'], p['nmed'], p['dia'], lam0, theta_rad, p['r'])
    fields = measurement.mie_complex_fields(
        p['npar'], p['nmed'], p['dia'], lam0, theta_rad, p['r'])

    assert np.allclose(fields.I_p, I_p)
    assert np.allclose(fields.I_s, I_s)
    assert np.allclose(fields.phase_p, T_p)
    assert np.allclose(fields.phase_s, T_s)
    assert np.allclose(np.abs(fields.E_p) ** 2, I_p)
    assert np.allclose(np.angle(fields.E_p), T_p)
    assert fields.sigma_s == sigma_s
