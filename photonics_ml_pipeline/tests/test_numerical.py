"""Numerical-correctness tests: beam physics, ABCD optics, dispersion."""
from __future__ import annotations

import numpy as np

from optics.abcd import free_space, propagate_q, q_at_waist, width_from_q
from photonics.dispersion import apply_dispersion, transfer_function
from physics.gaussian_beam import GaussianBeam


def test_width_matches_symbolic_relation() -> None:
    beam = GaussianBeam(wavelength_um=1.55, waist_um=10.0)
    zr = beam.rayleigh_range_um
    assert np.isclose(beam.width_um(zr), beam.waist_um * np.sqrt(2.0))  # w(zR) = sqrt(2) w0


def test_abcd_q_propagation_matches_beam_width() -> None:
    beam = GaussianBeam(wavelength_um=1.55, waist_um=12.0)
    zr = beam.rayleigh_range_um
    q0 = q_at_waist(zr)
    for z in (0.0, 100.0, 500.0, 1500.0):
        q = propagate_q(q0, free_space(z))
        assert np.isclose(width_from_q(q, beam.wavelength_um), float(beam.width_um(z)), rtol=1e-9)


def test_dispersion_is_unitary_energy_conserving() -> None:
    rng = np.random.default_rng(0)
    pulse = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    for D in (-5000.0, 600.0, 5000.0):
        out = apply_dispersion(pulse, D)
        assert np.isclose(np.linalg.norm(out), np.linalg.norm(pulse))


def test_transfer_function_is_all_pass() -> None:
    freq = np.fft.fftfreq(128)
    h = transfer_function(freq, 5000.0)
    assert np.allclose(np.abs(h), 1.0)
