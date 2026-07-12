"""Tests for physkit.units: the dimensional-analysis algebra."""
import pytest
from physkit import units as u
from physkit import constants as c


def test_derived_dimensions():
    assert u.ENERGY == u.MASS * u.LENGTH**2 / u.TIME**2
    assert u.FORCE == u.MASS * u.LENGTH / u.TIME**2
    assert u.VOLTAGE == u.ENERGY / u.CHARGE
    assert u.FREQUENCY == u.DIMENSIONLESS / u.TIME


def test_dimensionless():
    ratio = u.LENGTH / u.LENGTH
    assert ratio.is_dimensionless
    assert (u.ENERGY / u.ENERGY).is_dimensionless


def test_photon_energy_is_energy():
    # E = h f : [J s] * [1/s] = [J] = ENERGY
    h = u.Quantity(c.H, u.ACTION)
    f = u.Quantity(5e14, u.FREQUENCY)
    E = h * f
    E.to_dimension(u.ENERGY)            # raises if wrong
    assert E.dim == u.ENERGY


def test_de_broglie_is_length():
    # lambda = h / p : [J s] / ([kg][m/s]) = [m]
    h = u.Quantity(c.H, u.ACTION)
    p = u.Quantity(1e-24, u.MASS * u.VELOCITY)
    assert (h / p).dim == u.LENGTH


def test_addition_requires_same_dimension():
    e1 = u.Quantity(1.0, u.ENERGY)
    e2 = u.Quantity(2.0, u.ENERGY)
    assert (e1 + e2).value == 3.0
    with pytest.raises(ValueError):
        _ = u.Quantity(1.0, u.ENERGY) + u.Quantity(1.0, u.LENGTH)


def test_to_dimension_raises_on_mismatch():
    with pytest.raises(ValueError):
        u.Quantity(1.0, u.LENGTH).to_dimension(u.TIME)
