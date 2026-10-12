"""Tests for dgs.hydrogen_lines."""
import numpy as np
import pytest
import sympy as sp

from dgs import hydrogen_atom as ha
from dgs import hydrogen_lines as h


def test_question_answer():
    a, b = h.identify_line(121.6), h.identify_line(1282)
    assert (a["n"], a["m"], a["name"], a["region"]) == (2, 1, "Lyman alpha", "ultraviolet")
    assert (b["n"], b["m"], b["name"], b["region"]) == (5, 3, "Paschen beta", "infrared")
    assert a["energy_eV"] == pytest.approx(10.20, abs=0.01)
    assert b["energy_eV"] == pytest.approx(0.967, abs=0.002)
    assert a["runner_up"][2] > 10 and b["runner_up"][2] > 100     # unambiguous
    sym = h.solve_question_symbolic()
    assert [(n, m) for _, n, m, _, _ in sym] == [(2, 1), (5, 3)]
    assert sym[0][3] == sp.Rational(3, 4) and sym[1][3] == sp.Rational(16, 225)


def test_reduced_mass():
    assert h.rydberg_symbolic()[2] == 0
    assert h.rydberg_constant() == pytest.approx(10967758.3, rel=1e-8)
    assert h.wavelength_nm(2, 1, False) == pytest.approx(ha.transition_wavelength_nm(2, 1), rel=1e-8)
    assert h.wavelength_nm(2, 1) > h.wavelength_nm(2, 1, False)
    with pytest.raises(ValueError):
        h.wavelength_nm(1, 2)


def test_encoder_decoder():
    exact = [h.wavelength_nm(n, m) for n, m in [(2, 1), (3, 1), (3, 2), (4, 2), (5, 3), (4, 3)]]
    lat = h.encode(exact)
    assert lat["pairs"] == [(2, 1), (3, 1), (3, 2), (4, 2), (5, 3), (4, 3)]
    assert lat["R"] == pytest.approx(h.rydberg_constant(), rel=1e-12)
    assert 1 / h.reduced_mass_from_R(lat["R"]) == pytest.approx(h.M_P / h.M_E, rel=1e-6)
    assert h.decode(lat, (6, 2)) == pytest.approx(h.wavelength_nm(6, 2))


def test_animation(tmp_path):
    p, alt = h.emission_animation(str(tmp_path / "a.gif"), frames=12)
    from PIL import Image
    im = Image.open(p)
    assert im.n_frames >= 10 and "121.6" in alt and "1282.2" in alt
