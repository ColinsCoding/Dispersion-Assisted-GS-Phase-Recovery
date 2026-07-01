import numpy as np
import pytest
import sympy as sp
from dgs.organic_chemistry import (
    homo_lumo_to_wavelength,
    huckel_pi_chain, huckel_benzene,
    organic_semiconductor, lookup_functional_group,
    chromophore_absorption,
    pib_conjugated_chain,
    organic_chemistry_sympy_5,
    hc_eV_nm,
)


# ── HOMO-LUMO / wavelength ────────────────────────────────────────────

def test_homo_lumo_planck_relation():
    gap = 2.5
    r = homo_lumo_to_wavelength(gap)
    assert r["lambda_abs_nm"] == pytest.approx(hc_eV_nm / gap, rel=1e-6)


def test_homo_lumo_visible():
    r = homo_lumo_to_wavelength(2.0)   # 620 nm -- visible
    assert r["visible"] is True


def test_homo_lumo_uv_invisible():
    r = homo_lumo_to_wavelength(6.0)   # 207 nm -- UV, not visible
    assert r["visible"] is False


def test_homo_lumo_invalid():
    with pytest.raises(ValueError):
        homo_lumo_to_wavelength(0)


# ── Huckel pi chain ───────────────────────────────────────────────────

def test_huckel_gap_decreases_with_n():
    gaps = [huckel_pi_chain(N)["gap_eV"] for N in [2, 4, 6, 8]]
    for i in range(len(gaps) - 1):
        assert gaps[i] > gaps[i+1], f"gap did not decrease from N={2*(i+1)} to N={2*(i+2)}"


def test_huckel_ethylene_n2():
    r = huckel_pi_chain(2)
    # E = alpha + 2*beta*cos(k*pi/3): k=1: alpha+beta; k=2: alpha-beta
    # gap = (alpha-beta) - (alpha+beta) = -2*beta = 4.8 eV
    assert r["gap_eV"] == pytest.approx(4.8, abs=0.01)


def test_huckel_n6_visible():
    r = huckel_pi_chain(6)
    assert 380 < r["lambda_abs_nm"] < 700


def test_huckel_invalid_n():
    with pytest.raises(ValueError):
        huckel_pi_chain(1)


def test_huckel_returns_n_levels():
    N = 6
    r = huckel_pi_chain(N)
    assert len(r["E_k_eV"]) == N


# ── Huckel benzene ring ───────────────────────────────────────────────

def test_benzene_uv():
    b = huckel_benzene()
    assert b["lambda_abs_nm"] < 380   # UV, not visible


def test_benzene_6_levels():
    b = huckel_benzene()
    assert len(b["E_k_eV"]) == 6


def test_benzene_gap_positive():
    b = huckel_benzene()
    assert b["gap_eV"] > 0


# ── organic semiconductors ────────────────────────────────────────────

def test_pentacene_lookup():
    osc = organic_semiconductor("pentacene")
    assert osc["gap_eV"] == pytest.approx(1.9, abs=0.01)
    assert osc["HOMO_eV"] < osc["LUMO_eV"]


def test_organic_semiconductor_invalid():
    with pytest.raises(ValueError):
        organic_semiconductor("unobtanium")


def test_all_semiconductors_have_gap():
    for name in ["pentacene", "PCBM", "Alq3", "P3HT", "ITIC"]:
        osc = organic_semiconductor(name)
        assert osc["gap_eV"] > 0
        assert osc["HOMO_eV"] < osc["LUMO_eV"]


# ── functional groups ─────────────────────────────────────────────────

def test_lookup_alcohol():
    g = lookup_functional_group("alcohol")
    assert "-OH" in g["formula"]


def test_lookup_case_insensitive():
    g = lookup_functional_group("Alkane")
    assert "formula" in g


def test_lookup_invalid():
    with pytest.raises(ValueError):
        lookup_functional_group("unobtanium")


# ── Beer-Lambert ──────────────────────────────────────────────────────

def test_beer_lambert_zero_path():
    r = chromophore_absorption(10000, 0.01, 0)
    assert r["absorbance_A"] == pytest.approx(0.0)
    assert r["transmittance_T"] == pytest.approx(1.0)


def test_beer_lambert_transmittance_range():
    r = chromophore_absorption(30000, 0.02, 1e-6)
    assert 0 < r["transmittance_T"] <= 1.0


def test_beer_lambert_full_absorption():
    r = chromophore_absorption(100000, 1.0, 1.0)
    assert r["transmittance_T"] < 1e-10


def test_beer_lambert_invalid():
    with pytest.raises(ValueError):
        chromophore_absorption(-1, 0.01, 1)


# ── particle-in-a-box ─────────────────────────────────────────────────

def test_pib_wavelength_increases_with_n():
    wl2 = pib_conjugated_chain(2)["lambda_abs_nm"]
    wl4 = pib_conjugated_chain(4)["lambda_abs_nm"]
    assert wl4 > wl2   # longer chain -> longer wavelength


def test_pib_carotene_11():
    # PIB is a crude model: longer chains still have positive gap
    r = pib_conjugated_chain(11)
    assert r["lambda_abs_nm"] > 0
    assert r["gap_eV"] > 0


def test_pib_planck_consistency():
    r = pib_conjugated_chain(5)
    assert r["lambda_abs_nm"] == pytest.approx(hc_eV_nm / r["gap_eV"], rel=1e-4)


def test_pib_invalid():
    with pytest.raises(ValueError):
        pib_conjugated_chain(0)


# ── SymPy 5 ──────────────────────────────────────────────────────────

def test_organic_sympy_5_count():
    eqs = organic_chemistry_sympy_5()
    assert len(eqs) == 5


def test_organic_sympy_5_types():
    eqs = organic_chemistry_sympy_5()
    for k, eq in eqs.items():
        assert isinstance(eq, sp.Basic), f"{k} is not SymPy"
