"""The Python SEALS port against the lab MATLAB, plus the Rayleigh-Debye check.

tests/data/matlab_defaults.csv was written by seals/matlab/run_defaults.m
(MATLAB R2025b), the default case from the lab's main.m. Columns:
lambda, y, theta_deg(+mangle), I_p, I_s, phase_p, phase_s, I_rdg(MATLAB formula).
"""
import pathlib
import numpy as np
import pytest
from seals import scattering as S

DATA = pathlib.Path(__file__).parent / "data" / "matlab_defaults.csv"
M = np.loadtxt(DATA, delimiter=",")
LAM, Y, TH, IP, IS, TP, TS, IRD = M.T
p = S.DEFAULTS
LAM0 = (p["lam1"] + p["lam2"]) / 2
SIGMA_MATLAB = 1.640005703457e-10


def rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


def test_geometry_matches_matlab():
    y, th, valid = S.seals(p["d"], p["D"], p["a"], p["dcorr"], p["P"], p["NA"], LAM)
    assert valid.all()
    assert rel(y[:-1], Y[:-1]) < 1e-9          # last point is 0 by construction
    assert rel(th + p["mangle"], TH) < 1e-9


def test_mie_matches_matlab():
    sig, ip, is_, an, bn, tp, ts = S.mie(p["npar"], p["nmed"], p["dia"], LAM0, np.deg2rad(TH), p["r"])
    assert an.size == 32
    assert rel(ip, IP) < 1e-12 and rel(is_, IS) < 1e-12
    assert np.max(np.abs(np.angle(np.exp(1j * (tp - TP))))) < 1e-12
    assert np.max(np.abs(np.angle(np.exp(1j * (ts - TS))))) < 1e-12
    assert abs(sig - SIGMA_MATLAB) / SIGMA_MATLAB < 1e-12


def test_matlab_rdg_formula_reproduced_exactly():
    got = S.rayleigh_debye_matlab(p["dia"], LAM0, p["nmed"], p["npar"], np.deg2rad(TH), p["r"])
    assert rel(got, IRD) < 1e-9   # worst case sits near a zero of P(u)


@pytest.mark.parametrize("dia,npar", [(300e-9, 1.01), (800e-9, 1.01), (3000e-9, 1.002)])
def test_corrected_rdg_follows_mie_where_rdg_is_valid(dia, npar):
    lam, th = 1590e-9, np.linspace(0.05, np.pi - 0.05, 400)
    _, ip, is_, *_ = S.mie(npar, 1.0, dia, lam, th, 0.1)
    mie_I = ip + is_
    keep = mie_I > 1e-4 * mie_I.max()

    def shape_err(I):
        return np.max(np.abs(np.log10(I[keep] / I[0]) - np.log10(mie_I[keep] / mie_I[0])))

    fixed = shape_err(S.rayleigh_debye(dia, lam, 1.0, npar, th, 0.1))
    lab = shape_err(S.rayleigh_debye_matlab(dia, lam, 1.0, npar, th, 0.1))
    assert fixed < 0.06          # within ~15 % of Mie in shape
    assert fixed < lab


def test_corrected_rdg_absolute_level_matches_mie_for_tiny_sphere():
    # mie() keeps mie-2.m's convention, where I_p + I_s is TWICE the unpolarized
    # intensity (both field components carry cos(phi)). Unpolarized = half the sum.
    lam, th = 1590e-9, np.linspace(0.1, np.pi - 0.1, 50)
    _, ip, is_, *_ = S.mie(1.01, 1.0, 50e-9, lam, th, 0.1)
    assert rel(S.rayleigh_debye(50e-9, lam, 1.0, 1.01, th, 0.1), (ip + is_) / 2) < 1e-2
    assert rel(S.rayleigh_debye(50e-9, lam, 1.0, 1.01, th, 0.1), ip + is_) > 0.49
