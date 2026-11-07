import math

import numpy as np
import pytest
import sympy as sp

from dgs import mutual_inductance as mi


def test_mutual_and_k_round_trip():
    M = mi.mutual_inductance(4e-3, 9e-3, 0.5)
    assert M == pytest.approx(0.5 * 6e-3)
    assert mi.coupling_coefficient(4e-3, 9e-3, M) == pytest.approx(0.5)
    assert mi.coupling_coefficient(4e-3, 9e-3, -M) == pytest.approx(0.5)     # dot reversed


@pytest.mark.parametrize("bad", [0, -1e-3, float("nan"), float("inf")])
def test_rejects_bad_inductance(bad):
    with pytest.raises(ValueError, match="positive"):
        mi.mutual_inductance(bad, 1e-3, 0.5)


@pytest.mark.parametrize("k", [-0.1, 1.01])
def test_rejects_bad_k(k):
    with pytest.raises(ValueError, match="0 <= k <= 1"):
        mi.mutual_inductance(1e-3, 1e-3, k)


def test_rejects_M_above_bound():
    with pytest.raises(ValueError, match="unphysical"):
        mi.coupling_coefficient(1e-3, 4e-3, 2.1e-3)


def test_energy_nonnegative_iff_k_le_1():
    rng = np.random.default_rng(0)
    L1, L2 = 2e-3, 5e-3
    Mmax = math.sqrt(L1 * L2)
    I = rng.normal(size=(5000, 2))
    W = mi.energy(L1, L2, 0.999 * Mmax, I[:, 0], I[:, 1])
    assert W.min() >= 0
    assert mi.inductance_matrix_is_psd(L1, L2, Mmax)
    assert not mi.inductance_matrix_is_psd(L1, L2, 1.01 * Mmax)
    # with |M| > Mmax the ratio I1/I2 = -M/L1 gives negative energy
    M = 1.01 * Mmax
    assert mi.energy(L1, L2, M, -M / L1, 1.0) < 0


def test_sympy_energy_bound():
    d = mi.derive_energy_bound()
    L1, L2 = sp.symbols("L_1 L_2", positive=True)
    M, I2 = sp.symbols("M I_2", real=True)
    assert sp.simplify(d["x_star"] + M / L1) == 0
    assert sp.simplify(d["W_min"] - (L1 * L2 - M ** 2) * I2 ** 2 / (2 * L1)) == 0
    assert sp.expand(d["discriminant"] - (M ** 2 - L1 * L2)) == 0
    assert sp.expand(d["det"] - (L1 * L2 - M ** 2)) == 0


def test_series_recipe():
    L1, L2, M = 3e-3, 7e-3, 1.5e-3
    La, Lo = mi.series_aiding(L1, L2, M), mi.series_opposing(L1, L2, M)
    assert La == pytest.approx(13e-3) and Lo == pytest.approx(7e-3)
    assert mi.mutual_from_series(La, Lo) == pytest.approx(M)
    assert mi.k_from_series(La, Lo, L1, L2) == pytest.approx(M / math.sqrt(L1 * L2))


def test_open_short_recipe():
    L1, L2, k = 3e-3, 7e-3, 0.8
    Ls = mi.short_circuit_inductance(L1, L2, k)
    assert Ls == pytest.approx(L1 * 0.36)
    assert mi.k_from_open_short(L1, Ls) == pytest.approx(k)
    # matches Z_in with a short load, divided by j w
    w = 2 * math.pi * 1e3
    Zin = mi.input_impedance(L1, L2, mi.mutual_inductance(L1, L2, k), 0.0, w)
    assert (Zin / (1j * w)).real == pytest.approx(Ls)


def test_nested_solenoids_k():
    r = mi.nested_solenoids(100, 1e-4, 200, 1e-4, 0.1)
    assert r["k"] == pytest.approx(1.0)
    assert r["M"] == pytest.approx(math.sqrt(r["L1"] * r["L2"]))
    r = mi.nested_solenoids(100, 1e-4, 200, 4e-4, 0.1)
    assert r["k"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mi.nested_solenoids(100, 4e-4, 200, 1e-4, 0.1)


def test_elliptic_integrals():
    assert mi.elliptic_K(0) == pytest.approx(math.pi / 2)
    assert mi.elliptic_E(0) == pytest.approx(math.pi / 2)
    # K(1/sqrt2) = Gamma(1/4)^2 / (4 sqrt(pi)); E(1/sqrt2) from Legendre's relation
    K = math.gamma(0.25) ** 2 / (4 * math.sqrt(math.pi))
    assert mi.elliptic_K(1 / math.sqrt(2)) == pytest.approx(K, rel=1e-13)
    E = (math.pi / 2 + K ** 2) / (2 * K)
    assert mi.elliptic_E(1 / math.sqrt(2)) == pytest.approx(E, rel=1e-13)
    for m in (0.3, 0.7, 0.95):
        Kp, Ep = mi._agm_KE(math.sqrt(1 - m * m))
        Km, Em = mi._agm_KE(m)
        assert Em * Kp + Ep * Km - Km * Kp == pytest.approx(math.pi / 2, rel=1e-12)   # Legendre


def test_loops_far_field_dipole_limit():
    a, b, d = 0.01, 0.02, 1.0
    dipole = mi.MU0 * math.pi * a ** 2 * b ** 2 / (2 * d ** 3)
    assert mi.coaxial_loops_mutual(a, b, d) == pytest.approx(dipole, rel=1e-3)


@pytest.mark.parametrize("a,b,d", [(0.02, 0.03, 0.01), (0.05, 0.05, 0.02), (0.01, 0.04, 0.0)])
def test_neumann_matches_maxwell(a, b, d):
    Mx = mi.coaxial_loops_mutual(a, b, d)
    assert mi.neumann_coaxial(a, b, d) == pytest.approx(Mx, rel=1e-10)
    P1 = mi.circle_points(a, n=600)
    P2 = mi.circle_points(b, (0, 0, d), n=600)
    assert mi.neumann_polyline(P1, P2) == pytest.approx(Mx, rel=2e-4)


def test_neumann_polyline_symmetric_and_converges():
    P1 = mi.circle_points(0.02, n=300)
    P2 = mi.circle_points(0.03, (0.01, 0.005, 0.015), normal=(0.2, 0, 1), n=300)
    assert mi.neumann_polyline(P1, P2) == pytest.approx(mi.neumann_polyline(P2, P1), rel=1e-12)
    exact = mi.coaxial_loops_mutual(0.02, 0.03, 0.01)
    errs = [abs(mi.neumann_polyline(mi.circle_points(0.02, n=n),
                                    mi.circle_points(0.03, (0, 0, 0.01), n=n)) - exact) for n in (100, 200)]
    assert errs[1] < errs[0] / 3            # ~1/n^2


def test_transformer_k1_is_ideal():
    L1, L2 = 1.0, 0.01                      # 10:1 step-down
    M = mi.mutual_inductance(L1, L2, 1.0)
    w = 2 * math.pi * 60
    for ZL in (8.0, 100.0, 3 + 4j):
        assert mi.voltage_ratio(L1, L2, M, ZL, w) == pytest.approx(0.1, rel=1e-12)
    # huge L with fixed ratio: Z_in -> n^-2 Z_L = 100 Z_L
    big = mi.input_impedance(1e4, 1e2, 1e3, 8.0, w)
    assert big == pytest.approx(800.0, rel=1e-3)


def test_t_equivalent_reproduces_impedance_matrix():
    L1, L2, M = 2e-3, 0.5e-3, 0.8e-3
    a, b, c = mi.t_equivalent(L1, L2, M)
    w, ZL = 2 * math.pi * 1e5, 5.0
    # T network: series a, shunt c, series b then load
    Zb = 1j * w * b + ZL
    Zt = 1j * w * a + (1j * w * c * Zb) / (1j * w * c + Zb)
    assert Zt == pytest.approx(mi.input_impedance(L1, L2, M, ZL, w), rel=1e-12)


def test_leakage_split():
    r = mi.leakage_and_magnetizing(4e-3, 1e-3, 0.95)
    assert r["leak1"] + r["Lm1"] == pytest.approx(4e-3)
    assert r["n"] == pytest.approx(0.5)


def test_link_efficiency():
    assert mi.link_efficiency_max(0.0, 100, 100) == 0
    assert mi.link_efficiency_max(0.2, 100, 100) == pytest.approx(400 / (1 + math.sqrt(401)) ** 2)
    assert mi.link_efficiency_max(1.0, 1e4, 1e4) > 0.99


def test_spiral_coils_consistent():
    r = mi.spiral_radii(0.01, 0.02, 8)
    L = mi.spiral_self_inductance(r, 0.4e-3)
    M = mi.spiral_mutual(r, r, 0.005)
    k = mi.coupling_coefficient(L, L, M)
    assert 0.2 < k < 0.9
    assert mi.spiral_mutual(r, r, 0.02) < M                     # coupling falls with gap
    Moff = mi.spiral_mutual_offset(r[::4], r[::4], 0.005, 0.0, n=300)
    assert Moff == pytest.approx(mi.spiral_mutual(r[::4], r[::4], 0.005), rel=1e-3)
