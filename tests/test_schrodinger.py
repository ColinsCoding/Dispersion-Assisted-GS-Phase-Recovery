"""Tests for dgs/schrodinger.py — numpy-only 1-D Schrödinger solver."""
import numpy as np
from dgs.schrodinger import (
    solve, finite_square_well, sweep_depth, sweep_width, HBAR2_OVER_2M
)

# ── constant sanity ────────────────────────────────────────────────────────
def test_hbar2_over_2m():
    # ℏ²/2m ≈ 0.03810 eV·nm²  (standard QM textbook value)
    assert abs(HBAR2_OVER_2M - 0.03810) < 5e-4


# ── infinite square well analytic check ───────────────────────────────────
def test_infinite_well_eigenvalues():
    """
    Very deep well ≈ infinite square well.
    E_n = n² π² ℏ²/(2m L²),  n=1,2,3,...
    For L=10 nm: E_1 ≈ 0.03810 * π²/100 ≈ 0.003759 eV
    """
    L = 10.0
    V0 = 1000.0   # deep → approximate infinite well
    x = np.linspace(-15, 15, 2000)
    E, psi = solve(lambda x: finite_square_well(x, L, V0), x, n_states=3)

    E1_analytic = HBAR2_OVER_2M * np.pi**2 / L**2
    # tolerance ~1% (finite-difference grid error)
    assert abs(E[0] - E1_analytic) / E1_analytic < 0.02, \
        f"E1={E[0]:.5f} vs analytic={E1_analytic:.5f}"
    assert abs(E[1] - 4 * E1_analytic) / (4 * E1_analytic) < 0.02
    assert abs(E[2] - 9 * E1_analytic) / (9 * E1_analytic) < 0.02


# ── wavefunctions normalised ──────────────────────────────────────────────
def test_normalisation():
    x = np.linspace(-20, 20, 1000)
    dx = x[1] - x[0]
    E, psi = solve(lambda x: finite_square_well(x, 6.0, 0.5), x, n_states=2)
    for i in range(len(E)):
        norm = np.sum(psi[i]**2) * dx
        assert abs(norm - 1.0) < 1e-3, f"state {i} norm={norm:.6f}"


# ── energies ordered ──────────────────────────────────────────────────────
def test_energies_ascending():
    x = np.linspace(-20, 20, 800)
    E, _ = solve(lambda x: finite_square_well(x, 5.0, 1.0), x, n_states=4)
    assert all(E[i] < E[i+1] for i in range(len(E)-1))


# ── parameter sweeps return correct shapes ────────────────────────────────
def test_sweep_depth_shape():
    r = sweep_depth(L_nm=5.0, n_V=10, n_states=3)
    assert len(r["param"]) == 10
    assert len(r["energies"]) == 10
    assert len(r["thz_01"]) == 10


def test_sweep_width_shape():
    r = sweep_width(V0_eV=1.0, n_L=8, n_states=3)
    assert len(r["param"]) == 8


# ── THz range: GaAs-like well should have 0→1 transition in THz window ───
def test_thz_transition_in_window():
    """
    A 10 nm, 0.3 eV well (GaAs-like) should have 0→1 at a few THz.
    """
    r = sweep_depth(L_nm=10.0, V0_range=(0.3, 0.3), n_V=1, n_states=2)
    f = r["thz_01"][0]
    assert 0.1 < f < 50.0, f"THz transition {f:.2f} THz outside expected range"


if __name__ == "__main__":
    test_hbar2_over_2m()
    test_infinite_well_eigenvalues()
    test_normalisation()
    test_energies_ascending()
    test_sweep_depth_shape()
    test_sweep_width_shape()
    test_thz_transition_in_window()
    print("all tests passed")
