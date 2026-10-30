"""Numerical counterparts of lean/DispersionPhaseRetrieval.lean on the real dgs/gs_core.py code."""
import numpy as np

from dgs import gs_core as G

N, D1, D2 = 256, -5000.0, -5750.0
E_TRUE = np.exp(1j * np.pi / 2 * np.mod(np.arange(N) // 8 * 7 + 3, 4))
I1, I2 = np.abs(G.disperse(E_TRUE, D1))**2, np.abs(G.disperse(E_TRUE, D2))**2


def test_dispersion_is_unitary_group():
    y = np.exp(1j * np.random.default_rng(0).uniform(0, 2 * np.pi, N))
    assert np.isclose(np.linalg.norm(G.disperse(E_TRUE, D1)), np.linalg.norm(E_TRUE))          # H_norm
    assert np.allclose(G.undisperse(G.disperse(E_TRUE, D1), D1), E_TRUE, atol=1e-12)            # H_inverse
    assert np.allclose(G.disperse(G.disperse(E_TRUE, D1), D2), G.disperse(E_TRUE, D1 + D2), atol=1e-10)  # H_add
    assert np.isclose(np.linalg.norm(G.disperse(E_TRUE - y, D1)), np.linalg.norm(E_TRUE - y))   # dist_preserved


def test_plain_gs_gap_never_increases():
    PA = lambda E: G.undisperse(np.sqrt(I1) * np.exp(1j * np.angle(G.disperse(E, D1))), D1)
    PB = lambda E: G.undisperse(np.sqrt(I2) * np.exp(1j * np.angle(G.disperse(E, D2))), D2)
    x, gaps = PB(G.undisperse(np.sqrt(I1).astype(complex), D1)), []
    for _ in range(60):
        a = PA(x)
        gaps.append(np.linalg.norm(x - a))
        x = PB(a)
    assert np.all(np.diff(gaps) <= 1e-12)                                                      # error_nonincreasing
