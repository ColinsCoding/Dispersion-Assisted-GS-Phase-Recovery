"""Tests for dgs/coded_illumination_design.py (learned illumination for phase recovery)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from dgs import coded_illumination_design as cid


def test_fidelity_ignores_global_phase():
    x = torch.tensor(cid.random_samples(3, 16, 0))
    assert torch.allclose(cid.fidelity(x * np.exp(1j * 1.3), x), torch.ones(3), atol=1e-5)


def test_power_budget_fixed():
    m = cid._normalize(torch.randn(4, 32), torch)
    assert float(m.mean()) == pytest.approx(0.5, abs=1e-5) and float(m.min()) >= 0


def test_many_clean_masks_recover_phase():
    # sanity of the reconstructor itself: random masks + many iterations -> near-perfect recovery
    # (measured 0.976 at K=16, 1000 iterations; GS is slow, 200 iterations only reaches ~0.6)
    torch.manual_seed(0)
    x = torch.tensor(cid.random_samples(8, 32, 3))
    m = cid._normalize(0.5 * torch.randn(16, 32), torch)
    y = cid._measure(x, m, torch)
    assert float(cid.fidelity(cid.reconstruct(y, m, 1000, torch), x).mean()) > 0.9


def test_learned_masks_beat_random():
    r = cid.train_masks(K=4, seed=0, steps=150)
    assert r['fid_learned'] > 2 * r['fid_random']
    assert r['fid_learned'] > 0.25


def test_bad_args():
    with pytest.raises(ValueError):
        cid.train_masks(K=0)
