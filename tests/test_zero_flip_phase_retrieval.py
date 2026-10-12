"""Tests for dgs/zero_flip_phase_retrieval.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.zero_flip_phase_retrieval import (
    zeros_of, signal_from_zeros, flip_zero, all_flips, fourier_magnitude, distinguishability,
    symbolic_flip_identity, max_dispersion_without_wrap,
)

RNG = np.random.default_rng(1)
X = RNG.normal(size=6) + 1j * RNG.normal(size=6)


def test_zeros_roundtrip():
    assert np.allclose(signal_from_zeros(zeros_of(X), X[-1]), X)
    with pytest.raises(ValueError):
        zeros_of([1.0])


def test_single_flip_keeps_magnitude_changes_signal():
    for k in range(5):
        y = flip_zero(X, k)
        assert np.allclose(fourier_magnitude(y), fourier_magnitude(X), atol=1e-12)
        assert np.linalg.norm(y - X) > 1e-3
        zy = zeros_of(y)
        assert np.min(np.abs(zy - 1 / np.conj(zeros_of(X)[k]))) < 1e-9


def test_all_flips_share_one_fourier_magnitude():
    F = all_flips(X)
    assert len(F) == 2 ** 5
    M0 = fourier_magnitude(X)
    assert all(np.allclose(fourier_magnitude(f), M0, atol=1e-12) for f in F)
    # flipping every zero = time-reversed conjugate (the trivial ambiguity), up to a phase
    rev = np.conj(X[::-1])
    full = F[-1]
    c = np.vdot(rev, full) / np.vdot(rev, rev)
    assert abs(abs(c) - 1) < 1e-9 and np.allclose(full, c * rev, atol=1e-9)


def test_symbolic_identity():
    expr, (wr, wi, ar, ai) = symbolic_flip_identity()
    assert sp.simplify(expr + (ar**2 + ai**2 - 1) * (wr**2 + wi**2 - 1)) == 0
    assert sp.simplify(expr.subs(wi, sp.sqrt(1 - wr**2))) == 0              # zero on |w| = 1


def test_far_field_cannot_tell_flips_apart():
    F = all_flips(X)
    N = 65536
    assert 1e5 <= max_dispersion_without_wrap(N)
    near = distinguishability(F, 1000, N=N)
    far = distinguishability(F, 1e5, N=N)
    assert far < 0.2 * near
