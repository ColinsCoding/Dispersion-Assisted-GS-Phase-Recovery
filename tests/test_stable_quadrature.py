"""Smoke tests for dgs.stable_quadrature: quadrature accuracy, Fresnel vs scipy, stability primitives."""
import numpy as np
from dgs import stable_quadrature as sq


def test_gauss_legendre_polynomial_exact():
    # 3-point Gauss-Legendre is exact for degree <= 5; use a degree-5 polynomial
    f = lambda t: 3 * t**5 - 2 * t**3 + t - 4
    exact = 13.5                                   # int_{-1}^{2} (3t^5 - 2t^3 + t - 4) dt
    assert abs(sq.gauss_legendre(f, -1, 2, n=3) - exact) < 1e-10


def test_gauss_legendre_exp():
    assert abs(sq.gauss_legendre(np.exp, 0, 1, n=64) - (np.e - 1)) < 1e-12


def test_gauss_legendre_degenerate_interval():
    assert sq.gauss_legendre(np.sin, 2.0, 2.0) == 0.0


def test_adaptive_simpson_sin():
    assert abs(sq.adaptive_simpson(np.sin, 0.0, np.pi, tol=1e-12) - 2.0) < 1e-9


def test_adaptive_simpson_oscillatory():
    # cos(pi t^2/2) over [0, 4] oscillates ~4x; adaptive refinement must still nail it
    got = sq.adaptive_simpson(lambda t: np.cos(np.pi * t * t / 2.0), 0.0, 4.0, tol=1e-12)
    from scipy.special import fresnel
    _, C4 = fresnel(4.0)
    assert abs(got - C4) < 1e-8


def test_fresnel_matches_scipy():
    from scipy.special import fresnel                # scipy returns (S, C)
    for x in [0.3, 1.0, 2.5, 3.7, -1.7]:
        S_ref, C_ref = fresnel(x)
        assert abs(sq.fresnel_C(x) - C_ref) < 1e-8
        assert abs(sq.fresnel_S(x) - S_ref) < 1e-8


def test_fresnel_odd_and_origin():
    assert abs(sq.fresnel_C(0.0)) < 1e-12
    assert abs(sq.fresnel_S(0.0)) < 1e-12
    assert abs(sq.fresnel_C(-1.3) + sq.fresnel_C(1.3)) < 1e-10   # odd


def test_knife_edge_shadow_edge_is_quarter():
    # the defining result: intensity at the geometric shadow boundary is exactly I0/4
    assert abs(sq.knife_edge_intensity(0.0) - 0.25) < 1e-10


def test_knife_edge_shadow_and_light():
    assert sq.knife_edge_intensity(-4.0) < 0.02      # deep geometric shadow -> ~0
    assert 0.8 < sq.knife_edge_intensity(6.0) < 1.25 # lit region oscillates about 1 (first fringes)


def test_logsumexp_no_overflow():
    a = [1000.0, 1001.0, 1002.0]
    expected = 1002.0 + np.log(np.exp(-2.0) + np.exp(-1.0) + 1.0)
    assert abs(sq.logsumexp(a) - expected) < 1e-9
    assert np.isfinite(sq.logsumexp(a))
    with np.errstate(over="ignore"):                    # the naive route deliberately overflows
        assert not np.isfinite(np.log(np.sum(np.exp(a))))


def test_logsumexp_axis():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    out = sq.logsumexp(a, axis=0)
    assert out.shape == (2,)
    assert np.allclose(out, np.log(np.exp(0.0) + np.exp(1.0)))


def _naive_sum(xs):
    # an explicitly sequential accumulation -- CPython's built-in sum() now uses Neumaier
    # compensation internally, so it is not a valid "naive" baseline.
    s = 0.0
    for x in xs:
        s += x
    return s


def test_kahan_beats_naive():
    # each +0.5 is below the ulp/2 of 1e16, so sequential summation rounds it away and loses all 100;
    # Kahan's compensation accumulates the lost bits and recovers the exact 1e16 + 100.
    big = 1e16
    xs = [big] + [0.5] * 200
    true = big + 100.0                                # 1e16 + 100 is exactly representable
    kahan, naive = sq.kahan_sum(xs), _naive_sum(xs)
    assert kahan == true                              # compensated summation is exact here
    assert naive == big                               # sequential accumulation drops every 0.5
    assert abs(kahan - true) < abs(naive - true)      # -> strictly more accurate


def test_input_validation():
    for bad in (lambda: sq.gauss_legendre(np.sin, 0, 1, n=0),
                lambda: sq.adaptive_simpson(np.sin, 0, 1, tol=0.0),
                lambda: sq.adaptive_simpson(np.sin, 0, 1, max_depth=0),
                lambda: sq.logsumexp([])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
    print("all stable_quadrature tests passed")
