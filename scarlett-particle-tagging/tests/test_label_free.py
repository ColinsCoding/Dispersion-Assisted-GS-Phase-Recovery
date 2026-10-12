"""Tests for scarlett.label_free (synthetic SEALS label-free classification)."""
import numpy as np
import pytest
import sympy as sp

from scarlett import label_free as lf


def test_dataset_shapes_and_determinism():
    X, y, ang, names = lf.make_dataset(n_per_class=10, n_angles=20, seed=3)
    assert X.shape == (30, 20) and set(y) == {0, 1, 2} and len(names) == 3
    X2, *_ = lf.make_dataset(n_per_class=10, n_angles=20, seed=3)
    assert np.array_equal(X, X2)
    assert np.all((ang >= 0) & (ang <= np.pi))                            # valid scattering angles
    with pytest.raises(ValueError):
        lf.make_dataset(pixel_noise=-1)


def test_conventional_features_pick_two_columns():
    X, y, ang, _ = lf.make_dataset(n_per_class=5, n_angles=30)
    F, (f_deg, s_deg) = lf.conventional_features(X, ang)
    assert F.shape == (15, 2) and f_deg < s_deg
    assert f_deg == pytest.approx(np.degrees(ang).min())


def test_gini_maximum_symbolic():
    G, sol, Gmax = lf.gini_symbolic(3)
    assert Gmax == sp.Rational(2, 3)
    assert all(sol[s] == sp.Rational(1, 3) for s in sol if str(s).startswith("p"))
    g2, H, p = lf.gini_binary_vs_entropy()
    assert sp.simplify(g2 - 2 * p * (1 - p)) == 0
    assert sp.simplify(H.subs(p, sp.Rational(1, 2))) == 1


def test_full_spectrum_beats_two_angles():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    for gj in (0.0, 0.3):
        X, y, ang, _ = lf.make_dataset(gain_jitter=gj, n_per_class=100)
        tr, te = lf.split(len(y))
        F2, _ = lf.conventional_features(X, ang)
        acc2 = RandomForestClassifier(100, random_state=0).fit(F2[tr], y[tr]).score(F2[te], y[te])
        accS = RandomForestClassifier(100, random_state=0).fit(X[tr], y[tr]).score(X[te], y[te])
        assert accS >= 0.97 and accS > acc2 + 0.08


def test_csv_roundtrip(tmp_path):
    X, y, _, _ = lf.make_dataset(n_per_class=4, n_angles=8)
    p = lf.export_csv(tmp_path / "d.csv", X, y)
    back = np.loadtxt(p, delimiter=",")
    assert np.allclose(back[:, :-1], X, rtol=1e-7) and np.array_equal(back[:, -1], y)
