"""Tests for the physics models, uncertainty propagation, synthetic data and classifiers.

Run from the project root:  python -m pytest -q
Optional cross-checks (skipped if the tool is missing): MATLAB CSV, C program.
"""
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import sympy as sp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import physics_models as pm                         # noqa: E402
from src import synthetic_data as sd                         # noqa: E402
from src import uncertainty as un                            # noqa: E402
from src import classifiers as cl                            # noqa: E402


def test_photon_energy():
    assert pm.photon_energy_eV(1550e-9) == pytest.approx(0.79990, abs=1e-4)
    E = pm.photon_energy_symbolic().rhs
    h, c, lam = sp.symbols("h c lambda", positive=True)
    assert sp.simplify(E - h * c / lam) == 0
    with pytest.raises(ValueError):
        pm.photon_energy_J(0)


def test_beer_lambert():
    assert pm.beer_lambert(2.0, 100.0, 0.01) == pytest.approx(2.0 * np.exp(-1))
    sol = pm.beer_lambert_symbolic()
    I0, alpha, L = sp.symbols("I_0 alpha L", positive=True)
    assert sp.simplify(sol.rhs - I0 * sp.exp(-alpha * L)) == 0


def test_rc_and_rlc():
    t, R, C, V = sp.symbols("t R C V", positive=True)
    assert sp.simplify(pm.rc_symbolic().rhs - V * (1 - sp.exp(-t / (R * C)))) == 0
    assert pm.rc_step(1e-3, 5.0, 1e3, 1e-6) == pytest.approx(5 * (1 - np.exp(-1)))
    Z, w0 = pm.rlc_impedance_symbolic()
    Ls, Cs = sp.symbols("L C", positive=True)
    assert sp.simplify(w0 - 1 / sp.sqrt(Ls * Cs)) == 0
    L, Cv = 1e-9, 1e-12
    f0 = pm.resonance_hz(L, Cv)
    assert pm.series_rlc_impedance(f0, 50.0, L, Cv) == pytest.approx(50.0 + 0j, abs=1e-6)


def test_reflection_and_power():
    assert pm.reflection_coefficient(50.0) == 0
    assert pm.return_loss_db(0.1) == pytest.approx(20.0)
    assert pm.power_W(5.0, 100.0) == pytest.approx(0.25)
    assert sp.simplify(pm.power_symbolic().rhs - sp.Symbol("V", positive=True) ** 2 / sp.Symbol("R", positive=True)) == 0


def test_power_uncertainty_three_ways():
    P, sP = un.power_uncertainty(5.0, 0.05, 100.0, 1.0)
    _, sP_mc = un.power_uncertainty_mc(5.0, 0.05, 100.0, 1.0)
    assert sP == pytest.approx(0.25 * np.sqrt(0.02 ** 2 + 0.01 ** 2))
    assert sP_mc == pytest.approx(sP, rel=0.02)
    _, _, rel, (V, R, sV, sR) = un.power_uncertainty_symbolic()
    assert float(rel.subs({V: 5, R: 100, sV: 0.05, sR: 1})) == pytest.approx(sP / P)


def test_synthetic_data_is_deterministic_and_labelled():
    X1, y1, _ = sd.make_dataset(10, seed=3)
    X2, y2, _ = sd.make_dataset(10, seed=3)
    assert X1.equals(X2) and y1.equals(y2)
    assert set(y1) == set(sd.CLASSES) and list(X1.columns) == list(sd.FEATURES)
    with pytest.raises(ValueError):
        sd.make_measurement("not-a-class", np.random.default_rng(0))


def test_resonator_feature_physics():
    amp, phase = sd.microwave_resonant(np.random.default_rng(0))
    assert amp.min() < 0.3 and amp.max() > 0.9            # reflection dip at resonance


def test_classifiers_no_leakage_and_metrics():
    X, y, _ = sd.make_dataset(40, seed=0)
    Xtr, Xte, ytr, yte = cl.split(X, y)
    assert set(Xtr.index).isdisjoint(Xte.index)
    r = cl.evaluate(cl.decision_tree(), Xtr, Xte, ytr, yte, list(sd.CLASSES))
    assert 0 <= r["balanced_accuracy"] <= 1 and r["confusion"].sum() == len(yte)
    lo, hi = un.bootstrap_ci(yte, r["pred"], lambda a, b: np.mean(a == b), n_boot=200)
    assert lo <= r["accuracy"] <= hi + 1e-12


def test_matlab_csv_matches_python_if_present():
    csv = ROOT / "results" / "matlab_reference_features.csv"
    if not csv.exists():
        pytest.skip("run matlab/optical_rf_reference.m first")
    m = pd.read_csv(csv).set_index("quantity")["value"]
    py = pm.reference_features()
    for k, v in py.items():
        assert m[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k


def test_c_program_if_gcc_available(tmp_path):
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc not on PATH")
    exe = tmp_path / "uncertainty_demo"
    subprocess.run([gcc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(ROOT / "c" / "uncertainty_demo.c"),
                    "-lm", "-o", str(exe)], check=True, capture_output=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    assert "sigma_P = 0.005590 W" in out
