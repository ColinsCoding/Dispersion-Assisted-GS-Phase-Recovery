"""Tests for dgs/nuclide_notation.py (+ opt-in MATLAB cross-check: RUN_MATLAB=1)."""
import os
import pathlib
import shutil
import subprocess

import pytest
import sympy as sp

from dgs.nuclide_notation import (
    parse, particle_counts, symbolic_rules, check_options, KNIGHT_OPTIONS,
)


@pytest.mark.parametrize("s,expect", [
    ("16O2-", (8, 8, 10)), ("23Na+", (11, 12, 10)), ("238U", (92, 146, 92)),
    ("4He2+", (2, 2, 0)), ("35Cl-", (17, 18, 18)), ("56Fe3+", (26, 30, 23)), ("1H", (1, 0, 1)),
])
def test_counts(s, expect):
    assert particle_counts(s) == expect


def test_symbolic_rules():
    sol, (A, Z, q) = symbolic_rules()
    p, n, e = sp.symbols("p n e", integer=True)
    assert sol == {p: Z, n: A - Z, e: Z - q}


def test_exactly_one_option_correct_with_reasons():
    rows = check_options("16O2-", KNIGHT_OPTIONS)
    assert [r[2] for r in rows] == [True, False, False, False, False, False]
    assert "extra charge" in rows[1][3]
    assert "mass number" in rows[3][3] and "A instead of Z" in rows[3][3]


@pytest.mark.parametrize("bad", ["O16", "16Xx", "1O", "1H2+"])
def test_validation(bad):
    with pytest.raises(ValueError):
        particle_counts(bad)


@pytest.mark.skipif(os.environ.get("RUN_MATLAB") != "1" or not shutil.which("matlab")
                    and not pathlib.Path(r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe").exists(),
                    reason="set RUN_MATLAB=1 (MATLAB startup is slow)")
def test_matlab_twin_agrees():
    exe = shutil.which("matlab") or r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
    root = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run([exe, "-batch", "run('learn/matlab/nuclide_options.m')"], cwd=root,
                         capture_output=True, text=True, timeout=600).stdout
    assert "8 protons, 8 neutrons, 10 electrons" in out and "MATLAB checks passed" in out
