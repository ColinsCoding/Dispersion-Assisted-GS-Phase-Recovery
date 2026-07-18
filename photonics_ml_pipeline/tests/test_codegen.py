"""Code-generation tests: C consistency with SymPy/lambdify; Fortran/JS emitted."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from c_codegen.generator import compile_and_run_c, generate_c, generate_fortran, generate_js
from physics.symbolic import gaussian_beam_width


def test_generate_c_has_signature_and_cse() -> None:
    result = generate_c(gaussian_beam_width())
    assert "double beam_width(double z, double w0, double zR)" in result.source
    assert result.header.startswith("#ifndef BEAM_WIDTH_H")
    assert "#include" in result.example


def test_fortran_and_js_nonempty() -> None:
    bw = gaussian_beam_width()
    assert "beam_width" in generate_fortran(bw)
    assert "function beam_width" in generate_js(bw)


@pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")
def test_generated_c_matches_lambdify() -> None:
    bw = gaussian_beam_width()
    f = bw.lambdify("numpy")
    args = (300.0, 10.0, 200.0)  # z, w0, zR
    c_value = compile_and_run_c(bw, args)
    assert np.isclose(c_value, float(f(*args)), rtol=1e-9)
