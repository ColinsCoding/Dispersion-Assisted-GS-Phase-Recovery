"""Test magnetic-field energy (Griffiths 7.2.4): SymPy recreates W = 1/2 L I^2."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from griffiths import electrodynamics as ed

# 1. SymPy recreates Griffiths 7.30: integral_0^I L i di = 1/2 L I^2
L, I = sp.symbols("L I", positive=True)
assert sp.simplify(ed.magnetic_energy_inductor() - L * I**2 / 2) == 0

# 2. numeric inductor energy 1/2 L I^2
assert abs(ed.inductor_energy(10e-3, 2.0) - 0.5 * 10e-3 * 4) < 1e-15
assert ed.inductor_energy(0, 5) == 0.0
for bad in (lambda: ed.inductor_energy(-1, 2), lambda: ed.inductor_energy(1, -2)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("negative L/I should raise")

# 3. energy density u = B^2/(2 mu0): 1 T field ~ 398 kJ/m^3, scales as B^2
u1 = ed.magnetic_energy_density(1.0)
assert abs(u1 - 1.0 / (2 * ed._MU0)) < 1e-6
assert 3.9e5 < u1 < 4.0e5
assert abs(ed.magnetic_energy_density(2.0) / u1 - 4.0) < 1e-9   # B doubles -> u x4

print(f"TEST PASS  (SymPy: W = L*I^2/2 = Griffiths 7.30; u(1T)={u1/1e3:.0f} kJ/m^3)")
