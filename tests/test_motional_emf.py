"""Test motional EMF = B h v = the flux rule -dPhi/dt (Griffiths 7.11/7.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import sympy as sp
from griffiths import electrodynamics as ed

B, h, v, t = sp.symbols("B h v t", positive=True)
# moving loop: flux Phi = B*h*(v t) -> flux rule -dPhi/dt = -B h v
assert sp.simplify(ed.induced_emf(B*h*v*t, t) + ed.motional_emf(B, h, v)) == 0
# numeric generator: 0.5 T, 0.1 m, 3 m/s -> 0.15 V
assert abs(ed.motional_emf(0.5, 0.1, 3.0) - 0.15) < 1e-12
# steady loop (v=0) -> no EMF
assert ed.motional_emf(0.5, 0.1, 0.0) == 0.0
for bad in (lambda: ed.motional_emf(1, -1, 1), lambda: ed.motional_emf(1, 1, -1)):
    try: bad()
    except ValueError: pass
    else: raise AssertionError("negative h/v should raise")
print("TEST PASS  (motional EMF B*h*v = -dPhi/dt = the flux rule)")
