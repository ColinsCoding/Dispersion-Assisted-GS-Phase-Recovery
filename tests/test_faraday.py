"""Smoke-test Faraday EMF: changing flux -> nonzero loop integral -> KVL needs EMF."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from griffiths import electrodynamics as ed

t = sp.Symbol("t", real=True)
N, A, B0, w = sp.symbols("N A B_0 omega", positive=True)

# 1. symbolic Faraday: EMF = -dPhi/dt for Phi = N A B0 sin(omega t)
flux = N * A * B0 * sp.sin(w * t)
emf = ed.induced_emf(flux, t)
assert sp.simplify(emf - (-N * A * B0 * w * sp.cos(w * t))) == 0

# 2. a CONSTANT flux gives zero EMF (naive KVL holds)
assert ed.induced_emf(N * A * B0, t) == 0

# 3. numeric sinusoidal EMF: peak = N A B0 omega, 90 deg out of phase with flux
ts = np.linspace(0, 2 * np.pi, 60000)                      # ~600 samples/period (resolve peak)
e = ed.sinusoidal_emf(2, 0.01, 0.5, 100.0, ts)
assert abs(e.max() - 2 * 0.01 * 0.5 * 100.0) < 1e-3        # peak = N A B0 omega = 1.0
# flux ~ sin, emf ~ -cos: zero-crossings of emf at flux extrema (quadrature)
flux_num = 2 * 0.01 * 0.5 * np.sin(100.0 * ts)
i_fluxmax = np.argmax(flux_num)
assert abs(e[i_fluxmax]) < 0.02 * e.max()                  # EMF ~ 0 where flux peaks

# 4. corrected KVL: induced current I = EMF / R; zero when flux is steady
I = ed.loop_current(e, R=10.0)
assert np.allclose(I, e / 10.0)
assert np.all(ed.loop_current(ed.sinusoidal_emf(1, 1.0, 1.0, 0.0, ts), 10.0) == 0)  # omega=0 -> no EMF

# 5. faster change -> bigger EMF (it's the *rate*, not the flux, that matters)
slow = ed.sinusoidal_emf(1, 1.0, 1.0, 10.0, ts).max()
fast = ed.sinusoidal_emf(1, 1.0, 1.0, 50.0, ts).max()
assert abs(fast / slow - 5.0) < 1e-6                       # 5x frequency -> 5x EMF

# 6. validation
for bad in (lambda: ed.sinusoidal_emf(0, 1, 1, 1, 0.0), lambda: ed.loop_current(1.0, 0)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad input")

print(f"SMOKE PASS  (EMF=-dPhi/dt; peak=N*A*B0*omega; constant flux -> 0 EMF; "
      f"5x faster -> 5x EMF -- the rate is what breaks KVL)")
