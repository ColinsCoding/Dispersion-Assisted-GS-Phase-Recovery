"""Smoke-test Ohm + Gauss + continuity -> charge relaxation (griffiths.electrodynamics)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from griffiths import electrodynamics as ed

# 1. symbolic: the three laws combine to an exponential decay rho0 e^{-sigma t/eps}
ode, sol, tau = ed.charge_relaxation_ode()
t = sp.Symbol("t", real=True)
eps_, sig, rho0 = sp.symbols("epsilon sigma rho_0", positive=True)
assert sol.rhs == rho0 * sp.exp(-sig * t / eps_), sol.rhs
assert sp.simplify(tau - eps_ / sig) == 0
# the solution actually satisfies the ODE
assert sp.simplify(sol.rhs.diff(t) + (sig / eps_) * sol.rhs) == 0

# 2. numeric tau = eps/sigma; copper is famously sub-femtosecond
eps0, sigma_cu = 8.8541878128e-12, 5.96e7
tau_cu = ed.charge_relaxation_time(eps0, sigma_cu)
assert 1e-19 < tau_cu < 1e-18, tau_cu                  # ~1.5e-19 s

# 3. decay law: e-fold at t=tau, half at t=tau*ln2, monotone down
tau = ed.charge_relaxation_time(2.0, 4.0)              # tau = 0.5
assert abs(ed.charge_decay(1.0, tau, 2.0, 4.0) - np.exp(-1)) < 1e-12
assert abs(ed.charge_decay(1.0, tau * np.log(2), 2.0, 4.0) - 0.5) < 1e-12
ts = np.linspace(0, 5, 50)
d = ed.charge_decay(3.0, ts, 2.0, 4.0)
assert d[0] == 3.0 and np.all(np.diff(d) < 0)

# 4. validation
for bad in ((0, 1), (1, -1)):
    try:
        ed.charge_relaxation_time(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive eps/sigma should raise")

print(f"SMOKE PASS  (rho(t)=rho0 e^(-t/tau), tau=eps/sigma; copper tau={tau_cu:.2e} s)")
