"""Test dgs/maxwell_solid_state_continuity.py: the continuity equation
derived from Ampere-Maxwell + Gauss's law, and its two solid-state
applications (drift-diffusion carrier transport, dielectric relaxation).
Reuses dgs/causality.py's continuity_residual directly (not reimplemented)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.maxwell_solid_state_continuity import (
    derive_continuity_from_ampere_maxwell,
    drift_diffusion_carrier_density, verify_drift_diffusion_satisfies_pde,
    drift_diffusion_current_and_charge,
    dielectric_relaxation_time, derive_dielectric_relaxation_ode,
    dielectric_relaxation_decay,
)
from dgs.causality import continuity_residual

# 1. div(curl B) = 0 identically -- the vector-calculus identity the whole
#    derivation rests on
result = derive_continuity_from_ampere_maxwell()
assert result["div_curl_B_identity_holds"] is True
assert result["matches_div_J_plus_drho_dt"] is True

# 2. The drift-diffusion Gaussian Green's function must satisfy its PDE exactly
assert verify_drift_diffusion_satisfies_pde() is True

# 3. drift_diffusion_carrier_density bounds: D<=0 and t<=0 must raise
try:
    drift_diffusion_carrier_density(np.array([0.0]), np.array([1.0]), v=1.0, D=0.0)
    raise AssertionError("expected ValueError for D<=0")
except ValueError:
    pass
try:
    drift_diffusion_carrier_density(np.array([0.0]), np.array([-1.0]), v=1.0, D=1.0)
    raise AssertionError("expected ValueError for t<=0")
except ValueError:
    pass

# 4. Numeric continuity check: the reused continuity_residual must be small
#    (finite-difference-scale, not exactly 0) relative to the physical scale
#    of rho/J -- a real check, not a triviality (a spatially-uniform rho/J
#    would make this check vacuous; this one has genuine drift+spread structure)
x = np.linspace(-20, 20, 400)
t = np.linspace(0.5, 4.5, 300)
dd = drift_diffusion_current_and_charge(x, t, v=2.0, D=0.8, N0=1.0, charge=1.0)
res = continuity_residual(dd["rho"], dd["J"], x, t)
interior = res[5:-5, 5:-5]
rel_residual = np.max(np.abs(interior)) / np.max(np.abs(dd["rho"]))
assert rel_residual < 0.02, f"expected charge conservation to hold to <2%, got {rel_residual:.4f}"

# 5. drift_diffusion_current_and_charge bounds
try:
    drift_diffusion_current_and_charge(x, np.array([-1.0]), v=1.0, D=1.0)
    raise AssertionError("expected ValueError for t<=0")
except ValueError:
    pass

# 6. Dielectric relaxation: tau = eps/sigma, and rho(t) decays as rho0*exp(-t/tau)
tau = dielectric_relaxation_time(permittivity=1e-10, conductivity=1e-3)
assert abs(tau - 1e-7) < 1e-15

rho0 = 5.0
rho_t = dielectric_relaxation_decay(np.array([0.0, tau, 2 * tau]), rho0, tau)
assert abs(rho_t[0] - rho0) < 1e-12
assert abs(rho_t[1] - rho0 / np.e) < 1e-9, "at t=tau, charge should have decayed to 1/e of initial"
assert rho_t[2] < rho_t[1] < rho_t[0], "charge should decay monotonically"

# 7. The SymPy-solved ODE closed form must match the numeric decay function
ode_sol = derive_dielectric_relaxation_ode()
import sympy as sp
# Use the ACTUAL symbols from the solution (matching their exact
# assumptions, e.g. rho_0 is real=True not positive=True) rather than
# freshly-declared symbols of the same name -- SymPy symbols with
# differing assumptions are NOT interchangeable even with an identical name.
syms = {s.name: s for s in ode_sol.rhs.free_symbols}
analytic_at_tau = float(ode_sol.rhs.subs(syms["tau"], 1.0)
                         .subs({syms["t"]: 1.0, syms["rho_0"]: rho0}))
assert abs(analytic_at_tau - rho0 / np.e) < 1e-9

# 8. dielectric_relaxation_time/decay bounds
for bad_kwargs in [dict(permittivity=0.0, conductivity=1e-3), dict(permittivity=1e-10, conductivity=0.0)]:
    try:
        dielectric_relaxation_time(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass
try:
    dielectric_relaxation_decay(np.array([1.0]), rho0=1.0, tau=0.0)
    raise AssertionError("expected ValueError for tau<=0")
except ValueError:
    pass

print("all dgs.maxwell_solid_state_continuity tests passed")
