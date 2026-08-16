"""Test dgs/effective_mass_tensor.py: the generalized Newton's second law
a=(m*^-1).F for anisotropic (silicon valley) vs. isotropic (GaAs)
conduction bands, cross-checked by an independent semiclassical
trajectory simulation, plus a real bug this session caught: the
parallel-check's zero-force guard was silently firing on every realistic
force because it reused a dimensionless cosine tolerance as a Newtons-
scale magnitude threshold."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.effective_mass_tensor import (
    inverse_mass_tensor_numeric, silicon_valley_energy_symbolic,
    inverse_mass_tensor_symbolic, verify_silicon_valley_is_diagonal,
    tensor_acceleration, acceleration_is_parallel_to_force,
    gaas_inverse_mass_tensor, silicon_inverse_mass_tensor,
    verify_semiclassical_trajectory, HBAR, M_ELECTRON,
)

# 1. verify_silicon_valley_is_diagonal: must pass
assert verify_silicon_valley_is_diagonal() is True

# 2. gaas_inverse_mass_tensor: exactly isotropic (scaled identity)
GaAs_inv = gaas_inverse_mass_tensor(0.067)
expected_scalar = 1.0 / (0.067 * M_ELECTRON)
assert np.allclose(GaAs_inv, expected_scalar * np.eye(3))
try:
    gaas_inverse_mass_tensor(-1.0)
    raise AssertionError("expected ValueError for m_star <= 0")
except ValueError:
    pass

# 3. silicon_inverse_mass_tensor: exactly diag(1/(m_l*m0), 1/(m_t*m0), 1/(m_t*m0))
Si_inv = silicon_inverse_mass_tensor(0.98, 0.19)
expected_Si = np.diag([1/(0.98*M_ELECTRON), 1/(0.19*M_ELECTRON), 1/(0.19*M_ELECTRON)])
assert np.allclose(Si_inv, expected_Si)

# 4. tensor_acceleration: for GaAs (isotropic), a = F/m* exactly, in
#    whatever direction F points (a genuine tensor-equation sanity check,
#    not just "runs without error")
F = np.array([1e-18, 2e-18, -0.5e-18])
a_GaAs = tensor_acceleration(F, GaAs_inv)
assert np.allclose(a_GaAs, F * expected_scalar)

# 5. tensor_acceleration: shape mismatch and non-symmetric tensor must raise
try:
    tensor_acceleration(np.array([1.0, 2.0]), GaAs_inv)
    raise AssertionError("expected ValueError for shape mismatch")
except ValueError:
    pass
try:
    tensor_acceleration(np.array([1.0, 1.0, 1.0]), np.array([[1.0, 2.0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    raise AssertionError("expected ValueError for a non-symmetric inverse mass tensor")
except ValueError:
    pass

# 6. acceleration_is_parallel_to_force: TRUE for isotropic GaAs at ANY
#    force direction (a genuinely swept check, not one lucky angle)
for angle_deg in (0, 15, 45, 90, 137):
    theta = np.radians(angle_deg)
    F_dir = np.array([np.cos(theta), np.sin(theta), 0.0]) * 1e-18
    assert acceleration_is_parallel_to_force(F_dir, GaAs_inv) is True, f"failed at angle {angle_deg}"

# 7. acceleration_is_parallel_to_force: FALSE for anisotropic silicon at
#    an off-axis force -- THE bug this session caught: an earlier version
#    returned True here unconditionally because its zero-force guard
#    reused the cosine tolerance as a Newtons-scale magnitude threshold,
#    which a realistic atomic-scale force (~1e-18 N) always tripped
F_offaxis = np.array([1e-18, 1e-18, 0.0])
assert acceleration_is_parallel_to_force(F_offaxis, Si_inv) is False

# 8. acceleration_is_parallel_to_force: TRUE for silicon when F IS aligned
#    with a principal axis (the special case where even anisotropic mass
#    doesn't deflect the acceleration)
F_onaxis = np.array([1e-18, 0.0, 0.0])
assert acceleration_is_parallel_to_force(F_onaxis, Si_inv) is True
F_onaxis_y = np.array([0.0, 1e-18, 0.0])
assert acceleration_is_parallel_to_force(F_onaxis_y, Si_inv) is True

# 9. acceleration_is_parallel_to_force: a genuinely zero force must not
#    crash (the real zero-guard, now correctly scale-independent)
assert acceleration_is_parallel_to_force(np.zeros(3), Si_inv) is True

# 10. inverse_mass_tensor_numeric vs. inverse_mass_tensor_symbolic: the
#     numerical finite-difference Hessian must match the exact symbolic
#     one for the same silicon-valley model
import sympy as sp
E_sym, k_vars, params = silicon_valley_energy_symbolic()
kx0_s, ml_s, mt_s = params
hbar_s = sp.Symbol("hbar", positive=True)
# ml_s, mt_s are dimensionless ratios (m/m0) in the symbolic model --
# substitute actual kg masses, not bare ratios, or E_func's units won't
# match silicon_inverse_mass_tensor()'s SI-unit convention (the bug this
# test caught: substituting bare 0.98/0.19 gives inverse-mass-tensor
# entries in units of 1/(relative mass), off from the SI (1/kg) answer by
# a factor of M_ELECTRON)
E_numeric_expr = E_sym.subs({kx0_s: 0, ml_s: 0.98 * M_ELECTRON, mt_s: 0.19 * M_ELECTRON, hbar_s: HBAR})
E_func = sp.lambdify(k_vars, E_numeric_expr, "numpy")
E_func_np = lambda k: float(E_func(k[0], k[1], k[2]))

inv_mass_numeric = inverse_mass_tensor_numeric(E_func_np, np.array([0.0, 0.0, 0.0]))
assert np.allclose(inv_mass_numeric, Si_inv, rtol=1e-3), \
    f"numeric Hessian {inv_mass_numeric} doesn't match symbolic {Si_inv}"

# 11. verify_semiclassical_trajectory: independent finite-difference
#     trajectory check must pass for the silicon valley model
ok_traj = verify_semiclassical_trajectory(E_func_np, np.array([0.0, 0.0, 0.0]), F_offaxis)
assert ok_traj is True

print("all dgs.effective_mass_tensor tests passed")
