"""Test the free rigid body's Lagrangian/Noether derivation: phi is cyclic
(space-fixed symmetry) for ANY I1,I2,I3, psi is cyclic ONLY when I1==I2
(the tennis-racket-theorem asymmetric top has no such extra symmetry), and
the numerically-integrated Euler equations actually conserve p_psi=I3*omega3
exactly when that extra symmetry is present."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
import numpy as np
from dgs.lagrangian_rigid_body import (
    euler_angle_kinematics, free_rigid_body_lagrangian, noether_cyclic_check,
)
from dgs.gyroscopes import integrate_euler_rigid_body

# 1. omega1^2 + omega2^2 simplifies to the standard kinematic identity
kin = euler_angle_kinematics()
w1w2 = sp.simplify(kin["omega1"] ** 2 + kin["omega2"] ** 2)
expected = sp.sin(kin["theta"]) ** 2 * kin["phi"].diff(kin["t"]) ** 2 + kin["theta"].diff(kin["t"]) ** 2
assert sp.simplify(w1w2 - expected) == 0

# 2. phi is ALWAYS cyclic, for a fully generic asymmetric top
I1, I2, I3 = sp.symbols("I1 I2 I3", positive=True)
asym = free_rigid_body_lagrangian(I1, I2, I3)
phi_check = noether_cyclic_check(asym["L"], asym["phi"], asym["t"])
assert phi_check["cyclic"] is True
assert phi_check["noether_conservation_confirmed"] is True

# 3. psi is NOT cyclic for a generic asymmetric top (I1, I2, I3 all distinct symbols)
psi_check_asym = noether_cyclic_check(asym["L"], asym["psi"], asym["t"])
assert psi_check_asym["cyclic"] is False

# 4. psi IS cyclic once I1==I2 (symmetric top), with conserved momentum I3*omega3
I, I3_sym = sp.symbols("I I3", positive=True)
sym = free_rigid_body_lagrangian(I, I, I3_sym)
psi_check_sym = noether_cyclic_check(sym["L"], sym["psi"], sym["t"])
assert psi_check_sym["cyclic"] is True
assert psi_check_sym["noether_conservation_confirmed"] is True
expected_p_psi = I3_sym * sym["omega3"]
assert sp.simplify(psi_check_sym["conserved_momentum"] - expected_p_psi) == 0

# 5. Numeric cross-check: integrate the ACTUAL Euler equations
# (dgs.gyroscopes, independent implementation) for a symmetric top and
# confirm I3*omega3 really is exactly conserved -- not just an algebraic
# claim, an empirically verified one against a genuinely different code path
I1_val, I2_val, I3_val = 2.0, 2.0, 5.0
run = integrate_euler_rigid_body([0.3, 0.5, 4.0], I1_val, I2_val, I3_val, t_max=10.0, dt=0.0005)
p_psi_numeric = I3_val * run["omega"][:, 2]
assert (p_psi_numeric.max() - p_psi_numeric.min()) < 1e-9

# 6. And for a GENUINELY asymmetric top, I3*omega3 should NOT stay constant
# -- the symmetry that protected it in test 5 is gone
run_asym = integrate_euler_rigid_body([0.1, 3.0, 0.2], 1.0, 2.0, 3.0, t_max=10.0, dt=0.0005)
p_would_be_asym = 3.0 * run_asym["omega"][:, 2]
assert (p_would_be_asym.max() - p_would_be_asym.min()) > 0.1

print("all dgs.lagrangian_rigid_body tests passed")
