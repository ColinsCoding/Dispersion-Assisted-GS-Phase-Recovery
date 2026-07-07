"""Test dgs.charge_configurations: the dipole moment vector and its direction
(which way), the point-dipole field (along p on axis, opposite p on equator),
the leading-multipole classification with its 1/r falloff, the even/odd multipole
parity, and agreement of the exact Coulomb field with the ideal dipole far away."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import charge_configurations as cc

# a neutral dipole: +1 at +z, -1 at -z, separation a -> p = (0,0,a), pointing +z
a = 0.02
q = [1.0, -1.0]
pos = [[0, 0, a/2], [0, 0, -a/2]]

# 1. moments and direction
assert cc.net_charge(q) == 0.0
assert np.allclose(cc.dipole_moment(q, pos), [0, 0, a])
assert np.allclose(cc.dipole_direction(q, pos), [0, 0, 1])     # points - to +
# a config with no dipole has no direction
try:
    cc.dipole_direction([1.0, 1.0], pos); assert False         # net charge, symmetric -> p may be 0
except ValueError:
    pass

# 2. leading multipole: neutral pair -> dipole (1/r^2 potential, 1/r^3 field, odd)
lead = cc.leading_multipole(q, pos)
assert lead["term"] == "dipole" and lead["l"] == 1
assert lead["potential_falloff"] == 2 and lead["field_falloff"] == 3
assert lead["parity"] == "odd"
# add net charge -> monopole dominates
assert cc.leading_multipole([2.0, -1.0], pos)["term"] == "monopole"
# a linear quadrupole (+q, -2q, +q): net 0, dipole 0 -> quadrupole
quad_q = [1.0, -2.0, 1.0]
quad_pos = [[0, 0, a], [0, 0, 0], [0, 0, -a]]
assert np.isclose(cc.net_charge(quad_q), 0.0)
assert np.linalg.norm(cc.dipole_moment(quad_q, quad_pos)) < 1e-12
qlead = cc.leading_multipole(quad_q, quad_pos)
assert qlead["term"] == "quadrupole" and qlead["l"] == 2 and qlead["parity"] == "even"

# 3. multipole parity = (-1)^l (the even/odd operator eigenvalue)
assert cc.multipole_parity(0) == (1, "even")
assert cc.multipole_parity(1) == (-1, "odd")
assert cc.multipole_parity(2) == (1, "even")
assert cc.multipole_parity(3) == (-1, "odd")

# 4. point-dipole field: 2kp/r^3 along p on axis, -kp/r^3 opposite p on equator
p_unit = [0, 0, 1.0]
axis = cc.dipole_field(p_unit, [0, 0, 2.0], k=1.0)
assert np.allclose(axis, [0, 0, 2 * 1.0 / 8])                 # 2kp/r^3 = 0.25, +z
equ = cc.dipole_field(p_unit, [2.0, 0, 0], k=1.0)
assert np.allclose(equ, [0, 0, -1.0 / 8])                     # -kp/r^3 = -0.125, -z
# field falls as 1/r^3: r -> 2r divides magnitude by 8
E1 = np.linalg.norm(cc.dipole_field(p_unit, [0, 0, 1.0], k=1.0))
E2 = np.linalg.norm(cc.dipole_field(p_unit, [0, 0, 2.0], k=1.0))
assert np.isclose(E1 / E2, 8.0)

# 5. which way: exact field points along +p on the axis, opposite p on the equator
assert np.allclose(cc.field_direction(q, pos, [0, 0, 5.0]), [0, 0, 1], atol=1e-6)
assert np.allclose(cc.field_direction(q, pos, [5.0, 0, 0]), [0, 0, -1], atol=1e-6)

# 6. the exact Coulomb field matches the ideal point dipole far away
small = 1e-3
qs, ps = [1.0, -1.0], [[0, 0, small/2], [0, 0, -small/2]]
p_vec = cc.dipole_moment(qs, ps)
for fp in ([0, 0, 5.0], [5.0, 0, 0], [3.0, 0, 4.0]):
    exact = cc.coulomb_field(qs, ps, fp)
    ideal = cc.dipole_field(p_vec, fp)
    assert np.allclose(exact, ideal, rtol=1e-3)

# 7. a lone monopole field falls as 1/r^2
mono_q, mono_pos = [1.0], [[0, 0, 0]]
Em1 = np.linalg.norm(cc.coulomb_field(mono_q, mono_pos, [0, 0, 1.0]))
Em2 = np.linalg.norm(cc.coulomb_field(mono_q, mono_pos, [0, 0, 2.0]))
assert np.isclose(Em1 / Em2, 4.0)                            # 1/r^2 -> ratio 4

# 8. kwarg bounds
for bad in (lambda: cc.dipole_moment([1.0], [[0, 0, 0], [1, 0, 0]]),   # shape mismatch
            lambda: cc.coulomb_field([1.0], [[0, 0, 0]], [0, 0, 0]),   # on the charge
            lambda: cc.dipole_field([0, 0, 1], [0, 0, 0]),             # at the dipole
            lambda: cc.multipole_parity(-1)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_charge_configurations: all checks passed")
