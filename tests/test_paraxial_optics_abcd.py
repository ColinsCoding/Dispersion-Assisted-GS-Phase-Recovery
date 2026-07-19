"""Test paraxial optics in ABCD-matrix/Gaussian-beam form: the binomial-
expansion derivation, matrix unimodularity, Kogelnik's q-transformation,
and the two independent (must agree) cavity-stability criteria."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import paraxial_optics_abcd as po

# 1. the binomial-expansion derivation reproduces R = z + r^2/(2z) exactly
#    (the function itself asserts this internally; re-verify from outside too)
R_exact, R_paraxial = po.paraxial_approximation_derivation()
import sympy as sp
z, r = sp.symbols('z r', positive=True)
assert sp.simplify(R_paraxial - (z + r**2/(2*z))) == 0

# 2. free space and thin lens matrices (and their composition) are unimodular
assert po.is_unimodular(po.free_space_matrix(1.5))
assert po.is_unimodular(po.thin_lens_matrix(0.3))
M_system = po.compose_system(po.free_space_matrix(0.3), po.thin_lens_matrix(0.5),
                              po.free_space_matrix(0.2))
assert po.is_unimodular(M_system)

# 2b. the interface matrix is the DOCUMENTED exception: det = n1/n2, not 1,
# in this raw-angle convention -- verified as the actual value, not "close to 1"
n1, n2, R_interface = 1.0, 1.5, 0.2
M_interface = po.spherical_interface_matrix(n1, n2, R_interface)
assert abs(np.linalg.det(M_interface) - n1 / n2) < 1e-12
assert not po.is_unimodular(M_interface)   # correctly NOT unimodular

# 3. free space matrix with d=0 is the identity; composing with identity changes nothing
I = po.free_space_matrix(0.0)
assert np.allclose(I, np.eye(2))
M_with_identity = po.compose_system(I, M_system, I)
assert np.allclose(M_with_identity, M_system)

# 4. Gaussian beam q-parameter: at the waist (z=0), Re(q)=0 and Im(q)=z_R exactly
wavelength, w0 = 0.633e-6, 1e-3
q0 = po.gaussian_beam_q(0.0, w0, wavelength)
z_R_expected = np.pi * w0**2 / wavelength
assert abs(q0.real) < 1e-15
assert abs(q0.imag - z_R_expected) < 1e-9

# 5. propagating the beam through free space of distance d is the same as
#    just adding d to q (the trivial ABCD case), cross-checked against transform_q
d = 0.4
q_after_free_space = po.transform_q(po.free_space_matrix(d), q0)
assert abs(q_after_free_space - (q0 + d)) < 1e-9

# 6. recovering (w, R) from q and re-deriving q from (w, R) round-trips
#    (internal consistency of beam_waist_and_curvature vs gaussian_beam_q's
#    own definition, evaluated at a nonzero z so R is finite)
z_test = 0.1
q_z = z_test + 1j*z_R_expected
w_z, R_z = po.beam_waist_and_curvature(q_z, wavelength)
inv_q_reconstructed = complex(1/R_z, -wavelength/(np.pi*w_z**2))
assert abs(1/q_z - inv_q_reconstructed) < 1e-9

# 6b. Keplerian telescope: afocal (C~0), angular magnification = -f_o/f_e,
#     transverse magnification is exactly its reciprocal (A*D=1 when C=0)
f_o, f_e = 1.0, 0.05
M_scope = po.telescope_matrix(f_o, f_e)
assert abs(M_scope[1, 0]) < 1e-9   # afocal: C element ~ 0
m_ang = po.telescope_angular_magnification(f_o, f_e)
m_trans = po.telescope_transverse_magnification(f_o, f_e)
assert abs(m_ang - (-f_o / f_e)) < 1e-9
assert abs(m_trans - (-f_e / f_o)) < 1e-9
assert abs(m_ang * m_trans - 1.0) < 1e-9   # reciprocal relationship, verified directly

for bad in [(-1.0, 0.05), (1.0, 0.0)]:
    try:
        po.telescope_matrix(*bad)
        assert False, "should reject non-positive focal length"
    except ValueError:
        pass

# 6c. compound microscope: total magnification matches the standard
#     -(L/f_o)*(near_point/f_e) formula
f_o_m, f_e_m, L, near_point = 0.004, 0.025, 0.16, 0.25
M_total = po.microscope_magnification(f_o_m, f_e_m, L, near_point)
expected_total = -(L / f_o_m) * (near_point / f_e_m)
assert abs(M_total - expected_total) < 1e-6

# doubling the tube length should exactly double the total magnification
# (magnification is linear in L, holding everything else fixed)
M_total_2L = po.microscope_magnification(f_o_m, f_e_m, 2 * L, near_point)
assert abs(M_total_2L / M_total - 2.0) < 1e-6

for bad in [(-0.004, 0.025, 0.16), (0.004, -0.025, 0.16), (0.004, 0.025, -0.16)]:
    try:
        po.microscope_magnification(*bad)
        assert False, "should reject non-positive parameters"
    except ValueError:
        pass
try:
    po.microscope_magnification(f_o_m, f_e_m, L, near_point=-0.25)
    assert False, "should reject non-positive near_point"
except ValueError:
    pass

# 7. cavity stability: the g-parameter criterion and the eigenvalue criterion
#    must agree (is_stable_cavity asserts this internally) for BOTH a
#    genuinely stable and a genuinely unstable configuration
def round_trip(d, f):
    return po.compose_system(po.free_space_matrix(d), po.thin_lens_matrix(f),
                              po.free_space_matrix(d), po.thin_lens_matrix(f))

M_stable = round_trip(1.0, 0.8)
M_unstable = round_trip(1.0, 0.2)
assert po.is_stable_cavity(M_stable) == True
assert po.is_stable_cavity(M_unstable) == False

# stable case: eigenvalues genuinely on the unit circle
eigs_stable = np.linalg.eigvals(M_stable)
assert np.allclose(np.abs(eigs_stable), 1.0, atol=1e-6)
# unstable case: eigenvalues real, NOT on the unit circle, product = det = 1
eigs_unstable = np.linalg.eigvals(M_unstable)
assert np.all(np.abs(eigs_unstable.imag) < 1e-9)   # real eigenvalues
assert abs(eigs_unstable[0] * eigs_unstable[1] - 1.0) < 1e-6  # product = det(M) = 1
assert not np.allclose(np.abs(eigs_unstable), 1.0, atol=1e-6)

# 8. input validation
for bad_call in [
    lambda: po.free_space_matrix(-1.0),
    lambda: po.thin_lens_matrix(0.0),
    lambda: po.spherical_interface_matrix(-1.0, 1.5, 0.2),
    lambda: po.spherical_interface_matrix(1.0, 1.5, 0.0),
    lambda: po.compose_system(),
    lambda: po.gaussian_beam_q(0.0, -1.0, wavelength),
    lambda: po.beam_waist_and_curvature(q0, -1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.paraxial_optics_abcd tests passed")
