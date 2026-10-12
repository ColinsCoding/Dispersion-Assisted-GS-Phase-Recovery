"""Test dgs.quadrupole_wavefunction: Y_2^m against the textbook closed forms,
orthonormality by quadrature, hydrogen radial functions and <r^2>, the
quadrupole moments of the 3d states (analytic vs a numerical tensor that must
be symmetric and traceless), and the exact potential of the cloud approaching
monopole + quadrupole with the right 1/r^3 law."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import quadrupole_wavefunction as qw
from dgs import electrostatics_multipoles as em

theta = np.linspace(0.05, np.pi - 0.05, 37)[:, None]
phi = np.linspace(0.0, 2 * np.pi, 49)[None, :]

# 1. Y_2^m equals the closed forms of Griffiths' table (QM Sec. 4.1.2)
s, c = np.sin(theta), np.cos(theta)
assert np.allclose(qw.spherical_harmonic(2, 0, theta, phi), math.sqrt(5 / (16 * math.pi)) * (3 * c**2 - 1))
assert np.allclose(qw.spherical_harmonic(2, 1, theta, phi), -math.sqrt(15 / (8 * math.pi)) * s * c * np.exp(1j * phi))
assert np.allclose(qw.spherical_harmonic(2, -1, theta, phi), math.sqrt(15 / (8 * math.pi)) * s * c * np.exp(-1j * phi))
assert np.allclose(qw.spherical_harmonic(2, 2, theta, phi), math.sqrt(15 / (32 * math.pi)) * s**2 * np.exp(2j * phi))
assert np.allclose(qw.spherical_harmonic(2, -2, theta, phi), math.sqrt(15 / (32 * math.pi)) * s**2 * np.exp(-2j * phi))
assert np.allclose(qw.spherical_harmonic(0, 0, theta, phi), 1 / math.sqrt(4 * math.pi))

# 2. orthonormality over the sphere (Gauss-Legendre in cos(theta), uniform in phi)
mu, w = np.polynomial.legendre.leggauss(24)
ph = np.linspace(0, 2 * np.pi, 48, endpoint=False)
TH, PH = np.meshgrid(np.arccos(mu), ph, indexing="ij")
dA = w[:, None] * (2 * np.pi / ph.size)
states = [(l, m) for l in range(4) for m in range(-l, l + 1)]
for (l1, m1) in states:
    for (l2, m2) in states:
        overlap = np.sum(np.conj(qw.spherical_harmonic(l1, m1, TH, PH)) * qw.spherical_harmonic(l2, m2, TH, PH) * dA)
        assert abs(overlap - (1.0 if (l1, m1) == (l2, m2) else 0.0)) < 1e-12

# 3. the five real d orbitals: real, orthonormal, and equal to their Cartesian forms
x, y, z = np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)
cartesian = {"z2": math.sqrt(5 / (16 * math.pi)) * (3 * z**2 - 1), "xz": math.sqrt(15 / (4 * math.pi)) * x * z,
             "yz": math.sqrt(15 / (4 * math.pi)) * y * z, "xy": math.sqrt(15 / (4 * math.pi)) * x * y,
             "x2-y2": math.sqrt(15 / (16 * math.pi)) * (x**2 - y**2)}
for a in qw.D_ORBITALS:
    fa = qw.real_d_orbital(a, TH, PH)
    assert np.isrealobj(fa) and np.allclose(fa, cartesian[a])
    for b in qw.D_ORBITALS:
        assert abs(np.sum(fa * qw.real_d_orbital(b, TH, PH) * dA) - (a == b)) < 1e-12
# sum of the five squared = 5/(4 pi): a filled d subshell is spherically symmetric (Unsold)
assert np.allclose(sum(qw.real_d_orbital(a, TH, PH) ** 2 for a in qw.D_ORBITALS), 5 / (4 * math.pi))

# 4. radial functions: closed forms (Griffiths QM Sec. 4.2), normalization, <r^2>
r = np.linspace(0.0, 120.0, 120001)
assert np.allclose(qw.radial_wavefunction(1, 0, r), 2 * np.exp(-r))
assert np.allclose(qw.radial_wavefunction(2, 1, r), r * np.exp(-r / 2) / math.sqrt(24))
assert np.allclose(qw.radial_wavefunction(3, 2, r), 4 / (81 * math.sqrt(30)) * r**2 * np.exp(-r / 3))
for n, l in [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 2)]:
    R2 = qw.radial_wavefunction(n, l, r) ** 2
    assert math.isclose(np.trapezoid(R2 * r**2, r), 1.0, rel_tol=1e-6)
    assert math.isclose(np.trapezoid(R2 * r**4, r), qw.mean_r2(n, l), rel_tol=1e-6)
assert qw.mean_r2(3, 2) == 126 and qw.mean_r2(1, 0) == 3
assert math.isclose(qw.radial_wavefunction(3, 2, 6.0, a=2.0), qw.radial_wavefunction(3, 2, 3.0) / 2**1.5)  # scaling with a

# 5. quadrupole moments of the 3d states, and the closed-subshell sum rule
assert math.isclose(qw.mean_cos2(2, 0), 11 / 21) and math.isclose(qw.mean_cos2(0, 0), 1 / 3)
assert math.isclose(qw.quadrupole_moment_zz(3, 2, 0), 72.0)
assert math.isclose(qw.quadrupole_moment_zz(3, 2, 1), 36.0) and math.isclose(qw.quadrupole_moment_zz(3, 2, -1), 36.0)
assert math.isclose(qw.quadrupole_moment_zz(3, 2, 2), -72.0)
for n, l in [(2, 1), (3, 2), (4, 3)]:
    assert abs(sum(qw.quadrupole_moment_zz(n, l, m) for m in range(-l, l + 1))) < 1e-9
for n in (1, 2, 3):
    assert abs(qw.quadrupole_moment_zz(n, 0, 0)) < 1e-12              # s states are spherical
assert math.isclose(qw.quadrupole_moment_zz(3, 2, 0, a=2.0), 4 * 72.0)   # scales as a^2

# 6. numerical tensor: unit charge, symmetric, traceless, matches the closed form
for m, expected in [(0, 72.0), (1, 36.0), (2, -72.0)]:
    density = lambda rr, th, p, m=m: np.abs(qw.hydrogen_psi(3, 2, m, rr, th, p)) ** 2
    tensor, charge = qw.quadrupole_tensor(density, r_max=90.0)
    assert math.isclose(charge, 1.0, rel_tol=1e-6)
    assert np.allclose(tensor, tensor.T, atol=1e-8) and abs(np.trace(tensor)) < 1e-6
    assert np.allclose(np.diag(tensor), [-expected / 2, -expected / 2, expected], atol=1e-4)
# a real d_xy orbital is NOT axially symmetric: Theta_xx = Theta_yy but the lobes lie in the xy plane
density = lambda rr, th, p: (qw.radial_wavefunction(3, 2, rr) * qw.real_d_orbital("xy", th, p)) ** 2
tensor, _ = qw.quadrupole_tensor(density, r_max=90.0)
assert np.allclose(np.diag(tensor), [36.0, 36.0, -72.0], atol=1e-4)

# 7. same tensor convention as the point-charge module: a linear quadrupole (+q, -2q, +q) on z
theta_pc = em.quadrupole_moment([1.0, -2.0, 1.0], [[0, 0, 1.0], [0, 0, 0], [0, 0, -1.0]])
assert np.allclose(np.diag(theta_pc), [-2.0, -2.0, 4.0]) and abs(np.trace(theta_pc)) < 1e-12

# 8. potential of the cloud: far away it is monopole + quadrupole, error falling as 1/r^5
for m in (0, 1, 2):
    for angle in (0.0, 0.7, math.pi / 2):
        exact = qw.orbital_potential(3, 2, m, 200.0, angle)
        assert math.isclose(exact, qw.far_field_potential(3, 2, m, 200.0, angle), rel_tol=1e-5)
r_far = np.array([150.0, 300.0])
quad_part = qw.orbital_potential(3, 2, 0, r_far, 0.0) - 1 / r_far
assert math.isclose(quad_part[0] / quad_part[1], 8.0, rel_tol=0.01)          # 1/r^3: doubling r gives 1/8
residual = qw.orbital_potential(3, 2, 0, r_far, 0.0) - qw.far_field_potential(3, 2, 0, r_far, 0.0)
assert math.isclose(residual[0] / residual[1], 32.0, rel_tol=0.01)           # next term (L = 4) is 1/r^5
assert residual[0] > 0 and residual[0] / quad_part[0] < 0.02                 # and it is small out here
# an s state has no angular dependence at all, and V -> 1/r outside the cloud
v_s = qw.orbital_potential(1, 0, 0, 20.0, np.array([0.0, 1.0, 2.0]))
assert np.allclose(v_s, 1 / 20.0, rtol=1e-8)
# close in, the far-field formula must FAIL (it is only an expansion)
assert abs(qw.far_field_potential(3, 2, 0, 5.0, 0.0) / qw.orbital_potential(3, 2, 0, 5.0, 0.0) - 1) > 0.5

# 9. surface mesh: r = |Y|, sign marks the lobes
X, Y, Z, sign = qw.orbital_surface(lambda th, p: qw.real_d_orbital("z2", th, p), 61, 121)
assert X.shape == (61, 121) and set(np.unique(sign)) <= {-1.0, 0.0, 1.0}
assert math.isclose(np.sqrt(X**2 + Y**2 + Z**2).max(), math.sqrt(5 / (4 * math.pi)))   # lobe tip on the z axis
assert sign[0, 0] == 1.0 and sign[30, 0] == -1.0                    # positive lobes on z, negative ring at the equator
Xc, Yc, Zc, sc = qw.orbital_surface(lambda th, p: qw.spherical_harmonic(2, 2, th, p), 31, 61)
assert np.all(sc == 1.0) and np.allclose(Zc[15], 0.0, atol=1e-12)   # |Y_2^2| is a torus-like ring, no sign

# 10. bad inputs are refused
bad_calls = [
    lambda: qw.spherical_harmonic(2, 3, 0.1, 0.0), lambda: qw.spherical_harmonic(-1, 0, 0.1, 0.0),
    lambda: qw.real_d_orbital("zz", 0.1, 0.0), lambda: qw.radial_wavefunction(2, 2, 1.0),
    lambda: qw.radial_wavefunction(0, 0, 1.0), lambda: qw.radial_wavefunction(1, 0, -1.0),
    lambda: qw.radial_wavefunction(1, 0, 1.0, a=0.0), lambda: qw.orbital_surface(lambda t, p: t, 2, 10),
    lambda: qw.quadrupole_tensor(lambda r_, t, p: r_, r_max=-1.0),
    lambda: qw.orbital_potential(3, 2, 0, 0.0, 0.0), lambda: qw.orbital_potential(3, 2, 0, 1e6, 0.0),
    lambda: qw.far_field_potential(3, 2, 0, -1.0, 0.0),
]
for bad in bad_calls:
    try:
        bad(); assert False, "should have raised"
    except ValueError:
        pass

print("test_quadrupole_wavefunction: all checks passed")
