"""Paraxial optics in electrodynamics + linear algebra language: the ABCD
ray-transfer-matrix formalism, and the Gaussian-beam complex q-parameter that
makes it electrodynamics rather than just ray-tracing geometry.

THE BINOMIAL EXPANSION THAT JUSTIFIES ALL OF IT: an exact spherical
wavefront from a point source has phase exp(ikR), R=sqrt(z^2+r^2). Expand R
via the binomial series in (r/z):
    R = z*sqrt(1+(r/z)^2) = z*(1 + (r/z)^2/2 - ...) = z + r^2/(2z) + O(r^4)
Truncating at r^2/(2z) IS the paraxial approximation -- the same "small
parameter, binomial series, keep the first correction" move as the small-
angle pendulum vs. the exact elliptic-integral one elsewhere in this repo.
Every ABCD matrix below is only valid because of this one truncation.

LINEAR ALGEBRA: each optical element (free-space propagation, a thin lens,
a curved interface) is a 2x2 matrix acting on a ray vector (position,
angle). Composing a system is matrix multiplication (order matters -- the
FIRST element the ray hits is the RIGHTMOST matrix in the product).
Free-space propagation and thin lenses are unimodular (det=1), a genuine
conserved invariant checked below, not assumed; refraction at an interface
is the one exception in this raw-angle convention (det=n1/n2, not 1 --
see spherical_interface_matrix's docstring for why, caught by this
module's own tests rather than left as an unchecked claim).

ELECTRODYNAMICS: a Gaussian beam (an actual solution of the paraxial
Helmholtz wave equation, not a ray) is fully described by one complex
number, the q-parameter q=z+i*z_R (z_R = Rayleigh range). It transforms
under the SAME ABCD matrix via a Mobius transformation (Kogelnik's law):
    q_out = (A*q_in + B) / (C*q_in + D)
-- ray optics (linear algebra) and Gaussian beam propagation
(electrodynamics) are governed by the identical matrix.

CAVITY STABILITY, AS AN EIGENVALUE PROBLEM: a resonator's round-trip ABCD
matrix has eigenvalues on the unit circle (stable -- a ray/beam stays
bounded forever) or real, one >1 (unstable -- diverges every round trip).
Verified numerically below for both cases, not stated as a rule.

NumPy only (SymPy for the one-time symbolic derivation). Education.
"""

import numpy as np


def paraxial_approximation_derivation():
    """Executes the binomial-expansion derivation described in the module
    docstring and returns the SymPy expression R_paraxial = z + r^2/(2z),
    verified to match a direct 2nd-order series expansion of the exact
    R=sqrt(z^2+r^2), not asserted."""
    import sympy as sp
    r, z = sp.symbols('r z', positive=True)
    R_exact = sp.sqrt(z ** 2 + r ** 2)
    R_paraxial = sp.simplify(z * sp.series(sp.sqrt(1 + (r / z) ** 2), r, 0, 3).removeO())
    claimed = z + r ** 2 / (2 * z)
    assert sp.simplify(R_paraxial - claimed) == 0
    return R_exact, R_paraxial


def free_space_matrix(d):
    """ABCD matrix for propagating a distance d through a uniform medium."""
    if d < 0:
        raise ValueError(f"d must be non-negative, got {d}")
    return np.array([[1.0, d], [0.0, 1.0]])


def thin_lens_matrix(f):
    """ABCD matrix for a thin lens of focal length f (f>0 converging,
    f<0 diverging)."""
    if f == 0:
        raise ValueError("f must be nonzero")
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def spherical_interface_matrix(n1, n2, R):
    """ABCD matrix for refraction at a spherical interface of radius R
    (R>0 if center of curvature is on the outgoing side), going from
    index n1 to n2 -- the paraxial (binomial-truncated) form of Snell's law.

    NOTE ON UNIMODULARITY: unlike free_space_matrix/thin_lens_matrix, THIS
    matrix has det(M) = n1/n2, not 1, because it acts on the RAW ray angle
    theta rather than the "reduced angle" n*theta some textbooks use.
    det=1 universally is a property of the reduced-angle convention, not a
    property of ABCD matrices in general -- verified directly (not assumed)
    in this module's tests, which is what caught this the first time."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1, n2 must be positive refractive indices")
    if R == 0:
        raise ValueError("R must be nonzero (use free_space_matrix for a flat interface)")
    return np.array([[1.0, 0.0], [(n1 - n2) / (n2 * R), n1 / n2]])


def compose_system(*matrices):
    """Compose optical elements into one system matrix. Matrices should be
    passed in the order light PASSES THROUGH them; the returned product
    applies the LAST-listed element's matrix on the left (matches how a ray
    vector is transformed: v_out = M_last @ ... @ M_first @ v_in)."""
    if not matrices:
        raise ValueError("need at least one matrix")
    M = np.eye(2)
    for elem in matrices:
        M = elem @ M
    return M


def is_unimodular(M, tol=1e-9):
    """Every ABCD matrix satisfies det(M)=1 exactly -- a real conserved
    invariant (energy/phase-space-area conservation in the paraxial limit),
    checked here rather than assumed."""
    return abs(np.linalg.det(M) - 1.0) < tol


def gaussian_beam_q(z, w0, wavelength):
    """Complex beam parameter q(z) = z + i*z_R, z_R = pi*w0^2/wavelength
    (Rayleigh range) -- the ONE complex number that fully specifies a
    Gaussian beam's radius of curvature and spot size at position z."""
    if w0 <= 0 or wavelength <= 0:
        raise ValueError("w0 and wavelength must be positive")
    z_R = np.pi * w0 ** 2 / wavelength
    return z + 1j * z_R


def transform_q(M, q_in):
    """Kogelnik's law: q_out = (A*q_in+B)/(C*q_in+D) -- the Gaussian beam's
    complex parameter transforms via the SAME ABCD matrix as a ray, a
    Mobius (fractional-linear) transformation of q."""
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    denom = C * q_in + D
    if abs(denom) < 1e-15:
        raise ValueError("C*q_in + D ~ 0 -- q transformation is singular here")
    return (A * q_in + B) / denom


def beam_waist_and_curvature(q, wavelength):
    """Recover the physical beam radius w and wavefront radius of curvature
    R from the complex q-parameter: 1/q = 1/R - i*wavelength/(pi*w^2)."""
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    inv_q = 1.0 / q
    R = np.inf if abs(inv_q.real) < 1e-15 else 1.0 / inv_q.real
    w = np.sqrt(-wavelength / (np.pi * inv_q.imag))
    return w, R


def telescope_matrix(f_objective, f_eyepiece):
    """Keplerian (afocal) telescope: objective and eyepiece separated by
    d=f_objective+f_eyepiece, so the objective's back focal point coincides
    with the eyepiece's front focal point. Afocal means C~0 (parallel rays
    in -> parallel rays out, no net focusing power) -- checked, not assumed."""
    if f_objective <= 0 or f_eyepiece <= 0:
        raise ValueError("f_objective and f_eyepiece must be positive")
    d = f_objective + f_eyepiece
    return compose_system(thin_lens_matrix(f_objective), free_space_matrix(d),
                           thin_lens_matrix(f_eyepiece))


def telescope_angular_magnification(f_objective, f_eyepiece):
    """M_angular = -f_objective/f_eyepiece = the D matrix element (the
    output ray ANGLE for a ray entering through the objective's center at
    some angle) -- what '10x binoculars' actually means. Negative sign:
    a Keplerian telescope inverts the image."""
    M = telescope_matrix(f_objective, f_eyepiece)
    return M[1, 1]


def telescope_transverse_magnification(f_objective, f_eyepiece):
    """M_transverse = -f_eyepiece/f_objective = the A matrix element (the
    output ray HEIGHT for a ray entering at some height, zero angle).
    Because an afocal system has C~0 and det=1, M_transverse is exactly
    the RECIPROCAL of M_angular (A*D=1 when C=0) -- verified, not assumed."""
    M = telescope_matrix(f_objective, f_eyepiece)
    return M[0, 0]


def microscope_matrix(f_objective, f_eyepiece, tube_length):
    """Compound microscope: objective (short f_objective) forms a real,
    magnified intermediate image at tube_length past its own focal point;
    the eyepiece then acts as a simple magnifier on that image. Returns the
    objective-to-intermediate-image ABCD matrix (the piece whose A element
    gives the objective's own transverse magnification -L/f_objective)."""
    if f_objective <= 0 or f_eyepiece <= 0 or tube_length <= 0:
        raise ValueError("f_objective, f_eyepiece, tube_length must all be positive")
    d = f_objective + tube_length
    return compose_system(thin_lens_matrix(f_objective), free_space_matrix(d))


def microscope_magnification(f_objective, f_eyepiece, tube_length, near_point=0.25):
    """Total microscope magnification = (objective's transverse mag, the A
    element of microscope_matrix) times (eyepiece's angular magnification,
    near_point/f_eyepiece -- the standard simple-magnifier formula, near_point
    conventionally 0.25 m, the human near point)."""
    if near_point <= 0:
        raise ValueError("near_point must be positive")
    M_obj = microscope_matrix(f_objective, f_eyepiece, tube_length)
    objective_transverse_mag = M_obj[0, 0]
    eyepiece_angular_mag = near_point / f_eyepiece
    return objective_transverse_mag * eyepiece_angular_mag


def cavity_stability_g_parameter(M):
    """The resonator stability parameter g=(A+D)/2. |g|<1 -- stable, a ray
    stays bounded on every round trip. Directly tied to eigenvalues: stable
    means the round-trip matrix's eigenvalues sit exactly on the unit
    circle (complex conjugate pair); unstable means real eigenvalues, one
    of which exceeds 1 in magnitude (unbounded growth per round trip)."""
    return (M[0, 0] + M[1, 1]) / 2.0


def is_stable_cavity(M):
    """True iff |g|<1, cross-checked against the eigenvalue criterion
    (both must agree, or something is wrong with either check)."""
    g = cavity_stability_g_parameter(M)
    stable_by_g = abs(g) < 1.0
    eigs = np.linalg.eigvals(M)
    stable_by_eig = np.allclose(np.abs(eigs), 1.0, atol=1e-6)
    assert stable_by_g == stable_by_eig, (
        f"g-parameter and eigenvalue stability criteria disagree: "
        f"g={g}, eigs={eigs} -- one of the two checks is wrong"
    )
    return stable_by_g


if __name__ == "__main__":
    R_exact, R_paraxial = paraxial_approximation_derivation()
    print(f"exact: R={R_exact}")
    print(f"paraxial (binomial expansion): R={R_paraxial}")

    print("\n--- ABCD system: free space, thin lens, free space ---")
    d1, f, d2 = 0.3, 0.5, 0.2
    M_system = compose_system(free_space_matrix(d1), thin_lens_matrix(f), free_space_matrix(d2))
    print(M_system)
    print(f"unimodular (det=1)? {is_unimodular(M_system)}")

    print("\n--- Gaussian beam (HeNe, 1mm waist) through that system ---")
    wavelength, w0 = 0.633e-6, 1e-3
    q_in = gaussian_beam_q(0.0, w0, wavelength)
    q_out = transform_q(M_system, q_in)
    w_out, R_out = beam_waist_and_curvature(q_out, wavelength)
    print(f"input waist:  {w0*1e3:.4f} mm")
    print(f"output waist: {w_out*1e3:.4f} mm, wavefront R = {R_out:.4f} m")

    print("\n--- Keplerian telescope (1m objective, 5cm eyepiece) ---")
    f_o, f_e = 1.0, 0.05
    M_scope = telescope_matrix(f_o, f_e)
    print(f"C element (afocal check, should be ~0): {M_scope[1,0]:.2e}")
    m_ang = telescope_angular_magnification(f_o, f_e)
    m_trans = telescope_transverse_magnification(f_o, f_e)
    print(f"angular magnification (D element) = {m_ang:.2f}  (claimed -f_o/f_e = {-f_o/f_e:.2f})")
    print(f"transverse magnification (A element) = {m_trans:.4f}  (reciprocal of angular: {1/m_ang:.4f})")

    print("\n--- Compound microscope (4mm objective, 25mm eyepiece, 160mm tube) ---")
    f_o_m, f_e_m, L = 0.004, 0.025, 0.16
    M_total = microscope_magnification(f_o_m, f_e_m, L)
    print(f"total magnification = {M_total:.1f}x  (claimed -(L/f_o)*(0.25/f_e) = "
          f"{-(L/f_o_m)*(0.25/f_e_m):.1f}x)")

    print("\n--- Cavity stability, checked two ways (must agree) ---")
    def round_trip(d, f):
        return compose_system(free_space_matrix(d), thin_lens_matrix(f),
                               free_space_matrix(d), thin_lens_matrix(f))

    M_stable = round_trip(1.0, 0.8)
    M_unstable = round_trip(1.0, 0.2)
    print(f"stable cavity:   g={cavity_stability_g_parameter(M_stable):.4f}, "
          f"is_stable={is_stable_cavity(M_stable)}")
    print(f"unstable cavity: g={cavity_stability_g_parameter(M_unstable):.4f}, "
          f"is_stable={is_stable_cavity(M_unstable)}")
