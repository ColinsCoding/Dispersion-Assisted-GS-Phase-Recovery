"""Jones calculus -- polarization, rotation matrices, and chain rule in photonics.

THE CHAIN RULE IN PHOTONICS:
  A cascaded optical system (wave plate 1 -> rotator -> wave plate 2 -> polarizer)
  acts on the Jones vector E by successive LEFT-MULTIPLICATION:

    E_out = M_N * M_{N-1} * ... * M_2 * M_1 * E_in

  This is the CHAIN RULE for matrix-valued functions:
    d(f_N o ... o f_1)/dx = J_N * J_{N-1} * ... * J_1
  where J_k is the Jacobian of the k-th element.

  In photonics the Jacobians ARE the Jones matrices. Non-commutative (order matters!):
    QWP then HWP != HWP then QWP

JONES VECTORS (coherent, fully polarized light):
  E = [Ex, Ey]^T  (complex amplitudes of the x and y electric field components)

  Horizontal LP:    [1, 0]^T
  Vertical LP:      [0, 1]^T
  +45 LP:           [1, 1]^T / sqrt(2)
  -45 LP:           [1, -1]^T / sqrt(2)
  Right circular:   [1, -i]^T / sqrt(2)   (RCP, positive helicity)
  Left circular:    [1, +i]^T / sqrt(2)   (LCP, negative helicity)

JONES MATRICES (2x2 complex unitary matrices):
  Horizontal LP:    [[1,0],[0,0]]
  Vertical LP:      [[0,0],[0,1]]
  +45 LP:           0.5 * [[1,1],[1,1]]
  HWP (fast axis x): [[1,0],[0,-1]]         (rotates polarization 90 deg)
  QWP (fast axis x): [[1,0],[0,-i]]         (converts LP to circular)
  Rotation by phi:  [[cos,sin],[-sin,cos]]  (rotates the polarization axis)
  Phase retarder:   [[1,0],[0,e^(i*delta)]] (general wave plate)

ROTATING FRAME IN PHOTONICS:
  When a fiber or crystal is rotated by angle theta, the Jones matrix transforms as:
    M_rotated = R(-theta) * M * R(theta)
  where R(theta) = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
  This is a SIMILARITY TRANSFORM (rotation of coordinate frame).
  Same math as rotating reference frames in classical mechanics.

STOKES PARAMETERS (for partially polarized or unpolarized light):
  S0 = <|Ex|^2> + <|Ey|^2>    (total intensity)
  S1 = <|Ex|^2> - <|Ey|^2>    (horizontal vs vertical polarization)
  S2 = 2*Re(<Ex* Ey>)         (linear +45 vs -45 polarization)
  S3 = 2*Im(<Ex* Ey>)         (right circular vs left circular)
  Degree of polarization: DOP = sqrt(S1^2 + S2^2 + S3^2) / S0 in [0,1]
  Jones -> Stokes is valid only for pure states (DOP=1).

POINCARE SPHERE:
  (S1, S2, S3) lives on a sphere of radius S0. One full rotation of a wave plate
  corresponds to a rotation on the Poincare sphere. This is the SAME BERRY PHASE
  sphere as in quantum_information.py (Bloch sphere with renamed axes).

  Poles: N = RCP (S3=+S0), S = LCP (S3=-S0)
  Equator: linear polarizations (S3=0)

CONNECTION TO GS RECEIVER (this repo):
  The GS algorithm assumes the optical signal has a FIXED polarization when it
  enters the dispersive fiber. Polarization rotation from PMD (polarization-mode
  dispersion) appears as a time-varying Jones matrix on the fiber -- this is the
  main source of fading in a real fiber GS receiver. The Jones chain rule gives
  the transfer matrix: M_total = M_fiber * M_connector * M_CFBG.
"""
import numpy as np
import sympy as sp


# ── Jones vectors (normalized) ─────────────────────────────────────────

def jones_horizontal():
    return np.array([1, 0], dtype=complex)

def jones_vertical():
    return np.array([0, 1], dtype=complex)

def jones_linear(angle_deg):
    """Linearly polarized at angle_deg from the x-axis."""
    th = np.radians(angle_deg)
    return np.array([np.cos(th), np.sin(th)], dtype=complex)

def jones_right_circular():
    return np.array([1, -1j], dtype=complex) / np.sqrt(2)

def jones_left_circular():
    return np.array([1, 1j], dtype=complex) / np.sqrt(2)

def jones_elliptical(psi_deg, chi_deg):
    """General elliptical polarization state on the Poincare sphere.

    psi = orientation angle of the ellipse major axis (0-180 deg)
    chi = ellipticity angle (-45 to +45 deg); chi=0 is linear, chi=+/-45 is circular
    """
    psi = np.radians(psi_deg)
    chi = np.radians(chi_deg)
    Ex = np.cos(chi) * np.cos(psi) - 1j * np.sin(chi) * np.sin(psi)
    Ey = np.cos(chi) * np.sin(psi) + 1j * np.sin(chi) * np.cos(psi)
    return np.array([Ex, Ey], dtype=complex)


# ── Jones matrices ─────────────────────────────────────────────────────

def jones_matrix_linear_polarizer(angle_deg):
    """Jones matrix for a linear polarizer at angle_deg."""
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c*c, c*s], [c*s, s*s]], dtype=complex)

def jones_matrix_phase_retarder(delta_rad, fast_axis_deg=0.0):
    """General wave plate: phase retardation delta between fast and slow axes.

    For fast axis along x:
      M = [[1, 0], [0, e^(i*delta)]]  (slow axis picks up extra phase delta)

    For fast axis at angle theta:
      M_rotated = R(-theta) * M * R(theta)

    HWP: delta = pi -> M = diag(1, -1)  (with fast axis at x)
    QWP: delta = pi/2 -> M = diag(1, -i)  (with fast axis at x)
    """
    M = np.array([[1, 0], [0, np.exp(1j * delta_rad)]], dtype=complex)
    if fast_axis_deg != 0.0:
        M = jones_rotate_matrix(M, fast_axis_deg)
    return M

def jones_matrix_hwp(fast_axis_deg=0.0):
    """Half-wave plate: delta = pi."""
    return jones_matrix_phase_retarder(np.pi, fast_axis_deg)

def jones_matrix_qwp(fast_axis_deg=0.0):
    """Quarter-wave plate: delta = pi/2."""
    return jones_matrix_phase_retarder(np.pi / 2, fast_axis_deg)

def jones_matrix_rotation(angle_deg):
    """Rotation matrix R(theta): rotates the polarization coordinate frame.

    This is the ROTATING FRAME transform: if a birefringent element has its
    fast axis at angle theta, we rotate INTO its frame, apply the diagonal
    Jones matrix, then rotate BACK. That is the chain rule.
    """
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s], [-s, c]], dtype=complex)

def jones_rotate_matrix(M, angle_deg):
    """Transform Jones matrix M into a rotated coordinate frame.

    M_rotated = R(-theta) * M * R(theta)
    """
    R = jones_matrix_rotation(angle_deg)
    R_inv = jones_matrix_rotation(-angle_deg)
    return R_inv @ M @ R

def jones_matrix_faraday_rotator(rotation_deg):
    """Faraday rotator: non-reciprocal rotation (magneto-optic effect).

    Unlike a wave plate, reversing the direction of propagation does NOT
    undo the rotation (because it breaks time-reversal symmetry).
    Used in optical isolators.
    """
    return jones_matrix_rotation(rotation_deg)


# ── chain rule: cascaded optical system ────────────────────────────────

def jones_cascade(jones_matrices):
    """Compute the total Jones matrix for a cascade of optical elements.

    CHAIN RULE: M_total = M_N * M_{N-1} * ... * M_1
    Elements applied LEFT to RIGHT in the list, but multiplied right-to-left.

    Parameters
    ----------
    jones_matrices : list of 2x2 complex arrays
        Ordered list of Jones matrices from input to output.

    Returns
    -------
    M_total : 2x2 complex array
    """
    if not jones_matrices:
        return np.eye(2, dtype=complex)
    M = np.eye(2, dtype=complex)
    for Mk in reversed(jones_matrices):  # rightmost applied first
        M = Mk @ M
    return M


def jones_propagate(E_in, jones_matrices):
    """Propagate a Jones vector through a cascade of optical elements.

    E_out = M_total * E_in  where M_total = M_N * ... * M_1
    """
    M_total = jones_cascade(jones_matrices)
    E_out = M_total @ np.asarray(E_in, dtype=complex)
    return {"E_out": E_out, "M_total": M_total,
            "intensity": float(np.abs(E_out[0])**2 + np.abs(E_out[1])**2)}


# ── Stokes parameters and Poincare sphere ─────────────────────────────

def stokes_from_jones(E):
    """Compute Stokes parameters from a Jones vector (pure state, DOP=1).

    S = (S0, S1, S2, S3) where:
      S0 = |Ex|^2 + |Ey|^2    (total intensity)
      S1 = |Ex|^2 - |Ey|^2    (H-V linear)
      S2 = 2 Re(Ex* Ey)       (+45/-45 linear)
      S3 = 2 Im(Ex* Ey)       (circular)
    """
    E = np.asarray(E, dtype=complex)
    Ex, Ey = E[0], E[1]
    S0 = float(np.abs(Ex)**2 + np.abs(Ey)**2)
    S1 = float(np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = float(2 * np.real(np.conj(Ex) * Ey))
    S3 = float(2 * np.imag(np.conj(Ex) * Ey))
    DOP = np.sqrt(S1**2 + S2**2 + S3**2) / S0 if S0 > 1e-12 else 0.0
    return {"S0": S0, "S1": S1, "S2": S2, "S3": S3, "DOP": DOP}


def poincare_angles(E):
    """Map Jones vector to Poincare sphere (psi, chi) in degrees.

    psi = (1/2) * atan2(S2, S1)   orientation angle [0, 180)
    chi = (1/2) * arcsin(S3/S0)   ellipticity angle [-45, 45]
    """
    S = stokes_from_jones(E)
    psi_deg = 0.5 * np.degrees(np.arctan2(S["S2"], S["S1"]))
    chi_deg = 0.5 * np.degrees(np.arcsin(np.clip(S["S3"] / S["S0"], -1, 1)))
    return {"psi_deg": psi_deg, "chi_deg": chi_deg, "S": S}


def degree_of_polarization(E_ensemble):
    """DOP for an ensemble of Jones vectors (partially polarized light).

    Coherency matrix: J = <E E^dagger>
    DOP = sqrt(1 - 4*det(J)/tr(J)^2)
    """
    E_ens = np.array(E_ensemble, dtype=complex)   # shape (N, 2)
    J = np.zeros((2, 2), dtype=complex)
    for E in E_ens:
        E = E.reshape(2, 1)
        J += E @ E.conj().T
    J /= len(E_ens)
    tr_J = np.trace(J).real
    det_J = np.linalg.det(J).real
    DOP = float(np.sqrt(max(0, 1 - 4 * det_J / tr_J**2))) if tr_J > 1e-12 else 0.0
    return {"J": J, "DOP": DOP, "tr_J": tr_J, "det_J": det_J}


# ── fiber PMD model ────────────────────────────────────────────────────

def fiber_pmd_jones(DGD_ps, theta_deg, omega_rad_per_ps):
    """Jones matrix for a fiber section with polarization-mode dispersion (PMD).

    A birefringent fiber section has two principal states of polarization (PSPs)
    separated by a differential group delay (DGD) in picoseconds.
    The Jones matrix in the frequency domain is:

    M(omega) = R(-theta) * diag(e^(-i*omega*DGD/2), e^(+i*omega*DGD/2)) * R(theta)

    where theta is the orientation of the fast PSP axis and omega is the
    angular frequency offset from the carrier.

    For the GS receiver: PMD acts as a frequency-dependent polarization rotation.
    At the fading nulls (omega*DGD = pi), one polarization is completely attenuated.
    """
    if DGD_ps < 0:
        raise ValueError("DGD must be non-negative")
    phase = omega_rad_per_ps * DGD_ps / 2
    M_biref = np.array([[np.exp(-1j * phase), 0],
                         [0, np.exp(1j * phase)]], dtype=complex)
    return jones_rotate_matrix(M_biref, theta_deg)


# ── SymPy formalism ───────────────────────────────────────────────────

def jones_sympy_5():
    """Five key Jones calculus equations in SymPy."""
    theta, delta, psi, chi = sp.symbols('theta delta psi chi', real=True)
    Ex, Ey = sp.symbols('E_x E_y')
    M1, M2, E_in = sp.symbols('M_1 M_2 E_in')

    return {
        "Chain_rule_cascade":
            sp.Eq(sp.Symbol('E_out'),
                  sp.Symbol('M_N') * sp.Symbol('...') * M2 * M1 * E_in),
        "Phase_retarder":
            sp.Eq(sp.Symbol('M_WP'),
                  sp.Matrix([[1, 0], [0, sp.exp(sp.I * delta)]])),
        "Rotation_matrix":
            sp.Eq(sp.Symbol('R(theta)'),
                  sp.Matrix([[sp.cos(theta), sp.sin(theta)],
                              [-sp.sin(theta), sp.cos(theta)]])),
        "Stokes_S3_circular":
            sp.Eq(sp.Symbol('S3'),
                  2 * sp.im(sp.conjugate(Ex) * Ey)),
        "Rotated_matrix":
            sp.Eq(sp.Symbol('M_rot'),
                  sp.Symbol('R(-theta)') * sp.Symbol('M') * sp.Symbol('R(theta)')),
    }


if __name__ == "__main__":
    sp.init_printing(use_unicode=False)

    print("=== Jones vectors ===")
    states = {
        "H":    jones_horizontal(),
        "+45":  jones_linear(45),
        "RCP":  jones_right_circular(),
        "LCP":  jones_left_circular(),
    }
    for name, E in states.items():
        S = stokes_from_jones(E)
        print(f"  {name:>4}: S=({S['S0']:.2f},{S['S1']:+.2f},{S['S2']:+.2f},{S['S3']:+.2f})")

    print("\n=== Chain rule: LP -> QWP -> HWP ===")
    E_in = jones_linear(45)   # +45 LP
    M_qwp = jones_matrix_qwp(fast_axis_deg=0)
    M_hwp = jones_matrix_hwp(fast_axis_deg=22.5)  # HWP fast at 22.5 deg
    result = jones_propagate(E_in, [M_qwp, M_hwp])
    E_out = result["E_out"]
    print(f"  In:  +45 LP  = [{E_in[0]:.3f}, {E_in[1]:.3f}]")
    print(f"  Out: [{E_out[0]:.3f}, {E_out[1]:.3f}]")
    S_out = stokes_from_jones(E_out)
    print(f"  S3_out = {S_out['S3']:.3f}  (positive -> RCP, negative -> LCP)")

    print("\n=== Poincare sphere: RCP ===")
    p = poincare_angles(jones_right_circular())
    print(f"  psi = {p['psi_deg']:.1f} deg,  chi = {p['chi_deg']:.1f} deg")
    print(f"  (chi=+45 deg confirms RCP at north pole of Poincare sphere)")

    print("\n=== Fiber PMD at resonance: DGD=10 ps, omega=pi/10 ===")
    M_pmd = fiber_pmd_jones(10.0, theta_deg=30, omega_rad_per_ps=np.pi/10)
    E_h = jones_horizontal()
    E_pmd = M_pmd @ E_h
    print(f"  H in -> [{E_pmd[0]:.3f}, {E_pmd[1]:.3f}] after PMD")
    print(f"  Intensity: {np.abs(E_pmd[0])**2 + np.abs(E_pmd[1])**2:.4f} (conserved)")

    print("\n=== SymPy 5 ===")
    for k, eq in jones_sympy_5().items():
        print(f"  {k}: {eq}")
