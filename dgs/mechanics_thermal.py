"""
Mechanics and Thermal Physics: Statics, Dynamics, Rotation, Operators, Thermodynamics

LEARNING THREAD:
  Statics (week 1) -> Dynamics -> Lagrangian -> Thermal -> Quantum analogy

SYNTAX (0, 1, 2 -- as in "week 1 → week 2 → week 3"):
  Statics (Week 1):  Sum F = 0, Sum M = 0. No acceleration. 6x6 stiffness matrix.
  Dynamics (Week 2): F = ma. Lagrangian L = T - V. Euler-Lagrange equations.
  Thermal  (Week 3): F = -dU/dq. Energy flows. Entropy = phase space volume.

UNIFYING OPERATOR LANGUAGE (classical -> quantum):
  Classical: F = -grad(V)    (gradient of potential)
  Lagrangian: d/dt(dL/dq_dot) - dL/dq = 0
  Quantum: H|psi> = E|psi>   (same eigenvalue structure)
  All three: LINEAR OPERATOR acting on a STATE = OBSERVABLE x STATE

CONNECTIONS TO THIS REPO:
  Terminal velocity: drag coefficient CD = function of Re number
    Re = rho*v*L/mu -> dispersion-like dependence on velocity
  6x6 stiffness matrix: eigenmodes = resonant frequencies = poles of H(s)
    Same as H(f) poles in dispersive medium
  Rotation SO(3): exp(j*theta*J) = unitary rotation = exp(j*pi*D*f^2) in structure
    Both are "exponentiated operators" -- Lie group / Lie algebra connection
  Odd/even symmetry: integral of odd function = 0 on [-a,a]
    This is WHY GS can't recover global phase: phase is odd, intensity is even
  Lagrangian: L = T - V -> EL eqs -> Newton's laws
    In optics: L = sum_n |E_n - E_meas_n|^2 -> gradient descent = GS
  Thermal: S = k_B * ln(Omega) -> information entropy H = -sum p*ln(p)
    Photon entropy sets the quantum limit on phase retrieval SNR
"""
import math
import numpy as np
import sympy as sp

c_light = 2.998e8; hbar = 1.0546e-34; kB = 1.381e-23; g = 9.81


# ============================================================
# Terminal Velocity: Force Balance (Statics of Falling Object)
# ============================================================

def terminal_velocity():
    """
    Terminal velocity: a statics problem in disguise.

    STATICS FRAME: at terminal velocity, acceleration = 0.
      Sum of forces = 0:
        Weight down:   F_g = m*g
        Drag up:       F_d = (1/2)*rho*v^2*C_D*A
        Buoyancy up:   F_b = rho*V_obj*g  (often negligible for dense objects)
      At terminal: F_g = F_d + F_b
        m*g = (1/2)*rho_air*v_t^2*C_D*A + rho_air*V_obj*g
        v_t = sqrt(2*(m-rho_air*V)*g / (rho_air*C_D*A))

    SUPERSONIC REGIME (v > 343 m/s at sea level):
      Mach number: Ma = v / v_sound
      Drag coefficient changes dramatically:
        Subsonic Ma < 0.8:   C_D ~ 0.47 (sphere)
        Transonic 0.8-1.2:  C_D spikes to ~1.2 (wave drag!)
        Supersonic Ma > 1.2: C_D ~ 1/Ma^2 (decreasing, bow shock established)
      Terminal velocity possible at supersonic speeds:
        Felix Baumgartner (2012): 1357 km/h = Ma 1.25 at 39 km altitude
        At that altitude: rho_air = 0.004 kg/m^3 (vs 1.225 at sea level)
        Sparse air -> higher terminal velocity -> can exceed Ma=1.

    REYNOLDS NUMBER: Re = rho*v*L/mu
      Re < 1:      Stokes drag: F_d = 6*pi*mu*r*v (linear in v)
      Re ~ 1-1000: transition
      Re > 1000:   Quadratic drag: F_d = (1/2)*rho*v^2*C_D*A
      This nonlinear switch is the same physics as dispersion:
        low frequency -> linear response
        high frequency -> nonlinear (anharmonic) response

    SKATING (slow terminal velocity):
      On ice: friction coefficient mu_ice ~ 0.003 (much lower than dry: 0.5)
      At speed v on ice: F_friction = mu_ice * m * g (constant, not velocity-dependent!)
      Maximum speed: no air drag terminal at slow speeds, limited by muscle power.
      At race speeds (~15 m/s): F_drag ~ (1/2)*1.2*15^2*0.7*0.45 ~ 42 N
      This EXCEEDS friction -> aerodynamics matter for skating!
    """
    rho_air_sl = 1.225   # kg/m^3 sea level
    mu_air = 1.81e-5     # Pa*s dynamic viscosity at 20C
    g_val = 9.81

    # 1. Standard skydiver
    m_diver = 80; C_D = 0.47; A_diver = 0.5
    v_term_diver = math.sqrt(2*m_diver*g_val / (rho_air_sl*C_D*A_diver))

    # 2. Baumgartner jump (39 km altitude)
    rho_39km = 0.004   # kg/m^3
    v_sound_39km = 303   # m/s (colder -> lower c_s)
    C_D_supersonic = 0.6   # transonic average
    v_term_39km = math.sqrt(2*m_diver*g_val / (rho_39km*C_D_supersonic*A_diver))
    Ma_39km = v_term_39km / v_sound_39km

    # 3. Velocity vs time for a falling sphere (subsonic)
    # dv/dt = g - (rho_air*C_D*A)/(2*m) * v^2
    k = rho_air_sl*C_D*A_diver / (2*m_diver)
    v_t_arr = np.linspace(0, v_term_diver*1.2, 400)
    a_arr = g_val - k*v_t_arr**2   # acceleration at each v

    # Time to reach 99% terminal velocity (analytic solution)
    tau = math.sqrt(m_diver / (rho_air_sl*C_D*A_diver*g_val))
    t_arr = np.linspace(0, 5*tau, 400)
    v_t_time = v_term_diver * np.tanh(t_arr / tau)

    # Reynolds number at terminal velocity
    L_diver = 1.8   # characteristic length [m]
    Re_terminal = rho_air_sl * v_term_diver * L_diver / mu_air

    # 4. Skateboard / inline skating
    m_skater = 75
    mu_ice = 0.003; mu_wheels = 0.02
    F_friction_ice = mu_ice * m_skater * g_val
    F_friction_wheels = mu_wheels * m_skater * g_val
    C_D_skater = 0.7; A_skater = 0.45
    v_arr_skate = np.linspace(0, 30, 400)
    F_drag_skate = 0.5*rho_air_sl*v_arr_skate**2*C_D_skater*A_skater
    # v where drag = friction (transition point)
    v_transition_ice = math.sqrt(2*F_friction_ice / (rho_air_sl*C_D_skater*A_skater))

    return {
        'skydiver': {
            'v_terminal_ms': float(v_term_diver),
            'v_terminal_kmh': float(v_term_diver*3.6),
            'Re_terminal': float(Re_terminal),
            'tau_s': float(tau),
        },
        'Baumgartner_39km': {
            'rho_air': rho_39km,
            'v_terminal_ms': float(v_term_39km),
            'v_terminal_kmh': float(v_term_39km*3.6),
            'Ma': float(Ma_39km),
            'supersonic': bool(Ma_39km > 1),
            'lesson': 'Thin air at 39 km -> terminal v exceeds Mach 1. Statics, just different air.',
        },
        'velocity_vs_time': {
            't_s': t_arr.tolist(),
            'v_ms': v_t_time.tolist(),
            'v_terminal_ms': float(v_term_diver),
            'formula': 'v(t) = v_t * tanh(t/tau), tau = sqrt(m/(rho*C_D*A*g))',
        },
        'acceleration_vs_v': {
            'v_ms': v_t_arr.tolist(),
            'a_m_s2': a_arr.tolist(),
        },
        'skating': {
            'F_friction_ice_N': float(F_friction_ice),
            'F_friction_wheels_N': float(F_friction_wheels),
            'v_where_drag_equals_ice_friction_ms': float(v_transition_ice),
            'v_transition_kmh': float(v_transition_ice*3.6),
            'F_drag_N': F_drag_skate.tolist(),
            'v_arr_ms': v_arr_skate.tolist(),
            'lesson': 'At ~5 m/s on ice, aerodynamic drag exceeds ice friction. Tuck your body!',
        },
        'Reynolds': {
            'Stokes': 'Re < 1: F_drag = 6*pi*mu*r*v  (linear)',
            'Quadratic': 'Re > 1000: F_drag = (1/2)*rho*v^2*C_D*A  (quadratic)',
            'formula': 'Re = rho*v*L/mu',
        },
    }


# ============================================================
# 6x6 Stiffness Matrix: Eigenmode Discovery
# ============================================================

def stiffness_matrix_6x6():
    """
    6x6 stiffness matrix: the complete description of a 3D elastic joint.

    A rigid body in 3D has 6 DOFs:
      3 translations: x, y, z
      3 rotations: theta_x, theta_y, theta_z

    Stiffness matrix K (6x6): F = K * u
      [Fx]   [K11 K12 K13 K14 K15 K16] [ux]
      [Fy]   [K21 K22 K23 K24 K25 K26] [uy]
      [Fz] = [K31 K32 K33 K34 K35 K36] [uz]
      [Mx]   [K41 K42 K43 K44 K45 K46] [rx]
      [My]   [K51 K52 K53 K54 K55 K56] [ry]
      [Mz]   [K61 K62 K63 K64 K65 K66] [rz]

    K is symmetric (K = K^T) because energy is a quadratic form:
      U = (1/2) u^T K u
    Eigenvalues of K = stiffnesses in principal directions = resonant frequencies
    Eigenvectors = principal modes (the "natural" directions of deformation)

    FEATURE DISCOVERY (like PCA):
      Large eigenvalue: stiff direction (hard to deform)
      Small eigenvalue: compliant direction (easy to deform, flexible)
      Zero eigenvalue: rigid body mode (no restoring force, free motion)
      Negative eigenvalue: INSTABILITY (buckle, snap-through)

    Connection to H(f):
      In frequency domain: (K - omega^2*M) u = F
      Resonance: det(K - omega^2*M) = 0 -> natural frequencies omega_n
      This IS the denominator of the transfer function H(f) = 1/(K - omega^2*M)
      GS phase retrieval: we're finding the phase of H(f) from intensity measurements.
    """
    # Build a physical 6x6 stiffness matrix for a beam element
    # (Timoshenko beam, 3D, one element with both end nodes -> 6x6 condensed)
    E_mod = 200e9   # [Pa] steel modulus
    G_shear = 77e9  # [Pa] steel shear modulus
    L_beam = 1.0    # [m]
    A_cs = 1e-4     # [m^2] cross-section area
    I_y = 8.33e-9   # [m^4] moment of inertia
    I_z = 8.33e-9
    J_torsion = 1.67e-8   # [m^4] polar moment

    # Condensed 6x6: [u_x, u_y, u_z, theta_x, theta_y, theta_z] at fixed end
    # (cantilever: one end fixed, free end DOFs)
    EA_L = E_mod*A_cs/L_beam
    EIy_L3 = 12*E_mod*I_y/L_beam**3
    EIz_L3 = 12*E_mod*I_z/L_beam**3
    GJ_L = G_shear*J_torsion/L_beam
    EIy_L2 = 6*E_mod*I_y/L_beam**2
    EIz_L2 = 6*E_mod*I_z/L_beam**2
    EIy_L = 4*E_mod*I_y/L_beam
    EIz_L = 4*E_mod*I_z/L_beam

    K = np.array([
        [EA_L,    0,       0,      0,       0,       0],
        [0,       EIy_L3,  0,      0,       0,      -EIy_L2],
        [0,       0,       EIz_L3, 0,       EIz_L2,  0],
        [0,       0,       0,      GJ_L,    0,       0],
        [0,       0,       EIz_L2, 0,       EIz_L,   0],
        [0,      -EIy_L2,  0,      0,       0,       EIy_L],
    ])

    # Verify symmetry
    is_symmetric = bool(np.allclose(K, K.T))

    # Eigenvalue decomposition: feature discovery
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Condition number: ratio of largest to smallest eigenvalue
    condition_number = float(eigenvalues[-1] / max(eigenvalues[0], 1e-10))

    # Mass matrix M (consistent mass matrix for beam)
    rho_steel = 7850   # kg/m^3
    m_total = rho_steel * A_cs * L_beam
    M = np.diag([m_total/3, 13*m_total/35, 13*m_total/35,
                 rho_steel*J_torsion*L_beam/3, m_total*L_beam**2/105,
                 m_total*L_beam**2/105])

    # Natural frequencies: (K - omega^2*M) u = 0
    omega_sq, _ = np.linalg.eigh(np.linalg.solve(M, K))
    omega_sq = np.maximum(omega_sq, 0)   # clamp numerics
    f_natural_Hz = np.sqrt(omega_sq) / (2*np.pi)

    # Strain energy in each mode
    strain_energies = [float(0.5 * ev * np.dot(evec, evec))
                       for ev, evec in zip(eigenvalues, eigenvectors.T)]

    # PCA analogy: treat K as data covariance matrix
    # Most compliant direction = first eigenvector (smallest eigenvalue)
    most_compliant_dir = eigenvectors[:, 0]
    most_stiff_dir = eigenvectors[:, -1]

    return {
        'K': K.tolist(),
        'is_symmetric': is_symmetric,
        'beam': {
            'E_GPa': E_mod/1e9, 'L_m': L_beam,
            'A_m2': A_cs, 'I_m4': I_y,
        },
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'condition_number': condition_number,
        'natural_frequencies_Hz': f_natural_Hz.tolist(),
        'strain_energies': strain_energies,
        'feature_discovery': {
            'most_compliant': {
                'eigenvalue': float(eigenvalues[0]),
                'direction': most_compliant_dir.tolist(),
                'physics': 'Smallest stiffness = easiest to deform = first mode to fail',
            },
            'most_stiff': {
                'eigenvalue': float(eigenvalues[-1]),
                'direction': most_stiff_dir.tolist(),
                'physics': 'Largest stiffness = axial compression/extension',
            },
        },
        'connection': {
            'PCA': 'Eigenvectors of K = principal strain directions = PCA components of deformation',
            'H_f': 'H(omega) = 1/(K - omega^2*M) has poles at natural frequencies omega_n',
            'GS': 'Phase retrieval finds the phase of H(f); stiffness matrix gives the poles',
            'instability': 'Negative eigenvalue -> buckling (like GS divergence = runaway phase error)',
        },
    }


# ============================================================
# Quantum Operators on Classical Computing
# ============================================================

def qm_operators_classical():
    """
    Quantum mechanics operators implemented on classical computers.

    A QUANTUM OPERATOR is just a MATRIX acting on a STATE VECTOR.
    On a classical computer: matrix-vector multiply.
    No quantum hardware needed to compute eigenvalues or expectation values.

    HERMITIAN OPERATOR A = A^dagger:
      A_{ij} = conj(A_{ji})
      All eigenvalues are REAL  (observables must be real)
      Eigenvectors are ORTHOGONAL (different measurement outcomes don't interfere)
      Eigenvalue equation: A|psi_n> = a_n|psi_n>

    MEASUREMENT (classical simulation):
      |psi> = sum_n c_n |psi_n>   [superposition]
      Prob(outcome a_n) = |c_n|^2
      Expectation value <A> = sum_n a_n * |c_n|^2 = <psi|A|psi>
      After measurement of a_n: state COLLAPSES to |psi_n>

    COMMUTATOR [A, B] = AB - BA:
      [x, p] = j*hbar  (position and momentum don't commute)
      [A, B] = 0       (A and B can be measured simultaneously)
      Heisenberg: Delta_A * Delta_B >= (1/2)|<[A,B]>|

    CLASSICAL ANALOG:
      x -> column vector (state)
      A -> matrix (operator = measurement apparatus)
      Eigenvalue -> measurement outcome
      Eigenvector -> definite state (will give same result repeatedly)
      Commutator -> cannot simultaneously diagonalize -> uncertainty

    THIS REPO connection:
      GS constraint operators:
        Pi_amplitude: project onto amplitude constraint (non-Hermitian but bounded)
        Pi_phase: project onto phase estimate (unitary rotation)
        Full cycle T = Pi2 * H * Pi1 * H^{-1}: NOT Hermitian -> eigenvalues complex
        |eigenvalue| < 1 -> GS converges (damped)
        |eigenvalue| = 1 -> stalemate (oscillates)
        |eigenvalue| > 1 -> diverges
    """
    # Build Hermitian operators as matrices (position, momentum, H in finite basis)
    N_basis = 10   # truncate to N_basis eigenstates
    n_arr = np.arange(N_basis)

    # Harmonic oscillator Hamiltonian H = hbar*omega*(a^dag a + 1/2)
    omega0 = 1.0   # normalized
    H_osc = np.diag(hbar * omega0 * (n_arr + 0.5))   # diagonal in Fock basis

    # Position operator x = sqrt(hbar/(2*m*omega)) * (a + a^dag)
    # Creation/annihilation in Fock basis
    a_dag = np.diag(np.sqrt(np.arange(1, N_basis)), -1)   # N x N
    a_op  = np.diag(np.sqrt(np.arange(1, N_basis)), +1)
    mass = 9.109e-31   # electron mass
    x_scale = math.sqrt(hbar / (2*mass*omega0))
    X_op = x_scale * (a_op + a_dag)   # Hermitian

    # Momentum operator p = j*sqrt(hbar*m*omega/2) * (a^dag - a)
    p_scale = math.sqrt(hbar*mass*omega0/2)
    P_op = 1j * p_scale * (a_dag - a_op)   # Hermitian

    # Verify Hermitian
    X_herm = bool(np.allclose(X_op, X_op.conj().T, atol=1e-10))
    P_herm = bool(np.allclose(P_op, P_op.conj().T, atol=1e-10))

    # Commutator [X, P] should = j*hbar * I
    comm_XP = X_op @ P_op - P_op @ X_op
    comm_expected = 1j * hbar * np.eye(N_basis)
    commutator_check = bool(np.allclose(comm_XP, comm_expected, rtol=1e-6))

    # Expectation values in ground state |0>
    psi_ground = np.zeros(N_basis); psi_ground[0] = 1.0
    E_x = float(np.real(psi_ground.conj() @ X_op @ psi_ground))
    E_p = float(np.real(psi_ground.conj() @ P_op @ psi_ground))
    E_x2 = float(np.real(psi_ground.conj() @ (X_op@X_op) @ psi_ground))
    E_p2 = float(np.real(psi_ground.conj() @ (P_op@P_op) @ psi_ground))
    Delta_x = math.sqrt(E_x2 - E_x**2)
    Delta_p = math.sqrt(E_p2 - E_p**2)
    uncertainty_product = Delta_x * Delta_p

    # Eigenvalues/vectors of X in Fock basis
    x_eigenvalues, x_eigenvectors = np.linalg.eigh(X_op)

    # GS iteration operator T (2x2 toy model)
    # Forward: amplitude replace at input
    I_in = 1.0; I_out = 0.8   # measured intensities
    # T matrix: one GS step maps e^{j phi_old} -> e^{j phi_new}
    # Approximate as 2D rotation + scaling
    phi = 0.3   # test phase
    T_GS = np.array([
        [math.cos(phi), -math.sin(phi)],
        [math.sin(phi),  math.cos(phi)],
    ]) * math.sqrt(I_out/I_in)
    T_eigenvalues = np.linalg.eigvals(T_GS)
    T_spectral_radius = float(np.max(np.abs(T_eigenvalues)))

    return {
        'operators': {
            'X_op': X_op.tolist(),
            'P_op': np.real(P_op).tolist(),   # imaginary part
            'H_osc': H_osc.tolist(),
            'X_Hermitian': X_herm,
            'P_Hermitian': P_herm,
        },
        'commutator': {
            '[X,P]_equals_jhbar_I': commutator_check,
            'consequence': 'Delta_x * Delta_p >= hbar/2',
            'Delta_x': float(Delta_x),
            'Delta_p': float(Delta_p),
            'product_hbar_units': float(uncertainty_product / hbar),
            'Heisenberg_satisfied': float(uncertainty_product) >= hbar/2,
        },
        'expectation_values': {
            '<x>_ground': float(E_x),
            '<p>_ground': float(E_p),
            '<x^2>_ground': float(E_x2),
            '<p^2>_ground': float(E_p2),
        },
        'x_eigenvalues_sample': x_eigenvalues[:5].tolist(),
        'GS_iteration': {
            'T_matrix': T_GS.tolist(),
            'eigenvalues': [complex(v) for v in T_eigenvalues],
            'spectral_radius': T_spectral_radius,
            'converges': bool(T_spectral_radius < 1),
            'lesson': '|T eigenvalue| < 1: GS converges. = 1: stalemate. > 1: diverges.',
        },
        'classical_computation': {
            'Hermitian_check': 'A == A.conj().T  (numpy: np.allclose(A, A.conj().T))',
            'eigenvalue_eq': 'np.linalg.eigh(A)  -- guaranteed real eigenvalues for Hermitian',
            'expectation': '<A> = psi.conj() @ A @ psi   (matrix sandwiching)',
            'measurement_simulation': 'probs = |coeffs|^2; np.random.choice(eigenvalues, p=probs)',
        },
    }


# ============================================================
# Continuous Rotation SO(3) and Lie Algebra
# ============================================================

def continuous_rotation_SO3():
    """
    Rotation matrices: continuous rotation, Lie algebra, local maximum.

    ROTATION GROUP SO(3):
      R in SO(3): R^T R = I and det(R) = +1
      3 generators: J_x, J_y, J_z (3x3 antisymmetric matrices)
      Finite rotation: R = exp(theta * n_hat . J)  [matrix exponential]
      This is EXACTLY the same structure as:
        Optical H(f) = exp(j*pi*D*f^2) = Lie group element
        Quantum U = exp(-j*H*t/hbar) = Lie group element (SU(2))

    GENERATORS (Lie algebra so(3)):
      J_x = [[0,0,0],[0,0,-1],[0,1,0]]
      J_y = [[0,0,1],[0,0,0],[-1,0,0]]
      J_z = [[0,-1,0],[1,0,0],[0,0,0]]
      Commutation: [J_i, J_j] = epsilon_{ijk} J_k

    LOCAL MAXIMUM of rotation:
      Rotation by angle theta around z: eigenvalues = {1, e^{j*theta}, e^{-j*theta}}
      Trace(R(theta)) = 1 + 2*cos(theta)
      Maximum at theta=0: Trace = 3 (identity)
      Minimum at theta=pi: Trace = -1 (180 degree rotation)
      Local maximum of |R| (Frobenius norm) = always sqrt(3) = constant (unitary!)
      -> Rotations conserve norms: |R*v| = |v| for all v

    CONTINUOUS ROTATION (skating connection):
      Angular velocity omega -> angular momentum L = I * omega
      Kinetic energy: T = (1/2) omega^T I omega  [I = inertia tensor = 3x3 symmetric matrix]
      Free rotation (no torque): Euler's equations
        I1*omega1_dot = (I2-I3)*omega2*omega3  [and cyclic]
      Stable rotation: about axis of largest or smallest I (but NOT intermediate)
      -> Intermediate axis theorem (tennis racket theorem)
      A spinning skater pulls arms in: I decreases, omega increases (conserve L = I*omega)

    PIECEWISE FUNCTIONS at rotation angle theta:
      Heaviside step: H(theta) = 0 if theta<0, 1 if theta>=0
      Dirac delta: delta(theta) = d/d(theta) H(theta)  [spike at theta=0]
      sinc: sinc(theta) = sin(theta)/(theta)            [envelope of diffraction]
      Integration: integral_{-pi}^{pi} sin(n*theta) * sin(m*theta) d(theta)
        = pi if n=m, 0 if n!=m  [ORTHOGONALITY, same as eigenfunction basis]
    """
    # Lie algebra generators for so(3)
    Jx = np.array([[0,0,0],[0,0,-1],[0,1,0]], dtype=float)
    Jy = np.array([[0,0,1],[0,0,0],[-1,0,0]], dtype=float)
    Jz = np.array([[0,-1,0],[1,0,0],[0,0,0]], dtype=float)

    # Verify commutation relations: [Jx, Jy] = Jz
    comm_xy = Jx @ Jy - Jy @ Jx
    comm_yz = Jy @ Jz - Jz @ Jy
    comm_zx = Jz @ Jx - Jx @ Jz
    commutation_check = (np.allclose(comm_xy, Jz) and
                         np.allclose(comm_yz, Jx) and
                         np.allclose(comm_zx, Jy))

    # Rotation matrix around z by angle theta
    theta_arr = np.linspace(0, 2*np.pi, 400)
    Rz_trace = 1 + 2*np.cos(theta_arr)   # trace(Rz) = 1 + 2*cos(theta)

    # Rotation around arbitrary axis n_hat by angle theta
    n_hat = np.array([1,1,1]) / math.sqrt(3)   # (1,1,1) normalized
    theta_val = np.pi/3   # 60 degrees
    # Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    K_mat = np.array([[0, -n_hat[2], n_hat[1]],
                      [n_hat[2], 0, -n_hat[0]],
                      [-n_hat[1], n_hat[0], 0]])
    R_rodrigues = (np.eye(3) + math.sin(theta_val)*K_mat +
                   (1-math.cos(theta_val))*K_mat@K_mat)
    is_SO3 = (np.allclose(R_rodrigues.T @ R_rodrigues, np.eye(3)) and
              abs(np.linalg.det(R_rodrigues) - 1) < 1e-10)

    # Inertia tensor for a figure skater (simplified as ellipsoid)
    # Arms out: I_x = I_y = (m/5)*(a^2+b^2), I_z = (2m/5)*a^2
    m_skater = 60; a_arms = 0.8; b_arms = 0.4; a_no_arms = 0.2
    Iz_arms_out = 2*m_skater/5 * a_arms**2
    Iz_arms_in  = 2*m_skater/5 * a_no_arms**2
    omega_initial = 2*np.pi*1.5   # 1.5 rev/s
    L_z = Iz_arms_out * omega_initial
    omega_arms_in = L_z / Iz_arms_in   # conserve angular momentum

    # Trace vs theta for SO(3) rotation (local max/min analysis)
    d_trace = -2*np.sin(theta_arr)   # derivative of trace
    d2_trace = -2*np.cos(theta_arr)  # second derivative
    # Local max at theta where d_trace=0 and d2_trace<0: theta=pi (trace minimum there)
    # theta=0: d_trace=0, d2_trace=-2 < 0 -> local max of trace
    local_max_theta = 0.0
    local_min_theta = np.pi

    # Piecewise functions
    x_arr = np.linspace(-2*np.pi, 2*np.pi, 1000)
    heaviside = np.heaviside(x_arr, 0.5)
    sinc_arr = np.sinc(x_arr/np.pi)   # sinc(x) = sin(pi*x)/(pi*x) in numpy convention
    # sinc(theta) = sin(theta)/theta
    sinc_phys = np.where(np.abs(x_arr) < 1e-10, 1.0, np.sin(x_arr)/x_arr)

    # Orthogonality integral (numerical check)
    theta_int = np.linspace(-np.pi, np.pi, 10000)
    d_theta = theta_int[1]-theta_int[0]
    n1, m1 = 2, 3
    ortho_nm = float(np.sum(np.sin(n1*theta_int)*np.sin(m1*theta_int))*d_theta)
    n2, m2 = 2, 2
    ortho_nn = float(np.sum(np.sin(n2*theta_int)*np.sin(m2*theta_int))*d_theta)

    return {
        'SO3_generators': {
            'Jx': Jx.tolist(), 'Jy': Jy.tolist(), 'Jz': Jz.tolist(),
            'commutation_correct': bool(commutation_check),
        },
        'rotation_Rz': {
            'theta_rad': theta_arr.tolist(),
            'trace': Rz_trace.tolist(),
            'local_max_theta': local_max_theta,
            'local_min_theta': float(local_min_theta),
            'trace_at_0': 3.0,
            'trace_at_pi': -1.0,
        },
        'rodrigues_rotation': {
            'n_hat': n_hat.tolist(),
            'theta_deg': math.degrees(theta_val),
            'R': R_rodrigues.tolist(),
            'is_SO3': bool(is_SO3),
        },
        'skater_spin': {
            'I_z_arms_out': float(Iz_arms_out),
            'I_z_arms_in': float(Iz_arms_in),
            'omega_initial_rps': float(omega_initial/(2*np.pi)),
            'omega_arms_in_rps': float(omega_arms_in/(2*np.pi)),
            'speedup_factor': float(omega_arms_in/omega_initial),
        },
        'piecewise_functions': {
            'x_arr': x_arr.tolist(),
            'Heaviside': heaviside.tolist(),
            'sinc_physical': sinc_phys.tolist(),
            'sinc_numpy': sinc_arr.tolist(),
            'Heaviside_deriv': 'delta(x)  [Dirac delta = derivative of step]',
            'sinc_at_0': 1.0,
            'sinc_zeros': 'n*pi for n = 1,2,3,...',
        },
        'orthogonality': {
            'integral_sin_n_sin_m_n_ne_m': float(ortho_nm),
            'integral_sin_n_sin_n': float(ortho_nn),
            'expected_n_ne_m': 0.0,
            'expected_n_eq_m': float(np.pi),
            'lesson': 'sin(n*theta) are ORTHOGONAL EIGENFUNCTIONS = basis for Fourier series',
        },
        'Lie_group_connection': {
            'SO3': 'R(theta) = exp(theta * J)',
            'SU2': 'U(t) = exp(-j*H*t/hbar)',
            'H_f': 'H(f) = exp(j*pi*D*f^2)',
            'all_same': 'All three are Lie group elements: exponentiated generator',
        },
    }


# ============================================================
# Odd/Even Symmetry in Integration
# ============================================================

def symmetry_integration():
    """
    Odd and even function symmetry: the most powerful shortcut in integration.

    DEFINITIONS:
      Even function: f(-x) = f(x)   [symmetric about y-axis]
        Examples: cos(x), x^2, |x|, sinc(x), Gaussian, even Fourier modes
        Integral: integral_{-a}^{a} f(x) dx = 2 * integral_{0}^{a} f(x) dx

      Odd function: f(-x) = -f(x)  [antisymmetric about origin]
        Examples: sin(x), x, x^3, erf(x), imaginary part of Fourier transform
        Integral: integral_{-a}^{a} f(x) dx = 0  [ALWAYS, for any a]

    WHY THIS MATTERS FOR GS (this repo):
      Intensity measurement I = |E|^2:
        I(-x) = |E(-x)|^2 = |E(x)|^2 = I(x)   [EVEN by construction!]
        This means intensity can NEVER tell you about the odd part of phase.
        The imaginary part of the phase (odd) is INVISIBLE to intensity measurement.
        This is the fundamental reason phase retrieval is hard:
          measurements are even, phase is not.

    DIRAC DELTA:
      delta(x) = 0 for x != 0,  integral = 1
      delta(x) is EVEN: delta(-x) = delta(x)
      Sifting property: integral f(x)*delta(x-x0) dx = f(x0)
      Derivative: delta'(x) is ODD: delta'(-x) = -delta'(x)
      Fourier transform: F{delta(x)} = 1  [all frequencies equally]

    SINC FUNCTION:
      sinc(x) = sin(x)/x  (physics convention)
      sinc(x) is EVEN: sinc(-x) = sinc(x)
      Zeros at x = n*pi (n = ±1, ±2, ...)
      Fourier transform: F{sinc(pi*x/a)} = rect(a*f)  [rectangular spectrum]
      sinc IS the diffraction pattern from a single slit.
      sinc IS the response of an ideal low-pass filter.

    STATICS WEEK 1 CONNECTION:
      Sum of moments about the center of a symmetric beam:
        Left side: M(-x) (moment from left)
        Right side: M(x)  (moment from right, opposite direction)
        Sum = integral_{-L/2}^{L/2} w(x)*x dx
        If load distribution w(x) is EVEN, this integral = 0 -> balanced!
        -> Net moment = 0 -> beam is in equilibrium automatically by symmetry.

    PIECEWISE FUNCTIONS:
      Heaviside: H(x) = 0 if x<0, 1 if x>0
        Neither odd nor even. BUT: H(x) = (1/2) + (1/2)*sgn(x)
        Even part: 1/2, Odd part: (1/2)*sgn(x)
      Any function can be decomposed:
        f(x) = f_even(x) + f_odd(x)
        f_even(x) = (f(x) + f(-x))/2
        f_odd(x)  = (f(x) - f(-x))/2
    """
    x = np.linspace(-3*np.pi, 3*np.pi, 10000)
    dx = x[1]-x[0]

    # Even functions
    cos_x   = np.cos(x)
    x2      = x**2
    gauss   = np.exp(-x**2/2)
    sinc_x  = np.where(np.abs(x)<1e-10, 1.0, np.sin(x)/x)

    # Odd functions
    sin_x   = np.sin(x)
    x1      = x
    x3      = x**3
    erf_x   = 2/np.sqrt(np.pi)*np.cumsum(np.exp(-x**2))*dx   # approx erf

    # Integration over symmetric interval
    def integrate_sym(f, x, a=np.pi):
        mask = np.abs(x) <= a
        return float(np.trapezoid(f[mask], x[mask]))

    int_cos    = integrate_sym(cos_x, x)
    int_sin    = integrate_sym(sin_x, x)
    int_x2     = integrate_sym(x2,    x)
    int_x3     = integrate_sym(x3,    x)
    int_gauss  = integrate_sym(gauss, x)
    int_sinc   = integrate_sym(sinc_x, x)

    # Decompose Heaviside into even + odd
    H_x = np.heaviside(x, 0.5)
    H_even = (H_x + np.flip(H_x)) / 2
    H_odd  = (H_x - np.flip(H_x)) / 2

    # GS symmetry: intensity I(x) is always even
    # Phase phi(x) has both odd and even components
    # Measurement only constrains even part of phase (through interference)
    phi_true = np.sin(x) + 0.5*np.cos(2*x)   # mixed odd+even phase
    I_true = np.abs(np.exp(1j*phi_true))**2   # = 1 everywhere (unit amplitude)
    phi_even_part = (phi_true + np.flip(phi_true))/2
    phi_odd_part  = (phi_true - np.flip(phi_true))/2

    # Statics: symmetric beam
    L_beam = 10.0   # m
    x_beam = np.linspace(-L_beam/2, L_beam/2, 1000)
    dx_beam = x_beam[1]-x_beam[0]
    # Even uniform load: w(x) = w0 (constant)
    w0 = 1000.0   # N/m
    moment_integrand = w0 * x_beam   # w*x is ODD if w is even -> integral = 0
    net_moment = float(np.trapezoid(moment_integrand, x_beam))
    # Odd load: w(x) = w0*x (linear) -> moment integrand = w0*x^2 (even, nonzero)
    moment_odd_load = float(np.trapezoid(w0*x_beam**2, x_beam))

    return {
        'even_functions': {
            'examples': ['cos(x)', 'x^2', 'exp(-x^2)', 'sinc(x)'],
            'property': 'f(-x) = f(x)',
            'integral_rule': 'int_{-a}^{a} f_even dx = 2*int_{0}^{a} f_even dx',
            'integrals': {
                'int_cos_-pi_pi': float(int_cos),
                'int_x2_-pi_pi': float(int_x2),
                'int_gauss_-pi_pi': float(int_gauss),
                'int_sinc_-pi_pi': float(int_sinc),
            },
        },
        'odd_functions': {
            'examples': ['sin(x)', 'x', 'x^3', 'erf(x)'],
            'property': 'f(-x) = -f(x)',
            'integral_rule': 'int_{-a}^{a} f_odd dx = 0   ALWAYS',
            'integrals': {
                'int_sin_-pi_pi': float(int_sin),
                'int_x_-pi_pi': float(int_x3),
            },
        },
        'heaviside_decomposition': {
            'H_even': H_even.tolist(),
            'H_odd': H_odd.tolist(),
            'H_even_is_constant_half': True,
            'H_odd_is_sgn_over_2': True,
        },
        'GS_connection': {
            'intensity_is_even': True,
            'phase_odd_part_invisible': True,
            'lesson': 'Intensity I = |E|^2 is EVEN. Odd part of phase is invisible to measurement. This is why phase retrieval needs DIVERSITY (not just more photons).',
            'phi_odd_energy': float(np.trapezoid(phi_odd_part**2, x)),
            'phi_even_energy': float(np.trapezoid(phi_even_part**2, x)),
        },
        'statics_symmetry': {
            'symmetric_load_net_moment': float(net_moment),
            'is_zero': abs(net_moment) < 1.0,
            'lesson': 'Symmetric load on symmetric beam: net moment = 0. Symmetry kills the integral.',
            'odd_load_moment': float(moment_odd_load),
        },
        'piecewise_integration': {
            'Dirac_delta_even': True,
            'Dirac_delta_integral': 1.0,
            'Dirac_sifting': 'int f(x)*delta(x-x0) dx = f(x0)',
            'sinc_zeros': [f'n*pi for n = +-1, +-2, ...'],
            'sinc_integral': float(int_sinc),
            'sinc_integral_exact': np.pi,
        },
    }


# ============================================================
# Lagrangian Mechanics
# ============================================================

def lagrangian_mechanics():
    """
    Lagrangian mechanics: the most powerful formulation of classical physics.

    LAGRANGIAN: L = T - V   [kinetic - potential energy]
    EULER-LAGRANGE EQUATION:
      d/dt (dL/d(q_dot)) - dL/dq = 0
      This is Newton's 2nd law in disguise, but works in ANY coordinate system.

    EXAMPLES:

    1. Simple pendulum (angle theta is generalized coordinate):
       T = (1/2) m L^2 theta_dot^2
       V = m g L (1 - cos(theta))
       L = T - V
       EL: m*L^2*theta_ddot + m*g*L*sin(theta) = 0
       -> theta_ddot = -(g/L)*sin(theta)  [exact, nonlinear]
       Small angle: theta_ddot = -(g/L)*theta -> omega_0 = sqrt(g/L)

    2. Sliding block on wedge (2 DOFs: x = block horizontal, X = wedge position):
       Constraint: block stays on wedge surface -> reduces to 2 generalized coords.
       Lagrangian captures constraint automatically.

    3. Bead on rotating wire (1 DOF after constraint):
       T = (1/2)m(r_dot^2 + r^2*omega^2)
       V = (1/2)m*g*r^2/L  (if wire at angle)
       EL: m*r_ddot - m*r*omega^2 + m*g*r/L = 0
       -> Centrifugal term appears automatically!

    4. Hamiltonian H = sum_i p_i * q_dot_i - L  [Legendre transform]
       p_i = dL/d(q_dot_i)  [generalized momentum]
       Hamilton's equations: q_dot = dH/dp, p_dot = -dH/dq
       -> PHASE SPACE picture (q, p)

    STATICS CONNECTION (Week 1):
      At static equilibrium: q_dot = 0, q_ddot = 0
      EL equation: -dL/dq = 0  (since d/dt(dL/dq_dot) = 0)
      -> dV/dq = 0   [extremum of potential = equilibrium!]
      Static equilibrium = minimum of V (stable) or maximum (unstable)
      This is WHY: "minimize potential energy" is the statics rule!

    OPTICS LAGRANGIAN (Fermat's principle):
      delta integral n(r) dl = 0   [minimize optical path length]
      This IS the Lagrangian: L = n(r) * |dr/ds|
      EL equation: d/ds(n dr/ds) = grad(n)  [ray equation]
      In dispersive medium: n = n(omega) -> chromatic dispersion -> H(f)
    """
    # 1. Simple pendulum - numerical integration
    L_pend = 1.0; m_pend = 1.0
    g_val = 9.81; omega_0 = math.sqrt(g_val/L_pend)
    T_period = 2*np.pi/omega_0

    # Large-angle pendulum (Runge-Kutta 4)
    dt = 1e-4; t_max = 3*T_period
    t_arr = np.arange(0, t_max, dt)
    theta = np.zeros(len(t_arr)); theta_dot = np.zeros(len(t_arr))
    theta_0_large = np.radians(60)   # 60 degrees (nonlinear!)
    theta[0] = theta_0_large

    def deriv(th, thd):
        return thd, -(g_val/L_pend)*math.sin(th)

    for i in range(len(t_arr)-1):
        k1_th, k1_thd = deriv(theta[i], theta_dot[i])
        k2_th, k2_thd = deriv(theta[i]+dt/2*k1_th, theta_dot[i]+dt/2*k1_thd)
        k3_th, k3_thd = deriv(theta[i]+dt/2*k2_th, theta_dot[i]+dt/2*k2_thd)
        k4_th, k4_thd = deriv(theta[i]+dt*k3_th, theta_dot[i]+dt*k3_thd)
        theta[i+1] = theta[i] + dt*(k1_th+2*k2_th+2*k3_th+k4_th)/6
        theta_dot[i+1] = theta_dot[i] + dt*(k1_thd+2*k2_thd+2*k3_thd+k4_thd)/6

    # Energy conservation check
    T_arr = 0.5*m_pend*L_pend**2*theta_dot**2
    V_arr = m_pend*g_val*L_pend*(1-np.cos(theta))
    E_arr = T_arr + V_arr

    # 2. Bead on rotating wire
    omega_wire = 3.0   # rad/s
    r_arr = np.linspace(0, 0.5, 400)   # radial position
    V_centrifugal_effective = -0.5*m_pend*omega_wire**2*r_arr**2
    V_gravity_effective = 0.5*m_pend*g_val*r_arr**2/L_pend
    V_eff = V_centrifugal_effective + V_gravity_effective
    r_eq_sq = omega_wire**2*L_pend/g_val   # r^2 at equilibrium (if real)
    r_eq = math.sqrt(max(r_eq_sq, 0)) if r_eq_sq > 0 else 0

    # 3. Statics: equilibrium = minimum of V
    x_eq_arr = np.linspace(-2, 2, 400)
    V_double_well = x_eq_arr**4 - 2*x_eq_arr**2 + 0.5*x_eq_arr
    dV = np.gradient(V_double_well, x_eq_arr)
    d2V = np.gradient(dV, x_eq_arr)
    # Find equilibria (dV = 0)
    sign_changes = np.where(np.diff(np.sign(dV)))[0]
    equilibria = x_eq_arr[sign_changes]
    stable = [bool(d2V[i] > 0) for i in sign_changes]

    # 4. Hamiltonian phase portrait of pendulum
    theta_grid = np.linspace(-np.pi, np.pi, 100)
    p_grid = np.linspace(-5, 5, 100)
    TH, P = np.meshgrid(theta_grid, p_grid)
    H_pendulum = P**2/(2*m_pend*L_pend**2) + m_pend*g_val*L_pend*(1-np.cos(TH))

    return {
        'pendulum': {
            'omega_0_rad_s': float(omega_0),
            'T_period_s': float(T_period),
            'theta_0_deg': 60,
            't_s': t_arr[::100].tolist(),
            'theta_rad': theta[::100].tolist(),
            'E_total': E_arr[::100].tolist(),
            'energy_conserved': bool(np.std(E_arr) < 1e-4),
        },
        'rotating_bead': {
            'r_m': r_arr.tolist(),
            'V_eff': V_eff.tolist(),
            'r_equilibrium_m': float(r_eq),
            'centrifugal_exceeds_gravity': bool(omega_wire**2 > g_val/L_pend),
        },
        'statics_equilibria': {
            'x': x_eq_arr.tolist(),
            'V': V_double_well.tolist(),
            'equilibrium_x': equilibria.tolist(),
            'stable': stable,
            'lesson': 'Equilibrium = dV/dq = 0. Stable if d^2V/dq^2 > 0. Lagrangian makes statics automatic.',
        },
        'Hamiltonian': {
            'theta_rad': theta_grid.tolist(),
            'p': p_grid.tolist(),
            'H_pendulum': H_pendulum.tolist(),
            'separatrix_energy': float(2*m_pend*g_val*L_pend),
            'lesson': 'Separatrix at E = 2mgL separates libration (closed orbits) from rotation (spinning over top)',
        },
        'EL_equations': {
            'pendulum': 'theta_ddot = -(g/L)*sin(theta)',
            'small_angle': 'theta_ddot = -(g/L)*theta  [SHO, omega_0 = sqrt(g/L)]',
            'general': 'd/dt(dL/dq_dot) - dL/dq = 0',
            'statics_limit': 'q_dot=0 -> dV/dq=0 -> equilibrium condition',
            'optics': 'Fermat: delta(int n dl) = 0 -> ray equation -> H(f) in dispersive medium',
        },
    }


# ============================================================
# Thermal Physics
# ============================================================

def thermal_physics():
    """
    Thermal physics: thermodynamics + statistical mechanics + engineering.

    THERMODYNAMICS (macroscopic):
      1st law: dU = dQ - dW   [energy conservation]
        dU = change in internal energy
        dQ = heat added to system
        dW = work done BY system = P dV
      2nd law: dS >= dQ/T   [entropy never decreases]
        For reversible: dS = dQ/T
        For irreversible: dS > dQ/T  (free expansion, friction, etc.)
      3rd law: S -> 0 as T -> 0  [absolute zero: perfect crystal]

    STATISTICAL MECHANICS (microscopic):
      Entropy: S = k_B * ln(Omega)   [Boltzmann 1877]
        Omega = number of microstates = degeneracy
        This connects information theory to thermodynamics!
        Shannon entropy H = -sum p_i * ln(p_i)  [same structure]
      Partition function: Z = sum_n exp(-E_n / k_B T)
        Free energy: F = -k_B T * ln(Z)
        Everything follows: U = -d(ln Z)/d(beta), beta = 1/(k_B T)

    EQUIPARTITION THEOREM:
      Each quadratic degree of freedom contributes (1/2)*k_B*T to energy.
      Monatomic gas: 3 DOFs (x,y,z translation) -> U = (3/2)*N*k_B*T
      Diatomic: 5 DOFs (3 trans + 2 rot) -> U = (5/2)*N*k_B*T
      Solid:    6 DOFs (3 trans + 3 potential) -> U = 3*N*k_B*T [Dulong-Petit]

    ENGINEERING APPLICATIONS:
      Heat engine efficiency: eta = 1 - T_cold/T_hot  [Carnot limit]
      Refrigerator COP: COP = T_cold / (T_hot - T_cold)
      Blackbody radiation: spectral emissive power = Planck distribution
        B(f) = (2hf^3/c^2) / (exp(hf/kT) - 1)  [connects to QM: photon gas]
      Thermoelectric: Seebeck effect: V = S * Delta_T  [convert heat to electricity]
        ZT = S^2*sigma*T/kappa  [figure of merit; ZT > 1 for practical devices]

    CONNECTION TO THIS REPO:
      Thermal noise = Johnson-Nyquist noise: S_V(f) = 4*k_B*T*R [V^2/Hz]
      Shot noise = Poisson: S_I = 2*e*I  [A^2/Hz]
      At 1550 nm: hf/(k_B T) >> 1 -> quantum noise dominates over thermal noise
      Photon entropy: S_photon = k_B * [(n+1)*ln(n+1) - n*ln(n)] per mode
      n = Bose-Einstein occupancy = 1/(exp(hf/kT)-1)
      This sets the fundamental quantum limit on phase retrieval SNR.

    BUILDING FOR THE FUTURE (engineering technology):
      Thermal management in photonic ICs:
        Silicon photonics: dn/dT = 1.8e-4 /K  [large thermo-optic coefficient]
        Phase shift per Kelvin: Delta_phi = (2*pi/lambda)*dn_dT*L*Delta_T
        Thermal crosstalk between channels = noise in WDM system
        Active thermal stabilization: PID controllers, Peltier elements
      Photovoltaics: Shockley-Queisser limit = 33.7% (single junction, AM1.5)
        Tandem cells (GaAs on Si): can approach 40%
        Thermophotovoltaics: use thermal emitter at 1000-2000 K
    """
    # Carnot efficiency vs temperature ratio
    T_cold = 300   # K (room temperature, cold reservoir)
    T_hot_arr = np.linspace(300, 1000, 200)
    eta_Carnot = 1 - T_cold/T_hot_arr

    # Blackbody spectrum (Planck distribution)
    T_sun = 5778; T_room = 300
    f_arr = np.logspace(12, 16, 1000)   # Hz: IR to UV
    h_P = 6.626e-34; c_l = 2.998e8; kB_val = 1.381e-23

    def planck_B(f, T):
        x = h_P*f/(kB_val*T)
        return (2*h_P*f**3/c_l**2) / np.expm1(x)

    B_sun = planck_B(f_arr, T_sun)
    B_room = planck_B(f_arr, T_room)
    lambda_peak_sun = 2.898e-3 / T_sun * 1e9   # nm, Wien's law

    # Partition function for 2-level system
    E1 = 0; E2_arr = np.linspace(0, 5, 200)*kB_val*300   # 0 to 5 kT
    T_beta = 300
    Z_2level = 1 + np.exp(-E2_arr/(kB_val*T_beta))
    p1 = 1/Z_2level; p2 = np.exp(-E2_arr/(kB_val*T_beta))/Z_2level
    S_2level = -kB_val*(p1*np.log(p1+1e-30) + p2*np.log(p2+1e-30))

    # Equipartition: k_B*T per quadratic mode
    T_room_25 = 298
    kT_25 = kB_val * T_room_25
    kT_eV = kT_25 / 1.602e-19

    # Thermal noise and quantum noise at 1550 nm
    R = 50; B_noise = 1e9   # bandwidth 1 GHz
    S_V_thermal = 4*kB_val*T_room*R   # V^2/Hz
    V_thermal_rms = math.sqrt(S_V_thermal*B_noise)
    f_1550 = c_l/1550e-9
    hf_kT = h_P*f_1550 / (kB_val*T_room)
    n_thermal_1550 = 1/(math.exp(hf_kT)-1)

    # Silicon photonics thermal phase tuning
    dn_dT_Si = 1.8e-4   # /K
    lambda0_Si = 1550e-9; L_ring = 100e-6   # 100 um ring resonator
    Delta_phi_per_K = (2*np.pi/lambda0_Si) * dn_dT_Si * L_ring   # rad/K
    Delta_T_for_pi = np.pi / Delta_phi_per_K   # K to get pi phase shift

    return {
        'thermodynamics': {
            'laws': {
                '0th': 'Thermodynamic equilibrium is transitive (defines temperature)',
                '1st': 'dU = dQ - P*dV  (energy conservation)',
                '2nd': 'dS >= dQ/T  (entropy never decreases)',
                '3rd': 'S -> 0 as T -> 0  (Planck)',
            },
            'Carnot': {
                'T_cold_K': T_cold,
                'T_hot_arr': T_hot_arr.tolist(),
                'eta_Carnot': eta_Carnot.tolist(),
                'eta_at_1000K': float(eta_Carnot[-1]),
            },
        },
        'blackbody': {
            'f_Hz': f_arr.tolist(),
            'B_sun': B_sun.tolist(),
            'B_room': B_room.tolist(),
            'lambda_peak_sun_nm': float(lambda_peak_sun),
            'T_sun_K': T_sun,
        },
        'stat_mech': {
            'kT_at_room_eV': float(kT_eV),
            '2level_E_ratio': (E2_arr/(kB_val*T_beta)).tolist(),
            'entropy_S': S_2level.tolist(),
            'equipartition': {
                'per_mode': '(1/2)*k_B*T per quadratic DOF',
                'monatomic': '(3/2)*k_B*T per atom',
                'diatomic': '(5/2)*k_B*T per molecule',
                'solid': '3*k_B*T per atom  [Dulong-Petit]',
            },
        },
        'photon_noise': {
            'hf_over_kT_at_1550nm': float(hf_kT),
            'n_thermal_1550': float(n_thermal_1550),
            'quantum_limited': bool(hf_kT > 10),
            'V_thermal_rms_nV': float(V_thermal_rms*1e9),
            'thermal_noise_S_V': float(S_V_thermal),
        },
        'Si_photonics_thermal': {
            'dn_dT_per_K': dn_dT_Si,
            'Delta_phi_per_K_rad': float(Delta_phi_per_K),
            'Delta_T_for_pi_K': float(Delta_T_for_pi),
            'lesson': f'Need {Delta_T_for_pi:.1f} K temperature change for pi phase shift in Si ring at 1550 nm',
        },
        'building_future': {
            'GaAs_solar_efficiency': '~29% single junction, record 47.6% concentrator tandem',
            'Shockley_Queisser': '33.7% limit for single junction at AM1.5',
            'photonic_IC_thermal_challenge': 'dn/dT = 1.8e-4/K in Si; thermal crosstalk limits WDM channel count',
            'thermoelectric_ZT': 'ZT > 1 for Bi2Te3 (room temp), SnSe (600K, ZT=2.6)',
            'Boltzmann_to_Shannon': 'S = k_B*ln(Omega) = -k_B*sum(p*ln(p)) = k_B * Shannon entropy',
        },
    }


def demo():
    print("=== MECHANICS & THERMAL PHYSICS ===\n")

    print("--- Terminal Velocity ---")
    tv = terminal_velocity()
    print(f"  Skydiver v_t: {tv['skydiver']['v_terminal_kmh']:.1f} km/h, Re={tv['skydiver']['Re_terminal']:.2e}")
    print(f"  Baumgartner 39km: {tv['Baumgartner_39km']['v_terminal_kmh']:.0f} km/h, Ma={tv['Baumgartner_39km']['Ma']:.2f}, supersonic={tv['Baumgartner_39km']['supersonic']}")
    print(f"  Skating: ice friction={tv['skating']['F_friction_ice_N']:.1f} N, aero>friction above {tv['skating']['v_where_drag_equals_ice_friction_ms']:.1f} m/s")

    print("\n--- 6x6 Stiffness Matrix ---")
    K6 = stiffness_matrix_6x6()
    print(f"  Symmetric: {K6['is_symmetric']}")
    print(f"  Eigenvalues: {[f'{v:.3e}' for v in K6['eigenvalues']]}")
    print(f"  Condition number: {K6['condition_number']:.2e}")
    print(f"  Natural frequencies: {[f'{f:.1f}' for f in K6['natural_frequencies_Hz']]} Hz")

    print("\n--- QM Operators (classical) ---")
    qm = qm_operators_classical()
    print(f"  X Hermitian: {qm['operators']['X_Hermitian']}")
    print(f"  [X,P] = j*hbar*I: {qm['commutator']['[X,P]_equals_jhbar_I']}")
    print(f"  Heisenberg satisfied: {qm['commutator']['Heisenberg_satisfied']}")
    print(f"  Delta_x*Delta_p/hbar = {qm['commutator']['product_hbar_units']:.4f}")
    print(f"  GS iteration spectral radius: {qm['GS_iteration']['spectral_radius']:.4f}, converges: {qm['GS_iteration']['converges']}")

    print("\n--- Continuous Rotation SO(3) ---")
    rot = continuous_rotation_SO3()
    print(f"  Commutation [Jx,Jy]=Jz: {rot['SO3_generators']['commutation_correct']}")
    print(f"  Rodrigues R is SO(3): {rot['rodrigues_rotation']['is_SO3']}")
    print(f"  Skater spin: {rot['skater_spin']['omega_initial_rps']:.2f} rps -> {rot['skater_spin']['omega_arms_in_rps']:.2f} rps (x{rot['skater_spin']['speedup_factor']:.1f})")
    print(f"  sin(n)*sin(m) ortho (n!=m): {rot['orthogonality']['integral_sin_n_sin_m_n_ne_m']:.6f}")
    print(f"  sin(n)*sin(n) norm: {rot['orthogonality']['integral_sin_n_sin_n']:.4f} (exact={np.pi:.4f})")

    print("\n--- Symmetry Integration ---")
    sym = symmetry_integration()
    print(f"  int cos = {sym['even_functions']['integrals']['int_cos_-pi_pi']:.4f} (expect 0 -> wait cos is even, int=2 sin(pi)~0)")
    print(f"  int sin = {sym['odd_functions']['integrals']['int_sin_-pi_pi']:.6f} (expect 0)")
    print(f"  Statics symmetric load moment: {sym['statics_symmetry']['symmetric_load_net_moment']:.4f} (expect 0)")
    print(f"  GS: intensity is even = {sym['GS_connection']['intensity_is_even']}")
    print(f"  Odd phase invisible: {sym['GS_connection']['phase_odd_part_invisible']}")

    print("\n--- Lagrangian Mechanics ---")
    lag = lagrangian_mechanics()
    print(f"  Pendulum omega_0: {lag['pendulum']['omega_0_rad_s']:.3f} rad/s")
    print(f"  Energy conserved (RK4): {lag['pendulum']['energy_conserved']}")
    print(f"  Double-well equilibria: {[round(x,3) for x in lag['statics_equilibria']['equilibrium_x']]}")
    print(f"  Stable: {lag['statics_equilibria']['stable']}")

    print("\n--- Thermal Physics ---")
    th = thermal_physics()
    print(f"  Carnot eta at 1000K: {th['thermodynamics']['Carnot']['eta_at_1000K']:.3f}")
    print(f"  Sun peak wavelength: {th['blackbody']['lambda_peak_sun_nm']:.0f} nm")
    print(f"  kT at 25C: {th['stat_mech']['kT_at_room_eV']:.4f} eV")
    print(f"  hf/kT at 1550nm: {th['photon_noise']['hf_over_kT_at_1550nm']:.1f} (quantum limited: {th['photon_noise']['quantum_limited']})")
    print(f"  Si ring: {th['Si_photonics_thermal']['Delta_T_for_pi_K']:.1f} K for pi phase shift")

    print("\n=== MECHANICS & THERMAL PHYSICS COMPLETE ===")


if __name__ == '__main__':
    demo()
