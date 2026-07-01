"""Quantum information science -- graduate EE level.

SYMPY DATA OBJECTS (the vocabulary before the physics):
  sp.Symbol('x')         -- a free variable; no value, just a name
  sp.Function('f')(x)    -- a symbolic function application
  sp.Matrix([[...]])     -- matrix of symbolic expressions
  sp.Expr                -- base class of all symbolic expressions
  sp.Rational(1,2)       -- exact fractions (not floating point)
  sp.I, sp.pi, sp.oo    -- imaginary unit, pi, infinity as exact objects
  sp.Eq(lhs, rhs)        -- a symbolic equation (not assignment)

MEASUREMENT TYPES in quantum mechanics:
  Projective measurement: {Pi} where sum Pi = I, Pi^2 = Pi (PVM)
  POVM (positive operator-valued measure): {M_i}, sum M_i^dag M_i = I
  Weak measurement: partial information, partial collapse
  Homodyne: measures quadrature X or P (optical, relevant to this repo)
  Heterodyne: measures BOTH X and P simultaneously (back-action tradeoff)

  In this repo: the GS algorithm is a PROJECTIVE MEASUREMENT on function
  space -- it projects the current estimate onto the constraint set
  (amplitude replacement IS the projector). Each GS iteration is a
  measurement that collapses the wave function of solutions.

QUBIT AND QUANTUM GATES:
  |0> = [1,0]^T,  |1> = [0,1]^T  (computational basis)
  Pauli matrices: X (NOT), Y (phase+flip), Z (phase)
  Hadamard H: |0> -> (|0>+|1>)/sqrt(2)  (superposition)
  CNOT: entangling gate, |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
  Phase gate S: diag(1, i),  T gate: diag(1, exp(i*pi/4))

BELL STATES (maximally entangled 2-qubit states):
  |Phi+> = (|00> + |11>) / sqrt(2)
  |Phi-> = (|00> - |11>) / sqrt(2)
  |Psi+> = (|01> + |10>) / sqrt(2)
  |Psi-> = (|01> - |10>) / sqrt(2)

TOPOLOGICAL QUANTUM COMPUTING:
  Majorana zero modes (MZMs): exotic quasiparticles at ends of a
  topological superconductor wire. Two MZMs gamma_1 and gamma_2
  form one non-local qubit: {gamma_1, gamma_2} = 2*delta_12.
  The qubit is DELOCALIZED -- local noise cannot flip it.
  This is topological protection: information is stored in the TOPOLOGY
  (winding number) of the ground state, not in a local degree of freedom.

  Anyon braiding: the operation of swapping two anyons around each other
  (braiding) applies a UNITARY GATE. For Ising anyons (which include
  Majorana modes): braiding two Majoranas applies exp(i*pi/4 * gamma_1*gamma_2).
  This is naturally fault-tolerant because braiding is a global operation.

  Physical platforms (2025 state of the art):
    Microsoft: InAs/Al heterostructure nanowires + topological qubits
    Kitaev chain model: p-wave superconductor chain with MZMs at ends
    Nu = 5/2 fractional quantum Hall state: non-Abelian anyons (theorized)

CONNECTION TO PHOTONICS (this repo):
  Topological photonics uses the same mathematical framework (Chern numbers,
  Berry phase) to create edge states in photonic crystals that are immune
  to backscattering. The Berry phase module in vector_calculus_torsion.py
  is the foundational calculation for BOTH topological insulators and
  topological quantum computing.
"""
import numpy as np
import sympy as sp


# ── SymPy data objects reference ─────────────────────────────────────

def sympy_data_objects_demo():
    """Demonstrate the basic SymPy data object types with physical examples."""
    x, t = sp.symbols('x t', real=True)
    hbar, m, omega = sp.symbols('hbar m omega', positive=True)

    # Symbol: a free variable
    psi = sp.Function('psi')(x, t)

    # Expr: any symbolic expression
    hamiltonian = -hbar**2 / (2*m) * sp.diff(psi, x, 2) + sp.Rational(1,2)*m*omega**2*x**2*psi

    # Matrix: quantum state as column vector
    ket_0 = sp.Matrix([1, 0])
    ket_1 = sp.Matrix([0, 1])

    # Pauli matrices (exact rational entries)
    sigma_x = sp.Matrix([[0, 1], [1, 0]])
    sigma_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sigma_z = sp.Matrix([[1, 0], [0, -1]])

    return {
        "Symbol_example": x,
        "Function_example": psi,
        "Hamiltonian_Expr": hamiltonian,
        "ket_0": ket_0, "ket_1": ket_1,
        "sigma_x": sigma_x, "sigma_y": sigma_y, "sigma_z": sigma_z,
        "Rational_half": sp.Rational(1, 2),
        "I_exact": sp.I,
        "pi_exact": sp.pi,
    }


# ── single qubit ──────────────────────────────────────────────────────

def qubit_state(theta, phi):
    """Single qubit pure state on the Bloch sphere.

    |psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>

    theta in [0, pi], phi in [0, 2*pi).
    Returns as numpy complex array [alpha, beta].
    """
    if not (0 <= theta <= np.pi):
        raise ValueError("theta must be in [0, pi]")
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([alpha, beta], dtype=complex)


def bloch_vector(psi):
    """Bloch vector (rx, ry, rz) for a single qubit state.

    r = <psi|sigma|psi> where sigma = (X, Y, Z)
    For a pure state: |r| = 1. Mixed state: |r| < 1.
    """
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    rx = 2 * np.real(psi[0].conj() * psi[1])
    ry = 2 * np.imag(psi[0].conj() * psi[1])
    rz = np.abs(psi[0])**2 - np.abs(psi[1])**2
    return np.array([rx, ry, rz])


# ── quantum gates ─────────────────────────────────────────────────────

GATE_I  = np.eye(2, dtype=complex)
GATE_X  = np.array([[0,1],[1,0]], dtype=complex)           # Pauli X (NOT)
GATE_Y  = np.array([[0,-1j],[1j,0]], dtype=complex)        # Pauli Y
GATE_Z  = np.array([[1,0],[0,-1]], dtype=complex)          # Pauli Z
GATE_H  = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)  # Hadamard
GATE_S  = np.array([[1,0],[0,1j]], dtype=complex)          # Phase (sqrt Z)
GATE_T  = np.array([[1,0],[0,np.exp(1j*np.pi/4)]], dtype=complex)  # T gate
GATE_CNOT = np.array([[1,0,0,0],[0,1,0,0],
                       [0,0,0,1],[0,0,1,0]], dtype=complex)  # 2-qubit CNOT


def apply_gate(gate, state):
    """Apply a unitary gate matrix to a quantum state vector."""
    state = np.asarray(state, dtype=complex)
    gate = np.asarray(gate, dtype=complex)
    if gate.shape[0] != len(state):
        raise ValueError(f"Gate shape {gate.shape} incompatible with state length {len(state)}")
    result = gate @ state
    return result / np.linalg.norm(result)


def commutator_pauli_sympy():
    """Verify [sigma_i, sigma_j] = 2*i*epsilon_ijk*sigma_k symbolically."""
    sx = sp.Matrix([[0,1],[1,0]])
    sy = sp.Matrix([[0,-sp.I],[sp.I,0]])
    sz = sp.Matrix([[1,0],[0,-1]])
    XY = sx*sy - sy*sx
    YZ = sy*sz - sz*sy
    ZX = sz*sx - sx*sz
    # Return the residual matrices -- all should be zero matrices
    return {
        "[X,Y] = 2i*Z": sp.simplify(XY - 2*sp.I*sz),
        "[Y,Z] = 2i*X": sp.simplify(YZ - 2*sp.I*sx),
        "[Z,X] = 2i*Y": sp.simplify(ZX - 2*sp.I*sy),
    }


# ── Bell states ───────────────────────────────────────────────────────

def bell_states():
    """The four maximally entangled 2-qubit Bell states.

    Created by Hadamard on qubit 1 followed by CNOT:
    |Phi+> = (|00> + |11>) / sqrt(2)
    |Phi-> = (|00> - |11>) / sqrt(2)
    |Psi+> = (|01> + |10>) / sqrt(2)
    |Psi-> = (|01> - |10>) / sqrt(2)
    """
    s = 1 / np.sqrt(2)
    return {
        "Phi_plus":  np.array([s, 0, 0, s], dtype=complex),   # |00> + |11>
        "Phi_minus": np.array([s, 0, 0, -s], dtype=complex),  # |00> - |11>
        "Psi_plus":  np.array([0, s, s, 0], dtype=complex),   # |01> + |10>
        "Psi_minus": np.array([0, s, -s, 0], dtype=complex),  # |01> - |10>
    }


def create_bell_state(which="Phi_plus"):
    """Create a Bell state from |00> using H then CNOT."""
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)
    # Apply H to qubit 1: H tensor I
    HI = np.kron(GATE_H, GATE_I)
    state = HI @ ket_00
    state = GATE_CNOT @ state
    # Phase corrections for the four Bell states
    if which == "Phi_plus":
        pass
    elif which == "Phi_minus":
        state = np.kron(GATE_Z, GATE_I) @ state
    elif which == "Psi_plus":
        state = np.kron(GATE_I, GATE_X) @ state
    elif which == "Psi_minus":
        state = np.kron(GATE_Z, GATE_X) @ state
    else:
        raise ValueError(f"Unknown Bell state '{which}'")
    return state / np.linalg.norm(state)


# ── density matrix and entanglement entropy ───────────────────────────

def density_matrix(psi):
    """Density matrix rho = |psi><psi| for a pure state."""
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    return psi @ psi.conj().T


def partial_trace(rho, keep=0, dims=(2, 2)):
    """Partial trace of a 2-qubit density matrix over one subsystem.

    keep=0: keep qubit A (trace out B)
    keep=1: keep qubit B (trace out A)

    Returns the reduced density matrix of the kept qubit.
    """
    d0, d1 = dims
    rho4 = rho.reshape(d0, d1, d0, d1)
    if keep == 0:
        # rho_A[i,j] = sum_a rho[i,a,j,a]
        return np.einsum('iaja->ij', rho4)
    else:
        # rho_B[i,j] = sum_a rho[a,i,a,j]
        return np.einsum('aiaj->ij', rho4)


def von_neumann_entropy(rho):
    """Von Neumann entropy S = -Tr(rho * log2(rho)).

    For a pure state: S = 0.
    For a maximally mixed state: S = log2(d) (maximum).
    For a maximally entangled Bell state: reduced rho is I/2, S = 1 ebit.
    """
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]   # remove numerical zeros
    S = -np.sum(eigvals * np.log2(eigvals))
    return float(S)


def entanglement_entropy(psi_2qubit):
    """Entanglement entropy of a 2-qubit pure state (in ebits).

    E(psi) = S(rho_A) = S(rho_B) where rho_A = Tr_B(|psi><psi|)
    Range: 0 (product state) to 1 (maximally entangled Bell state).
    """
    rho = density_matrix(psi_2qubit)
    rho_A = partial_trace(rho, keep=0)
    return von_neumann_entropy(rho_A)


def is_entangled(psi_2qubit, threshold=1e-6):
    """Check if a 2-qubit state is entangled (E > 0)."""
    E = entanglement_entropy(psi_2qubit)
    return E > threshold


# ── quantum teleportation ─────────────────────────────────────────────

def quantum_teleportation_circuit(psi_to_send):
    """Simulate the 3-step quantum teleportation protocol.

    Alice wants to send qubit state |psi> = alpha|0> + beta|1> to Bob.
    They share a Bell pair |Phi+> = (|00> + |11>)/sqrt(2).

    Step 1: Alice applies CNOT(her_qubit -> her_bell_qubit)
    Step 2: Alice applies H to her qubit
    Step 3: Alice measures (2 classical bits), sends to Bob
    Step 4: Bob applies corrections based on classical bits
    Bob's qubit is now in state |psi> (teleported).

    Returns: Bob's final state (should match psi_to_send up to global phase).
    """
    psi = np.asarray(psi_to_send, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    alpha, beta = psi[0], psi[1]

    # Initial 3-qubit state: |psi>_A tensor |Phi+>_BC
    bell = bell_states()["Phi_plus"]
    state = np.kron(psi, bell)   # [alpha,beta] x [1,0,0,1]/sqrt(2) -- 8 components

    # Step 1: CNOT on qubits A,B (control=A qubit 0, target=B qubit 1)
    # CNOT_{01} in 3-qubit space: acts on qubits 0,1 (leave qubit 2)
    CNOT_AB = np.kron(GATE_CNOT, GATE_I)
    state = CNOT_AB @ state

    # Step 2: Hadamard on qubit A
    H_A = np.kron(GATE_H, np.eye(4))
    state = H_A @ state

    # Step 3: Measure qubits A, B (collapse to one of 4 outcomes)
    # Reshape to [qubit_A, qubit_B, qubit_C]
    outcomes = {}
    state_cube = state.reshape(2, 2, 2)
    for ma in range(2):
        for mb in range(2):
            bob_unnorm = state_cube[ma, mb, :]
            prob = np.sum(np.abs(bob_unnorm)**2)
            if prob > 1e-12:
                bob = bob_unnorm / np.sqrt(prob)
                # Step 4: Bob's corrections
                if mb == 1:
                    bob = GATE_X @ bob
                if ma == 1:
                    bob = GATE_Z @ bob
                outcomes[(ma, mb)] = {"bob_state": bob, "prob": prob}

    # All outcomes give the same Bob state = psi (up to global phase)
    return outcomes


# ── topological quantum computing ─────────────────────────────────────

def kitaev_chain_hamiltonian_sympy(N=4):
    """Kitaev chain Hamiltonian in SymPy for a 1D p-wave superconductor.

    H = -mu * sum_j c_j^dag c_j
        - t * sum_j (c_j^dag c_{j+1} + h.c.)
        + Delta * sum_j (c_j c_{j+1} + h.c.)

    At the topological phase (|mu| < 2t, Delta != 0): Majorana zero modes
    appear at the ends of the chain.
    Phase boundary: |mu| = 2|t|

    Returns the Hamiltonian as a string of symbolic terms (not a full matrix
    -- that requires 2^N dimensions).
    """
    mu, t, Delta = sp.symbols('mu t Delta', real=True)
    c = [sp.Symbol(f'c_{j}') for j in range(N)]
    cdag = [sp.Symbol(f'c_dag_{j}') for j in range(N)]

    H_mu = -mu * sum(cdag[j]*c[j] for j in range(N))
    H_t  = -t  * sum(cdag[j]*c[j+1] + cdag[j+1]*c[j] for j in range(N-1))
    H_D  = Delta * sum(c[j]*c[j+1] + cdag[j+1]*cdag[j] for j in range(N-1))
    H = H_mu + H_t + H_D

    return {
        "H": H,
        "topological_phase_condition":
            sp.Eq(sp.Abs(mu), 2*sp.Abs(t)),
        "MZM_gamma1":
            sp.Eq(sp.Symbol('gamma_1'), c[0] + cdag[0]),
        "MZM_gamma2":
            sp.Eq(sp.Symbol('gamma_2'), sp.I*(cdag[0] - c[0])),
        "Majorana_anticommutator":
            "{gamma_i, gamma_j} = 2 * delta_ij  (Clifford algebra)",
        "Topological_protection":
            "Qubit stored in non-local parity of two MZMs -- immune to local perturbations",
    }


def chern_number_2d(k_x_arr, k_y_arr, H_func):
    """Numerical Chern number (topological invariant) for a 2D band.

    C = (1/2*pi) * integral_{BZ} F_xy dk_x dk_y
    where F_xy = d_x A_y - d_y A_x is the Berry curvature.
    A_mu = -i <u_k|d/dk_mu|u_k> is the Berry connection.

    A non-zero Chern number C=1 indicates a TOPOLOGICAL PHASE with
    edge states immune to smooth deformations.

    H_func(kx, ky) should return a 2x2 Hermitian matrix (for a 2-band model).
    """
    n_x, n_y = len(k_x_arr), len(k_y_arr)
    dkx = k_x_arr[1] - k_x_arr[0]
    dky = k_y_arr[1] - k_y_arr[0]

    def ground_state(kx, ky):
        H = H_func(kx, ky)
        _, vecs = np.linalg.eigh(H)
        return vecs[:, 0]   # lowest energy eigenstate

    # Compute Berry curvature on a discrete grid using link variables
    F_total = 0.0
    for ix in range(n_x - 1):
        for iy in range(n_y - 1):
            kx, ky = k_x_arr[ix], k_y_arr[iy]
            u00 = ground_state(kx,      ky)
            u10 = ground_state(kx+dkx,  ky)
            u01 = ground_state(kx,      ky+dky)
            u11 = ground_state(kx+dkx,  ky+dky)
            # Fukui method: arg of the PRODUCT of link variables, maps to (-pi, pi]
            # This gives exact integer Chern numbers on coarse grids
            F_plaq = np.angle(
                np.vdot(u00, u10) *
                np.vdot(u10, u11) *
                np.vdot(u11, u01) *
                np.vdot(u01, u00)
            )
            F_total += F_plaq

    chern = F_total / (2 * np.pi)
    return {"chern_number": round(chern), "chern_raw": chern}


# ── QI SymPy formalism ────────────────────────────────────────────────

def quantum_information_sympy_5():
    """Five key QI equations in SymPy."""
    alpha, beta = sp.symbols('alpha beta', complex=True)
    theta, phi_s = sp.symbols('theta phi', real=True, positive=True)
    rho_s = sp.Symbol('rho')
    S_s = sp.Symbol('S')
    gamma1, gamma2 = sp.symbols('gamma_1 gamma_2')

    return {
        "Qubit_state":
            sp.Eq(sp.Symbol('|psi>'),
                  sp.cos(theta/2)*sp.Symbol('|0>') +
                  sp.exp(sp.I*phi_s)*sp.sin(theta/2)*sp.Symbol('|1>')),
        "Von_Neumann_entropy":
            sp.Eq(S_s, -sp.Symbol('Tr(rho*log2(rho))')),
        "Bell_Phi_plus":
            sp.Eq(sp.Symbol('|Phi+>'),
                  (sp.Symbol('|00>') + sp.Symbol('|11>')) / sp.sqrt(2)),
        "Majorana_anticommutator":
            sp.Eq(gamma1*gamma2 + gamma2*gamma1, 2*sp.Symbol('delta_12')),
        "Chern_number":
            sp.Eq(sp.Symbol('C'),
                  sp.Symbol('1/(2*pi)') *
                  sp.Symbol('integral_BZ_F_xy_dk')),
    }


if __name__ == "__main__":
    print("=== SymPy data objects ===")
    demo = sympy_data_objects_demo()
    print(f"  sigma_x:\n{demo['sigma_x']}")
    print(f"  pi exact: {demo['pi_exact']}")
    print(f"  Rational(1,2): {demo['Rational_half']}")

    print("\n=== Pauli commutators [X,Y]=2iZ ===")
    comms = commutator_pauli_sympy()
    for k, eq in comms.items():
        ok = eq.rhs == sp.zeros(2)
        print(f"  {k}: {ok}")

    print("\n=== Bell states and entanglement entropy ===")
    bs = bell_states()
    for name, state in bs.items():
        E = entanglement_entropy(state)
        print(f"  {name}: E = {E:.4f} ebit  (entangled: {is_entangled(state)})")

    print("\n=== Quantum teleportation ===")
    psi_alice = qubit_state(np.pi/3, np.pi/4)
    outcomes = quantum_teleportation_circuit(psi_alice)
    for (ma, mb), info in outcomes.items():
        fidelity = abs(np.vdot(psi_alice, info["bob_state"]))**2
        print(f"  outcome ({ma},{mb}): P={info['prob']:.3f}, "
              f"fidelity with original={fidelity:.4f}")

    print("\n=== Chern number: Haldane-like 2-band model ===")
    def haldane_H(kx, ky, m=0.5, t1=1.0):
        hx = np.sin(kx)
        hy = np.sin(ky)
        hz = m - t1*(np.cos(kx) + np.cos(ky))
        return np.array([[hz, hx-1j*hy],[hx+1j*hy, -hz]])

    kx = np.linspace(-np.pi, np.pi, 30)
    ky = np.linspace(-np.pi, np.pi, 30)
    C = chern_number_2d(kx, ky, haldane_H)
    print(f"  Chern number (m=0.5): C = {C['chern_number']}  (topological if C=+-1)")

    C_trivial = chern_number_2d(kx, ky, lambda kx,ky: haldane_H(kx,ky,m=3.0))
    print(f"  Chern number (m=3.0): C = {C_trivial['chern_number']} (trivial)")

    print("\n=== Kitaev chain Majorana modes ===")
    kit = kitaev_chain_hamiltonian_sympy(4)
    print(f"  Phase boundary: {kit['topological_phase_condition']}")
    print(f"  {kit['Topological_protection']}")

    print("\n=== SymPy 5 ===")
    for k, eq in quantum_information_sympy_5().items():
        print(f"  {k}: {eq}")
