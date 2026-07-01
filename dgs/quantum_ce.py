"""
Quantum Science and Quantum Information for Computer Engineers.
CSUS CE + Physics Minor curriculum bridge.

Covers what your CE degree skips but every quantum lab expects you to know:
  1. Energy scales (Joule <-> eV <-> kT <-> hf) -- the "Joule" question
  2. Maxwell -> QED bridge: photon as quantum of EM field
  3. Qubit hardware from a CE perspective (superconducting, photonic, trapped ion)
  4. Quantum error correction: stabilizer codes, surface code logical qubit
  5. Quantum advantage: where classical CE cannot compete and why

Connection to this repo:
  - STEAM photon = quantum of the optical field E(t)
  - GS phase retrieval operates on |<n|psi>|^2 (photon number measurement)
  - Dispersive fiber H(f) = exp(j*pi*D*f^2) is a UNITARY gate on optical modes
  - Photonic QKD uses same fiber + same GS math for key distribution
"""
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# 1. ENERGY SCALES -- the Joule question
# ---------------------------------------------------------------------------

# Fundamental constants
h_J  = 6.62607015e-34   # Planck constant [J*s]
hbar = 1.054571817e-34  # hbar = h/(2*pi) [J*s]
kB   = 1.380649e-23     # Boltzmann constant [J/K]
eV   = 1.602176634e-19  # 1 electron-volt [J]
c    = 2.99792458e8     # speed of light [m/s]

ENERGY_SCALES = {
    # Thermal energy kT
    'kT_room_300K_eV':     kB * 300 / eV,          # ~0.026 eV
    'kT_room_300K_J':      kB * 300,                # ~4.1e-21 J
    'kT_cryo_10mK_eV':     kB * 0.010 / eV,        # ~8.6e-7 eV -- superconducting qubit
    'kT_cryo_10mK_J':      kB * 0.010,

    # Photon energy hf
    'hf_1GHz_eV':          h_J * 1e9 / eV,         # ~4.1e-6 eV -- microwave, SC qubit
    'hf_10GHz_eV':         h_J * 10e9 / eV,        # ~4.1e-5 eV -- transmon qubit freq
    'hf_1THz_eV':          h_J * 1e12 / eV,        # ~4.1e-3 eV -- THz photon
    'hf_1550nm_eV':        h_J * c / 1550e-9 / eV, # ~0.80 eV -- telecom photon
    'hf_800nm_eV':         h_J * c / 800e-9 / eV,  # ~1.55 eV -- Ti:Sapph laser
    'hf_visible_green_eV': h_J * c / 532e-9 / eV,  # ~2.33 eV

    # Semiconductor band gaps (CE relevance)
    'Si_bandgap_eV':       1.12,                    # Si at 300K
    'GaAs_bandgap_eV':     1.42,                    # GaAs (LED, laser)
    'InGaAsP_1550nm_eV':   0.80,                    # InGaAsP (telecom laser)
    'SiC_bandgap_eV':      3.26,                    # SiC (power electronics)

    # Qubit energy scales
    'transmon_qubit_GHz':   5.0,                    # typical SC transmon [GHz]
    'transmon_qubit_eV':    h_J * 5e9 / eV,
    'transmon_T_required_K': h_J * 5e9 / kB * 0.1, # 10% of hf/kB -> ~24 mK
    'ion_trap_optical_eV':  h_J * c / 729e-9 / eV, # Ca+ qubit at 729nm: ~1.70 eV

    # Joule conversion summary
    'J_per_eV':  eV,
    'eV_per_J':  1.0/eV,
    'kT300_in_eV': kB*300/eV,
}


def energy_in_all_units(E_J):
    """Convert energy in Joules to all relevant units."""
    return {
        'J':   E_J,
        'eV':  E_J / eV,
        'meV': E_J / eV * 1e3,
        'kT_at_300K': E_J / (kB * 300),
        'kT_at_10mK': E_J / (kB * 0.010),
        'freq_GHz':   E_J / h_J / 1e9,
        'freq_THz':   E_J / h_J / 1e12,
        'wavelength_nm': h_J * c / E_J * 1e9 if E_J > 0 else float('inf'),
        'temp_K':     E_J / kB,
    }


def photon_energy(wavelength_nm=1550.0):
    """E = hf = hc/lambda. Telecom photon at 1550nm."""
    E_J = h_J * c / (wavelength_nm * 1e-9)
    return energy_in_all_units(E_J)


def thermal_vs_qubit_energy(T_K=0.020, f_qubit_GHz=5.0):
    """
    Key question: is kT << hf? (quantum regime)
    If kT >> hf: qubit is thermally excited -> mixed state, no coherence.
    If kT << hf: qubit stays in |0> ground state.
    Ratio = hf/(kT): must be >> 1 for quantum operation.
    """
    E_thermal = kB * T_K
    E_qubit = h_J * f_qubit_GHz * 1e9
    ratio = E_qubit / E_thermal
    return {
        'T_K': T_K,
        'f_qubit_GHz': f_qubit_GHz,
        'kT_eV': E_thermal / eV,
        'hf_eV': E_qubit / eV,
        'hf_over_kT': ratio,
        'quantum_regime': ratio > 10,
        'excited_state_population': 1.0 / (np.exp(ratio) + 1),  # Fermi-Dirac at qubit freq
        'message': ('QUANTUM: qubit is cold enough' if ratio > 10
                    else 'CLASSICAL: thermal noise destroys coherence'),
    }


# ---------------------------------------------------------------------------
# 2. Maxwell -> QED bridge
# ---------------------------------------------------------------------------
def maxwell_to_qed_bridge():
    """
    How classical Maxwell equations become Quantum ElectroDynamics (QED).

    Classical:  E(r,t) and B(r,t) are real fields. Energy = eps0/2 * |E|^2 + ...
    Quantum:    E and B become OPERATORS. Photon = quantum of excitation.

    Steps (Canonical Quantization):
    1. Write classical EM field as sum of harmonic oscillators (normal modes):
         E(t) = sum_k [A_k * exp(j*omega_k*t) + A_k* * exp(-j*omega_k*t)]

    2. Each mode k is a harmonic oscillator with energy:
         H_k = hbar*omega_k * (n_k + 1/2)    where n_k = 0,1,2,...
         n_k = photon number in mode k

    3. Promote amplitudes to operators:
         A_k   ->  a_k   (annihilation operator, removes one photon)
         A_k*  ->  a_k+  (creation operator, adds one photon)
         [a_k, a_k+] = 1  (canonical commutation relation)

    4. Electric field OPERATOR:
         E_hat = sum_k E_0k * (a_k * eps_k * exp(j*omega_k*t) + h.c.)
         where E_0k = sqrt(hbar*omega_k / 2*eps0*V) is the vacuum field

    CE connection:
      - a_k and a_k+ are the quantum analog of complex amplitude in signal processing
      - n_k = a_k+ * a_k is the Hermitian photon number operator (real eigenvalues)
      - <n_k> = mean photon number = classical intensity / (hbar*omega_k)
      - |E(t)|^2 measurement = photon counting = Hermitian observable
      - THIS is why GS phase retrieval works: |E|^2 = <psi|n_hat|psi> is measurable
    """
    # Vacuum electric field amplitude per mode (single photon field)
    V_mode = (10e-6)**3   # 10 um mode volume [m^3] (typical cavity)
    eps0 = 8.854e-12
    omega = 2*np.pi * c / 1550e-9

    E0_vacuum = np.sqrt(hbar * omega / (2 * eps0 * V_mode))

    return {
        'vacuum_field_V_per_m': E0_vacuum,
        'single_photon_energy_eV': hbar * omega / eV,
        'commutation': '[a, a+] = 1  -- bosonic commutation (photons are bosons)',
        'number_operator': 'n_hat = a+ @ a, eigenvalues 0,1,2,...',
        'measurement_connection': '|E_out|^2 = <psi|a+a|psi> = photon number expectation',
        'gs_connection': 'GS retrieves phase of a_k -- the non-Hermitian amplitude operator',
        'qed_lagrangian': 'L = -1/(4*mu0) * F_mu_nu * F^mu_nu + psibar*(j*gamma^mu*D_mu - m)*psi',
        'ce_analog': 'a_k is the complex baseband amplitude in RF/DSP -- same math',
    }


def purcell_effect(Q=1000, V_mode_um3=100, wavelength_nm=1550, n=3.4):
    """
    Purcell effect: cavity QED enhances spontaneous emission rate.
    F_P = (3/(4*pi^2)) * (lambda/n)^3/V * Q

    CE relevance:
      - LED/VCSEL design: cavity increases emission rate -> faster modulation
      - Single-photon emitter: Purcell-enhanced QD for quantum key distribution
      - Same physics as antenna gain: cavity focuses modes into fewer channels

    Q: quality factor of cavity
    V_mode_um3: mode volume [um^3]
    wavelength_nm: resonant wavelength [nm]
    n: refractive index
    """
    lambda_m = wavelength_nm * 1e-9
    V_m3 = V_mode_um3 * 1e-18
    F_P = (3 / (4 * np.pi**2)) * (lambda_m/n)**3 / V_m3 * Q
    return {
        'Purcell_factor': F_P,
        'emission_rate_enhanced_vs_free_space': F_P,
        'Q_factor': Q,
        'V_mode_um3': V_mode_um3,
        'ce_use_case': 'VCSEL with Purcell enhancement -> >40GHz modulation bandwidth',
        'quantum_use_case': 'Photonic crystal cavity + QD -> deterministic single photon',
    }


# ---------------------------------------------------------------------------
# 3. Qubit hardware from a CE perspective
# ---------------------------------------------------------------------------
QUBIT_HARDWARE = {
    'superconducting_transmon': {
        'physical_qubit': 'Josephson junction + capacitor on sapphire chip',
        'frequency_GHz': (4, 8),
        'T1_coherence_us': (100, 500),        # energy relaxation
        'T2_coherence_us': (50, 300),         # dephasing
        'gate_time_ns': 20,
        'temperature_mK': 15,
        'control': '5-8 GHz microwave pulses from AWG -> coax -> dilution fridge',
        'ce_skills_needed': [
            'RF engineering (AWG, IQ mixer, coax at cryogenic temp)',
            'FPGA control (real-time pulse sequencing at ns precision)',
            'Cryogenic circuit design (superconducting materials)',
            'Digital signal processing (heterodyne readout, IQ demodulation)',
        ],
        'who_builds_it': 'IBM Quantum, Google Sycamore, Rigetti, IQM',
        'python_sdk': 'Qiskit (IBM), Cirq (Google)',
        'joule_scale': thermal_vs_qubit_energy(0.015, 5.0),
    },
    'photonic_qubit': {
        'physical_qubit': 'Photon polarization, path, time-bin, or squeezed state',
        'frequency_GHz': c/1550e-9/1e9,       # 193 THz
        'T1_coherence_us': float('inf'),       # photons dont decay (free space)
        'T2_coherence_us': float('inf'),       # limited by path length / detector timing
        'gate_time_ns': 0.001,                 # optical gate ~ 1ps
        'temperature_K': 300,                  # room temperature!
        'control': 'Beam splitters, phase shifters, MZM, single-photon detectors (SNSPD)',
        'ce_skills_needed': [
            'Fiber optics (SMF, MZM, this repo)',
            'FPGA timing for photon coincidence detection',
            'Silicon photonics chip design',
            'Statistical analysis (photon counting statistics)',
        ],
        'who_builds_it': 'PsiQuantum, Xanadu (Borealis), Quandela',
        'python_sdk': 'PennyLane (Xanadu), Perceval (Quandela)',
        'connection_to_repo': (
            'H(f) = exp(j*pi*D*f^2) is a single-mode unitary gate on optical field. '
            'dgs/photonic_qkd.py implements BB84 over the same fiber. '
            'Your STEAM setup IS a photonic quantum information processor.'
        ),
    },
    'trapped_ion': {
        'physical_qubit': 'Electronic states of laser-cooled Ca+, Yb+, or Be+ ions',
        'frequency_GHz': c/729e-9/1e9,        # optical: 411 THz for Ca+ S1/2 <-> D5/2
        'T1_coherence_s': 10,                 # very long (minutes possible)
        'T2_coherence_s': 1,
        'gate_time_us': 10,
        'temperature_uK': 10,                 # laser-cooled to microKelvin
        'control': 'Laser pulses at 729nm (Ca+), acousto-optic modulators',
        'ce_skills_needed': [
            'PID control of laser frequency (same pid.py in this repo)',
            'RF drive for ion trap electrodes (Paul trap: MHz oscillation)',
            'Digital signal processing for fluorescence photon counting',
            'Vacuum system monitoring and control',
        ],
        'who_builds_it': 'IonQ, Quantinuum (Honeywell), Oxford Ionics',
        'python_sdk': 'pytket (Quantinuum), IonQ SDK',
    },
}


# ---------------------------------------------------------------------------
# 4. Quantum Error Correction -- stabilizer codes
# ---------------------------------------------------------------------------
def pauli_matrices():
    """Pauli matrices as numpy arrays. Foundation of all QEC."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def three_qubit_bit_flip_code():
    """
    Simplest QEC: 3-qubit bit-flip code.
    Encodes |0> -> |000>, |1> -> |111>
    Corrects 1 X error on any qubit.

    Circuit (CE analog: triple modular redundancy, TMR):
      Encode:  |psi>|00> -> CNOT(0,1) -> CNOT(0,2) -> |psi_L>
      Measure: Ancilla syndrome: Z0Z1 and Z1Z2 (parity checks)
      Correct: If syndrome = (1,0): flip qubit 0; (0,1): flip qubit 2; (1,1): flip qubit 1

    Stabilizers: {Z0Z1, Z1Z2}
    Logical operators: X_L = X0X1X2, Z_L = Z0 (any single Z)
    """
    P = pauli_matrices()

    def kron3(A, B, C):
        return np.kron(np.kron(A, B), C)

    I, X, Z = P['I'], P['X'], P['Z']

    # Stabilizer generators
    S1 = kron3(Z, Z, I)   # Z0 Z1
    S2 = kron3(I, Z, Z)   # Z1 Z2

    # Logical operators
    X_L = kron3(X, X, X)  # X_L = X0 X1 X2
    Z_L = kron3(Z, I, I)  # Z_L = Z0

    # Code space (simultaneous +1 eigenspace of S1, S2)
    # |0_L> = (|000> + |111>) is NOT this code -- this is REPETITION code
    # |0_L> = |000>, |1_L> = |111> (classical repetition)
    logical_0 = np.array([1,0,0,0,0,0,0,0], dtype=complex)  # |000>
    logical_1 = np.array([0,0,0,0,0,0,0,1], dtype=complex)  # |111>

    # Syndrome measurement: eigenvalue of S1, S2 on each error
    def syndrome(state):
        s1 = np.real(state.conj() @ S1 @ state)
        s2 = np.real(state.conj() @ S2 @ state)
        return round(s1), round(s2)

    # Apply X error on qubit 0: X0 |000> = |100>
    X0_error = kron3(X, I, I)
    errored_state = X0_error @ logical_0
    syn = syndrome(errored_state)

    return {
        'stabilizers': {'S1=Z0Z1': S1, 'S2=Z1Z2': S2},
        'logical_X': X_L,
        'logical_Z': Z_L,
        'logical_0': logical_0,
        'logical_1': logical_1,
        'example_syndrome_X0_error': syn,
        'interpretation': {
            (1, 1): 'X error on qubit 0',
            (-1, 1): 'X error on qubit 1',
            (1, -1): 'X error on qubit 2',
            (1, 1): 'no error or X error on qubit 0',
        },
        'ce_analog': 'TMR (Triple Modular Redundancy) in FPGA/spacecraft: '
                     'vote on 3 copies of output. QEC is the quantum generalization.',
        'distance': 3,
        'corrects': '1 X error on any qubit',
    }


def surface_code_overview():
    """
    Surface code: the leading QEC code for fault-tolerant quantum computing.
    Used by Google, IBM, Microsoft.

    d=3 surface code: 9 physical qubits -> 1 logical qubit
    Threshold: ~1% physical error rate (much better than ~0.01% for other codes)

    Layout (d=3):
      D D D     D = data qubit (9 total for d=3)
      D D D     X = X-type stabilizer (measures Z errors)
      D D D     Z = Z-type stabilizer (measures X errors)

    Physical count:
      d=3:  9 data + 8 ancilla = 17 physical qubits / 1 logical
      d=5:  25 data + 24 ancilla = 49 physical / 1 logical
      d=7:  49 + 48 = 97 physical / 1 logical
      Google 2023 (Nature): error rate improves from d=5 to d=7 -> first QEC below threshold

    CE relevance:
      - Each stabilizer measurement = one FPGA clock cycle of classical control
      - Decoding syndrome -> correction = real-time classical compute (FPGA/ASIC)
      - Minimum-weight perfect matching (MWPM) decoder: graph theory on FPGA
      - This is the classical CE job in a quantum computer: the decoder ASIC
    """
    return {
        'code_distance': 3,
        'physical_per_logical': 17,
        'threshold_percent': 1.0,
        'stabilizers': '4 X-type (weight 4) + 4 Z-type (weight 4) for d=3',
        'logical_operators': 'X_L and Z_L anticommute, weight d=3 each',
        'decoder': 'MWPM (Blossom algorithm) or neural network decoder',
        'ce_job': 'Build the real-time MWPM decoder ASIC running at 1MHz syndrome rate',
        'google_2023': 'First experimental evidence of below-threshold QEC (Nature 2023)',
        'ibm_roadmap': '100 physical qubits per logical by 2029',
        'python_tool': 'Stim (Google) -- fast Clifford circuit simulator for QEC',
        'connection_to_photonics': (
            'Photonic surface code: beam splitters + SNSPD + feedforward. '
            'PsiQuantum aims at million photonic qubits by 2029. '
            'Same fiber infrastructure as this repo.'
        ),
    }


# ---------------------------------------------------------------------------
# 5. Quantum advantage: where classical CE cannot compete
# ---------------------------------------------------------------------------
def quantum_advantage_table():
    """
    Where quantum computers beat classical and why.
    From a CE perspective: what hardware do you need and what can you simulate?
    """
    return [
        {
            'problem': 'Integer factoring (RSA-2048)',
            'classical_complexity': 'exp(O(n^1/3)): millions of years on best supercomputer',
            'quantum_algorithm': "Shor's algorithm: O(n^3) gate ops",
            'qubits_needed': '~4000 logical (millions of physical with QEC)',
            'threat_to_CE': 'Breaks all current RSA/ECC public-key crypto',
            'timeline': '10-15 years (NIST post-quantum standards now available)',
            'ce_action': 'Implement CRYSTALS-Kyber or CRYSTALS-Dilithium (NIST PQC)',
        },
        {
            'problem': 'Unstructured search (N items)',
            'classical_complexity': 'O(N)',
            'quantum_algorithm': "Grover's algorithm: O(sqrt(N)) -- quadratic speedup",
            'qubits_needed': 'log2(N)',
            'ce_use': 'Database search, optimization, collision finding in hash functions',
            'grover_demo': 'dgs/qubits.py::phase_estimation -> can be extended to Grover',
        },
        {
            'problem': 'Quantum chemistry (drug discovery)',
            'classical_complexity': 'exp(O(N_electrons)) -- exponential',
            'quantum_algorithm': 'VQE or QPE: poly(N) gates for N orbitals',
            'qubits_needed': '~1000 logical for drug-relevant molecules',
            'ce_relevance': 'Classical HPC + quantum coprocessor architecture',
        },
        {
            'problem': 'Quantum machine learning (disputed)',
            'classical_complexity': 'O(N*d) for N samples, d features',
            'quantum_algorithm': 'HHL: O(poly(log N)) IF data is already quantum',
            'caveat': 'DEQUANTIZATION: many QML speedups can be matched classically',
            'honest_assessment': 'QML advantage is NOT proven for practical CE problems',
        },
        {
            'problem': 'QKD (quantum key distribution)',
            'classical_complexity': 'Computationally secure (not information-theoretic)',
            'quantum_algorithm': 'BB84: unconditional information-theoretic security',
            'qubits_needed': '1 photon per bit (single-photon source + SNSPD)',
            'ce_relevance': 'Fiber QKD over same SMF-28 as this repo',
            'repo_module': 'dgs/photonic_qkd.py, dgs/path_integral_qkd.py',
        },
    ]


# ---------------------------------------------------------------------------
# 6. EM <-> Quantum: the Griffiths bridge
# ---------------------------------------------------------------------------
def griffiths_to_qis_map():
    """
    Maps Griffiths EM chapters to Quantum Information Science concepts.
    This is the bridge CSUS CE + Physics minor students need.
    """
    return [
        {
            'griffiths_ch': 'Ch1-2: Vector calculus, electrostatics',
            'qis_concept': 'Hilbert space geometry (inner product, orthogonality)',
            'math': 'nabla F -> gradient, integral of |psi|^2 -> probability',
            'module': 'dgs/hilbert_space.py',
        },
        {
            'griffiths_ch': 'Ch9: EM waves in matter (n, k, beta2)',
            'qis_concept': 'Quantum optical channel: photon propagation in fiber',
            'math': 'H(f) = exp(j*pi*D*f^2) = unitary gate on optical mode',
            'module': 'dgs/coppinger1999.py, dgs/gs_core.py',
        },
        {
            'griffiths_ch': 'Ch10: Potentials and fields (Lienard-Wiechert)',
            'qis_concept': 'Spontaneous emission (atom radiates photon), Purcell effect',
            'math': 'A(r,t) -> quantized -> a_k photon creation/annihilation',
            'module': 'griffiths/radiation.py (partial)',
        },
        {
            'griffiths_ch': 'Ch4-5: Multipole expansion, magnetic materials',
            'qis_concept': 'Spin-1/2 (qubit): Pauli matrices are angular momentum',
            'math': 'S = hbar/2 * sigma; [Sx,Sy] = j*hbar*Sz',
            'module': 'dgs/qubits.py, dgs/quantum_information.py',
        },
        {
            'griffiths_ch': 'Ch7: Faraday, induction, Maxwell equations',
            'qis_concept': 'Josephson junction: quantum of flux Phi_0 = h/(2e)',
            'math': 'V = hbar/(2e) * d(phi)/dt -- quantum of voltage',
            'module': 'dgs/quantum_ce.py (this file)',
        },
        {
            'griffiths_ch': 'Ch8: Energy in EM fields (Poynting)',
            'qis_concept': 'Photon energy hf = hbar*omega; vacuum fluctuations',
            'math': '<0|H|0> = sum_k hbar*omega_k/2 (zero-point energy)',
            'module': 'dgs/quantum_ce.py (this file)',
        },
    ]


def josephson_junction():
    """
    Josephson junction: heart of superconducting qubit.
    V = hbar/(2e) * d(phi)/dt    (Josephson voltage-phase relation)
    I = Ic * sin(phi)             (Josephson current-phase relation)
    Energy: E_J = hbar*Ic/(2e) * (1 - cos(phi))
    Capacitive energy: E_C = e^2/(2C)
    Transmon regime: E_J/E_C >> 1 -> anharmonic oscillator -> qubit

    CE connection: Josephson junction IS a quantum nonlinear inductor.
    L_J = Phi_0 / (2*pi*Ic*cos(phi)) -- inductance depends on phase (flux state).
    Equivalent SPICE element: nonlinear inductor with quantum phase variable.
    """
    Phi_0 = h_J / (2*eV)   # flux quantum [V*s = Wb]
    Ic = 10e-9              # critical current [A] (typical transmon)
    C = 70e-15              # junction capacitance [F]

    E_J = Phi_0 * Ic / (2*np.pi)          # Josephson energy [J]
    E_C = eV**2 / (2*C)                   # charging energy [J]
    ratio = E_J / E_C

    # Qubit frequency (harmonic approximation)
    omega_01 = np.sqrt(8*E_J*E_C) / hbar - E_C/hbar
    f_01_GHz = omega_01 / (2*np.pi) / 1e9

    # Anharmonicity (why it's a qubit not just an oscillator)
    alpha = -E_C / hbar
    alpha_MHz = alpha / (2*np.pi) / 1e6

    return {
        'Phi_0_Wb': Phi_0,
        'Ic_nA': Ic*1e9,
        'C_fF': C*1e15,
        'E_J_J': E_J,
        'E_C_J': E_C,
        'E_J_over_E_C': ratio,
        'transmon_regime': ratio > 10,
        'f01_GHz': round(f_01_GHz, 3),
        'anharmonicity_MHz': round(alpha_MHz, 1),
        'ce_interpretation': (
            f'Transmon qubit = LC oscillator where L is a Josephson junction. '
            f'f01={f_01_GHz:.1f}GHz transition (|0>-|1>). '
            f'f12={f_01_GHz + alpha_MHz/1e3:.3f}GHz (|1>-|2>) -- different frequency '
            f'allows selective addressing: this is the qubit.'
        ),
        'spice_model': 'L_J(phi) in parallel with C: same equations as SPICE RLC',
    }


def demo():
    print("=== ENERGY SCALES (Joule question) ===")
    for k, v in ENERGY_SCALES.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4g}")

    print("\n=== Photon at 1550nm ===")
    p = photon_energy(1550)
    for k, v in p.items():
        print(f"  {k}: {v:.4g}")

    print("\n=== Qubit thermal check (15mK, 5GHz transmon) ===")
    t = thermal_vs_qubit_energy(0.015, 5.0)
    print(f"  hf/kT = {t['hf_over_kT']:.1f}")
    print(f"  Excited state population: {t['excited_state_population']:.2e}")
    print(f"  {t['message']}")

    print("\n=== Maxwell -> QED ===")
    bridge = maxwell_to_qed_bridge()
    print(f"  Vacuum field E0 = {bridge['vacuum_field_V_per_m']:.2f} V/m")
    print(f"  GS connection: {bridge['gs_connection']}")

    print("\n=== Josephson Junction (transmon qubit) ===")
    jj = josephson_junction()
    print(f"  E_J/E_C = {jj['E_J_over_E_C']:.1f} (transmon: {jj['transmon_regime']})")
    print(f"  f01 = {jj['f01_GHz']} GHz")
    print(f"  Anharmonicity = {jj['anharmonicity_MHz']} MHz")
    print(f"  CE: {jj['ce_interpretation'][:80]}...")

    print("\n=== 3-Qubit Bit-Flip Code ===")
    qec = three_qubit_bit_flip_code()
    print(f"  X0 error syndrome: {qec['example_syndrome_X0_error']}")
    print(f"  CE analog: {qec['ce_analog']}")

    print("\n=== Surface Code ===")
    sc = surface_code_overview()
    print(f"  d=3: {sc['physical_per_logical']} physical / 1 logical qubit")
    print(f"  Threshold: {sc['threshold_percent']}% physical error rate")
    print(f"  CE job: {sc['ce_job']}")

    print("\n=== Griffiths -> QIS Map ===")
    for entry in griffiths_to_qis_map():
        print(f"  {entry['griffiths_ch']}")
        print(f"    -> {entry['qis_concept']}")

    print("\n=== Quantum Advantage ===")
    for entry in quantum_advantage_table():
        print(f"  {entry['problem']}: {entry['quantum_algorithm'][:60]}")


if __name__ == '__main__':
    demo()
