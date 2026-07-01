import numpy as np
import pytest
import sympy as sp
from dgs.quantum_information import (
    sympy_data_objects_demo,
    qubit_state, bloch_vector,
    GATE_X, GATE_Y, GATE_Z, GATE_H, GATE_S, GATE_T,
    apply_gate,
    commutator_pauli_sympy,
    bell_states, create_bell_state, entanglement_entropy, is_entangled,
    density_matrix, partial_trace, von_neumann_entropy,
    quantum_teleportation_circuit,
    kitaev_chain_hamiltonian_sympy,
    chern_number_2d,
    quantum_information_sympy_5,
)


# ── SymPy data objects ────────────────────────────────────────────────

def test_sympy_demo_returns_sigma_matrices():
    demo = sympy_data_objects_demo()
    sx = demo["sigma_x"]
    assert sx[0, 1] == 1 and sx[1, 0] == 1
    assert sx[0, 0] == 0 and sx[1, 1] == 0


def test_sympy_demo_pauli_y_imaginary():
    demo = sympy_data_objects_demo()
    sy = demo["sigma_y"]
    assert sy[0, 1] == -sp.I


def test_sympy_demo_rational_half():
    demo = sympy_data_objects_demo()
    assert demo["Rational_half"] == sp.Rational(1, 2)


# ── qubit state ───────────────────────────────────────────────────────

def test_qubit_state_north_pole():
    psi = qubit_state(0.0, 0.0)
    assert np.abs(psi[0] - 1.0) < 1e-10
    assert np.abs(psi[1]) < 1e-10


def test_qubit_state_south_pole():
    psi = qubit_state(np.pi, 0.0)
    assert np.abs(psi[0]) < 1e-10
    assert np.abs(np.abs(psi[1]) - 1.0) < 1e-10


def test_qubit_state_equator_normalized():
    psi = qubit_state(np.pi / 2, np.pi / 4)
    assert np.abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_qubit_state_invalid_theta():
    with pytest.raises(ValueError):
        qubit_state(-0.1, 0.0)


def test_bloch_vector_north_pole():
    psi = qubit_state(0.0, 0.0)
    r = bloch_vector(psi)
    assert r[2] == pytest.approx(1.0, abs=1e-6)


def test_bloch_vector_pure_state_unit_norm():
    psi = qubit_state(np.pi / 3, np.pi / 4)
    r = bloch_vector(psi)
    assert np.linalg.norm(r) == pytest.approx(1.0, abs=1e-6)


# ── gates ─────────────────────────────────────────────────────────────

def test_pauli_x_flips():
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = apply_gate(GATE_X, ket0)
    assert np.allclose(ket1, [0, 1])


def test_pauli_z_phase():
    ket1 = np.array([0, 1], dtype=complex)
    result = apply_gate(GATE_Z, ket1)
    assert np.allclose(result, [0, -1])


def test_hadamard_creates_superposition():
    ket0 = np.array([1, 0], dtype=complex)
    psi = apply_gate(GATE_H, ket0)
    assert np.abs(np.abs(psi[0]) - 1/np.sqrt(2)) < 1e-10
    assert np.abs(np.abs(psi[1]) - 1/np.sqrt(2)) < 1e-10


def test_hadamard_self_inverse():
    ket1 = np.array([0, 1], dtype=complex)
    psi = apply_gate(GATE_H, apply_gate(GATE_H, ket1))
    assert np.allclose(psi, [0, 1], atol=1e-10)


def test_gate_incompatible_shape():
    with pytest.raises(ValueError):
        apply_gate(GATE_X, np.array([1, 0, 0], dtype=complex))


def test_pauli_unitarity():
    for G in [GATE_X, GATE_Y, GATE_Z, GATE_H]:
        prod = G @ G.conj().T
        assert np.allclose(prod, np.eye(2), atol=1e-10)


# ── Pauli commutators ─────────────────────────────────────────────────

def test_pauli_commutators_satisfied():
    comms = commutator_pauli_sympy()
    zeros = sp.zeros(2)
    for k, residual in comms.items():
        assert residual == zeros, f"{k} not satisfied"


# ── Bell states ───────────────────────────────────────────────────────

def test_bell_states_normalized():
    bs = bell_states()
    for name, state in bs.items():
        assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10, f"{name} not normalized"


def test_bell_states_orthogonal():
    bs = bell_states()
    states = list(bs.values())
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            overlap = np.abs(np.vdot(states[i], states[j]))
            assert overlap < 1e-10, f"Bell states {i} and {j} not orthogonal"


def test_create_bell_state_phi_plus():
    state = create_bell_state("Phi_plus")
    target = bell_states()["Phi_plus"]
    fidelity = np.abs(np.vdot(target, state))**2
    assert fidelity == pytest.approx(1.0, abs=1e-6)


def test_create_bell_state_all_four():
    names = ["Phi_plus", "Phi_minus", "Psi_plus", "Psi_minus"]
    targets = bell_states()
    for name in names:
        state = create_bell_state(name)
        fidelity = np.abs(np.vdot(targets[name], state))**2
        assert fidelity == pytest.approx(1.0, abs=0.01), f"{name} mismatch"


def test_create_bell_state_invalid():
    with pytest.raises(ValueError):
        create_bell_state("invalid_name")


# ── entanglement ──────────────────────────────────────────────────────

def test_bell_state_entanglement_entropy_one_ebit():
    bs = bell_states()
    for name, state in bs.items():
        E = entanglement_entropy(state)
        assert E == pytest.approx(1.0, abs=1e-4), f"{name}: E={E}"


def test_product_state_zero_entanglement():
    ket0 = np.array([1, 0], dtype=complex)
    product = np.kron(ket0, ket0)
    E = entanglement_entropy(product)
    assert E == pytest.approx(0.0, abs=1e-6)


def test_is_entangled_bell_state():
    state = bell_states()["Phi_plus"]
    assert is_entangled(state) is True


def test_is_not_entangled_product_state():
    product = np.kron(np.array([1,0], dtype=complex), np.array([0,1], dtype=complex))
    assert is_entangled(product) is False


def test_von_neumann_entropy_maximally_mixed():
    rho = np.eye(2, dtype=complex) / 2  # maximally mixed = I/2
    S = von_neumann_entropy(rho)
    assert S == pytest.approx(1.0, abs=1e-6)


def test_von_neumann_entropy_pure_state_zero():
    psi = np.array([1, 0], dtype=complex)
    rho = density_matrix(psi)
    S = von_neumann_entropy(rho)
    assert S == pytest.approx(0.0, abs=1e-6)


# ── quantum teleportation ─────────────────────────────────────────────

def test_teleportation_four_outcomes():
    psi = qubit_state(np.pi / 3, np.pi / 4)
    outcomes = quantum_teleportation_circuit(psi)
    assert len(outcomes) == 4


def test_teleportation_probabilities_sum_to_one():
    psi = qubit_state(np.pi / 3, np.pi / 4)
    outcomes = quantum_teleportation_circuit(psi)
    total_prob = sum(info["prob"] for info in outcomes.values())
    assert total_prob == pytest.approx(1.0, abs=1e-6)


def test_teleportation_fidelity_perfect():
    psi = qubit_state(np.pi / 3, np.pi / 4)
    outcomes = quantum_teleportation_circuit(psi)
    for (ma, mb), info in outcomes.items():
        fidelity = np.abs(np.vdot(psi, info["bob_state"]))**2
        assert fidelity == pytest.approx(1.0, abs=0.01), \
            f"Teleportation failed for outcome ({ma},{mb}): fidelity={fidelity}"


# ── Kitaev chain ──────────────────────────────────────────────────────

def test_kitaev_chain_returns_dict():
    kit = kitaev_chain_hamiltonian_sympy(4)
    assert "H" in kit
    assert "topological_phase_condition" in kit
    assert "MZM_gamma1" in kit


def test_kitaev_phase_boundary_is_equation():
    kit = kitaev_chain_hamiltonian_sympy(4)
    assert isinstance(kit["topological_phase_condition"], sp.Basic)


# ── Chern number ──────────────────────────────────────────────────────

def _haldane_H(kx, ky, m=0.5):
    hx = np.sin(kx)
    hy = np.sin(ky)
    hz = m - (np.cos(kx) + np.cos(ky))
    return np.array([[hz, hx-1j*hy],[hx+1j*hy, -hz]])


def test_chern_number_topological_phase():
    # endpoint=False: BZ is periodic, don't double-count the boundary
    kx = np.linspace(-np.pi, np.pi, 25, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, 25, endpoint=False)
    C = chern_number_2d(kx, ky, lambda kx, ky: _haldane_H(kx, ky, m=0.5))
    assert abs(C["chern_number"]) == 1


def test_chern_number_trivial_phase():
    kx = np.linspace(-np.pi, np.pi, 25, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, 25, endpoint=False)
    # m=5.0 is deep in the trivial phase (|m| > 2*t1 = 2)
    C = chern_number_2d(kx, ky, lambda kx, ky: _haldane_H(kx, ky, m=5.0))
    assert C["chern_number"] == 0


# ── SymPy 5 ──────────────────────────────────────────────────────────

def test_qi_sympy_5_count():
    eqs = quantum_information_sympy_5()
    assert len(eqs) == 5


def test_qi_sympy_5_types():
    eqs = quantum_information_sympy_5()
    for k, eq in eqs.items():
        assert isinstance(eq, sp.Basic), f"{k} is not a SymPy expression"
