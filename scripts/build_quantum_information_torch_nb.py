"""Build notebooks/quantum_information_torch.ipynb -- dgs/quantum_information.py's
qubit/gate/entanglement/teleportation content, reimplemented in torch tensors
(complex128) and cross-checked against the module's existing NumPy version,
with sp.init_printing() used throughout for the symbolic pieces (the module
itself never called it). Closes with a torch-autograd bonus the NumPy version
can't do at all: gradient-based single-qubit state preparation.

Sections:
  S1  Setup, sp.init_printing(), SymPy data objects + Pauli commutators
  S2  Torch qubit states + Bloch vector, cross-checked vs NumPy
  S3  Torch gates, cross-checked vs NumPy
  S4  Bell states via torch (H then CNOT), cross-checked
  S5  Density matrix, partial trace, entanglement entropy via torch
  S6  Quantum teleportation via torch
  S7  Torch-autograd bonus: gradient-based qubit state preparation
  S8  Symbolic QI formalism, rendered
  S9  Summary: numpy vs torch agreement across everything
"""

import json, pathlib

NB = pathlib.Path("notebooks/quantum_information_torch.ipynb")
NB.parent.mkdir(exist_ok=True)

cells = []
def md(src): cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "execution_count": None,
                              "metadata": {}, "outputs": [], "source": src})


# ── S1 ────────────────────────────────────────────────────────────────────────
md("""# Quantum Information Science -- SymPy-Verified, Torch Tensors

[`dgs/quantum_information.py`](../dgs/quantum_information.py) already has
qubits, gates, Bell states, density matrices, entanglement entropy, quantum
teleportation, and topological QC -- all in NumPy, and its symbolic pieces
never called `sp.init_printing()`. This notebook: (1) turns on proper
rendering for the symbolic content, (2) reimplements the numerical content
in **torch tensors** (`complex128`), cross-checked line-for-line against the
module's own NumPy functions, and (3) does one thing NumPy can't: use
**autograd** to gradient-descend a qubit into a target state, an actual use
of torch beyond "NumPy with extra steps."
""")

code("""\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

import numpy as np
import torch
import sympy as sp
import pandas as pd

from dgs import quantum_information as qi

sp.init_printing()
torch.set_default_dtype(torch.float64)
CDTYPE = torch.complex128
checks = []   # (name, numpy_result, torch_result, match) collected throughout

print("numpy", np.__version__, "| torch", torch.__version__, "| sympy", sp.__version__)
""")

md("""## SymPy data objects + Pauli commutators, properly rendered

`qi.sympy_data_objects_demo()` and `qi.commutator_pauli_sympy()` already
exist; this is the first time their output gets `init_printing`."""),

code("""\
demo = qi.sympy_data_objects_demo()
demo["sigma_x"]
""")

code("""\
demo["Hamiltonian_Expr"]
""")

code("""\
comms = qi.commutator_pauli_sympy()
for name, residual in comms.items():
    ok = residual == sp.zeros(2)
    print(f"{name}:  residual = {residual.tolist()}   verified: {ok}")
""")

# ── S2: torch qubit states ────────────────────────────────────────────────────
md("""## Torch Qubit States + Bloch Vector

$|\\psi\\rangle=\\cos(\\theta/2)|0\\rangle+e^{i\\phi}\\sin(\\theta/2)|1\\rangle$,
reimplemented with torch tensors, cross-checked against `qi.qubit_state` /
`qi.bloch_vector` (NumPy) at several points on the Bloch sphere."""),

code("""\
def qubit_state_torch(theta, phi):
    theta_t = torch.as_tensor(theta, dtype=torch.float64)
    phi_t = torch.as_tensor(phi, dtype=torch.float64)
    alpha = torch.cos(theta_t / 2).to(CDTYPE)
    beta = (torch.exp(1j * phi_t) * torch.sin(theta_t / 2)).to(CDTYPE)
    return torch.stack([alpha, beta])

def bloch_vector_torch(psi):
    psi = psi / torch.linalg.norm(psi)
    rx = 2 * torch.real(psi[0].conj() * psi[1])
    ry = 2 * torch.imag(psi[0].conj() * psi[1])
    rz = torch.abs(psi[0])**2 - torch.abs(psi[1])**2
    return torch.stack([rx, ry, rz]).real

test_angles = [(0.0, 0.0), (np.pi/2, 0.0), (np.pi/2, np.pi/2), (np.pi/3, np.pi/4), (np.pi, 0.0)]
for theta, phi in test_angles:
    psi_np = qi.qubit_state(theta, phi)
    r_np = qi.bloch_vector(psi_np)
    psi_t = qubit_state_torch(theta, phi)
    r_t = bloch_vector_torch(psi_t).numpy()
    match = np.allclose(r_np, r_t, atol=1e-10)
    checks.append(("bloch_vector", theta, match))
    print(f"theta={theta:.3f} phi={phi:.3f}  numpy r={r_np}  torch r={r_t}  match={match}")
""")

# ── S3: torch gates ──────────────────────────────────────────────────────────
md("## Torch Gates, Cross-Checked Against `qi.GATE_*`"),

code("""\
X_t = torch.tensor([[0,1],[1,0]], dtype=CDTYPE)
Y_t = torch.tensor([[0,-1j],[1j,0]], dtype=CDTYPE)
Z_t = torch.tensor([[1,0],[0,-1]], dtype=CDTYPE)
H_t = torch.tensor([[1,1],[1,-1]], dtype=CDTYPE) / torch.sqrt(torch.tensor(2.0, dtype=CDTYPE))
S_t = torch.tensor([[1,0],[0,1j]], dtype=CDTYPE)
T_t = torch.tensor([[1,0],[0,torch.exp(1j*torch.tensor(np.pi/4))]], dtype=CDTYPE)
CNOT_t = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=CDTYPE)

gate_pairs = [("X", qi.GATE_X, X_t), ("Y", qi.GATE_Y, Y_t), ("Z", qi.GATE_Z, Z_t),
              ("H", qi.GATE_H, H_t), ("S", qi.GATE_S, S_t), ("T", qi.GATE_T, T_t),
              ("CNOT", qi.GATE_CNOT, CNOT_t)]
for name, g_np, g_t in gate_pairs:
    match = np.allclose(g_np, g_t.numpy(), atol=1e-10)
    checks.append(("gate", name, match))
    print(f"{name}: match={match}")

def apply_gate_torch(gate, state):
    result = gate.to(CDTYPE) @ state.to(CDTYPE)
    return result / torch.linalg.norm(result)

psi0 = qubit_state_torch(0.0, 0.0)
psi_plus = apply_gate_torch(H_t, psi0)
print("H|0> =", psi_plus.numpy(), " (expect (1,1)/sqrt2)")
""")

# ── S4: Bell states ────────────────────────────────────────────────────────────
md("""## Bell States via Torch (H then CNOT)

`qi.create_bell_state` reimplemented with `torch.kron` instead of
`np.kron`."""),

code("""\
def create_bell_state_torch(which="Phi_plus"):
    ket00 = torch.tensor([1,0,0,0], dtype=CDTYPE)
    HI = torch.kron(H_t, torch.eye(2, dtype=CDTYPE))
    state = HI @ ket00
    state = CNOT_t @ state
    if which == "Phi_minus":
        state = torch.kron(Z_t, torch.eye(2, dtype=CDTYPE)) @ state
    elif which == "Psi_plus":
        state = torch.kron(torch.eye(2, dtype=CDTYPE), X_t) @ state
    elif which == "Psi_minus":
        state = torch.kron(Z_t, X_t) @ state
    elif which != "Phi_plus":
        raise ValueError(f"unknown Bell state {which!r}")
    return state / torch.linalg.norm(state)

for which in ["Phi_plus", "Phi_minus", "Psi_plus", "Psi_minus"]:
    s_np = qi.create_bell_state(which)
    s_t = create_bell_state_torch(which).numpy()
    match = np.allclose(np.abs(s_np), np.abs(s_t), atol=1e-10)   # up to global phase
    checks.append(("bell_state", which, match))
    print(f"{which}: numpy={np.round(s_np,3)}  torch={np.round(s_t,3)}  match(|.|)={match}")
""")

# ── S5: entanglement entropy ────────────────────────────────────────────────────
md("""## Density Matrix, Partial Trace, Entanglement Entropy -- Torch

`torch.linalg.eigvalsh` handles complex Hermitian matrices directly (returns
real eigenvalues), and `torch.einsum` does the partial trace exactly like
`np.einsum`."""),

code("""\
def density_matrix_torch(psi):
    return torch.outer(psi, psi.conj())

def partial_trace_torch(rho, keep=0, dims=(2,2)):
    d0, d1 = dims
    rho4 = rho.reshape(d0, d1, d0, d1)
    if keep == 0:
        return torch.einsum('iaja->ij', rho4)
    return torch.einsum('aiaj->ij', rho4)

def von_neumann_entropy_torch(rho):
    eigvals = torch.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return -torch.sum(eigvals * torch.log2(eigvals)).item()

def entanglement_entropy_torch(psi_2qubit):
    rho = density_matrix_torch(psi_2qubit)
    rho_A = partial_trace_torch(rho, keep=0)
    return von_neumann_entropy_torch(rho_A)

for which in ["Phi_plus", "Phi_minus", "Psi_plus", "Psi_minus"]:
    E_np = qi.entanglement_entropy(qi.create_bell_state(which))
    E_t = entanglement_entropy_torch(create_bell_state_torch(which))
    match = np.isclose(E_np, E_t, atol=1e-8)
    checks.append(("entanglement_entropy", which, match))
    print(f"{which}: numpy E={E_np:.4f}  torch E={E_t:.4f}  match={match}")

# a PRODUCT state (not entangled) should give E=0 both ways
product_state = np.kron(qi.qubit_state(0.7, 0.3), qi.qubit_state(1.1, 0.9))
product_state_t = torch.tensor(product_state, dtype=CDTYPE)
E_np_prod = qi.entanglement_entropy(product_state)
E_t_prod = entanglement_entropy_torch(product_state_t)
print(f"\\nproduct state (unentangled): numpy E={E_np_prod:.2e}  torch E={E_t_prod:.2e}  (both ~0)")
checks.append(("entanglement_entropy", "product_state", E_np_prod < 1e-6 and E_t_prod < 1e-6))
""")

# ── S6: teleportation ────────────────────────────────────────────────────────
md("""## Quantum Teleportation via Torch

Reimplements `qi.quantum_teleportation_circuit` with torch tensors; checks
that every one of the 4 measurement outcomes gives Bob a state with fidelity
1 to Alice's original (up to the classical corrections), matching the NumPy
version outcome-by-outcome."""),

code("""\
def quantum_teleportation_circuit_torch(psi_to_send):
    psi = psi_to_send.to(CDTYPE)
    psi = psi / torch.linalg.norm(psi)
    bell = create_bell_state_torch("Phi_plus")
    state = torch.kron(psi, bell)

    I2 = torch.eye(2, dtype=CDTYPE)
    CNOT_AB = torch.kron(CNOT_t, I2)
    state = CNOT_AB @ state
    H_A = torch.kron(H_t, torch.eye(4, dtype=CDTYPE))
    state = H_A @ state

    outcomes = {}
    state_cube = state.reshape(2, 2, 2)
    for ma in range(2):
        for mb in range(2):
            bob_unnorm = state_cube[ma, mb, :]
            prob = torch.sum(torch.abs(bob_unnorm)**2).item()
            if prob > 1e-12:
                bob = bob_unnorm / np.sqrt(prob)
                if mb == 1:
                    bob = X_t @ bob
                if ma == 1:
                    bob = Z_t @ bob
                outcomes[(ma, mb)] = {"bob_state": bob, "prob": prob}
    return outcomes

psi_alice_np = qi.qubit_state(np.pi/3, np.pi/4)
psi_alice_t = qubit_state_torch(np.pi/3, np.pi/4)

outcomes_np = qi.quantum_teleportation_circuit(psi_alice_np)
outcomes_t = quantum_teleportation_circuit_torch(psi_alice_t)

for key in outcomes_np:
    fid_np = abs(np.vdot(psi_alice_np, outcomes_np[key]["bob_state"]))**2
    fid_t = (torch.abs(torch.vdot(psi_alice_t, outcomes_t[key]["bob_state"]))**2).item()
    match = np.isclose(fid_np, 1.0, atol=1e-8) and np.isclose(fid_t, 1.0, atol=1e-8)
    checks.append(("teleportation_fidelity", str(key), match))
    print(f"outcome {key}: numpy fidelity={fid_np:.6f}  torch fidelity={fid_t:.6f}  P={outcomes_np[key]['prob']:.3f}")
""")

# ── S7: autograd bonus ────────────────────────────────────────────────────────
md("""## Torch-Autograd Bonus: Gradient-Descent Qubit State Preparation

Something the NumPy version simply cannot do: treat $(\\theta,\\phi)$ as
trainable parameters and gradient-descend them so the prepared state matches
a target state, using `torch.optim.Adam` on the infidelity
$1-|\\langle\\text{target}|\\psi\\rangle|^2$. **Fidelity**, not the raw angles, is
the thing that should converge -- $(\\theta,\\phi)$ has gauge redundancy
(different angle pairs can give the same physical state up to global phase),
so checking angle-for-angle equality would be the wrong test."""),

code("""\
target_theta, target_phi = 1.9, 2.3
target = qubit_state_torch(target_theta, target_phi).detach()

theta_param = torch.tensor(0.1, requires_grad=True)
phi_param = torch.tensor(0.1, requires_grad=True)
opt = torch.optim.Adam([theta_param, phi_param], lr=0.1)

history = []
for step in range(300):
    opt.zero_grad()
    psi = qubit_state_torch(theta_param, phi_param)
    fidelity = torch.abs(torch.vdot(target, psi))**2
    loss = 1 - fidelity
    loss.backward()
    opt.step()
    history.append(fidelity.item())

print(f"final fidelity: {history[-1]:.10f}")
print(f"recovered (theta,phi) = ({theta_param.item():.4f}, {phi_param.item():.4f})")
print(f"target    (theta,phi) = ({target_theta:.4f}, {target_phi:.4f})")
print("(angles need not match exactly -- gauge freedom -- fidelity->1 is the real check)")
checks.append(("autograd_state_prep", "fidelity", history[-1] > 1 - 1e-8))
""")

code("""\
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(history, lw=2, color="steelblue")
ax.axhline(1.0, color="firebrick", ls="--", lw=1, label="perfect fidelity")
ax.set_xlabel("gradient step"); ax.set_ylabel("fidelity")
ax.set_title("Autograd-driven qubit state preparation")
ax.legend()
plt.tight_layout()
plt.savefig("qubit_autograd_fidelity.png", dpi=90)
plt.show()
""")

# ── S8: symbolic QI formalism ────────────────────────────────────────────────
md("## Symbolic QI Formalism, Rendered"),

code("""\
for name, eq in qi.quantum_information_sympy_5().items():
    print(name)
    display(eq)
""")

# ── S9: summary ────────────────────────────────────────────────────────────────
md("## Summary: NumPy vs. Torch Agreement"),

code("""\
df = pd.DataFrame(checks, columns=["category", "case", "match"])
print(df.to_string(index=False))
assert df["match"].all(), "every numpy/torch cross-check must agree"
print(f"\\nAll {len(df)} numpy/torch cross-checks agree, across "
      f"{df['category'].nunique()} categories.")
""")

# ── finalize ─────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "cells": cells,
}

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB}  ({len(cells)} cells)")
print(f"Execute (needs torch -> py 3.12): "
      f"py -3.12 -m jupyter nbconvert --to notebook --execute --inplace \"{NB}\"")
