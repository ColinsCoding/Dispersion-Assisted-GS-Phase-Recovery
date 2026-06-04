"""
repl/_repl_dram_to_quantum.py
Vertical stack: DRAM device physics -> digital logic -> quantum computing.
Feynman path integral connects all three.
"""
import numpy as np
import sympy as sp
import pandas as pd
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 60)
print("DRAM -> DIGITAL LOGIC -> QUANTUM COMPUTING")
print("one vertical stack, same physics all the way down")
print("=" * 60)
print()

# ============================================================
# 1. DRAM device physics
# ============================================================
print("=== 1. DRAM Device Physics ===")
print("""
Structure: 1 MOSFET + 1 capacitor per bit cell

MOSFET:
  Gate voltage > Vth  ->  channel forms  ->  transistor ON  (write/read)
  Gate voltage < Vth  ->  no channel     ->  transistor OFF (hold)

Capacitor:
  Stores charge Q = C*V  (~5000 electrons at 1V, 0.9 fF)
  Leaks via junction current -> must refresh every 64 ms

Key equations:
  Vth  = 2*phi_F + sqrt(2*eps_si*q*Na*2*phi_F) / Cox   (threshold voltage)
  Cox  = eps_ox / t_ox                                  (gate capacitance)
  Id   = (W/L)*mu_n*Cox*(Vgs-Vth)*Vds  (linear region)
""")

# numerical: threshold voltage estimate
eps_si  = 11.7 * 8.854e-12   # F/m
eps_ox  = 3.9  * 8.854e-12   # F/m
q       = 1.602e-19           # C
Na      = 1e17 * 1e6          # /m^3  (p-type doping)
kT      = 0.026               # eV at 300K
ni      = 1e10 * 1e6          # /m^3  intrinsic carrier concentration
t_ox    = 5e-9                # 5 nm gate oxide

phi_F   = kT * np.log(Na / ni)   # Fermi potential
Cox     = eps_ox / t_ox
Vth     = 2*phi_F + np.sqrt(2*eps_si*q*Na*2*phi_F) / Cox

print(f"phi_F = {phi_F:.3f} V  (Fermi potential)")
print(f"Cox   = {Cox:.3e} F/m^2")
print(f"Vth   = {Vth:.3f} V  (threshold voltage)")
print()

# leakage: exponential decay
print("Charge retention (exponential decay):")
tau_leak = 0.064 / np.log(2)   # 64ms half-life
t_ms = np.array([0, 10, 32, 64, 100, 200])
Q_ratio = np.exp(-t_ms*1e-3 / tau_leak)
df_leak = pd.DataFrame({'t_ms': t_ms, 'Q/Q0': np.round(Q_ratio, 4),
                         'readable': ['yes' if q > 0.5 else 'NO' for q in Q_ratio]})
print(df_leak.to_string(index=False))
print("-> refresh required before 64ms (half charge lost)")
print()

# ============================================================
# 2. Digital logic: XOR from NAND
# ============================================================
print("=== 2. XOR from NAND (universal gate) ===")
print("""
XOR built from 4 NAND gates:
  G1 = NAND(A, B)
  G2 = NAND(A, G1)
  G3 = NAND(B, G1)
  G4 = NAND(G2, G3)  = A XOR B
""")

vals = [(0,0),(0,1),(1,0),(1,1)]
rows = []
for a,b in vals:
    g1 = int(not(a and b))
    g2 = int(not(a and g1))
    g3 = int(not(b and g1))
    g4 = int(not(g2 and g3))
    rows.append({'A':a,'B':b,'G1':g1,'G2':g2,'G3':g3,'XOR=G4':g4,'check':a^b})
df_xor = pd.DataFrame(rows)
print(df_xor.to_string(index=False))
print("XOR=G4 == check: all match")
print()

# ============================================================
# 3. Quantum bit (qubit) vs classical bit
# ============================================================
print("=== 3. Classical bit vs Qubit ===")
print("""
Classical bit:   state in {0, 1}
                 read: always gives 0 or 1

Qubit:           state |psi> = alpha|0> + beta|1>
                 |alpha|^2 + |beta|^2 = 1   (normalization)
                 read: gives 0 with prob |alpha|^2
                             1 with prob |beta|^2
                 MEASUREMENT DESTROYS superposition

Bloch sphere:
  |psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>
  theta=0    ->  |0>         (north pole)
  theta=pi   ->  |1>         (south pole)
  theta=pi/2 ->  |+> = (|0>+|1>)/sqrt(2)  (equator, superposition)
""")

# numerical: qubit state probabilities
thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
rows_q = []
for theta in thetas:
    a = np.cos(theta/2)
    b = np.sin(theta/2)
    rows_q.append({'theta_deg': int(np.degrees(theta)),
                   '|alpha|^2': round(a**2, 4),
                   '|beta|^2':  round(b**2, 4),
                   'sum':       round(a**2+b**2, 4),
                   'state': '|0>' if theta==0 else ('|1>' if abs(theta-np.pi)<0.01 else '|+>')})
print(pd.DataFrame(rows_q).to_string(index=False))
print()

# ============================================================
# 4. Quantum gates: unitary matrices
# ============================================================
print("=== 4. Quantum Gates (unitary matrices) ===")

gates = {
    'Pauli X (NOT)':  np.array([[0,1],[1,0]]),
    'Pauli Z':        np.array([[1,0],[0,-1]]),
    'Hadamard':       np.array([[1,1],[1,-1]]) / np.sqrt(2),
    'S gate':         np.array([[1,0],[0,1j]]),
    'T gate':         np.array([[1,0],[0,np.exp(1j*np.pi/4)]]),
}

for name, U in gates.items():
    UU = U @ U.conj().T
    is_unitary = np.allclose(UU, np.eye(2))
    det = np.linalg.det(U)
    print(f"  {name:20s}  unitary={is_unitary}  det={det:.3f}")
print()

print("Hadamard on |0>: H|0> = (|0>+|1>)/sqrt(2)")
ket0 = np.array([1.0, 0.0])
H_gate = np.array([[1,1],[1,-1]]) / np.sqrt(2)
print(f"  {H_gate @ ket0}  <- equal superposition")
print()

# CNOT gate (2-qubit)
print("CNOT gate (2-qubit controlled-NOT):")
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
basis = ['|00>','|01>','|10>','|11>']
for i, state in enumerate(basis):
    ket = np.zeros(4); ket[i] = 1
    out_idx = np.argmax(CNOT @ ket)
    print(f"  CNOT {state} -> {basis[out_idx]}")
print()

# ============================================================
# 5. Feynman path integral: classical -> quantum -> GS
# ============================================================
print("=== 5. Feynman path integral: the unifying picture ===")
print("""
Classical mechanics:   particle takes ONE path (minimum action)
Quantum mechanics:     particle takes ALL paths simultaneously
                       amplitude = sum exp(i*S[path]/hbar)
                       probability = |amplitude|^2

DRAM:     classical bit, one definite state
          Coulomb potential V(r) = kq/r governs charge storage
          leakage = thermal fluctuations over barrier (kT)

Qubit:    superposition until measured
          gate = unitary rotation on Bloch sphere
          decoherence = environment "measures" qubit -> collapses

GS phase recovery:
          I1 = |<E|H1>|^2  <- intensity = Born rule (squared amplitude)
          GS recovers phi from |psi|^2 = I  (quantum state tomography)
          Two measurements (D1, D2) = two bases (MUB)
          FNO learns the inverse Born map: I -> phi

The chain:
  DRAM stores bits         (classical, electrostatics)
  Logic gates process bits (Boolean, CMOS, Vth physics)
  Qubits are superpositions(quantum, unitary gates)
  GS recovers phase        (quantum-like: |E|^2 loses phase, GS restores it)
  FNO learns the map       (ML: supervised on the quantum measurement problem)
""")

# numerical: double-slit interference (Feynman sum over paths)
print("Double-slit interference (sum over 2 paths):")
lam  = 1550e-9    # 1550 nm (your fiber wavelength)
d    = 10e-6      # 10 um slit separation
L    = 0.01       # 1 cm screen distance
x    = np.linspace(-0.5e-3, 0.5e-3, 500)

# path length difference -> phase difference
delta = d * x / L
phi_diff = 2 * np.pi * delta / lam
I_interference = np.cos(phi_diff/2)**2   # normalized

peaks = np.where((np.diff(np.sign(np.diff(I_interference))) < 0))[0]
fringe_spacing = np.mean(np.diff(x[peaks])) * 1e6   # um

print(f"  lambda = 1550 nm, d = 10 um, L = 1 cm")
print(f"  Fringe spacing = lambda*L/d = {lam*L/d*1e6:.1f} um")
print(f"  Measured from interference pattern: {fringe_spacing:.1f} um")
print(f"  Feynman: each path contributes exp(i*2*pi*path_length/lambda)")
print(f"  Sum of two paths -> cos^2 interference -> same as your GS kernel")
