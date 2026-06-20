"""Smoke-test the qubit / quantum-gate simulator."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import qubits as q

# 1. single-qubit gates: X flips, Z phases, H makes superposition
assert np.allclose(q.apply_1q(q.ket("0"), q.X, 0), q.ket("1"))      # X|0> = |1>
assert np.allclose(q.apply_1q(q.ket("1"), q.Z, 0), -q.ket("1"))     # Z|1> = -|1>
h0 = q.apply_1q(q.ket("0"), q.H, 0)
assert np.allclose(np.abs(h0)**2, [0.5, 0.5])                        # H|0> -> equal superposition

# 2. gate identities: H^2 = I, X^2 = I, S^2 = Z, T^2 = S
assert np.allclose(q.apply_1q(h0, q.H, 0), q.ket("0"))               # H involution
assert np.allclose(q.S @ q.S, q.Z)
assert np.allclose(q.T @ q.T, q.S)

# 3. all gates are unitary (U U^dagger = I) -- reversible, norm-preserving
for G in (q.X, q.Y, q.Z, q.H, q.S, q.T):
    assert np.allclose(G @ G.conj().T, np.eye(2), atol=1e-12)

# 4. CNOT is a reversible XOR: |10>->|11>, |11>->|10>, |0x> unchanged
assert np.allclose(q.cnot(q.ket("10"), 0, 1), q.ket("11"))
assert np.allclose(q.cnot(q.ket("11"), 0, 1), q.ket("10"))
assert np.allclose(q.cnot(q.ket("01"), 0, 1), q.ket("01"))
assert np.allclose(q.cnot(q.cnot(q.ket("10"), 0, 1), 0, 1), q.ket("10"))   # CNOT^2 = I

# 5. Bell state: entangled, only 00 and 11 occur, each 1/2
psi = q.bell_state()
assert np.allclose(psi, [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
p = q.measure_probs(psi)
assert abs(p[0] - 0.5) < 1e-12 and abs(p[3] - 0.5) < 1e-12
assert p[1] == 0 and p[2] == 0                                        # never 01 or 10 (correlated)
counts = q.sample(psi, 4000, np.random.default_rng(0))
assert set(counts) <= {"00", "11"}                                    # perfect correlation
assert abs(counts.get("00", 0) / 4000 - 0.5) < 0.05

# 6. norm is preserved through a circuit
state = q.ket("000")
for op in [(q.H, 0)]:
    state = q.apply_1q(state, *op)
state = q.cnot(state, 0, 1)
state = q.cnot(state, 1, 2)                                           # GHZ-style spread
assert abs(np.linalg.norm(state) - 1.0) < 1e-12

# 7. validation
for bad in (lambda: q.ket("0x1"), lambda: q.cnot(q.ket("00"), 0, 0),
            lambda: q.apply_1q(q.ket("0"), q.X, 5)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad input")

print(f"SMOKE PASS  (X/Z/H + S^2=Z, T^2=S; CNOT=reversible XOR; "
      f"Bell pair {dict(counts)} -- only 00/11, entangled)")
