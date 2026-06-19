"""Smoke-test the quantum 'algebra tools': observables, expectation values, Bloch vector."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import qubits as q

# 1. observables are Hermitian; gates are unitary (the two algebra families)
for A in (q.X, q.Y, q.Z):
    assert q.is_hermitian(A)
for U in (q.X, q.H, q.S, q.T):
    assert q.is_unitary(U)
assert not q.is_hermitian(q.T)           # T is unitary but not Hermitian
assert not q.is_unitary(np.array([[1, 1], [0, 1]]))   # shear is neither

# 2. <Z> reads the bit: +1 for |0>, -1 for |1>, 0 for an equal superposition
assert abs(q.expectation(q.ket("0"), q.Z) - 1) < 1e-12
assert abs(q.expectation(q.ket("1"), q.Z) + 1) < 1e-12
plus = q.apply_1q(q.ket("0"), q.H, 0)    # |+>
assert abs(q.expectation(plus, q.Z)) < 1e-12
assert abs(q.expectation(plus, q.X) - 1) < 1e-12     # <X> = +1 for |+>

# 3. expectation of a Hermitian observable is real
assert abs(q.expectation(plus, q.Y).imag) < 1e-12

# 4. Bloch vector: the three numbers that fix a pure qubit; unit length
assert np.allclose(q.bloch_vector(q.ket("0")), [0, 0, 1])
assert np.allclose(q.bloch_vector(q.ket("1")), [0, 0, -1])
assert np.allclose(q.bloch_vector(plus), [1, 0, 0])
i_state = q.apply_1q(q.ket("0"), q.S @ q.H, 0)        # |+i>
assert np.allclose(q.bloch_vector(i_state), [0, 1, 0], atol=1e-9)
for s in (q.ket("0"), plus, i_state):
    assert abs(np.linalg.norm(q.bloch_vector(s)) - 1.0) < 1e-9   # pure -> on the sphere

# 5. eigenvalue connection: <A> of an eigenstate equals its eigenvalue
#    Z|1> = -1|1>, so <Z> = -1 exactly (measurement is deterministic there)
assert abs(q.expectation(q.ket("1"), q.Z) - (-1.0)) < 1e-12

print(f"SMOKE PASS  (<Z|0>=+1, <Z|1>=-1, <X|+>=+1; Bloch(|+>)= "
      f"{np.round(q.bloch_vector(plus),3).tolist()}; observables Hermitian, gates unitary)")
