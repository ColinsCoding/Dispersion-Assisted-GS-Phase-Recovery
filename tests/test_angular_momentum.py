"""Test dgs.angular_momentum: the su(2) algebra. m = j..-j, Jz eigenvalues, the
Casimir J^2=j(j+1)I, the closing commutators [Jx,Jy]=iJz (cyclic), the ladder
operators (J+=J-dagger, raise/lower m, annihilate the end rungs), and j=1/2
reproducing the Pauli matrices."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import angular_momentum as am

# 1. m values: 2j+1 of them, descending from j to -j
assert np.allclose(am.m_values(1.0), [1, 0, -1])
assert np.allclose(am.m_values(0.5), [0.5, -0.5])
assert len(am.m_values(2.0)) == 5

# 2. Jz is diagonal with the m's as eigenvalues
for j in (0.5, 1.0, 1.5, 2.0):
    Jz = am.jz(j)
    assert np.allclose(np.diag(Jz), am.m_values(j))
    assert np.allclose(np.sort(np.linalg.eigvalsh(Jz)), np.sort(am.m_values(j)))

# 3. the Casimir J^2 = j(j+1) I
for j in (0.5, 1.0, 1.5, 2.0, 2.5):
    d = int(2 * j + 1)
    assert np.allclose(am.j_squared(j), j * (j + 1) * np.eye(d))

# 4. the commutators close: [Jx,Jy]=iJz, [Jy,Jz]=iJx, [Jz,Jx]=iJy
for j in (0.5, 1.0, 1.5, 2.0):
    Jx, Jy, Jz = am.jx(j), am.jy(j), am.jz(j)
    assert np.allclose(am.commutator(Jx, Jy), 1j * Jz)
    assert np.allclose(am.commutator(Jy, Jz), 1j * Jx)
    assert np.allclose(am.commutator(Jz, Jx), 1j * Jy)
    # Jx, Jy, Jz are Hermitian (observables)
    for J in (Jx, Jy, Jz):
        assert np.allclose(J, J.conj().T)

# 5. ladder operators
j = 1.0
Jp, Jm = am.jplus(j), am.jminus(j)
assert np.allclose(Jp, Jm.conj().T)                       # J+ = J-dagger
# J+ raises m: J+|1,0> = sqrt(2)|1,+1>
assert np.allclose(Jp @ np.eye(3)[1], np.sqrt(2) * np.eye(3)[0])
# J+ annihilates the top rung, J- the bottom rung
assert np.allclose(Jp @ np.eye(3)[0], 0)                  # |1,+1> is highest
assert np.allclose(Jm @ np.eye(3)[2], 0)                  # |1,-1> is lowest
# ladder coefficient sqrt(j(j+1)-m(m+1)); zero at m=j
assert np.isclose(am.raising_coefficient(1.0, 0.0), np.sqrt(2))
assert am.raising_coefficient(1.0, 1.0) == 0.0

# 6. j=1/2 IS the Pauli matrices (the qubit)
sx, sy, sz = am.pauli_matrices()
assert np.allclose(sx, [[0, 1], [1, 0]])
assert np.allclose(sy, [[0, -1j], [1j, 0]])
assert np.allclose(sz, [[1, 0], [0, -1]])
assert np.allclose(am.commutator(sx, sy), 2j * sz)        # [sx,sy]=2i sz
# each Pauli squares to the identity
for s in (sx, sy, sz):
    assert np.allclose(s @ s, np.eye(2))

# 7. dimensions
assert am.jz(0.0).shape == (1, 1)                         # j=0: a single state
assert am.jz(2.5).shape == (6, 6)

# 8. kwarg bounds: j must be a non-negative integer or half-integer
for bad in (-0.5, 0.3, 1.2):
    try:
        am.m_values(bad); assert False
    except ValueError:
        pass

print("test_angular_momentum: all checks passed")
