"""Quantum angular momentum: the su(2) algebra that spin, rotation, and light share.

Angular momentum in quantum mechanics is three operators that do NOT commute:
        [Jx, Jy] = i Jz,   [Jy, Jz] = i Jx,   [Jz, Jx] = i Jy   (hbar = 1),
so you cannot know two components at once. What you CAN know together is the total
J^2 and one component (say Jz), which share eigenstates |j, m>:
        J^2 |j,m> = j(j+1) |j,m>,     Jz |j,m> = m |j,m>,
with m running j, j-1, ..., -j (that's 2j+1 states). The LADDER operators
        J+/- = Jx +/- i Jy,   J+/- |j,m> = sqrt(j(j+1) - m(m+/-1)) |j,m+/-1>
step m up and down and annihilate the top/bottom rung -- exactly the raising/lowering
structure of dgs.quantum_oscillator's a-dagger/a, now for spin.

This one algebra shows up everywhere:
  * j = 1/2 gives Jx, Jy, Jz = the Pauli matrices over 2 -- the qubit (dgs.qubits).
  * its conserved charge is the angular momentum that a central force keeps constant
    (dgs.symmetry_physics, Noether from rotational symmetry).
  * its spatial eigenstates are the spherical harmonics Y_l^m, and a light beam with a
    helical phase e^{i l phi} carries orbital angular momentum l per photon -- the
    "wave" face of the same integer m.

Everything is built from the ladder coefficients and checked: J^2 = j(j+1) I, the
commutators close, J+ = J-dagger, and j=1/2 reproduces the Paulis. NumPy only; py-3.13.
"""

import numpy as np


def _validate_j(j):
    """j must be a non-negative integer or half-integer (so 2j is a non-negative
    integer and the multiplet has 2j+1 states)."""
    if j < 0 or not np.isclose(2 * j, round(2 * j)):
        raise ValueError("j must be a non-negative integer or half-integer")
    return int(round(2 * j + 1))            # dimension


def m_values(j):
    """The 2j+1 magnetic quantum numbers m = j, j-1, ..., -j (descending)."""
    d = _validate_j(j)
    return np.array([j - k for k in range(d)], dtype=float)


def raising_coefficient(j, m):
    """<j,m+1| J+ |j,m> = sqrt(j(j+1) - m(m+1)) -- zero at the top rung m=j."""
    val = j * (j + 1) - m * (m + 1)
    return np.sqrt(max(val, 0.0))


def jz(j):
    """Jz = diag(m): the measurable component, its eigenvalues are the m's."""
    return np.diag(m_values(j)).astype(complex)


def jplus(j):
    """Raising operator J+ (maps |m> -> |m+1>, annihilates the top state)."""
    m = m_values(j)
    d = len(m)
    J = np.zeros((d, d), dtype=complex)
    for k in range(d):
        if k - 1 >= 0:                       # |m+1> sits at the lower index k-1
            J[k - 1, k] = raising_coefficient(j, m[k])
    return J


def jminus(j):
    """Lowering operator J- = J+dagger (maps |m> -> |m-1>)."""
    return jplus(j).conj().T


def jx(j):
    """Jx = (J+ + J-)/2."""
    return (jplus(j) + jminus(j)) / 2


def jy(j):
    """Jy = (J+ - J-)/(2i)."""
    return (jplus(j) - jminus(j)) / (2 * 1j)


def j_squared(j):
    """J^2 = Jx^2 + Jy^2 + Jz^2 = j(j+1) I -- the Casimir invariant."""
    Jx, Jy, Jz = jx(j), jy(j), jz(j)
    return Jx @ Jx + Jy @ Jy + Jz @ Jz


def commutator(A, B):
    """[A, B] = AB - BA."""
    A = np.asarray(A); B = np.asarray(B)
    return A @ B - B @ A


def pauli_matrices():
    """The three Pauli matrices (sigma_x, sigma_y, sigma_z) = 2 * (Jx,Jy,Jz) at
    j=1/2 -- the qubit's angular momentum."""
    return 2 * jx(0.5), 2 * jy(0.5), 2 * jz(0.5)


if __name__ == "__main__":
    for j in (0.5, 1.0, 1.5, 2.0):
        Jx, Jy, Jz = jx(j), jy(j), jz(j)
        d = int(2 * j + 1)
        c_ok = np.allclose(commutator(Jx, Jy), 1j * Jz)
        j2_ok = np.allclose(j_squared(j), j * (j + 1) * np.eye(d))
        print(f"j={j}: dim {d}, m = {m_values(j)},  [Jx,Jy]=iJz? {c_ok},  "
              f"J^2=j(j+1)I? {j2_ok}")

    print("\nj=1/2 gives the Pauli matrices:")
    sx, sy, sz = pauli_matrices()
    print("  sigma_x =\n", sx.real)
    print("  sigma_z =\n", sz.real)
    print("  [sx, sy] = 2i sz ?",
          np.allclose(commutator(sx, sy), 2j * sz))

    # ladder action: J+ raises m and annihilates the top rung
    j = 1.0
    top = np.eye(3)[0]                       # |j=1, m=+1>
    print(f"\nJ+ on the top state (m=+1) -> {np.round((jplus(j) @ top).real, 3)} (annihilated)")
    mid = np.eye(3)[1]                       # |1, 0>
    print(f"J+ on |1,0> -> {np.round((jplus(j) @ mid).real, 3)} = sqrt(2)|1,+1>")
