"""The generalized (Robertson) uncertainty principle -- Griffiths QM section 3.4.

For ANY two observables A, B in a state psi, the spreads obey
    sigma_A sigma_B >= | (1/2i) <[A, B]> | ,   [A, B] = AB - BA.
The famous cases are special cases of this one inequality:
  * position & momentum: [x, p] = i hbar  ->  sigma_x sigma_p >= hbar/2 (Heisenberg);
  * spin: [sigma_x, sigma_y] = 2 i sigma_z  ->  sigma_Sx sigma_Sy >= |<Sz>|.
Whenever two operators do NOT commute you cannot make both spreads small at once; when
they commute the bound is zero (simultaneous eigenstates exist). It is all linear
algebra -- operators are matrices, states are vectors, expectations are inner products
(Griffiths Chapter 3). This is the OPERATOR form; dgs.uncertainty is the Fourier
(pulse time-bandwidth) form of the same idea. NumPy. Education.
"""

import numpy as np

# Pauli matrices (Griffiths' fundamental-equations sheet). Spin S = (hbar/2) * sigma.
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _normalize(state):
    s = np.asarray(state, dtype=complex)
    return s / np.linalg.norm(s)


def expectation(op, state):
    """<psi|A|psi> for a state vector (normalized internally)."""
    s = _normalize(state)
    return complex(s.conj() @ (np.asarray(op, complex) @ s))


def variance(op, state):
    """sigma_A^2 = <A^2> - <A>^2 (real and >= 0 for a Hermitian A)."""
    op = np.asarray(op, complex)
    a = expectation(op, state)
    a2 = expectation(op @ op, state)
    return float((a2 - a * a).real)


def std(op, state):
    """sigma_A = sqrt(variance)."""
    return np.sqrt(max(variance(op, state), 0.0))


def commutator(A, B):
    """[A, B] = AB - BA. Zero iff A and B share an eigenbasis (compatible observables)."""
    A, B = np.asarray(A, complex), np.asarray(B, complex)
    return A @ B - B @ A


def uncertainty_bound(A, B, state):
    """Right-hand side of Robertson: |(1/2i) <[A,B]>| -- the lower bound on the product
    of spreads in this state."""
    return abs(expectation(commutator(A, B), state) / (2j))


def check_uncertainty(A, B, state):
    """Returns (sigma_A*sigma_B, bound, holds): verifies sigma_A sigma_B >= bound."""
    lhs = std(A, state) * std(B, state)
    rhs = uncertainty_bound(A, B, state)
    return lhs, rhs, bool(lhs >= rhs - 1e-9)


def position_momentum_uncertainty(x, psi, hbar=1.0):
    """sigma_x sigma_p for a wavefunction psi(x) on grid x, with p = -i hbar d/dx
    (evaluated in momentum space via the FFT). Returns (sigma_x sigma_p, hbar/2). A
    Gaussian saturates it at hbar/2 -- the continuous [x, p] = i hbar special case."""
    x = np.asarray(x, float)
    psi = np.asarray(psi, complex)
    dx = x[1] - x[0]
    P = np.abs(psi) ** 2; P /= P.sum()
    xbar = np.sum(x * P); sx = np.sqrt(np.sum((x - xbar) ** 2 * P))
    k = 2 * np.pi * np.fft.fftfreq(len(x), dx)            # p = hbar k
    Pp = np.abs(np.fft.fft(psi)) ** 2; Pp /= Pp.sum()
    kbar = np.sum(k * Pp); sk = np.sqrt(np.sum((k - kbar) ** 2 * Pp))
    return float(sx * hbar * sk), hbar / 2


if __name__ == "__main__":
    print("[sigma_x, sigma_y] = 2 i sigma_z :",
          np.allclose(commutator(SIGMA_X, SIGMA_Y), 2j * SIGMA_Z))
    up = np.array([1, 0])                                  # spin up along z
    lhs, rhs, ok = check_uncertainty(SIGMA_X, SIGMA_Y, up)
    print(f"spin |up_z>: sigma_x sigma_y = {lhs:.3f} >= |<sigma_z>| = {rhs:.3f}  (ok={ok}, equality)")
    plus = np.array([1, 1])                                # spin along +x: <sigma_z>=0 -> bound 0
    print(f"spin |+x>:   bound = {uncertainty_bound(SIGMA_X, SIGMA_Y, plus):.3f} (sigma_x is sharp here)")
    t = np.linspace(-12, 12, 4096); g = np.exp(-t**2 / 2)
    prod, bound = position_momentum_uncertainty(t, g)
    print(f"Gaussian: sigma_x sigma_p = {prod:.4f}  (Heisenberg minimum hbar/2 = {bound})")
