"""Why J = L + S is the SHARP (conserved) observable once spin-orbit
coupling is present, while L and S individually are NOT -- verified
directly by explicit angular-momentum matrix commutators (the real,
rigorous QM textbook argument), not just asserted.

THE PHYSICS: with a spin-orbit Hamiltonian term H_SO = xi * L.S
(L.S = Lx*Sx + Ly*Sy + Lz*Sz), the individual operators L_z and S_z do
NOT commute with H_SO -- so m_l and m_s stop being good quantum numbers.
But L^2, S^2, J_z, and J^2 all DO commute with H_SO -- so l, s, j, and
m_j remain sharp/conserved. This is exactly why spectroscopic notation
switches from |l, m_l, s, m_s> to term symbols ^(2S+1)L_J (e.g. 2P_3/2)
once spin-orbit coupling matters.

REAL APPLICATION -- MRI: proton (nuclear spin I=1/2) precession is the
SAME Larmor physics as dgs.electron_spin_resonance's electron ESR, just
with the proton's gyromagnetic ratio instead of the electron's. Real
clinical MRI scanners: 1.5 T -> ~63.9 MHz proton Larmor frequency,
3 T -> ~127.7 MHz -- both checked here against the real published
proton gyromagnetic ratio (42.577 MHz/T).

NUMERIC-TYPE NOTE: angular momentum quantum numbers for fermions (J, m_J)
are HALF-integers (1/2, 3/2, ...), not integers -- representing them as
plain floats works but can accumulate rounding error under repeated
arithmetic. quantum_number_as_fraction() shows the exact alternative
(Python's fractions.Fraction) for when exactness matters (e.g. checking
selection rules by equality rather than a tolerance)."""

from fractions import Fraction

import numpy as np

L_LETTERS = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}
PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T = 42.577  # real published value


def angular_momentum_operators(j):
    """Standard textbook construction of J_x, J_y, J_z, J^2 matrices for
    angular momentum quantum number j (integer or half-integer), in the
    |j, m> basis (2j+1 states) via raising/lowering operators."""
    if j <= 0:
        raise ValueError("j must be positive")
    if abs(2 * j - round(2 * j)) > 1e-9:
        raise ValueError("j must be an integer or half-integer")
    dim = int(round(2 * j + 1))
    m_vals = [j - k for k in range(dim)]   # descending: j, j-1, ..., -j
    Jz = np.diag(m_vals).astype(complex)
    Jplus = np.zeros((dim, dim), dtype=complex)
    for k in range(dim - 1):
        m = m_vals[k + 1]   # state being raised FROM
        coeff = np.sqrt(j * (j + 1) - m * (m + 1))
        Jplus[k, k + 1] = coeff   # |j,m> -> |j,m+1>, which sits at index k
    Jminus = Jplus.conj().T
    Jx = (Jplus + Jminus) / 2.0
    Jy = (Jplus - Jminus) / (2.0j)
    J2 = (j * (j + 1)) * np.eye(dim, dtype=complex)
    return Jx, Jy, Jz, J2


def commutator(A, B):
    return A @ B - B @ A


def spin_orbit_hamiltonian(L, S, xi=1.0):
    """H_SO = xi * L.S = xi*(Lx(x)Sx + Ly(x)Sy + Lz(x)Sz), built on the
    combined (2L+1)*(2S+1)-dimensional space via Kronecker products."""
    if xi <= 0:
        raise ValueError("xi must be positive")
    Lx, Ly, Lz, L2 = angular_momentum_operators(L)
    Sx, Sy, Sz, S2 = angular_momentum_operators(S)
    H_SO = xi * (np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz))
    return H_SO


def verify_good_quantum_numbers(L, S, xi=1.0):
    """Computes ||[H_SO, X]|| for X in {L_z, S_z, J_z, L^2, S^2, J^2} on
    the combined L(x)S space. L_z and S_z should NOT commute (nonzero
    norm); J_z, L^2, S^2, and J^2 SHOULD commute (~zero norm) -- this is
    the actual, checkable content of "J is conserved, L and S are not"."""
    Lx, Ly, Lz, L2 = angular_momentum_operators(L)
    Sx, Sy, Sz, S2 = angular_momentum_operators(S)
    dimL, dimS = Lx.shape[0], Sx.shape[0]
    IL, IS = np.eye(dimL, dtype=complex), np.eye(dimS, dtype=complex)

    H_SO = spin_orbit_hamiltonian(L, S, xi)
    Lz_full = np.kron(Lz, IS)
    Sz_full = np.kron(IL, Sz)
    Jz_full = Lz_full + Sz_full
    L2_full = np.kron(L2, IS)
    S2_full = np.kron(IL, S2)
    J2_full = L2_full + S2_full + 2 * H_SO   # (L+S)^2 = L^2 + S^2 + 2*L.S

    results = {}
    for name, op in [("L_z", Lz_full), ("S_z", Sz_full), ("J_z", Jz_full),
                      ("L^2", L2_full), ("S^2", S2_full), ("J^2", J2_full)]:
        norm = np.linalg.norm(commutator(H_SO, op))
        results[name] = {"commutator_norm": float(norm), "conserved": norm < 1e-9}
    return results


def allowed_J_values(L, S):
    """Clebsch-Gordan coupling: allowed total-J values run from |L-S| to
    L+S in integer steps."""
    if L < 0 or S < 0:
        raise ValueError("L and S must be non-negative")
    j_min, j_max = abs(L - S), L + S
    n_values = int(round(j_max - j_min)) + 1
    return [j_min + k for k in range(n_values)]


def term_symbol_string(L, S, J):
    """Build the spectroscopic term symbol ^(2S+1)L_J, e.g. L=1,S=0.5,J=1.5
    -> '2P3/2'."""
    if L < 0 or S < 0 or J < 0:
        raise ValueError("L, S, J must be non-negative")
    L_int = int(round(L))
    if L_int not in L_LETTERS:
        raise ValueError(f"no letter code defined for L={L}")
    multiplicity = int(round(2 * S + 1))
    J_frac = Fraction(int(round(2 * J)), 2)
    J_str = str(J_frac.numerator) if J_frac.denominator == 1 else f"{J_frac.numerator}/{J_frac.denominator}"
    return f"{multiplicity}{L_LETTERS[L_int]}{J_str}"


def proton_larmor_frequency_hz(B_tesla, gyromagnetic_ratio_mhz_per_t=PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T):
    """Same Larmor physics as dgs.electron_spin_resonance, with the
    proton's (not electron's) gyromagnetic ratio -- this IS the MRI
    resonance condition."""
    if B_tesla <= 0:
        raise ValueError("B_tesla must be positive")
    if gyromagnetic_ratio_mhz_per_t <= 0:
        raise ValueError("gyromagnetic_ratio_mhz_per_t must be positive")
    return gyromagnetic_ratio_mhz_per_t * 1e6 * B_tesla


def quantum_number_as_fraction(j):
    """Half-integer angular momentum quantum numbers represented EXACTLY
    as Python Fractions (via the doubled-integer 2j), rather than floats
    that can drift under repeated arithmetic."""
    if j < 0:
        raise ValueError("j must be non-negative")
    if abs(2 * j - round(2 * j)) > 1e-9:
        raise ValueError("j must be an integer or half-integer")
    return Fraction(int(round(2 * j)), 2)


if __name__ == "__main__":
    print("=== Is L, S, or J the sharp observable under spin-orbit coupling? ===\n")
    L, S = 1, 0.5   # a p-electron with spin
    results = verify_good_quantum_numbers(L, S)
    for name, r in results.items():
        status = "CONSERVED" if r["conserved"] else "NOT conserved"
        print(f"  [H_SO, {name}]  norm = {r['commutator_norm']:.2e}  -> {status}")
    print("\nExactly as expected: L_z, S_z are NOT conserved (m_l, m_s not good")
    print("quantum numbers); J_z, L^2, S^2, J^2 ARE conserved (l, s, j, m_j are).\n")

    print("=== Allowed J values and term symbols, L=1 (P), S=1/2 ===\n")
    for J in allowed_J_values(L, S):
        print(f"  J = {quantum_number_as_fraction(J)}  ->  term symbol: "
              f"{term_symbol_string(L, S, J)}")

    print("\n=== MRI: same Larmor physics as ESR, proton not electron ===\n")
    for B in [1.5, 3.0]:
        f = proton_larmor_frequency_hz(B)
        print(f"  B = {B} T  ->  proton Larmor frequency = {f/1e6:.2f} MHz "
              f"(real clinical scanner: {'~63.9' if B == 1.5 else '~127.7'} MHz)")

    print("\n=== Numeric-type note: half-integer J as exact Fraction vs float ===\n")
    j_float = 1.5
    j_frac = quantum_number_as_fraction(j_float)
    print(f"J as float: {j_float}  (exact equality checks risk float rounding)")
    print(f"J as Fraction: {j_frac}  (exact by construction -- safe for == checks)")
    print("(A future MATLAB port would use the same doubled-integer trick --")
    print(" MATLAB has no built-in Fraction type, so store 2*J as an integer.)")
