"""Quantum error correction, from parity checks to a matching decoder -- and the
refrigerator it runs in.

  * REPETITION / BIT-FLIP CODE. Stabilizers Z1Z2, Z2Z3; syndromes locate one
    X error. A distance-d code fails when more than half the qubits flip:
        p_L = sum_{k > d/2} C(d,k) p^k (1-p)^{d-k};   d=3: p_L = 3p^2 - 2p^3.
    Below the crossover, bigger codes are better: a THRESHOLD.
  * PHASE-FLIP CODE. Conjugate by Hadamards: Z errors become X errors.
  * HAMMING [7,4] / STEANE. Parity-check matrix whose columns are 1..7 in
    binary: the syndrome of a single error IS the error position (GF(2)).
    Steane's quantum code uses the same matrix for both X and Z checks.
  * DECODING IS A GRAPH PROBLEM. With noisy syndrome measurements, record
    detection events (syndrome changes) in space-time and pair them up by
    minimum-weight perfect matching (networkx). Logical error rate vs physical
    error rate for d = 3, 5, 7 crosses at the threshold.
  * OVERHEAD. Surface-code heuristic p_L ~ A (p/p_th)^((d+1)/2) -> physical
    qubits 2 d^2 - 1 per logical qubit for a target p_L.
  * THE HEAT ENGINE. QEC pumps entropy out of the qubits; the refrigerator
    pumps the heat out of the cryostat. Carnot integrals in SymPy, the
    refrigerator's COP T_c / (T_h - T_c), and the Landauer floor for resetting
    syndrome ancillas.

All free software (numpy, sympy, networkx). Synthetic Monte Carlo. Education.
"""
import itertools

import numpy as np
import sympy as sp

K_B = 1.380649e-23

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], float)
Z = np.array([[1, 0], [0, -1]], float)
Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def kron(*ops):
    out = np.array([[1.0]])
    for o in ops:
        out = np.kron(out, o)
    return out


def op_on(op, q, n):
    return kron(*[op if i == q else I2 for i in range(n)])


# ── 1. three-qubit bit-flip code ─────────────────────────────────────

def bit_flip_stabilizers():
    return kron(Z, Z, I2), kron(I2, Z, Z)


def syndrome_table():
    """Syndrome (eigenvalues of Z1Z2, Z2Z3) for no error and each single X error."""
    S1, S2 = bit_flip_stabilizers()
    zero_L = np.zeros(8)
    zero_L[0] = 1
    table = {}
    for name, E in [("none", np.eye(8))] + [(f"X on qubit {q}", op_on(X, q, 3)) for q in range(3)]:
        psi = E @ zero_L
        table[name] = (int(round(psi @ S1 @ psi)), int(round(psi @ S2 @ psi)))
    return table


def correction_for(syndrome):
    lut = {(1, 1): None, (-1, 1): 0, (-1, -1): 1, (1, -1): 2}
    return lut[tuple(syndrome)]


def encode_bit_flip(alpha, beta):
    psi = np.zeros(8, complex)
    psi[0], psi[7] = alpha, beta
    return psi


def run_bit_flip(alpha, beta, p, rng=None):
    """Encode, apply independent X errors with probability p, measure the
    stabilizers (deterministic on X-error states), correct.
    Returns (fidelity, logical_error). A logical error leaves X_L|psi>, which
    can still overlap the original strongly (|2 Re(a b*)|^2), so 1 - fidelity
    UNDER-counts failures; logical_error checks for X_L|psi> directly."""
    rng = np.random.default_rng(rng)
    psi0 = encode_bit_flip(alpha, beta)
    psi = psi0.copy()
    for q in range(3):
        if rng.random() < p:
            psi = op_on(X, q, 3) @ psi
    S1, S2 = bit_flip_stabilizers()
    syn = (int(np.sign(np.real(psi.conj() @ S1 @ psi))), int(np.sign(np.real(psi.conj() @ S2 @ psi))))
    q = correction_for(syn)
    if q is not None:
        psi = op_on(X, q, 3) @ psi
    XL = kron(X, X, X)
    logical_err = bool(np.allclose(psi, XL @ psi0)) and not np.allclose(XL @ psi0, psi0)
    return float(abs(np.vdot(psi0, psi)) ** 2), logical_err


def logical_error_symbolic(d=3):
    """p_L = sum_{k=(d+1)/2}^{d} C(d,k) p^k (1-p)^(d-k) for odd d."""
    if d < 1 or d % 2 == 0:
        raise ValueError("d must be odd and positive")
    p = sp.symbols("p", positive=True)
    return sp.expand(sum(sp.binomial(d, kk) * p**kk * (1 - p) ** (d - kk) for kk in range((d + 1) // 2, d + 1)))


def code_capacity_crossover():
    """Solve p_L(d=3) = p: the code helps only for p < crossover."""
    p = sp.symbols("p", positive=True)
    sols = sp.solve(sp.Eq(logical_error_symbolic(3), p), p)
    return sorted(s for s in sols if 0 < s < 1)


def logical_error(p, d):
    from math import comb
    return sum(comb(d, kk) * p**kk * (1 - p) ** (d - kk) for kk in range((d + 1) // 2, d + 1))


# ── 2. phase-flip code ───────────────────────────────────────────────

def hadamard_conjugation_check():
    """H X H = Z and H Z H = X; so H^{(x)3} maps bit-flip stabilizers ZZ to XX."""
    Hs = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    Xs, Zs = sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[1, 0], [0, -1]])
    return sp.simplify(Hs * Xs * Hs - Zs), sp.simplify(Hs * Zs * Hs - Xs)


# ── 3. Hamming [7,4] and the Steane code (GF(2) linear algebra) ──────

def hamming_parity_check():
    """3x7 matrix whose column j (1-based) is j in binary (MSB first)."""
    return np.array([[(j >> b) & 1 for j in range(1, 8)] for b in (2, 1, 0)], dtype=int)


def hamming_syndrome(error_vec):
    """H e mod 2, read as a binary number = position (1..7) of a single error, 0 = none."""
    s = hamming_parity_check() @ np.asarray(error_vec, int) % 2
    return int(s[0] * 4 + s[1] * 2 + s[2])


def hamming_codewords():
    """All 16 codewords: null space of H over GF(2) (brute force over 2^7)."""
    Hm = hamming_parity_check()
    return [np.array(v) for v in itertools.product((0, 1), repeat=7) if not (Hm @ np.array(v) % 2).any()]


def min_distance(codewords):
    w = [int(c.sum()) for c in codewords if c.any()]
    return min(w)


def steane_check_commutation():
    """Steane code: X-type and Z-type stabilizers from the same Hamming rows.
    An X-check and a Z-check commute iff their supports overlap evenly;
    H H^T = 0 (mod 2) guarantees it."""
    Hm = hamming_parity_check()
    return Hm @ Hm.T % 2


# ── 4. decoding as graph matching, with measurement errors ───────────

def sample_repetition_history(d, rounds, p, q, rng):
    """Phenomenological noise on a distance-d repetition code: each round every
    data qubit flips with prob p, every one of the d-1 parity measurements is
    wrong with prob q; a final perfect round reads the data directly.
    Returns detection events (round, check) and the true final data flips."""
    data = np.zeros(d, dtype=int)
    prev = np.zeros(d - 1, dtype=int)
    events = []
    for r in range(rounds):
        data ^= (rng.random(d) < p).astype(int)
        synd = (data[:-1] ^ data[1:]) ^ (rng.random(d - 1) < q).astype(int)
        for c in np.nonzero(synd ^ prev)[0]:
            events.append((r, int(c)))
        prev = synd
    final = data[:-1] ^ data[1:]                      # perfect final readout
    for c in np.nonzero(final ^ prev)[0]:
        events.append((rounds, int(c)))
    return events, data


def matching_decode(events, d, rounds):
    """Minimum-weight perfect matching on detection events. Each event may also
    match to the nearer code boundary (left of check 0 or right of check d-2).
    Edge weight = space distance + time distance. Returns the set of data
    qubits the decoder flips."""
    import networkx as nx
    G = nx.Graph()
    for i, (r1, c1) in enumerate(events):
        for j in range(i + 1, len(events)):
            r2, c2 = events[j]
            G.add_edge(("e", i), ("e", j), weight=-(abs(c1 - c2) + abs(r1 - r2)))
        # private boundary node per event
        wl, wr = c1 + 1, (d - 1) - c1
        G.add_edge(("e", i), ("b", i), weight=-min(wl, wr))
    for i in range(len(events)):                     # boundary nodes may pair freely at zero cost
        for j in range(i + 1, len(events)):
            G.add_edge(("b", i), ("b", j), weight=0)
    M = nx.max_weight_matching(G, maxcardinality=True)
    flips = np.zeros(d, dtype=int)
    for u, v in M:
        if u[0] == "b" and v[0] == "b":
            continue
        if u[0] == "b":
            u, v = v, u
        c1 = events[u[1]][1]
        if v[0] == "b":                               # to the nearer boundary
            qubits = range(0, c1 + 1) if c1 + 1 <= (d - 1) - c1 else range(c1 + 1, d)
        else:
            c2 = events[v[1]][1]
            lo, hi = sorted((c1, c2))
            qubits = range(lo + 1, hi + 1)
        for qq in qubits:
            flips[qq] ^= 1
    return flips


def logical_failure(d, rounds, p, q, rng):
    events, data = sample_repetition_history(d, rounds, p, q, rng)
    corrected = data ^ matching_decode(events, d, rounds)
    # after correction the data must be all-equal; a logical error = all ones
    if corrected.any() and not corrected.all():
        raise RuntimeError("decoder left a non-codeword")
    return bool(corrected.all())


def threshold_sweep(ds=(3, 5, 7), ps=(0.02, 0.05, 0.08, 0.11, 0.14), trials=400, seed=0):
    """Logical failure rate vs physical error rate (p = q, rounds = d)."""
    rng = np.random.default_rng(seed)
    out = {}
    for d in ds:
        out[d] = [np.mean([logical_failure(d, d, p, p, rng) for _ in range(trials)]) for p in ps]
    return np.asarray(ps), out


# ── 5. overhead ──────────────────────────────────────────────────────

def surface_code_distance(p, target_pL, p_th=1e-2, A=0.1):
    """Smallest odd d with A (p/p_th)^((d+1)/2) <= target (heuristic scaling)."""
    if not 0 < p < p_th:
        raise ValueError("need 0 < p < p_th")
    d = 3
    while A * (p / p_th) ** ((d + 1) / 2) > target_pL:
        d += 2
    return d


def surface_code_qubits(d):
    """Rotated surface code: d^2 data + d^2 - 1 measurement qubits."""
    return 2 * d * d - 1


# ── 6. the heat engine and the refrigerator ──────────────────────────

def carnot_cycle_symbolic():
    """Ideal-gas Carnot cycle: isothermal expansion at T_h (V1->V2), adiabat to
    T_c (V2->V3), isothermal compression (V3->V4), adiabat back. Integrals of
    p dV for the isotherms, adiabat relation T V^(gamma-1) = const.
    Returns (Q_h, Q_c, efficiency)."""
    n, R, Th, Tc, V1, V2, V, g = sp.symbols("n R T_h T_c V_1 V_2 V gamma", positive=True)
    Qh = sp.expand_log(sp.integrate(n * R * Th / V, (V, V1, V2)), force=True)
    V3 = V2 * (Th / Tc) ** (1 / (g - 1))
    V4 = V1 * (Th / Tc) ** (1 / (g - 1))
    Qc = sp.simplify(sp.expand_log(-sp.integrate(n * R * Tc / V, (V, V3, V4)), force=True))
    eta = sp.simplify(sp.expand_log((Qh - Qc) / Qh, force=True))
    return sp.factor(Qh), sp.factor(Qc), eta


def carnot_cop_symbolic():
    """Refrigerator run backwards: COP = Q_c / W = T_c / (T_h - T_c)."""
    Th, Tc = sp.symbols("T_h T_c", positive=True)
    eta = 1 - Tc / Th
    return sp.simplify((1 - eta) / eta)


def wall_power_for_cooling(Q_c_W, T_c, T_h=300.0, carnot_fraction=1.0):
    """Room-temperature work needed to remove Q_c at T_c (ideal Carnot unless
    carnot_fraction < 1)."""
    if T_c <= 0 or T_c >= T_h or not 0 < carnot_fraction <= 1:
        raise ValueError("need 0 < T_c < T_h and 0 < carnot_fraction <= 1")
    return Q_c_W * (T_h - T_c) / T_c / carnot_fraction


def qec_landauer_heat(n_logical, d, cycle_hz, T_c):
    """Landauer floor for resetting syndrome ancillas: (d^2 - 1) ancillas per
    logical qubit per cycle, k T ln 2 each, dumped at the cold stage [W]."""
    return n_logical * (d * d - 1) * cycle_hz * K_B * T_c * np.log(2)
