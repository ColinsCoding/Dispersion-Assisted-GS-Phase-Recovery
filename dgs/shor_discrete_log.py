"""Shor's algorithm, the OTHER half: discrete logarithm, not factoring --
and why that half, not the factoring half, is what actually threatens
Ethereum (and Bitcoin).

dgs/quantum_modulo.py already covers Shor's algorithm for FACTORING
(breaking RSA, whose security rests on the hardness of factoring N=p*q).
But Ethereum's signatures are ECDSA on the elliptic curve secp256k1,
whose security rests on a DIFFERENT hard problem: the elliptic curve
DISCRETE LOGARITHM problem (ECDLP) -- given a public key Q=x*G on the
curve, find the private key x. Shor's original 1994 paper actually
describes TWO algorithms, factoring and discrete log, and the discrete
log one is the one relevant here. This module implements the discrete-
log version for the simpler case of a prime-order cyclic group Z_p*
(multiplication mod a prime p) -- the direct, well-established stepping
stone to the elliptic-curve case, which replaces "multiply mod p" with
"add points on a curve" but keeps the SAME period-finding structure.

THE ALGORITHM: given g (a generator of a subgroup of prime order r) and
h = g^x mod p, define f(a,b) = g^a * h^(-b) mod p = g^(a - x*b) mod p.
This is constant along lines a - x*b = const (mod r) in the (a,b)
plane -- a 2D generalization of the 1D periodicity Shor's factoring
algorithm exploits. The quantum step (2D QFT) turns "which line" into a
frequency-domain peak at (k1,k2) satisfying k2 ≡ -x*k1 (mod r); solving
for x from a SINGLE such peak (r prime, so k1 has a modular inverse) via
x = -k2 * k1^{-1} mod r.

WHAT THIS MODULE IS, AND ISN'T: a classical simulation of the algorithm
STRUCTURE (like quantum_modulo.shor_factor_classical is for factoring),
verified against brute-force discrete log on small examples -- not a
real quantum circuit, and not an implementation of the actual elliptic-
curve group law ECDSA uses (a further, well-documented generalization
this module does not attempt). The real-world threat estimate for
secp256k1 itself is cited literature, not computed here: recent circuit
constructions estimate roughly 1200-2330 LOGICAL qubits (with tens of
millions of Toffoli gates) to break a 256-bit curve like secp256k1
(Roetteler et al. 2017, arXiv:1706.06752; Häner et al. 2020,
arXiv:2001.09580), and, after fault-tolerant error correction overhead,
current estimates for PHYSICAL qubits range from roughly 500,000 (given
future, not-yet-achieved gate error rates) up to the tens of millions
for a same-day break with today's best demonstrated error rates -- see
this module's secp256k1_quantum_threat_summary() for the exact figures
and sources, kept separate from anything this module actually computes.
"""
import numpy as np


def discrete_log_brute_force(g, h, p, r=None):
    """The trusted, independent baseline: try every x in [0, r) and check
    g^x == h (mod p). O(r) -- what a classical computer is stuck with."""
    if r is None:
        r = p - 1
    for x in range(r):
        if pow(g, x, p) == h:
            return x
    return None


def multiplicative_order(g, p):
    """The order of g in Z_p* (smallest r with g^r == 1 mod p) -- reused
    conceptually from dgs.quantum_modulo, restated here to keep this
    module's dependencies minimal."""
    r, val = 1, g % p
    while val != 1:
        val = (val * g) % p
        r += 1
        if r > p:
            return None
    return r


def find_prime_order_generator(p, r):
    """Find an element of Z_p* with EXACT multiplicative order r (r must
    divide p-1). Using a prime r keeps every nonzero frequency bin
    invertible mod r, which is what makes recovering x from a single
    2D-QFT peak below straightforward (the standard textbook
    simplification; a composite-order group needs combining several
    measurement samples via the CRT instead, not implemented here)."""
    if (p - 1) % r != 0:
        raise ValueError(f"r={r} must divide p-1={p-1}")
    for cand in range(2, p):
        if multiplicative_order(cand, p) == r:
            return cand
    raise ValueError(f"no element of order {r} found mod {p}")


def discrete_log_period_grid(g, h, p, r):
    """The 2D array f(a,b) = g^a * h^(-b) mod p, a,b in [0,r) -- the
    function Shor's discrete-log algorithm period-finds on. Verified
    (in tests) to satisfy f(a,b) == g^((a-x*b) mod r) exactly, for the
    true x with h=g^x, before trusting anything built on top of it."""
    h_inv = pow(h, p - 2, p)   # h^{-1} mod p via Fermat's little theorem (p prime)
    F = np.zeros((r, r))
    for a in range(r):
        for b in range(r):
            F[a, b] = (pow(g, a, p) * pow(h_inv, b, p)) % p
    return F


def shor_discrete_log_classical(g, h, p, r, a0=None, b0=None, seed=0):
    """Classically simulate the 2D-QFT step of Shor's discrete-log
    algorithm and recover x, given g (order r, r prime), h=g^x mod p.

    a0, b0 pick which measurement outcome to simulate collapsing to
    (defaults to a random point, standing in for the genuinely random
    outcome a real quantum measurement would give) -- verified in tests
    to recover the correct x regardless of which (a0,b0) is used.
    """
    if not _is_prime(r):
        raise ValueError("this classical demonstration requires a PRIME order r "
                          "(composite r needs combining multiple measurement samples via CRT, not implemented here)")
    rng = np.random.default_rng(seed)
    if a0 is None:
        a0 = int(rng.integers(0, r))
    if b0 is None:
        b0 = int(rng.integers(0, r))

    F = discrete_log_period_grid(g, h, p, r)
    target = F[a0, b0]
    comb = (F == target).astype(complex)   # the "collapsed state": all (a,b) consistent with the measured value

    power = np.abs(np.fft.fft2(comb)) ** 2
    threshold = 1e-6 * power.max()
    candidates = set()
    for k1 in range(r):
        for k2 in range(r):
            if power[k1, k2] > threshold and k1 != 0:
                candidates.add((-k2 * pow(k1, -1, r)) % r)

    return {
        "candidates": candidates,
        "a0": a0, "b0": b0,
        "unique": len(candidates) == 1,
        "x_recovered": next(iter(candidates)) if len(candidates) == 1 else None,
    }


def _is_prime(n):
    if n < 2:
        return False
    for k in range(2, int(n ** 0.5) + 1):
        if n % k == 0:
            return False
    return True


def secp256k1_quantum_threat_summary():
    """Cited literature estimates for breaking Ethereum/Bitcoin's actual
    curve (secp256k1, a 256-bit elliptic curve) via the REAL elliptic-
    curve generalization of the algorithm demonstrated above -- NOT
    computed by this module, stated here only as sourced facts so the
    toy demonstration above isn't mistaken for a real threat estimate."""
    return {
        "curve": "secp256k1 (used by Bitcoin and Ethereum ECDSA signatures)",
        "hard_problem": "elliptic curve discrete logarithm problem (ECDLP)",
        "logical_qubits_estimate": "~1200-2330 (varies by circuit construction)",
        "gate_count_estimate": "~70-90 million Toffoli gates",
        "physical_qubits_estimate": "~500,000 (optimistic future error rates) to ~13 million (today's demonstrated error rates, same-day break)",
        "sources": [
            "Roetteler, Naehrig, Svore, Lauter (2017), 'Quantum Resource Estimates for Computing "
            "Elliptic Curve Discrete Logarithms', arXiv:1706.06752",
            "Haner, Jaques, Naehrig, Roetteler, Soeken (2020), 'Improved Quantum Circuits for "
            "Elliptic Curve Discrete Logarithms', arXiv:2001.09580",
        ],
        "caveat": ("These figures describe the REAL elliptic-curve group law on secp256k1. The "
                   "classical demonstration in this module (shor_discrete_log_classical) solves "
                   "the simpler Z_p* multiplicative-group case, the standard pedagogical stepping "
                   "stone -- it is not itself an implementation of, or threat estimate for, ECDSA."),
    }


if __name__ == "__main__":
    print("=== SHOR'S ALGORITHM, DISCRETE LOG HALF ===\n")

    p, r = 23, 11
    g = find_prime_order_generator(p, r)
    print(f"working in a prime-order-{r} subgroup of Z_{p}*, generator g={g}")

    x_true = 7
    h = pow(g, x_true, p)
    print(f"h = g^x mod p = {h}  (secret x = {x_true}, to be recovered)")

    x_brute = discrete_log_brute_force(g, h, p, r)
    print(f"\nbrute-force discrete log (trusted baseline): x = {x_brute}")
    assert x_brute == x_true

    result = shor_discrete_log_classical(g, h, p, r, seed=1)
    print(f"Shor discrete-log (classical simulation of the 2D-QFT step): "
          f"x = {result['x_recovered']}  (measurement sample a0={result['a0']}, b0={result['b0']})")
    assert result["x_recovered"] == x_true
    print("[OK] recovered the correct discrete log from the period-finding structure")

    print("\n=== WHY THIS MATTERS FOR ETHEREUM: CITED, NOT COMPUTED ===")
    threat = secp256k1_quantum_threat_summary()
    print(f"curve: {threat['curve']}")
    print(f"hard problem: {threat['hard_problem']}")
    print(f"logical qubits (literature estimate): {threat['logical_qubits_estimate']}")
    print(f"physical qubits (literature estimate): {threat['physical_qubits_estimate']}")
    for s in threat["sources"]:
        print(f"  source: {s}")
    print(f"\ncaveat: {threat['caveat']}")

    print("\n=== DONE ===")
