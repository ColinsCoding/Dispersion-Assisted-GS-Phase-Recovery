"""Abstract algebra over finite sets: groups, rings, and fields.

Check the axioms by brute force (finite structures, so exhaustive verification
is honest), build Cayley tables, and construct the running examples -- Z/nZ,
the symmetric/parity groups, and the finite fields GF(p). Standalone (math/CS).
The point: each rung of group -> ring -> field adds an operation you can rely
on, and the same Z/26 ring and GF(p) fields power this repo's cipher and coding
work.
"""

import itertools


# ── the group axioms, checked exhaustively ──────────────────────────
def is_closed(elements, op):
    """True if `op` never produces a result outside `elements` (closure axiom)."""
    s = set(elements)
    return all(op(a, b) in s for a in elements for b in elements)


def is_associative(elements, op):
    """True if (a*b)*c == a*(b*c) for every triple (the associativity axiom)."""
    return all(op(op(a, b), c) == op(a, op(b, c))
               for a in elements for b in elements for c in elements)


def is_commutative(elements, op):
    """True if a*b == b*a for every pair (the abelian/commutative condition)."""
    return all(op(a, b) == op(b, a) for a in elements for b in elements)


def identity(elements, op):
    """Return the identity element, or None."""
    for e in elements:
        if all(op(e, a) == a and op(a, e) == a for a in elements):
            return e
    return None


def inverses(elements, op):
    """Return {a: a^-1} if every element is invertible, else None."""
    e = identity(elements, op)
    if e is None:
        return None
    inv = {}
    for a in elements:
        found = next((b for b in elements if op(a, b) == e and op(b, a) == e), None)
        if found is None:
            return None
        inv[a] = found
    return inv


def is_group(elements, op):
    """Closure + associativity + identity + inverses."""
    return (is_closed(elements, op) and is_associative(elements, op)
            and identity(elements, op) is not None
            and inverses(elements, op) is not None)


def is_abelian_group(elements, op):
    """A group whose operation also commutes (a*b == b*a for all a, b)."""
    return is_group(elements, op) and is_commutative(elements, op)


# ── rings and fields ────────────────────────────────────────────────
def is_ring(elements, add, mul):
    """(R,+) abelian group, (R,*) associative & closed, * distributes over +."""
    if not is_abelian_group(elements, add):
        return False
    if not (is_closed(elements, mul) and is_associative(elements, mul)):
        return False
    left = all(mul(a, add(b, c)) == add(mul(a, b), mul(a, c))
               for a in elements for b in elements for c in elements)
    right = all(mul(add(b, c), a) == add(mul(b, a), mul(c, a))
                for a in elements for b in elements for c in elements)
    return left and right


def is_field(elements, add, mul):
    """A commutative ring whose nonzero elements form a group under * (so you can
    divide -- no zero divisors)."""
    if not is_ring(elements, add, mul) or not is_commutative(elements, mul):
        return False
    zero = identity(elements, add)
    nonzero = [e for e in elements if e != zero]
    return bool(nonzero) and is_group(nonzero, mul)


def classify(elements, add, mul=None):
    """Summarise the structure: group/abelian, and (if mul given) ring/field."""
    out = {"is_group": is_group(elements, add),
           "is_abelian": is_abelian_group(elements, add)}
    if mul is not None:
        out["is_ring"] = is_ring(elements, add, mul)
        out["is_field"] = is_field(elements, add, mul)
    return out


def cayley_table(elements, op):
    """The operation table as a list of rows (indices into `elements`)."""
    idx = {e: i for i, e in enumerate(elements)}
    return [[idx[op(a, b)] for b in elements] for a in elements]


# ── the standard examples ───────────────────────────────────────────
def Zn(n):
    """Elements of Z/nZ."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return list(range(n))


def add_mod(n):
    """Return the addition-mod-n operation: (a, b) -> (a + b) % n."""
    return lambda a, b: (a + b) % n


def mul_mod(n):
    """Return the multiplication-mod-n operation: (a, b) -> (a * b) % n."""
    return lambda a, b: (a * b) % n


def is_prime(p):
    """Trial-division primality test: p is prime iff no divisor up to sqrt(p)."""
    return p > 1 and all(p % k for k in range(2, int(p**0.5) + 1))


def symmetric_group(n):
    """Permutations of {0,...,n-1} as tuples; the op is `compose`."""
    return list(itertools.permutations(range(n)))


def compose(p, q):
    """Permutation composition (p after q): (p o q)(i) = p[q[i]]."""
    return tuple(p[q[i]] for i in range(len(p)))


def parity_group():
    """The inversion/parity group {+1, -1} under multiplication, isomorphic to
    Z/2Z -- the group behind Griffiths' pseudovectors (eigenvalues = parity)."""
    return [1, -1], (lambda a, b: a * b)
