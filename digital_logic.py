"""Digital logic: binary adders, Boolean gate algebra, CMOS, and Gray code.

Half/full adders built from gates, the ripple-carry adder assembled with a loop,
carry-lookahead generate/propagate, Gray-code conversion, and SymPy Boolean
verification that the gate-level sum/carry equal arithmetic addition. Standalone
(EE/CS, not the griffiths physics engine).
"""

import sympy as sp
from sympy import symbols
from sympy.logic.boolalg import Xor, simplify_logic


# ── gate-level adders (operate on 0/1 ints) ─────────────────────────
def half_adder(a, b):
    """Half adder: sum = a XOR b, carry = a AND b."""
    return (a ^ b, a & b)


def full_adder(a, b, cin):
    """Full adder from two half adders; cout = OR of the two carries."""
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    return (s2, c1 | c2)


def ripple_carry_add(a_bits, b_bits):
    """N-bit ripple-carry adder. a_bits, b_bits are LSB-first lists of 0/1.

    Returns (sum_bits_LSB_first, final_carry). The carry ripples through a loop --
    the structure (and the latency) of the simplest hardware adder.
    """
    if any(x not in (0, 1) for x in a_bits + b_bits):
        raise ValueError("inputs must be lists of bits (0 or 1)")
    n = max(len(a_bits), len(b_bits))
    a = list(a_bits) + [0] * (n - len(a_bits))
    b = list(b_bits) + [0] * (n - len(b_bits))
    carry, out = 0, []
    for i in range(n):
        s, carry = full_adder(a[i], b[i], carry)
        out.append(s)
    return out, carry


# ── integer <-> bit-list helpers ────────────────────────────────────
def bits_to_int(bits):
    """LSB-first bit list -> integer."""
    return sum(b << i for i, b in enumerate(bits))


def int_to_bits(x, n):
    """Integer -> LSB-first bit list of width n."""
    if x < 0:
        raise ValueError("x must be non-negative")
    return [(x >> i) & 1 for i in range(n)]


# ── carry-lookahead (the faster adder) ──────────────────────────────
def carry_lookahead(a_bits, b_bits):
    """Carry-lookahead generate/propagate: g_i = a_i & b_i, p_i = a_i ^ b_i,
    c_{i+1} = g_i | (p_i & c_i). Computes all carries, returns (sum_bits, cout).

    Same result as ripple-carry but the carries are expressed in parallel -- the
    basis of fast adders (log-depth instead of linear)."""
    n = max(len(a_bits), len(b_bits))
    a = list(a_bits) + [0] * (n - len(a_bits))
    b = list(b_bits) + [0] * (n - len(b_bits))
    g = [a[i] & b[i] for i in range(n)]
    p = [a[i] ^ b[i] for i in range(n)]
    carries = [0] * (n + 1)
    for i in range(n):
        carries[i + 1] = g[i] | (p[i] & carries[i])
    out = [p[i] ^ carries[i] for i in range(n)]
    return out, carries[n]


# ── Gray code ───────────────────────────────────────────────────────
def to_gray(n):
    """Binary -> reflected Gray code: g = n XOR (n >> 1)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n ^ (n >> 1)


def from_gray(g):
    """Gray code -> binary (cumulative XOR of the prefix)."""
    if g < 0:
        raise ValueError("g must be non-negative")
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


# ── SymPy Boolean verification ──────────────────────────────────────
def adder_boolean():
    """Symbolic full-adder equations. Returns (sum_expr, carry_expr) with the
    carry simplified to the majority function."""
    a, b, cin = symbols("a b c_in")
    s = Xor(a, b, cin)
    cout = simplify_logic((a & b) | (b & cin) | (a & cin))
    return s, cout


# ── CMOS gate cost ──────────────────────────────────────────────────
CMOS_TRANSISTORS = {
    "NOT": 2, "NAND2": 4, "NOR2": 4, "AND2": 6, "OR2": 6,
    "XOR2": 12, "XNOR2": 12,            # static CMOS XOR ~ 12T (or 8-10T optimised)
}


def cmos_cost(gate):
    """Transistor count of a static-CMOS gate (NAND/NOR are the cheap primitives)."""
    g = gate.upper()
    if g not in CMOS_TRANSISTORS:
        raise ValueError(f"unknown gate {gate!r}; known: {sorted(CMOS_TRANSISTORS)}")
    return CMOS_TRANSISTORS[g]


# ── the 7 basic gates (operate on 0/1 ints) ─────────────────────────
def _bit(x):
    if x not in (0, 1):
        raise ValueError("gate inputs must be 0 or 1")
    return x


def NOT(a):  return 1 - _bit(a)
def AND(a, b):  return _bit(a) & _bit(b)
def OR(a, b):   return _bit(a) | _bit(b)
def NAND(a, b): return 1 - (_bit(a) & _bit(b))
def NOR(a, b):  return 1 - (_bit(a) | _bit(b))
def XOR(a, b):  return _bit(a) ^ _bit(b)
def XNOR(a, b): return 1 - (_bit(a) ^ _bit(b))

# the canonical seven; NAND and NOR are each functionally complete (universal)
GATES = {"NOT": NOT, "AND": AND, "OR": OR, "NAND": NAND, "NOR": NOR,
         "XOR": XOR, "XNOR": XNOR}


def gate(name, *inputs):
    """Apply a named basic gate. NOT takes 1 input, the rest take 2."""
    g = GATES.get(name.upper())
    if g is None:
        raise ValueError(f"unknown gate {name!r}; known: {sorted(GATES)}")
    return g(*inputs)


def truth_table(name):
    """Return the truth table of a basic gate as a list of (inputs..., output)."""
    import itertools
    arity = 1 if name.upper() == "NOT" else 2
    return [(*ins, gate(name, *ins)) for ins in itertools.product((0, 1), repeat=arity)]


# ── decoder / encoder / mux (routing and selecting) ─────────────────
def decoder(sel_bits):
    """n select bits (LSB-first) -> one-hot list of length 2^n.

    The address decoder: turns a binary address into a single asserted line.
    This is literally an instruction decoder when the address is an opcode.
    """
    idx = bits_to_int(sel_bits)
    out = [0] * (2 ** len(sel_bits))
    out[idx] = 1
    return out


def priority_encoder(lines):
    """2^n input lines -> (index-of-highest-set as n-bit LSB-first list, valid).

    The inverse of a decoder: collapses one-hot (or many-hot) back to a binary
    index, choosing the highest-priority (highest-index) active line. valid=0
    when no line is set.
    """
    if any(x not in (0, 1) for x in lines):
        raise ValueError("lines must be 0/1")
    n = (len(lines) - 1).bit_length()
    hi = -1
    for i, v in enumerate(lines):
        if v:
            hi = i
    if hi < 0:
        return [0] * n, 0
    return int_to_bits(hi, n), 1


def mux(data, sel_bits):
    """2^n data inputs, n select bits -> the selected data value (data[sel])."""
    idx = bits_to_int(sel_bits)
    if idx >= len(data):
        raise ValueError("select index out of range for data length")
    return data[idx]


def demux(value, sel_bits):
    """Route a single value to one of 2^n outputs chosen by the select bits."""
    out = [0] * (2 ** len(sel_bits))
    out[bits_to_int(sel_bits)] = value
    return out


# ── ALU + a tiny instruction set (gates -> adder -> ALU -> ISA) ──────
# Opcode table: the "instruction set" the ALU understands. Each opcode is a
# binary address that a decoder() turns into the one control line that gates
# the chosen datapath -- the mental model of how a CPU executes an instruction.
ALU_OPS = {0: "ADD", 1: "SUB", 2: "AND", 3: "OR", 4: "XOR", 5: "NOT", 6: "PASS"}
_ALU_NAME_TO_OP = {v: k for k, v in ALU_OPS.items()}


def _add_with_carry(a_bits, b_bits, cin):
    n = max(len(a_bits), len(b_bits))
    a = list(a_bits) + [0] * (n - len(a_bits))
    b = list(b_bits) + [0] * (n - len(b_bits))
    carry, out = cin, []
    for i in range(n):
        s, carry = full_adder(a[i], b[i], carry)
        out.append(s)
    return out, carry


def alu(op, a_bits, b_bits=None):
    """A 1-instruction-wide ALU. `op` is an opcode int or its name (see ALU_OPS).

    Returns (result_bits_LSB_first, flags) where flags has zero / carry /
    negative. SUB is two's-complement add (a + ~b + 1); AND/OR/XOR/NOT are the
    bitwise basic gates; ADD is the ripple-carry adder. This is the whole
    arithmetic/logic core of a processor in one function.
    """
    name = ALU_OPS[op] if isinstance(op, int) else op.upper()
    if name not in _ALU_NAME_TO_OP:
        raise ValueError(f"unknown ALU op {op!r}; known: {sorted(_ALU_NAME_TO_OP)}")
    a = list(a_bits)
    n = len(a)
    b = list(b_bits) if b_bits is not None else [0] * n
    b += [0] * (n - len(b))
    a += [0] * (len(b) - n)
    n = max(n, len(b))
    carry = 0

    if name == "ADD":
        res, carry = _add_with_carry(a, b, 0)
    elif name == "SUB":
        res, carry = _add_with_carry(a, [1 - x for x in b], 1)   # a + ~b + 1
    elif name == "AND":
        res = [AND(a[i], b[i]) for i in range(n)]
    elif name == "OR":
        res = [OR(a[i], b[i]) for i in range(n)]
    elif name == "XOR":
        res = [XOR(a[i], b[i]) for i in range(n)]
    elif name == "NOT":
        res = [NOT(a[i]) for i in range(n)]
    else:  # PASS
        res = a[:]

    flags = {"zero": int(all(x == 0 for x in res)),
             "carry": carry,
             "negative": res[-1] if res else 0}
    return res, flags
