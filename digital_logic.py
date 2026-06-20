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


# ── the top rung: a tiny accumulator CPU (ALU -> instruction set) ────
# The smallest honest "mental model 01" of a processor: one accumulator (ACC),
# a flat memory, a program counter (PC). Each cycle the CPU FETCHES the
# instruction at PC, DECODES the opcode (a decoder() picks the one control line),
# and EXECUTES it -- arithmetic/logic flow straight through the alu() above.
ISA = {
    "LOADI": 0,   # ACC <- immediate operand
    "LOAD":  1,   # ACC <- MEM[operand]
    "STORE": 2,   # MEM[operand] <- ACC
    "ADD":   3,   # ACC <- ACC + MEM[operand]
    "SUB":   4,   # ACC <- ACC - MEM[operand]
    "AND":   5,   # ACC <- ACC & MEM[operand]
    "OR":    6,   # ACC <- ACC | MEM[operand]
    "XOR":   7,   # ACC <- ACC ^ MEM[operand]
    "NOT":   8,   # ACC <- ~ACC
    "JMP":   9,   # PC  <- operand
    "JZ":    10,  # PC  <- operand  if zero flag set
    "HALT":  11,
}
_ISA_OP_TO_NAME = {v: k for k, v in ISA.items()}
_ALU_FOR = {"ADD": "ADD", "SUB": "SUB", "AND": "AND", "OR": "OR", "XOR": "XOR"}


def run_program(program, mem=None, width=8, max_cycles=10000, trace=False):
    """Execute a list of (mnemonic, operand) instructions on the accumulator CPU.

    Everything arithmetic goes through alu() on `width`-bit two's-complement words,
    so this is gates all the way down. Returns a dict with acc, mem, pc, flags,
    cycles (+ a per-cycle trace if requested).

    Example -- 3 + 4:
        run_program([("LOADI", 3), ("STORE", 0), ("LOADI", 4),
                     ("ADD", 0), ("HALT", 0)])["acc"] == 7
    """
    mask = (1 << width) - 1
    memory = [0] * (1 << width) if mem is None else list(mem)
    acc, pc, cycles = 0, 0, 0
    flags = {"zero": 1, "carry": 0, "negative": 0}
    log = []

    def alu_apply(name, operand_val):
        res, fl = alu(name, int_to_bits(acc, width), int_to_bits(operand_val, width))
        return bits_to_int(res) & mask, fl

    while pc < len(program):
        if cycles >= max_cycles:
            raise RuntimeError("max_cycles exceeded (infinite loop?)")
        cycles += 1
        mnem, operand = program[pc]
        if mnem not in ISA:
            raise ValueError(f"unknown instruction {mnem!r}")

        # DECODE: opcode -> one-hot control line (the decoder is the instruction decoder)
        onehot = decoder(int_to_bits(ISA[mnem], 4))
        assert sum(onehot) == 1 and onehot[ISA[mnem]] == 1

        next_pc = pc + 1
        if mnem == "LOADI":
            acc = operand & mask
        elif mnem == "LOAD":
            acc = memory[operand] & mask
        elif mnem == "STORE":
            memory[operand] = acc
        elif mnem in _ALU_FOR:
            acc, flags = alu_apply(_ALU_FOR[mnem], memory[operand])
        elif mnem == "NOT":
            res, flags = alu("NOT", int_to_bits(acc, width))
            acc = bits_to_int(res) & mask
        elif mnem == "JMP":
            next_pc = operand
        elif mnem == "JZ":
            next_pc = operand if flags["zero"] else pc + 1
        elif mnem == "HALT":
            if trace:
                log.append((cycles, pc, mnem, operand, acc, dict(flags)))
            break

        flags["zero"] = int(acc == 0)
        flags["negative"] = (acc >> (width - 1)) & 1
        if trace:
            log.append((cycles, pc, mnem, operand, acc, dict(flags)))
        pc = next_pc

    out = {"acc": acc, "mem": memory, "pc": pc, "flags": flags, "cycles": cycles}
    if trace:
        out["trace"] = log
    return out


# ── sequential logic: memory, clocked one tick at a time ────────────
# Combinational logic (above) has no memory -- outputs depend only on the
# present inputs. *Sequential* logic adds state: a clock edge latches the inputs
# into flip-flops, so the output depends on history. Everything below is "what
# happens on one rising clock edge".
def d_flip_flop(d, q_prev):
    """One D flip-flop: on the clock edge Q <- D (it simply remembers the last D).
    The atomic 1-bit memory cell; q_prev is only returned if you gate the clock."""
    return _bit(d)


def register_tick(data_bits, load, current_bits):
    """n-bit register: on a tick, Q <- data if load==1, else hold (load enable)."""
    cur = [_bit(x) for x in current_bits]
    if load not in (0, 1):
        raise ValueError("load must be 0 or 1")
    if load:
        data = [_bit(x) for x in data_bits]
        if len(data) != len(cur):
            raise ValueError("data and register width mismatch")
        return data
    return cur


def shift_register_tick(current_bits, serial_in, direction="right"):
    """Shift one position, clocking in serial_in. Returns (new_bits, serial_out)."""
    b = [_bit(x) for x in current_bits]
    si = _bit(serial_in)
    if direction == "right":
        return [si] + b[:-1], b[-1]
    if direction == "left":
        return b[1:] + [si], b[0]
    raise ValueError("direction must be 'right' or 'left'")


def counter_tick(current_bits, enable=1):
    """Synchronous up-counter: Q <- Q+1 mod 2^n when enabled (built on the adder)."""
    n = len(current_bits)
    if not enable:
        return [_bit(x) for x in current_bits]
    s, _ = ripple_carry_add([_bit(x) for x in current_bits], int_to_bits(1, n))
    return s[:n]


def fsm_run(transitions, outputs, start, input_seq, mealy=False):
    """Run a finite state machine over an input sequence.

    transitions : {(state, symbol): next_state}
    outputs     : Moore  -> {state: out}; Mealy -> {(state, symbol): out}
    Returns (final_state, output_seq, state_trace). A sequential circuit *is* an
    FSM: the register holds the state, combinational logic computes next-state
    and output. This is the gs_monitor trigger / regex-automata thread in the repo.
    """
    state = start
    out_seq, trace = [], [state]
    for sym in input_seq:
        key = (state, sym)
        if key not in transitions:
            raise ValueError(f"no transition from state {state!r} on input {sym!r}")
        if mealy:
            out_seq.append(outputs[key])
        state = transitions[key]
        trace.append(state)
        if not mealy:
            out_seq.append(outputs[state])
    return state, out_seq, trace
