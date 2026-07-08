"""Integers as finite state machines: divisibility and addition over the bits.

A finite state machine has no memory beyond its current state -- yet with the RIGHT
state it can do arithmetic on arbitrarily long binary numbers. Two classic constructions
tie integers, booleans, and 01 strings together:

  DIVISIBILITY BY N. Read a binary number most-significant-bit first. Appending a bit b
  updates the value to 2*value + b, so its REMAINDER mod N updates the same way:
        state' = (2 * state + b) mod N.
  Keep only the remainder (one of N states) and you have a DFA that ACCEPTS exactly the
  binary strings that are multiples of N -- checking divisibility of an unbounded number
  with N states of memory. The accepting state is remainder 0.

  ADDITION. A serial adder reads two numbers least-significant-bit first; its only state
  is the CARRY (0 or 1). Each step is a full adder:
        sum_bit = (a + b + carry) mod 2,   carry' = (a + b + carry) // 2.
  Integer addition is a 2-state finite machine over the bits -- the carry is the state,
  the sum bit is the output.

So "combine integers, boolean, 01, FSM" is literally this: an integer property (divisible?
sum?) computed by a machine whose alphabet is {0,1}, whose output is boolean, and whose
memory is a single bounded state. Verified against Python's own %/+ on many inputs.
Complements dgs.digital_logic.fsm_run and dgs.functional_verification (FSM checking).
Pure Python; py-3.13.
"""


# ----------------------------------------------------------------------
# Divisibility-by-N DFA: state = remainder mod N
# ----------------------------------------------------------------------

def divisibility_fsm(n):
    """Build the DFA recognizing binary multiples of N. Returns dict with states
    0..N-1, transition delta[(state, bit)] = (2*state + bit) % N, start state 0,
    and accepting set {0}. Reading MSB-first, the state IS the value mod N."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    delta = {(s, b): (2 * s + b) % n for s in range(n) for b in (0, 1)}
    return {"n_states": n, "delta": delta, "start": 0, "accept": {0}}


def run_fsm(machine, bits):
    """Run a DFA over a sequence of input symbols. Returns (final_state, accepted)."""
    state = machine["start"]
    for b in bits:
        key = (state, b)
        if key not in machine["delta"]:
            raise ValueError(f"no transition from {state} on {b!r}")
        state = machine["delta"][key]
    return state, state in machine["accept"]


def is_divisible_by(binary_string, n):
    """True iff the binary string (MSB-first) is a multiple of N, decided by the
    divisibility DFA -- one bit at a time, N states of memory, no big-int needed."""
    bits = _parse_bits(binary_string)
    _, accepted = run_fsm(divisibility_fsm(n), bits)
    return accepted


def remainder_trace(binary_string, n):
    """The DFA state (= value mod N) after each bit -- shows the state tracking the
    running remainder exactly. Returns the list of remainders."""
    bits = _parse_bits(binary_string)
    machine = divisibility_fsm(n)
    state = machine["start"]
    trace = []
    for b in bits:
        state = machine["delta"][(state, b)]
        trace.append(state)
    return trace


def _parse_bits(binary_string):
    bits = [int(c) for c in str(binary_string)]
    if any(b not in (0, 1) for b in bits):
        raise ValueError("binary string must contain only 0 and 1")
    return bits


# ----------------------------------------------------------------------
# Serial adder FSM: state = carry
# ----------------------------------------------------------------------

def serial_adder_step(carry, a_bit, b_bit):
    """One step of the serial adder (a full adder): returns (sum_bit, next_carry)
    for input bits a,b and the incoming carry. This is the entire state machine."""
    for v in (carry, a_bit, b_bit):
        if v not in (0, 1):
            raise ValueError("carry and bits must be 0 or 1")
    total = a_bit + b_bit + carry
    return total % 2, total // 2


def add_via_fsm(a, b):
    """Add two non-negative integers by running the serial adder over their bits,
    LSB first, carrying state between steps -- integer addition as a 2-state FSM.
    Returns the sum (equal to a + b)."""
    if a < 0 or b < 0:
        raise ValueError("a and b must be non-negative")
    carry = 0
    width = max(a.bit_length(), b.bit_length())
    result = 0
    for i in range(width):
        s_bit, carry = serial_adder_step(carry, (a >> i) & 1, (b >> i) & 1)
        result |= s_bit << i
    if carry:
        result |= 1 << width               # final carry-out is the top bit
    return result


if __name__ == "__main__":
    print("divisibility-by-3 DFA (state = value mod 3):")
    for s in ("110", "111", "1001", "0", "1111"):
        val = int(s, 2)
        print(f"  {s:>5} = {val:2d}: divisible by 3? {is_divisible_by(s, 3)}  "
              f"(actual {val % 3 == 0}),  remainder trace {remainder_trace(s, 3)}")

    print("\ndivisibility across several N on '11010' (=26):")
    for n in (2, 3, 5, 13):
        print(f"  divisible by {n:2d}? {is_divisible_by('11010', n)}  "
              f"(26 % {n} = {26 % n})")

    print("\nserial adder FSM (state = carry):")
    for a, b in [(13, 9), (255, 1), (0, 0), (170, 85)]:
        print(f"  {a} + {b} = {add_via_fsm(a, b)}  (== {a + b}? {add_via_fsm(a, b) == a + b})")
