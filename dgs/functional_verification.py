"""Functional verification: proving a design does what the spec says (EDA).

Synthesis turns intent into gates; VERIFICATION asks the opposite question -- does
the built thing actually match the spec? It is where most real chip effort goes,
and it splits by whether the design has MEMORY:

  * COMBINATIONAL (no state): output depends only on current inputs, so you can
    verify EXHAUSTIVELY -- check the design under test (DUT) against a golden
    reference on every input. For a small block this is a complete proof of
    equivalence (the idea behind formal equivalence checking).

  * SEQUENTIAL (has state / memory): output depends on the HISTORY of inputs, so
    a truth table is not enough -- you must drive input SEQUENCES and compare the
    DUT's observable output TRAJECTORY against the reference over time. The state
    space is 2^(flip-flops), far too large to enumerate, so real verification
    leans on constrained-random stimulus plus COVERAGE (did we exercise every
    state/transition?) and ASSERTIONS (did any property ever break?).

The point of a testbench is to CATCH BUGS, so this module also injects deliberate
bugs (a counter that saturates instead of wrapping; a sequence detector that fires
on the wrong pattern) and shows the harness flags them with the exact failing
sequence and cycle -- error checking that takes responsibility for the whole
input space, not a happy-path demo. Reuses dgs.digital_logic DUTs (the bit-level
counter, the FSM engine) and can verify the dgs.mux_design_views mux. NumPy-free,
pure Python; py-3.13.
"""

from itertools import product
import random

from dgs import digital_logic as dl


# ----------------------------------------------------------------------
# Combinational: exhaustive equivalence checking
# ----------------------------------------------------------------------

def exhaustive_equivalence(dut, ref, arg_domains):
    """Check dut(*args) == ref(*args) for EVERY combination of arguments drawn
    from arg_domains (a list of iterables). For a combinational block with a
    small input space this is a complete equivalence proof. Returns dict with
    passed, n_checked, and the list of mismatching (args, dut_out, ref_out)."""
    if not arg_domains:
        raise ValueError("need at least one argument domain")
    domains = [list(d) for d in arg_domains]
    mismatches = []
    n = 0
    for args in product(*domains):
        n += 1
        d, r = dut(*args), ref(*args)
        if d != r:
            mismatches.append((args, d, r))
    return {"passed": not mismatches, "n_checked": n, "mismatches": mismatches}


# ----------------------------------------------------------------------
# Sequential: drive sequences, compare observable output trajectories
# ----------------------------------------------------------------------

def run_sequential(step, init_state, inputs):
    """Clock a state machine: step(state, x) -> (next_state, output). Returns
    (state_trace, output_trace); state_trace includes the initial state, so it
    is one longer than output_trace. This is the register-plus-logic loop that
    every sequential circuit is."""
    state = init_state
    states = [state]
    outs = []
    for x in inputs:
        state, out = step(state, x)
        states.append(state)
        outs.append(out)
    return states, outs


def verify_sequential(dut_step, ref_step, init_state, input_seqs,
                      ref_init_state=None):
    """Black-box functional verification of stateful hardware: for each input
    sequence, run the DUT and the reference and compare their OBSERVABLE OUTPUT
    trajectories (internal state encodings may differ -- outputs are the spec).
    Returns dict with passed, n_sequences, and on failure the failing sequence
    and the first cycle where outputs diverge."""
    ref_init = init_state if ref_init_state is None else ref_init_state
    for seq in input_seqs:
        _, dut_out = run_sequential(dut_step, init_state, seq)
        _, ref_out = run_sequential(ref_step, ref_init, seq)
        for i, (a, b) in enumerate(zip(dut_out, ref_out)):
            if a != b:
                return {"passed": False, "n_sequences": len(input_seqs),
                        "failing_sequence": list(seq), "first_divergence": i,
                        "dut_output": a, "ref_output": b}
    return {"passed": True, "n_sequences": len(input_seqs),
            "failing_sequence": None, "first_divergence": None}


def constrained_random_sequences(n_seqs, length, alphabet=(0, 1), seed=0):
    """Constrained-random stimulus: n_seqs input sequences of the given length,
    each symbol drawn from alphabet. The workhorse of modern verification --
    you cannot enumerate the state space, so you sample it, reproducibly."""
    if n_seqs < 1 or length < 1:
        raise ValueError("n_seqs and length must be >= 1")
    rng = random.Random(seed)
    return [[rng.choice(alphabet) for _ in range(length)] for _ in range(n_seqs)]


# ----------------------------------------------------------------------
# Coverage and assertions
# ----------------------------------------------------------------------

def observed_transitions(step, init_state, input_seqs):
    """Collect the set of (state, input) transitions a stimulus set actually
    exercises on a reference machine -- the raw material for coverage."""
    seen = set()
    for seq in input_seqs:
        state = init_state
        for x in seq:
            seen.add((state, x))
            state, _ = step(state, x)
    return seen


def transition_coverage(all_transitions, observed):
    """Fraction of the machine's (state, input) transitions the tests hit.
    100% transition coverage is a real sign-off metric; the MISSED set tells
    you what stimulus is still needed."""
    all_t = set(all_transitions)
    if not all_t:
        raise ValueError("all_transitions is empty")
    hit = all_t & set(observed)
    return {"coverage": len(hit) / len(all_t), "missed": sorted(all_t - hit)}


def check_property(trace, predicate):
    """Assertion checker: does `predicate(item)` hold at every step of a trace?
    Returns (holds, first_violation_index) -- the cycle a property breaks, or
    None if it never does. This is the SystemVerilog-assertion idea in miniature."""
    for i, item in enumerate(trace):
        if not predicate(item):
            return {"holds": False, "first_violation": i}
    return {"holds": True, "first_violation": None}


# ----------------------------------------------------------------------
# Devices under test + golden references (with deliberate bugs)
# ----------------------------------------------------------------------

def counter_ref(n_bits):
    """Golden mod-2^n up-counter, arithmetic spec: step(s, en) -> (next, out)
    with next = (s+1) mod 2^n when enabled, output = current count."""
    N = 2 ** n_bits
    def step(s, en):
        return ((s + 1) % N if en else s), s
    return step


def counter_dut(n_bits):
    """The RTL counter under test: the bit-level ripple counter from
    dgs.digital_logic.counter_tick, wrapped to integer state. An INDEPENDENT
    implementation from counter_ref, so agreement is a real equivalence."""
    def step(s, en):
        bits = dl.int_to_bits(s, n_bits)
        nxt = dl.bits_to_int(dl.counter_tick(bits, enable=en))
        return nxt, s
    return step


def counter_buggy(n_bits):
    """A BUGGY counter that SATURATES at the top instead of wrapping to 0 --
    the classic off-by-one at the rollover boundary a testbench must catch."""
    N = 2 ** n_bits
    def step(s, en):
        if not en:
            return s, s
        return (s + 1 if s < N - 1 else s), s      # bug: no wrap
    return step


# "101" Mealy sequence detector: transition + output tables
SEQ101_TRANSITIONS = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 1,
                      (2, 0): 0, (2, 1): 1}
SEQ101_OUTPUTS = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0,
                  (2, 0): 0, (2, 1): 1}


def seq101_ref():
    """Golden '101' detector as an explicit 3-state Mealy FSM (S0, S1=saw 1,
    S2=saw 10); output 1 exactly when the incoming bit completes '101'."""
    def step(s, x):
        return SEQ101_TRANSITIONS[(s, x)], SEQ101_OUTPUTS[(s, x)]
    return step


def seq101_dut():
    """Independent '101' detector: a 2-bit shift register remembering the last
    two inputs, firing when (prev2, prev1, x) == (1, 0, 1). Different STATE
    encoding from seq101_ref, so matching outputs is genuine verification."""
    def step(state, x):
        p2, p1 = state
        out = 1 if (p2, p1, x) == (1, 0, 1) else 0
        return (p1, x), out
    return step


def seq101_buggy():
    """A BUGGY detector that fires on '111' instead of '101' -- catchable by
    any sequence containing the real pattern."""
    def step(state, x):
        p2, p1 = state
        out = 1 if (p2, p1, x) == (1, 1, 1) else 0
        return (p1, x), out
    return step


if __name__ == "__main__":
    from dgs import mux_design_views as mx

    # combinational: exhaustively verify the mux's gate netlist vs its spec
    res = exhaustive_equivalence(mx.mux2_structural, mx.mux2_behavioral,
                                 [(0, 1)] * 3)
    print(f"mux exhaustive equivalence: passed={res['passed']} "
          f"({res['n_checked']} input combinations checked)")

    # sequential: verify the RTL counter vs the arithmetic spec on random stimulus
    n = 3
    seqs = constrained_random_sequences(200, 30, alphabet=(0, 1), seed=1)
    good = verify_sequential(counter_dut(n), counter_ref(n), 0, seqs)
    print(f"counter DUT vs spec: passed={good['passed']} "
          f"over {good['n_sequences']} random enable sequences")

    # the buggy counter must be CAUGHT, with the failing cycle
    bad = verify_sequential(counter_buggy(n), counter_ref(n), 0,
                            [[1] * 10])          # count past the rollover
    print(f"buggy (saturating) counter caught? {not bad['passed']} "
          f"at cycle {bad['first_divergence']} "
          f"(DUT {bad['dut_output']} vs spec {bad['ref_output']})")

    # FSM: independent 101 detector vs the golden FSM, plus coverage + assertion
    seqs = constrained_random_sequences(300, 20, alphabet=(0, 1), seed=2)
    fsm_ok = verify_sequential(seq101_dut(), seq101_ref(), (0, 0), seqs,
                               ref_init_state=0)
    cov = transition_coverage(SEQ101_TRANSITIONS.keys(),
                              observed_transitions(seq101_ref(), 0, seqs))
    print(f"101 detector DUT vs FSM: passed={fsm_ok['passed']}, "
          f"transition coverage {cov['coverage']*100:.0f}%")
    caught = verify_sequential(seq101_buggy(), seq101_ref(), (0, 0),
                               [[1, 0, 1]], ref_init_state=0)
    print(f"buggy (111) detector caught? {not caught['passed']} "
          f"at cycle {caught['first_divergence']}")

    # assertion: the counter's output must always stay in [0, 2^n)
    _, outs = run_sequential(counter_ref(n), 0, [1] * 20)
    prop = check_property(outs, lambda v: 0 <= v < 2 ** n)
    print(f"assertion 'count in range' holds? {prop['holds']}")
