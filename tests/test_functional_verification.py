"""Test dgs.functional_verification: exhaustive combinational equivalence (and
that a wrong DUT fails), sequential verification of the RTL counter vs its
arithmetic spec, bug injection getting CAUGHT with the exact failing cycle,
reproducible constrained-random stimulus, transition coverage, and assertions.
Pure Python."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import functional_verification as fv
from dgs import mux_design_views as mx

# 1. combinational: the mux netlist is exhaustively equivalent to its spec...
res = fv.exhaustive_equivalence(mx.mux2_structural, mx.mux2_behavioral, [(0, 1)] * 3)
assert res["passed"] and res["n_checked"] == 8 and res["mismatches"] == []
# ...and a wrong DUT (ignores the select line) is flagged with mismatches
bad_mux = lambda a, b, s: a
res_bad = fv.exhaustive_equivalence(bad_mux, mx.mux2_behavioral, [(0, 1)] * 3)
assert not res_bad["passed"] and len(res_bad["mismatches"]) > 0
# every reported mismatch is real
for args, d, r in res_bad["mismatches"]:
    assert d != r and bad_mux(*args) == d and mx.mux2_behavioral(*args) == r

# 2. run_sequential: state trace is one longer than the output trace
states, outs = fv.run_sequential(fv.counter_ref(3), 0, [1, 1, 1])
assert len(states) == len(outs) + 1 == 4
assert outs == [0, 1, 2] and states == [0, 1, 2, 3]

# 3. sequential: the bit-level RTL counter matches the arithmetic spec on
#    random enable sequences (an independent-implementation equivalence)
seqs = fv.constrained_random_sequences(200, 30, alphabet=(0, 1), seed=1)
good = fv.verify_sequential(fv.counter_dut(3), fv.counter_ref(3), 0, seqs)
assert good["passed"] and good["first_divergence"] is None

# 4. bug injection is CAUGHT with the exact failing cycle and values
bad = fv.verify_sequential(fv.counter_buggy(3), fv.counter_ref(3), 0, [[1]*10])
assert not bad["passed"]
assert bad["first_divergence"] == 8         # 0..7 then rollover: buggy saturates
assert bad["dut_output"] == 7 and bad["ref_output"] == 0
assert bad["failing_sequence"] == [1]*10

# 5. constrained-random stimulus is reproducible (same seed -> same sequences)
a = fv.constrained_random_sequences(10, 12, seed=7)
b = fv.constrained_random_sequences(10, 12, seed=7)
c = fv.constrained_random_sequences(10, 12, seed=8)
assert a == b and a != c
assert len(a) == 10 and all(len(s) == 12 for s in a)

# 6. transition coverage: full when stimulus hits every (state,input),
#    partial with the correct missed set otherwise
full = fv.observed_transitions(fv.seq101_ref(), 0,
                               fv.constrained_random_sequences(300, 20, seed=2))
cov = fv.transition_coverage(fv.SEQ101_TRANSITIONS.keys(), full)
assert cov["coverage"] == 1.0 and cov["missed"] == []
thin = fv.observed_transitions(fv.seq101_ref(), 0, [[0, 0, 0]])   # only (0,0)
cov2 = fv.transition_coverage(fv.SEQ101_TRANSITIONS.keys(), thin)
assert abs(cov2["coverage"] - 1/6) < 1e-9
assert (0, 0) not in cov2["missed"] and (2, 1) in cov2["missed"]

# 7. FSM: independent '101' detector matches the golden FSM; the '111' bug is
#    caught on the very pattern it should have detected
fsm_ok = fv.verify_sequential(fv.seq101_dut(), fv.seq101_ref(), (0, 0),
                              fv.constrained_random_sequences(300, 20, seed=3),
                              ref_init_state=0)
assert fsm_ok["passed"]
caught = fv.verify_sequential(fv.seq101_buggy(), fv.seq101_ref(), (0, 0),
                              [[1, 0, 1]], ref_init_state=0)
assert not caught["passed"] and caught["first_divergence"] == 2

# 8. assertions: an in-range property holds; a false one reports the first
#    violating cycle
_, outs = fv.run_sequential(fv.counter_ref(3), 0, [1]*20)
assert fv.check_property(outs, lambda v: 0 <= v < 8)["holds"]
viol = fv.check_property(outs, lambda v: v != 3)      # count reaches 3 at cycle 3
assert not viol["holds"] and viol["first_violation"] == 3

# 9. kwarg bounds
for bad_call in (lambda: fv.exhaustive_equivalence(bad_mux, bad_mux, []),
                 lambda: fv.constrained_random_sequences(0, 5),
                 lambda: fv.constrained_random_sequences(5, 0),
                 lambda: fv.transition_coverage([], set())):
    try:
        bad_call()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("test_functional_verification: all checks passed")
