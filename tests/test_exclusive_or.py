"""Test dgs.exclusive_or: inclusive vs exclusive OR differ only at (1,1),
XOR-reduce is odd parity, actual-voltage mapping + noise margins, and the
learnability arc -- AND/OR are linearly separable but XOR/XNOR (the parity
pair) are not, a single perceptron cannot drive XOR to zero error, and a
2->H->1 MLP does solve it. NumPy + SciPy (both on py-3.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from itertools import product
from dgs import exclusive_or as xo
from dgs import digital_logic as dl

# 1. inclusive vs exclusive OR: agree everywhere except (1,1)
rows = xo.or_xor_tables()
assert rows == [(0, 0, 0, 0), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 0)]
disagree = [(a, b) for a, b, o, x in rows if o != x]
assert disagree == [(1, 1)]
assert xo.or_xor_disagreement() == (1, 1)

# 2. XOR-reduce == odd parity == popcount mod 2, and matches digital_logic
assert xo.xor_reduce([1, 0, 1, 1, 0]) == 1        # three 1s -> odd -> 1
assert xo.xor_reduce([1, 1]) == 0                 # two 1s -> even -> 0
for n in range(1, 11):
    assert xo.xor_is_odd_parity(n)
# consistency with the repo's parity_bit: XOR-reduce is the even-parity bit
for bits in product((0, 1), repeat=5):
    assert xo.xor_reduce(bits) == dl.parity_bit(list(bits), even=True)

# 3. actual voltage: 0/1 -> rails, and noise margins
v = xo.logic_to_voltage([0, 1, 1, 0], v_dd=1.8)
assert np.allclose(v, [0.0, 1.8, 1.8, 0.0])
nmh, nml = xo.noise_margins(v_oh=1.7, v_ol=0.1, v_ih=1.2, v_il=0.6)
assert np.isclose(nmh, 0.5) and np.isclose(nml, 0.5)
# an inverted level ordering must be rejected
try:
    xo.noise_margins(v_oh=1.0, v_ol=0.1, v_ih=1.2, v_il=0.6)  # V_IH > V_OH
    assert False
except ValueError:
    pass

# 4. linear separability, decided rigorously by LP: the parity functions are
#    the ONLY 2-input booleans that fail (XOR and XNOR)
funcs = {
    "AND":  lambda a, b: a & b,
    "OR":   lambda a, b: a | b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR":  lambda a, b: 1 - (a | b),
    "XOR":  lambda a, b: a ^ b,
    "XNOR": lambda a, b: 1 - (a ^ b),
}
sep = {name: xo.is_linearly_separable(*xo.boolean_dataset(f)) for name, f in funcs.items()}
assert sep == {"AND": True, "OR": True, "NAND": True, "NOR": True,
               "XOR": False, "XNOR": False}

# 5. perceptron: converges on OR (separable), never clears XOR (not separable)
w, b, errs_or, conv_or = xo.perceptron_train(*xo.boolean_dataset(funcs["OR"]),
                                             epochs=100, seed=0)
assert conv_or and errs_or[-1] == 0
_, _, errs_xor, conv_xor = xo.perceptron_train(*xo.boolean_dataset(funcs["XOR"]),
                                               epochs=200, seed=0)
assert not conv_xor and errs_xor[-1] > 0            # the wall: nonzero forever

# 6. the hidden layer clears the wall: a 2->H->1 MLP solves XOR exactly
res = xo.mlp_xor_train(hidden=8, epochs=4000, lr=0.5, seed=0)
assert res["accuracy"] == 1.0
assert res["final_loss"] < 1e-2
assert list(res["rounded"]) == [0, 1, 1, 0]
# loss actually decreased a lot from start to finish (it learned)
assert res["losses"][-1] < 0.05 * res["losses"][0]

# 7. kwarg bounds
for bad in (lambda: xo.xor_reduce([0, 2, 1]),
            lambda: xo.xor_is_odd_parity(0),
            lambda: xo.logic_to_voltage([0, 1], v_dd=0.0),
            lambda: xo.logic_to_voltage([0, 3]),
            lambda: xo.mlp_xor_train(hidden=1)):
    try:
        bad()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("test_exclusive_or: all checks passed")
