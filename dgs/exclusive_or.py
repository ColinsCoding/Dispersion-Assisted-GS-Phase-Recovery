"""Inclusive OR vs exclusive OR: one gate splits into parity and neural nets.

The two ORs differ in exactly ONE row of the truth table:
    a b | OR (inclusive) | XOR (exclusive)
    0 0 |      0         |      0
    0 1 |      1         |      1
    1 0 |      1         |      1
    1 1 |      1         |      0     <-- the only disagreement
"Inclusive" OR means "at least one is true"; "exclusive" OR means "an ODD
number are true" -- so a chain of XORs IS a parity check: XOR-reduce a bit
string and you get 1 iff it holds an odd count of 1s. That makes XOR the gate
form of the even/odd decomposition in dgs.even_odd and the parity bit in
dgs.digital_logic.

Three layers of the same idea, each building on the last:
  1. LOGIC. Inclusive vs exclusive OR, and n-input XOR = odd parity.
  2. ACTUAL VOLTAGE. Those 0s and 1s are really V_SS and V_DD on a wire; a
     real gate has to hold outputs cleanly apart (V_OH, V_OL) and still read
     noisy inputs (V_IH, V_IL). The slack is the NOISE MARGIN -- the volts of
     garbage a stage can tolerate before a 1 looks like a 0.
  3. NETWORKING. XOR is the textbook function a single-layer perceptron
     CANNOT learn, because its two 1-rows sit on the diagonal and no straight
     line separates them from the two 0-rows -- XOR is not linearly separable.
     AND/OR are; XOR/XNOR (the parity pair) are the only 2-input functions
     that are not. The fix is a HIDDEN LAYER: a 2->H->1 net folds the plane so
     the diagonal becomes separable. This is why neural nets need depth, and
     it is the same even/odd structure showing up as a learnability wall.

Separability is decided rigorously by linear programming (scipy.optimize.
linprog); the perceptron and the tiny MLP are pure NumPy so the whole thing
runs on py-3.13 (a PyTorch version is a drop-in, but not needed here).
"""

import numpy as np
from itertools import product


# ----------------------------------------------------------------------
# 1. Logic: inclusive vs exclusive, and XOR = odd parity
# ----------------------------------------------------------------------

def or_xor_tables():
    """Both truth tables side by side as rows (a, b, OR, XOR). They agree
    everywhere except (1,1), where inclusive OR is 1 but exclusive OR is 0."""
    rows = []
    for a, b in product((0, 1), repeat=2):
        rows.append((a, b, a | b, a ^ b))
    return rows


def or_xor_disagreement():
    """The single input where inclusive and exclusive OR differ: (1,1).
    Returned as the (a, b) tuple -- 'both true' is the only case 'at least
    one' and 'exactly an odd number' disagree on."""
    return next((a, b) for a, b, o, x in or_xor_tables() if o != x)


def xor_reduce(bits):
    """Fold XOR across a bit string: the running parity. Equals 1 iff the
    string holds an ODD number of 1s, so this is exactly odd-parity /
    popcount mod 2 -- the same quantity dgs.digital_logic.parity_bit uses."""
    bits = [int(b) for b in bits]
    if any(b not in (0, 1) for b in bits):
        raise ValueError("bits must be 0/1")
    acc = 0
    for b in bits:
        acc ^= b
    return acc


def xor_is_odd_parity(n):
    """Verify over ALL 2^n inputs that chained XOR equals popcount mod 2 --
    the exact statement 'exclusive OR generalizes to a parity check'.
    Returns True. n in 1..16."""
    if not 1 <= n <= 16:
        raise ValueError("n must be in 1..16")
    for bits in product((0, 1), repeat=n):
        if xor_reduce(bits) != sum(bits) % 2:
            return False
    return True


# ----------------------------------------------------------------------
# 2. Actual voltage: logic levels and noise margins
# ----------------------------------------------------------------------

def logic_to_voltage(bits, v_dd=1.8, v_ss=0.0):
    """Turn abstract 0/1 into the volts actually on the wire: 0 -> V_SS,
    1 -> V_DD. The whole point of digital design is that these two rails are
    far enough apart to survive noise (see noise_margins)."""
    if v_dd <= v_ss:
        raise ValueError("v_dd must exceed v_ss")
    bits = np.asarray(bits)
    if not np.all(np.isin(bits, (0, 1))):
        raise ValueError("bits must be 0/1")
    return np.where(bits == 1, v_dd, v_ss).astype(float)


def noise_margins(v_oh, v_ol, v_ih, v_il):
    """The volts of noise a logic stage tolerates. A gate drives HIGH to at
    least V_OH and LOW to at most V_OL; the next gate still reads HIGH above
    V_IH and LOW below V_IL. The slack on each side is
        NM_H = V_OH - V_IH   (a driven 1 can lose this many volts, still read 1)
        NM_L = V_IL - V_OL   (a driven 0 can gain this many volts, still read 0).
    Both must be positive or the family is unusable. Returns (NM_H, NM_L)."""
    if not (v_oh > v_ih > v_il > v_ol):
        raise ValueError("levels must satisfy V_OH > V_IH > V_IL > V_OL")
    return v_oh - v_ih, v_il - v_ol


# ----------------------------------------------------------------------
# 3. Networking: the perceptron wall and the hidden-layer fix
# ----------------------------------------------------------------------

def boolean_dataset(func):
    """(X, y) for a 2-input boolean function given as func(a, b). X is the
    four input rows, y in {-1,+1} for the separability tests. func may be
    e.g. lambda a,b: a & b (AND), a | b (OR), a ^ b (XOR)."""
    X = np.array(list(product((0, 1), repeat=2)), float)
    y = np.array([1 if func(int(a), int(b)) else -1 for a, b in X])
    return X, y


def is_linearly_separable(X, y):
    """Rigorously decide if labels y in {-1,+1} on points X can be split by a
    hyperplane, via LP feasibility: does there exist (w, b) with
        y_i (w . x_i + b) >= 1  for every i ?
    Feasible <=> linearly separable (scipy.optimize.linprog). AND and OR are
    feasible; XOR and XNOR are NOT -- the parity functions are the only
    2-input ones this fails on."""
    from scipy.optimize import linprog
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    # unknowns z = [w (d), b (1)]; constraint -y_i*(w.x_i + b) <= -1
    A_ub = -y[:, None] * np.hstack([X, np.ones((n, 1))])
    b_ub = -np.ones(n)
    res = linprog(c=np.zeros(d + 1), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * (d + 1), method="highs")
    return bool(res.success)


def perceptron_train(X, y, epochs=100, lr=1.0, seed=0):
    """The classic perceptron learning rule on labels y in {-1,+1}. Returns
    (weights, bias, errors_per_epoch, converged). By the perceptron
    convergence theorem it reaches zero errors IFF the data is linearly
    separable -- so it learns OR but CANNOT drive XOR's error to zero, no
    matter how long it runs."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, X.shape[1])
    b = 0.0
    errors = []
    for _ in range(epochs):
        err = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:          # misclassified
                w += lr * yi * xi
                b += lr * yi
                err += 1
        errors.append(err)
    return w, b, errors, errors[-1] == 0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def mlp_xor_train(hidden=8, epochs=4000, lr=0.5, seed=0):
    """Train a 2 -> hidden -> 1 network (tanh hidden, sigmoid output) by
    plain-NumPy backprop to compute XOR -- the function a single layer can't.
    The hidden layer bends the input plane so XOR's diagonal 1s become
    linearly separable in hidden space. Returns dict with predictions,
    rounded outputs, final loss, accuracy (1.0 = solved) and the loss curve.
    This is the 'torch networking' idea; NumPy keeps it py-3.13-native."""
    if hidden < 2:
        raise ValueError("need hidden >= 2 (1 hidden unit cannot fold XOR)")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0, (2, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0, (hidden, 1)); b2 = np.zeros(1)
    losses = []
    for _ in range(epochs):
        h = np.tanh(X @ W1 + b1)                 # forward
        out = _sigmoid(h @ W2 + b2)
        losses.append(float(np.mean((out - y) ** 2)))
        dout = (out - y) * out * (1 - out)       # backprop
        dW2 = h.T @ dout; db2 = dout.sum(0)
        dh = (dout @ W2.T) * (1 - h ** 2)
        dW1 = X.T @ dh; db1 = dh.sum(0)
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1
    pred = out.ravel()
    rounded = (pred > 0.5).astype(int)
    return {
        "pred": pred, "rounded": rounded,
        "target": y.ravel().astype(int),
        "final_loss": losses[-1],
        "accuracy": float(np.mean(rounded == y.ravel().astype(int))),
        "losses": losses,
    }


if __name__ == "__main__":
    print("a b | OR | XOR")
    for a, b, o, x in or_xor_tables():
        print(f" {a} {b} |  {o} |  {x}")
    print("only disagreement at (a,b) =", or_xor_disagreement(),
          "(inclusive says 1, exclusive says 0)")
    print("XOR-reduce [1,0,1,1,0] =", xor_reduce([1, 0, 1, 1, 0]),
          "(odd # of 1s -> 1); is XOR odd-parity for all n<=8?",
          all(xor_is_odd_parity(n) for n in range(1, 9)))

    nmh, nml = noise_margins(v_oh=1.7, v_ol=0.1, v_ih=1.2, v_il=0.6)
    print(f"\nactual voltage (1.8 V CMOS-ish): logic [0,1,1,0] -> "
          f"{logic_to_voltage([0,1,1,0])} V;  noise margins NM_H={nmh:.2f} V, NM_L={nml:.2f} V")

    print("\nlinearly separable?  AND:", is_linearly_separable(*boolean_dataset(lambda a,b: a&b)),
          " OR:", is_linearly_separable(*boolean_dataset(lambda a,b: a|b)),
          " XOR:", is_linearly_separable(*boolean_dataset(lambda a,b: a^b)),
          "(XOR is the parity wall)")

    Xor = boolean_dataset(lambda a, b: a ^ b)
    _, _, errs, conv = perceptron_train(*Xor, epochs=100)
    print("single perceptron on XOR: converged?", conv,
          f"(still {errs[-1]} errors after 100 epochs)")
    res = mlp_xor_train()
    print(f"2->8->1 MLP on XOR: accuracy {res['accuracy']:.0%}, "
          f"final loss {res['final_loss']:.1e}, outputs {np.round(res['pred'],2)} "
          f"vs target {res['target']} -- the hidden layer clears the wall.")
