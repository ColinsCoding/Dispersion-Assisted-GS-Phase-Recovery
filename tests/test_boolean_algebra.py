"""Test dgs.boolean_algebra: TruthTable, Quine-McCluskey minimization, K-map,
and the torch LogicNet learner. No prior test file existed for this module."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.boolean_algebra import TruthTable, minimize_sop, kmap, LogicNet

# 1. TruthTable: minterms/maxterms partition all 2^n rows correctly
tt_and = TruthTable(2, lambda A, B: A & B)
assert tt_and.minterms == [3]              # only A=1,B=1 (row index 3) is true
assert tt_and.maxterms == [0, 1, 2]

tt_or = TruthTable(2, lambda A, B: A | B)
assert tt_or.minterms == [1, 2, 3]
assert set(tt_or.minterms) | set(tt_or.maxterms) == set(range(4))

# 2. minimize_sop: known minimal forms for textbook functions
assert minimize_sop(tt_and.minterms, n_vars=2) == "F = AB"
assert minimize_sop([], n_vars=2) == "F = 0"

# 3-variable majority function: minimal SOP should have exactly 3 two-literal
# terms (AB, BC, AC in some order) -- the classic majority-gate result
tt_maj = TruthTable(3, lambda A, B, C: (A & B) | (B & C) | (A & C))
sop_maj = minimize_sop(tt_maj.minterms, n_vars=3)
assert sop_maj.count("+") == 2              # 3 terms => 2 plus signs
for term in sop_maj.replace("F = ", "").split(" + "):
    assert len(term) == 2                   # each term is exactly 2 literals

# 3. don't-cares: a don't-care that WOULD extend a term should be usable to
# simplify further than without it
minterms_dc = [1, 3]      # A=0: rows 1,3 -> ~AB minterm pattern (n_vars=2: A B)
sop_no_dc = minimize_sop(minterms_dc, n_vars=2)
sop_with_dc = minimize_sop(minterms_dc, n_vars=2, dc=[0, 2])   # don't-care makes it "F=1"
assert sop_with_dc == "F = 1"
assert sop_no_dc != "F = 1"                 # the don't-cares were necessary for that simplification

# 4. every minterm the function claims true must actually evaluate true
# against minimize_sop's own SOP string, re-derived independently via
# TruthTable on the parsed expression (round-trip check)
def eval_sop(sop_str, n_vars, bits):
    # sop_str looks like "F = AB + ~AC"; evaluate against a bit tuple
    names = [chr(ord('A') + i) for i in range(n_vars)]
    val = {n: b for n, b in zip(names, bits)}
    rhs = sop_str.split("F = ")[1]
    if rhs == "0":
        return 0
    if rhs == "1":
        return 1
    result = 0
    for term in rhs.split(" + "):
        t = 1
        i = 0
        while i < len(term):
            if term[i] == "~":
                t &= (1 - val[term[i+1]])
                i += 2
            else:
                t &= val[term[i]]
                i += 1
        result |= t
    return result

for bits, expected in tt_maj.rows:
    assert eval_sop(sop_maj, 3, bits) == expected, (bits, expected, sop_maj)

# 5. kmap: must not raise for supported sizes, and must print something for
# unsupported sizes rather than raise
kmap(tt_maj)             # 3-var: should run without error
kmap(TruthTable(5, lambda a,b,c,d,e: a))   # unsupported size: prints a message, no raise

# 6. LogicNet (torch): learns a simple function (AND) to near-perfect accuracy
tt_and3 = TruthTable(3, lambda A, B, C: A & B & C)
net = LogicNet(n_inputs=3, hidden=8)
net.fit(tt_and3, epochs=800, lr=0.05)

import torch
X = torch.tensor([list(bits) for bits, _ in tt_and3.rows], dtype=torch.float32)
Y = torch.tensor([float(o) for _, o in tt_and3.rows], dtype=torch.float32)
with torch.no_grad():
    preds = (net(X) >= 0.5).float()
accuracy = (preds == Y).float().mean().item()
assert accuracy == 1.0, f"LogicNet should learn 3-input AND exactly, got accuracy={accuracy}"

print("TEST PASS  (TruthTable partitions minterms/maxterms correctly; minimize_sop matches "
      "known textbook forms incl. majority gate and don't-cares; SOP round-trips against the "
      "original truth table; kmap runs for supported/unsupported sizes; torch LogicNet learns "
      "3-input AND to 100% accuracy)")
