"""Build notebooks/boolean_logic_first_pass.ipynb -- a first pass at digital
logic Boolean equations from dgs/boolean_algebra.py: truth table -> minimal
SOP (Quine-McCluskey) -> K-map -> a torch neural net that learns the SAME
function from raw examples, no minimization involved. The contrast between
"exact minimal equation" and "learned black box" is the point: some
functions (XOR) resist compact SOP forms and are famously hard for a small
neural net for the same underlying reason (no linearly-separable structure).

Sections:
  S1  Setup
  S2  Basic gates: AND, OR, XOR, NAND
  S3  Majority function (3-input)
  S4  XOR parity (4-input) -- SOP can't compress it
  S5  Half-adder and full-adder -- real digital-logic circuits
  S6  torch LogicNet learns each function from raw examples
  S7  Summary: equation complexity vs. learning difficulty
"""

import json, pathlib

NB = pathlib.Path("notebooks/boolean_logic_first_pass.ipynb")
NB.parent.mkdir(exist_ok=True)

cells = []
def md(src): cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "execution_count": None,
                              "metadata": {}, "outputs": [], "source": src})


# ── S1 ────────────────────────────────────────────────────────────────────────
md("""# Digital Logic: Boolean Equations -- First Pass

Three ways to describe the same Boolean function, from
[`dgs/boolean_algebra.py`](../dgs/boolean_algebra.py):

1. **Truth table** -- the ground truth, all $2^n$ input rows.
2. **Minimal SOP equation** -- Quine-McCluskey minimization, the smallest
   sum-of-products expression that reproduces the table exactly.
3. **A trained neural net** (`LogicNet`, torch) -- learns the same
   input-output mapping from raw examples, no Boolean algebra involved at
   all.

The interesting result isn't that all three agree (they will, by
construction) -- it's *how hard* step 2 and step 3 are for different
functions. **XOR-like functions resist compact SOP equations and are
famously hard for small neural nets for the same underlying reason**: no
linearly-separable structure to exploit. Seeing that pattern show up twice,
in two completely different formalisms, is the actual content of this
notebook.
""")

code("""\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

from dgs.boolean_algebra import TruthTable, minimize_sop, kmap, LogicNet
import torch
import pandas as pd

summary_rows = []
print("torch", torch.__version__)
""")

# ── S2: basic gates ────────────────────────────────────────────────────────
md("## Basic Gates: AND, OR, XOR, NAND"),

code("""\
gates = {
    "AND":  (2, lambda A, B: A & B),
    "OR":   (2, lambda A, B: A | B),
    "XOR":  (2, lambda A, B: A ^ B),
    "NAND": (2, lambda A, B: not (A and B)),
}

for name, (n, fn) in gates.items():
    tt = TruthTable(n, fn)
    print(f"--- {name} ---")
    tt.show()
    print(minimize_sop(tt.minterms, n_vars=n))
    print()
""")

# ── S3: majority ────────────────────────────────────────────────────────────
md("## Majority Function (3-input) -- Classic 2-Literal-Term SOP"),

code("""\
tt_maj = TruthTable(3, lambda A, B, C: (A & B) | (B & C) | (A & C))
tt_maj.show(["A", "B", "C"])
sop_maj = minimize_sop(tt_maj.minterms, n_vars=3)
print(sop_maj)
kmap(tt_maj, var_names=["A", "B", "C"])
""")

# ── S4: XOR parity ────────────────────────────────────────────────────────────
md("""## XOR Parity (4-input) -- SOP Can't Compress It

4-input parity is true on exactly the minterms with an odd number of 1s --
a checkerboard pattern on the K-map. No two true cells are ever adjacent, so
Quine-McCluskey can't combine *any* of them: **the minimal SOP is just the
full sum of all 8 minterms**, no compression at all. This is the textbook
worst case for SOP minimization."""),

code("""\
tt_xor4 = TruthTable(4, lambda A, B, C, D: A ^ B ^ C ^ D)
sop_xor4 = minimize_sop(tt_xor4.minterms, n_vars=4)
n_terms = sop_xor4.count("+") + 1
print(sop_xor4)
print(f"\\nnumber of minterms true: {len(tt_xor4.minterms)} out of {2**4}")
print(f"number of SOP terms: {n_terms}  (no compression -- every true minterm needed its own term)")
kmap(tt_xor4, var_names=["A", "B", "C", "D"])
""")

# ── S5: adders ────────────────────────────────────────────────────────────────
md("""## Half-Adder and Full-Adder -- Real Digital-Logic Circuits

Sum and Carry, derived from their truth tables the same way, not quoted from
a textbook."""),

code("""\
print("=== Half adder: Sum = A xor B, Carry = A and B ===")
tt_sum_half = TruthTable(2, lambda A, B: A ^ B)
tt_carry_half = TruthTable(2, lambda A, B: A & B)
print("Sum:  ", minimize_sop(tt_sum_half.minterms, n_vars=2))
print("Carry:", minimize_sop(tt_carry_half.minterms, n_vars=2))

print("\\n=== Full adder: A, B, Cin -> Sum, Cout ===")
tt_sum_full = TruthTable(3, lambda A, B, Cin: A ^ B ^ Cin)
tt_cout_full = TruthTable(3, lambda A, B, Cin: (A & B) | (Cin & (A ^ B)))
print("Sum: ", minimize_sop(tt_sum_full.minterms, n_vars=3))
print("Cout:", minimize_sop(tt_cout_full.minterms, n_vars=3))
kmap(tt_cout_full, var_names=["A", "B", "Cin"])
""")

# ── S6: torch LogicNet ──────────────────────────────────────────────────────
md("""## `LogicNet` Learns Each Function From Raw Examples

No Boolean algebra involved here -- just $2^n$ (input, output) pairs and
gradient descent. Track final accuracy and epochs-to-converge for every
function above; the harder-to-minimize functions (XOR-like) should also be
the harder-to-learn ones."""),

code("""\
functions_to_learn = [
    ("AND",          2, lambda A, B: A & B),
    ("OR",           2, lambda A, B: A | B),
    ("XOR",          2, lambda A, B: A ^ B),
    ("Majority-3",   3, lambda A, B, C: (A & B) | (B & C) | (A & C)),
    ("XOR-parity-4", 4, lambda A, B, C, D: A ^ B ^ C ^ D),
    ("Full-adder-Cout", 3, lambda A, B, Cin: (A & B) | (Cin & (A ^ B))),
]

def train_and_track(n, fn, epochs=1500, lr=0.05, hidden=8, seed=0):
    \"\"\"Manual training loop (not LogicNet.fit, which only prints) so the
    epoch at which the net FIRST reaches 100% accuracy can be tracked --
    final accuracy alone saturates to 1.0 for every function by epochs=1500
    and hides exactly the difficulty gap this notebook is trying to show.\"\"\"
    torch.manual_seed(seed)
    tt = TruthTable(n, fn)
    net = LogicNet(n_inputs=n, hidden=hidden)
    X = torch.tensor([list(bits) for bits, _ in tt.rows], dtype=torch.float32)
    Y = torch.tensor([float(o) for _, o in tt.rows], dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    first_perfect_epoch = None
    for ep in range(epochs):
        opt.zero_grad()
        pred = net(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        if first_perfect_epoch is None:
            with torch.no_grad():
                if ((net(X) >= 0.5).float() == Y).all():
                    first_perfect_epoch = ep
    final_acc = (((net(X) >= 0.5).float() == Y).float().mean()).item()
    return tt, final_acc, first_perfect_epoch

for name, n, fn in functions_to_learn:
    tt, final_acc, first_perfect = train_and_track(n, fn)
    sop = minimize_sop(tt.minterms, n_vars=n)
    n_sop_terms = sop.count('+') + 1 if sop != 'F = 0' else 0
    fp_str = str(first_perfect) if first_perfect is not None else "never (in 1500 epochs)"
    print(f"{name:18s}  SOP terms={n_sop_terms}  first-100%-epoch={fp_str}  final_acc={final_acc:.3f}")
    summary_rows.append({"function": name, "n_inputs": n, "sop_terms": n_sop_terms,
                          "first_100pct_epoch": first_perfect, "final_accuracy": final_acc})
""")

# ── S7: summary ────────────────────────────────────────────────────────────────
md("## Summary: Equation Complexity vs. Learning Difficulty"),

code("""\
df = pd.DataFrame(summary_rows).sort_values("sop_terms").reset_index(drop=True)
df
""")

code("""\
correlation = df["sop_terms"].corr(df["first_100pct_epoch"])
print(f"correlation between SOP term count and epochs-to-first-100%: {correlation:.3f}")
""")

md("""`first_100pct_epoch` (not final accuracy, which saturates to 1.0 for
every function given enough epochs and hides the gap) is where the pattern
shows up: functions with few SOP terms (AND, OR -- linearly separable, a
single hyperplane divides true from false) hit 100% accuracy almost
immediately, while XOR-like functions, needing the full minterm count in
SOP, also take visibly longer to learn exactly. Both are downstream of the
same property of the function -- linear separability -- so it isn't a
coincidence that it shows up in two unrelated formalisms (combinational
logic minimization and gradient descent) as two different-looking
symptoms.""")

# ── finalize ─────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "cells": cells,
}

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB}  ({len(cells)} cells)")
print(f"Execute (needs torch -> py 3.12): "
      f"py -3.12 -m jupyter nbconvert --to notebook --execute --inplace \"{NB}\"")
