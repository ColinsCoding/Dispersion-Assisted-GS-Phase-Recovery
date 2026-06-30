"""Boolean algebra: truth tables, SOP/POS, Quine-McCluskey minimization,
K-map display, and a PyTorch logic learner for N variables.

Usage:
    from dgs.boolean_algebra import TruthTable, minimize_sop, kmap, LogicNet

    # 3-variable majority function
    tt = TruthTable(3, lambda A, B, C: (A & B) | (B & C) | (A & C))
    tt.show()
    print(minimize_sop(tt.minterms, n_vars=3))
    kmap(tt, var_names=["A", "B", "C"])

    # Learn any truth table with torch (py -3.12)
    net = LogicNet(n_inputs=3)
    net.fit(tt)
    net.predict_all()
"""

from __future__ import annotations
import itertools
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Truth table
# ---------------------------------------------------------------------------

class TruthTable:
    """Enumerate all 2^n input combinations and record outputs."""

    def __init__(self, n_vars: int, fn: Callable[..., int]):
        if n_vars < 1 or n_vars > 16:
            raise ValueError("n_vars must be 1..16")
        self.n = n_vars
        self.rows: List[tuple[tuple[int, ...], int]] = []
        for bits in itertools.product((0, 1), repeat=n_vars):
            out = int(bool(fn(*bits)))
            self.rows.append((bits, out))

    @property
    def minterms(self) -> List[int]:
        return [i for i, (_, o) in enumerate(self.rows) if o == 1]

    @property
    def maxterms(self) -> List[int]:
        return [i for i, (_, o) in enumerate(self.rows) if o == 0]

    def show(self, var_names: Optional[List[str]] = None) -> None:
        names = var_names or [chr(ord('A') + i) for i in range(self.n)]
        header = "  ".join(names) + "  |  F"
        print(header)
        print("-" * len(header))
        for bits, out in self.rows:
            row = "   ".join(str(b) for b in bits)
            print(f"{row}  |  {out}")


# ---------------------------------------------------------------------------
# Quine-McCluskey: find prime implicants, then greedy cover → minimal SOP
# ---------------------------------------------------------------------------

def _ones(x: int) -> int:
    return bin(x).count('1')

def _differ_by_one_bit(a: int, b: int) -> int:
    """Return the differing bit position (0-indexed from LSB) or -1."""
    diff = a ^ b
    if diff and not (diff & (diff - 1)):      # exactly one bit
        return diff.bit_length() - 1
    return -1

def minimize_sop(minterms: List[int], n_vars: int, *, dc: Optional[List[int]] = None) -> str:
    """Quine-McCluskey minimization. Returns a human-readable SOP string.

    dc: don't-care minterms (contribute to implicants but not cover).
    """
    if not minterms:
        return "F = 0"

    dc = dc or []
    all_terms = sorted(set(minterms) | set(dc))

    # Each implicant: frozenset of covered minterms, mask of fixed bits
    # Represent as (value, mask) where mask bit=1 means variable is fixed.
    implicants: List[tuple[frozenset, int, int]] = [
        (frozenset([m]), m, (1 << n_vars) - 1) for m in all_terms
    ]
    prime_implicants: List[tuple[frozenset, int, int]] = []
    used: set = set()

    while implicants:
        next_level: List[tuple[frozenset, int, int]] = []
        combined: set[int] = set()
        for i in range(len(implicants)):
            for j in range(i + 1, len(implicants)):
                cov_i, val_i, mask_i = implicants[i]
                cov_j, val_j, mask_j = implicants[j]
                if mask_i != mask_j:
                    continue
                bit = _differ_by_one_bit(val_i, val_j)
                if bit == -1:
                    continue
                new_mask = mask_i & ~(1 << bit)
                new_val  = val_i & new_mask
                new_cov  = cov_i | cov_j
                combined.add(i); combined.add(j)
                entry = (new_cov, new_val, new_mask)
                if entry not in next_level:
                    next_level.append(entry)
        for idx, imp in enumerate(implicants):
            if idx not in combined:
                prime_implicants.append(imp)
        implicants = next_level

    # Greedy essential-prime-implicant cover
    must_cover = set(minterms)      # don't-cares don't need to be covered
    essential: List[tuple[frozenset, int, int]] = []
    uncovered = set(must_cover)

    # Find essential PIs: only PI covering a given minterm
    for m in list(uncovered):
        covers = [(cov, val, mask) for (cov, val, mask) in prime_implicants if m in cov and cov <= must_cover | set(dc) ]
        if len(covers) == 1:
            pi = covers[0]
            if pi not in essential:
                essential.append(pi)
                uncovered -= pi[0]

    # Greedy pick for remaining
    while uncovered:
        best = max(prime_implicants, key=lambda p: len(p[0] & uncovered))
        essential.append(best)
        uncovered -= best[0]

    return _sop_string(essential, n_vars)


def _sop_string(implicants: List[tuple[frozenset, int, int]], n_vars: int) -> str:
    var_names = [chr(ord('A') + i) for i in range(n_vars)]
    terms = []
    for _, val, mask in implicants:
        literals = []
        for bit in range(n_vars - 1, -1, -1):
            if mask & (1 << bit):
                v = var_names[n_vars - 1 - bit]
                literals.append(v if (val >> bit) & 1 else f"~{v}")
        terms.append("".join(literals) if literals else "1")
    return "F = " + " + ".join(terms)


# ---------------------------------------------------------------------------
# Karnaugh map (2, 3, or 4 variables)
# ---------------------------------------------------------------------------

_GRAY = {
    2: [0, 1],
    4: [0, 1, 3, 2],
}

def kmap(tt: TruthTable, var_names: Optional[List[str]] = None) -> None:
    """Print a K-map for 2-, 3-, or 4-variable truth tables."""
    n = tt.n
    if n not in (2, 3, 4):
        print(f"K-map only supported for 2–4 variables (got {n})")
        return

    names = var_names or [chr(ord('A') + i) for i in range(n)]
    output = {i: o for i, (_, o) in enumerate(tt.rows)}

    if n == 2:
        col_var, row_var = names[1], names[0]
        col_order = _GRAY[2]
        print(f"     {col_var}=0  {col_var}=1")
        for r in _GRAY[2]:
            cells = " ".join(str(output[r * 2 + c]) for c in col_order)
            print(f"{row_var}={r}  {cells}")
    elif n == 3:
        # rows: A; cols: BC in Gray order
        row_var = names[0]
        col_label = names[1] + names[2]
        col_order = _GRAY[4][:4]   # 00 01 11 10
        print(f"  \\ {col_label}  00  01  11  10")
        for r in _GRAY[2]:
            row = []
            for bc in [0, 1, 3, 2]:
                idx = r * 4 + bc
                row.append(str(output[idx]))
            print(f"  {row_var}={r}       " + "   ".join(row))
    else:
        # rows: AB; cols: CD in Gray order
        ab_label, cd_label = names[0] + names[1], names[2] + names[3]
        print(f"  \\ {cd_label}  00  01  11  10")
        for ab in [0, 1, 3, 2]:
            row = []
            for cd in [0, 1, 3, 2]:
                a, b = (ab >> 1) & 1, ab & 1
                c, d = (cd >> 1) & 1, cd & 1
                idx = a * 8 + b * 4 + c * 2 + d
                row.append(str(output[idx]))
            a_bit, b_bit = (ab >> 1) & 1, ab & 1
            print(f"  {ab_label}={a_bit}{b_bit}      " + "   ".join(row))


# ---------------------------------------------------------------------------
# PyTorch logic learner: learns any Boolean function with N inputs
# (py -3.12 only; torch not available on 3.13)
# ---------------------------------------------------------------------------

def LogicNet(n_inputs: int, hidden: int = 16):
    """Return a tiny MLP that can learn any N-input Boolean function.

    n_inputs: number of Boolean variables (no upper limit, but 2^N rows
              are generated for training).
    hidden:   width of the single hidden layer.

    Returns a fitted-object factory: call .fit(tt) then .predict_all().
    """
    import torch
    import torch.nn as nn

    class _LogicNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

        def fit(self, tt: TruthTable, epochs: int = 2000, lr: float = 0.01):
            X = torch.tensor([list(bits) for bits, _ in tt.rows], dtype=torch.float32)
            Y = torch.tensor([float(o) for _, o in tt.rows], dtype=torch.float32)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.BCELoss()
            for ep in range(epochs):
                opt.zero_grad()
                loss = loss_fn(self(X), Y)
                loss.backward()
                opt.step()
                if ep % 500 == 0:
                    print(f"  epoch {ep:4d}  loss={loss.item():.4f}")
            return self

        def predict_all(self, tt: Optional[TruthTable] = None, threshold: float = 0.5):
            import torch
            if tt is None:
                X = torch.tensor(
                    list(itertools.product([0.0, 1.0], repeat=n_inputs)),
                    dtype=torch.float32,
                )
            else:
                X = torch.tensor([list(bits) for bits, _ in tt.rows], dtype=torch.float32)
            with torch.no_grad():
                preds = (self(X) >= threshold).int().tolist()
            print("Input → Predicted")
            for row, pred in zip(itertools.product([0, 1], repeat=n_inputs), preds):
                print(f"  {list(row)} → {pred}")

    return _LogicNet()


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 3-variable majority ===")
    tt = TruthTable(3, lambda A, B, C: (A & B) | (B & C) | (A & C))
    tt.show()
    print()
    print(minimize_sop(tt.minterms, n_vars=3))
    print()
    kmap(tt)

    print()
    print("=== 4-variable XOR parity ===")
    tt4 = TruthTable(4, lambda A, B, C, D: A ^ B ^ C ^ D)
    print(minimize_sop(tt4.minterms, n_vars=4))
    print()
    kmap(tt4)

    print()
    print("=== 2-variable NAND ===")
    tt2 = TruthTable(2, lambda A, B: not (A and B))
    tt2.show()
    print(minimize_sop(tt2.minterms, n_vars=2))
    kmap(tt2)
