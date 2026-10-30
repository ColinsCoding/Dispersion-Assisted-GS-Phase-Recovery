"""Linear algebra proofs in SymPy, the same math as attention in torch, and supply and demand.

SymPy proves the identities for fully symbolic n x n matrices (every entry its own symbol), so a
zero difference is a proof for that size; lean/LinearAlgebraProofs.lean proves them for every size.

    prove_identities(n)    det(AB) = det A det B, (AB)^T = B^T A^T, tr(AB) = tr(BA), (AB)^-1 = B^-1 A^-1
    cauchy_schwarz(n)      Lagrange identity: |u|^2 |v|^2 - (u.v)^2 is a sum of squares
    projection(v)          P = v v^T / v^T v is symmetric and idempotent; 1 - P is too
    market(...)            linear supply and demand as a 2x2 system A [Q, P] = b (photonics example)
    attention(Q, K, V)     softmax(Q K^T / sqrt(d)) V in torch, checked against torch's own kernel
    market_c(...)          the same 2x2 solve compiled with gcc (dgs/c/market/market.c)
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
C_SRC = ROOT / "dgs" / "c" / "market" / "market.c"


def _check_n(n):
    if not isinstance(n, int) or not 1 <= n <= 4:
        raise ValueError(f"n must be an int from 1 to 4 (symbolic det grows like n!), got {n!r}")


def symbolic_matrix(name, n):
    """n x n matrix whose entries are independent symbols name_ij."""
    _check_n(n)
    return sp.Matrix(n, n, lambda i, j: sp.Symbol(f"{name}_{i + 1}{j + 1}"))


# ---- identities ------------------------------------------------------------------------------
def prove_identities(n=2):
    """Each identity as (lhs - rhs) simplified; a dict of name -> True when the difference is 0."""
    A, B = symbolic_matrix("a", n), symbolic_matrix("b", n)
    checks = {
        "det(AB) = det(A) det(B)": sp.expand((A * B).det() - A.det() * B.det()),
        "(AB)^T = B^T A^T": (A * B).T - B.T * A.T,
        "tr(AB) = tr(BA)": sp.expand((A * B).trace() - (B * A).trace()),
        "det(A^T) = det(A)": sp.expand(A.T.det() - A.det()),
    }
    if n <= 3:   # symbolic inverses get large fast
        checks["(AB)^-1 = B^-1 A^-1"] = sp.simplify((A * B).inv() - B.inv() * A.inv())
    return {k: (v == sp.zeros(*v.shape) if isinstance(v, sp.MatrixBase) else v == 0) for k, v in checks.items()}


def cauchy_schwarz(n=2):
    """Return (gap, sum_of_squares) with gap = |u|^2|v|^2 - (u.v)^2; they are equal, so gap >= 0."""
    _check_n(n)
    u, v = sp.symbols(f"u1:{n + 1}", real=True), sp.symbols(f"v1:{n + 1}", real=True)
    gap = sum(x**2 for x in u) * sum(y**2 for y in v) - sum(x * y for x, y in zip(u, v))**2
    sos = sum((u[i] * v[j] - u[j] * v[i])**2 for i in range(n) for j in range(i + 1, n))
    return sp.expand(gap), sp.expand(sos)


def projection(v):
    """Orthogonal projection onto span(v). Returns P and a dict of the properties that hold."""
    v = sp.Matrix(v)
    if v.shape[1] != 1 or all(x == 0 for x in v):
        raise ValueError("v must be a nonzero column vector")
    P = v * v.T / (v.T * v)[0]
    I = sp.eye(v.shape[0])
    props = {"P^T = P": sp.simplify(P.T - P) == sp.zeros(*P.shape),
             "P P = P": sp.simplify(P * P - P) == sp.zeros(*P.shape),
             "(1-P)(1-P) = 1-P": sp.simplify((I - P) * (I - P) - (I - P)) == sp.zeros(*P.shape),
             "P v = v": sp.simplify(P * v - v) == sp.zeros(*v.shape)}
    return P, props


def kernel_midpoint(A):
    """Null space basis of A, and a check that the midpoint of two null vectors is still null."""
    A = sp.Matrix(A)
    basis = A.nullspace()
    if not basis:
        return basis, True
    v, w = basis[0], (basis[-1] * 3 + basis[0])
    return basis, A * ((v + w) / 2) == sp.zeros(A.shape[0], 1)


# ---- supply and demand -----------------------------------------------------------------------
def market(a, b, c, d):
    """Demand Q = a - b P, supply Q = c + d P. Solve [[1, b], [1, -d]] [Q, P] = [a, c].

    b, d are the slopes (b > 0: people buy less at a higher price; d > 0: makers sell more).
    Returns dict with the matrix A, vector rhs, equilibrium P and Q, and det A = -(b + d).
    """
    a, b, c, d = map(sp.sympify, (a, b, c, d))
    for name, s in (("b", b), ("d", d)):
        if s.is_number and not s > 0:
            raise ValueError(f"slope {name} must be positive, got {s}")
    A = sp.Matrix([[1, b], [1, -d]])
    rhs = sp.Matrix([a, c])
    Q, P = A.LUsolve(rhs)
    P, Q = sp.simplify(P), sp.simplify(Q)
    if P.is_number and Q.is_number and (P < 0 or Q < 0):
        raise ValueError(f"no market: equilibrium P={P}, Q={Q} is negative (supply starts above demand)")
    return {"A": A, "rhs": rhs, "P": P, "Q": Q, "det": sp.simplify(A.det())}


# Illustrative (made-up) numbers: a photonics part such as a 1550 nm DFB laser diode.
# Q in thousands of units per month, P in dollars.
PHOTONICS_MARKET = {"a": 120, "b": 0.4, "c": 20, "d": 0.6}


def market_shift(a, b, c, d, demand_shift=0, supply_shift=0):
    """Equilibrium before and after shifting demand (a) and supply (c) by the given amounts."""
    before = market(a, b, c, d)
    after = market(a + demand_shift, b, c + supply_shift, d)
    return {"before": (before["P"], before["Q"]), "after": (after["P"], after["Q"]),
            "dP": sp.simplify(after["P"] - before["P"]), "dQ": sp.simplify(after["Q"] - before["Q"])}


def _gcc():
    gcc = shutil.which("gcc") or next((p for p in [r"C:\msys64\mingw64\bin\gcc.exe"] if os.path.exists(p)), None)
    if gcc is None:
        raise RuntimeError("gcc not found (MinGW-w64 at C:\\msys64\\mingw64\\bin, or build-essential)")
    return gcc


def market_c(a, b, c, d, demand_shift=0.0, supply_shift=0.0):
    """Compile dgs/c/market/market.c (C89, -Werror) and run it. Returns dict of floats."""
    gcc = _gcc()
    env = dict(os.environ, PATH=str(Path(gcc).parent) + os.pathsep + os.environ.get("PATH", ""))
    exe = Path(tempfile.gettempdir()) / "dgs_market" / ("market.exe" if os.name == "nt" else "market")
    if not exe.exists() or exe.stat().st_mtime < C_SRC.stat().st_mtime:
        exe.parent.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run([gcc, "-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", str(C_SRC), "-o", str(exe)],
                            capture_output=True, text=True, env=env)
        if cp.returncode:
            raise RuntimeError("gcc failed:\n" + cp.stderr[-2000:])
    args = [str(x) for x in (a, b, c, d, demand_shift, supply_shift)]
    cp = subprocess.run([str(exe), *args], capture_output=True, text=True, env=env)
    if cp.returncode:
        raise ValueError(cp.stderr.strip() or cp.stdout.strip())
    return {k: float(v) for k, v in (line.split("=") for line in cp.stdout.split())}


# ---- attention -------------------------------------------------------------------------------
def softmax_symbolic(n=3):
    """Softmax of n symbolic scores; returns (weights, sum simplified, shift difference)."""
    _check_n(n)
    x, c = sp.symbols(f"x1:{n + 1}", real=True), sp.Symbol("c", real=True)
    w = [sp.exp(xi) / sum(sp.exp(xj) for xj in x) for xi in x]
    ws = [sp.exp(xi + c) / sum(sp.exp(xj + c) for xj in x) for xi in x]
    return w, sp.simplify(sum(w)), [sp.simplify(a - b) for a, b in zip(ws, w)]


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention: softmax(Q K^T / sqrt(d)) V. Returns (output, weights).

    Q: (..., n_q, d), K: (..., n_k, d), V: (..., n_k, d_v). mask: bool (..., n_q, n_k), True = keep.
    """
    import torch
    if Q.shape[-1] != K.shape[-1] or K.shape[-2] != V.shape[-2]:
        raise ValueError(f"shape mismatch: Q {tuple(Q.shape)}, K {tuple(K.shape)}, V {tuple(V.shape)}")
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)    # torch subtracts the row max first (softmax_shift)
    return weights @ V, weights


def check_attention(n=5, d=8, seed=0):
    """Compare attention() with torch's fused kernel and test the linear algebra facts. Returns dict."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    Q, K, V = (torch.randn(n, d, generator=g, dtype=torch.float64) for _ in range(3))
    out, w = attention(Q, K, V)
    perm = torch.randperm(n, generator=g)
    out_p, _ = attention(Q[perm], K, V)
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    out_c, w_c = attention(Q, K, V, mask=causal)
    return {
        "max |ours - torch|": (out - F.scaled_dot_product_attention(Q, K, V)).abs().max().item(),
        "causal max |ours - torch|": (out_c - F.scaled_dot_product_attention(Q, K, V, is_causal=True)).abs().max().item(),
        "rows sum to 1": torch.allclose(w.sum(-1), torch.ones(n, dtype=torch.float64)),
        "weights > 0": bool((w > 0).all()),
        "permute queries -> permuted output": torch.allclose(out_p, out[perm]),
        "causal: no weight on the future": bool((w_c.triu(1) == 0).all()),
        "output in convex hull of V (per column)": bool(((out >= V.min(0).values - 1e-12) &
                                                          (out <= V.max(0).values + 1e-12)).all()),
    }
