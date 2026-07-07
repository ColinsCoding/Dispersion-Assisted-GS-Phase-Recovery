"""Syntax vs semantics: a statement's MEANING is physics.

Every programming language separates two things a beginner blurs:

  SYNTAX   -- the grammar. Is the text a well-formed statement? Purely about
              shape: `sqrt(2)` parses, `sqrt(` does not. A parser decides this
              and builds an abstract syntax tree; it never computes anything.
  SEMANTICS-- the meaning. WHAT does the well-formed statement compute, and is
              that even defined? `log(-1)` is perfectly grammatical yet has no
              real value -- its semantics is constrained by the MATH (here, the
              physics) the symbol denotes.

This module makes the split concrete with a tiny expression language over four
functions -- transpose, sqrt, exp, log -- and then shows the punchline: the
SEMANTICS of those four operations is QUANTUM MECHANICS.

  transpose (conjugate)  -> the ADJOINT A-dagger. An observable is Hermitian,
                            A = A-dagger, which forces its eigenvalues (measured
                            values) to be REAL.
  sqrt                   -> NORMALIZATION. A state divided by sqrt(<psi|psi>) has
                            Born probabilities |psi|^2 summing to 1.
  exp                    -> UNITARY TIME EVOLUTION U = exp(-i H t / hbar), which
                            preserves the norm (probability is conserved).
  log                    -> VON NEUMANN ENTROPY S = -Tr(rho log rho): 0 for a
                            pure state, log(n) for the maximally mixed one.

So `y = exp(x)` is not just grammar plus a number: its semantics -- the thing the
statement MEANS -- is the physics operation the symbol stands for. Each identity
is verified in NumPy, and the four scalar ops are cross-checked in real C
(compiled with gcc, the same pattern as dgs.c_type_precision) so "quantum
mechanics and C code" is literal, not a metaphor. NumPy + stdlib; py-3.13.
"""

import os
import math
import subprocess

import numpy as np

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"


# ----------------------------------------------------------------------
# SYNTAX: tokenize + parse a tiny expression grammar into an AST
# ----------------------------------------------------------------------

class SyntaxErrorPL(Exception):
    """Raised when text is not well-formed -- a GRAMMAR violation."""


class SemanticError(Exception):
    """Raised when a well-formed statement has no defined value -- a MEANING
    (domain) violation, e.g. log of a non-positive number."""


_FUNCS = ("sqrt", "exp", "log")


def tokenize(s):
    """Split source text into tokens (numbers, function names, operators). A
    stray character is already a SYNTAX error."""
    tokens, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "+-*/()":
            tokens.append(c); i += 1
        elif c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            tokens.append(("num", float(s[i:j]))); i = j
        elif c.isalpha():
            j = i
            while j < len(s) and s[j].isalpha():
                j += 1
            name = s[i:j]
            if name not in _FUNCS:
                raise SyntaxErrorPL(f"unknown function {name!r}")
            tokens.append(("func", name)); i = j
        else:
            raise SyntaxErrorPL(f"illegal character {c!r}")
    return tokens


def parse(s):
    """Grammar (recursive descent):
        expr   := term (('+'|'-') term)*
        term   := factor (('*'|'/') factor)*
        factor := num | func '(' expr ')' | '(' expr ')'
    Returns an AST of nested tuples -- the SYNTAX, with no evaluation. Raises
    SyntaxErrorPL on any malformed input."""
    tokens = tokenize(s)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def eat(expected=None):
        nonlocal pos
        if pos >= len(tokens):
            raise SyntaxErrorPL("unexpected end of input")
        tok = tokens[pos]; pos += 1
        if expected is not None and tok != expected:
            raise SyntaxErrorPL(f"expected {expected!r}, got {tok!r}")
        return tok

    def expr():
        node = term()
        while peek() in ("+", "-"):
            op = eat()
            node = ("binop", op, node, term())
        return node

    def term():
        node = factor()
        while peek() in ("*", "/"):
            op = eat()
            node = ("binop", op, node, factor())
        return node

    def factor():
        tok = peek()
        if tok is None:
            raise SyntaxErrorPL("unexpected end of input")
        if tok in ("+", "-"):                       # unary sign: -1, log(-1)
            op = eat()
            return ("binop", op, ("num", 0.0), factor())
        if isinstance(tok, tuple) and tok[0] == "num":
            eat(); return tok
        if isinstance(tok, tuple) and tok[0] == "func":
            eat(); eat("("); arg = expr(); eat(")")
            return ("func", tok[1], arg)
        if tok == "(":
            eat(); node = expr(); eat(")")
            return node
        raise SyntaxErrorPL(f"unexpected token {tok!r}")

    tree = expr()
    if pos != len(tokens):
        raise SyntaxErrorPL(f"trailing tokens from position {pos}")
    return tree


def evaluate(ast):
    """Give the AST its SEMANTICS -- a number -- applying the domain rules the
    physics imposes (sqrt needs >= 0, log needs > 0, no divide by zero). Raises
    SemanticError when a grammatical expression is nonetheless undefined."""
    kind = ast[0]
    if kind == "num":
        return ast[1]
    if kind == "binop":
        _, op, l, r = ast
        a, b = evaluate(l), evaluate(r)
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/":
            if b == 0:
                raise SemanticError("division by zero")
            return a / b
    if kind == "func":
        _, name, arg = ast
        x = evaluate(arg)
        if name == "sqrt":
            if x < 0:
                raise SemanticError("sqrt of a negative number is not real "
                                    "(a probability cannot be negative)")
            return math.sqrt(x)
        if name == "log":
            if x <= 0:
                raise SemanticError("log of a non-positive number is undefined "
                                    "(an eigenvalue/probability must be > 0)")
            return math.log(x)
        if name == "exp":
            return math.exp(x)
    raise SemanticError(f"malformed AST node {ast!r}")


def classify(expr_str):
    """The whole point, in one call: is a statement a SYNTAX error, a SEMANTIC
    (domain) error, or OK with a value? Distinguishes 'not well-formed' from
    'well-formed but meaningless'."""
    try:
        ast = parse(expr_str)
    except SyntaxErrorPL as e:
        return {"kind": "syntax_error", "detail": str(e)}
    try:
        return {"kind": "ok", "value": evaluate(ast)}
    except SemanticError as e:
        return {"kind": "semantic_error", "detail": str(e)}


# ----------------------------------------------------------------------
# SEMANTICS = quantum mechanics: what the four operations MEAN
# ----------------------------------------------------------------------

def dagger(A):
    """Conjugate transpose A-dagger -- the ADJOINT. Transpose is the real
    special case; for complex operators you must also conjugate."""
    return np.asarray(A, complex).conj().T


def is_hermitian(A, tol=1e-9):
    """A = A-dagger: the defining property of a quantum observable."""
    A = np.asarray(A, complex)
    return bool(np.allclose(A, dagger(A), atol=tol))


def observable_eigenvalues_real(A, tol=1e-9):
    """Semantics of Hermiticity: a Hermitian operator's eigenvalues (the
    possible measured values) are REAL. Returns (all_real, eigenvalues)."""
    if not is_hermitian(A, tol):
        raise ValueError("operator is not Hermitian (not an observable)")
    w = np.linalg.eigvals(np.asarray(A, complex))
    return bool(np.all(np.abs(w.imag) < tol)), np.sort(w.real)


def normalize(psi):
    """Divide a state by sqrt(<psi|psi>) so |psi|^2 is a probability
    distribution -- the semantics of `sqrt` in QM (the Born rule's root)."""
    psi = np.asarray(psi, complex)
    n = math.sqrt(np.vdot(psi, psi).real)
    if n == 0:
        raise ValueError("zero vector cannot be normalized")
    return psi / n


def born_probabilities(psi):
    """|psi_i|^2 for a normalized state: they sum to 1."""
    psi = normalize(psi)
    return np.abs(psi) ** 2


def time_evolution(H, t, hbar=1.0):
    """Unitary evolution U = exp(-i H t / hbar), built from H's spectral
    decomposition (H = V L V-dagger => U = V exp(-i L t/hbar) V-dagger). The
    semantics of `exp` in QM. H must be Hermitian."""
    H = np.asarray(H, complex)
    if not is_hermitian(H):
        raise ValueError("H must be Hermitian to generate unitary evolution")
    w, V = np.linalg.eigh(H)
    phases = np.exp(-1j * w * t / hbar)
    return (V * phases) @ dagger(V)


def is_unitary(U, tol=1e-9):
    """U U-dagger = I: unitary evolution preserves total probability."""
    U = np.asarray(U, complex)
    return bool(np.allclose(U @ dagger(U), np.eye(len(U)), atol=tol))


def von_neumann_entropy(rho, base=None, tol=1e-12):
    """S = -Tr(rho log rho) = -sum lambda log lambda over the density matrix's
    eigenvalues -- the semantics of `log` in QM. 0 for a pure state, log(n) for
    the maximally mixed state. base=None gives nats; base=2 gives bits."""
    rho = np.asarray(rho, complex)
    if not is_hermitian(rho):
        raise ValueError("density matrix must be Hermitian")
    if not np.isclose(np.trace(rho).real, 1.0, atol=1e-6):
        raise ValueError("density matrix must have trace 1")
    w = np.linalg.eigvalsh(rho)
    if np.any(w < -1e-9):
        raise ValueError("density matrix must be positive semidefinite")
    w = w[w > tol]                             # 0 log 0 -> 0
    S = float(-np.sum(w * np.log(w)))
    return S if base is None else S / math.log(base)


# ----------------------------------------------------------------------
# quantum mechanics AND C code: the four scalar ops compiled with gcc
# ----------------------------------------------------------------------

C_SOURCE_QM_OPS = r"""
#include <stdio.h>
#include <math.h>

int main(void) {
    double x = 2.0;
    /* the three scalar operations whose QM meaning is normalization,
       time evolution, and entropy */
    printf("%.15e %.15e %.15e\n", sqrt(x), exp(x), log(x));
    /* transpose of a 2x2 (the real face of the adjoint): print A^T row-major */
    double A[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    printf("%.1f %.1f %.1f %.1f\n", A[0][0], A[1][0], A[0][1], A[1][1]);
    return 0;
}
"""


def gcc_available(gcc_path=GCC_DEFAULT):
    """Whether the C toolchain is present (so the C cross-check can run)."""
    return os.path.exists(gcc_path)


def compile_and_run_c(out_dir, gcc_path=GCC_DEFAULT):
    """Compile C_SOURCE_QM_OPS with gcc and run it, parsing sqrt/exp/log and the
    2x2 transpose it prints. Same gcc-subprocess pattern as
    dgs.c_type_precision. Returns a dict of the C results."""
    src = os.path.join(out_dir, "qm_ops.c")
    exe = os.path.join(out_dir, "qm_ops.exe")
    with open(src, "w") as f:
        f.write(C_SOURCE_QM_OPS)
    r = subprocess.run([gcc_path, "-O2", "-o", exe, src, "-lm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcc failed: {r.stderr}")
    out = subprocess.run([exe], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"C program failed: {out.stderr}")
    line0, line1 = out.stdout.strip().splitlines()
    sq, ex, lg = map(float, line0.split())
    transpose = list(map(float, line1.split()))
    return {"sqrt": sq, "exp": ex, "log": lg, "transpose": transpose}


if __name__ == "__main__":
    print("SYNTAX vs SEMANTICS")
    for e in ("sqrt(2) + log(exp(1))", "log(", "log(-1)", "1/(2-2)"):
        print(f"  {e:24s} -> {classify(e)}")

    print("\nSEMANTICS = quantum mechanics:")
    # transpose -> adjoint -> real eigenvalues of an observable
    A = np.array([[2, 1j], [-1j, 3]])
    real, eigs = observable_eigenvalues_real(A)
    print(f"  Hermitian A=A-dagger? {is_hermitian(A)}; eigenvalues real? {real} "
          f"-> {np.round(eigs, 3)}")
    # sqrt -> normalization
    p = born_probabilities([3, 4j])
    print(f"  sqrt: Born probabilities {np.round(p,3)} sum to {p.sum():.3f}")
    # exp -> unitary evolution, norm preserved
    H = np.array([[1.0, 0.5], [0.5, -1.0]])
    U = time_evolution(H, t=0.7)
    psi = normalize([1, 1j])
    print(f"  exp: U unitary? {is_unitary(U)}; norm before {np.linalg.norm(psi):.3f} "
          f"after {np.linalg.norm(U @ psi):.3f}")
    # log -> entropy
    pure = np.array([[1, 0], [0, 0]], float)
    mixed = np.eye(2) / 2
    print(f"  log: entropy pure={von_neumann_entropy(pure):.3f}, "
          f"maximally mixed={von_neumann_entropy(mixed, base=2):.3f} bits")

    print("\nquantum mechanics AND C code:")
    if gcc_available():
        scratch = os.environ.get("TEMP", ".")
        c = compile_and_run_c(scratch)
        print(f"  C: sqrt(2)={c['sqrt']:.6f} exp(2)={c['exp']:.6f} log(2)={c['log']:.6f}; "
              f"transpose {c['transpose']}")
        print(f"  matches Python? {np.isclose(c['sqrt'], math.sqrt(2)) and np.isclose(c['exp'], math.exp(2)) and np.isclose(c['log'], math.log(2))}")
    else:
        print("  (gcc not found -- skipping the C cross-check)")
