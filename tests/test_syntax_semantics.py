"""Test dgs.syntax_semantics: the syntax/semantics split (well-formed vs
meaningful), and that the semantics of transpose/sqrt/exp/log is quantum
mechanics -- adjoint & real eigenvalues, Born normalization, unitary evolution,
von Neumann entropy -- with a gcc cross-check of the scalar ops when available."""
import sys, pathlib, math, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import syntax_semantics as ss

# 1. SYNTAX: parsing builds an AST and evaluates; classify separates the cases
assert ss.parse("1+2")[0] == "binop"
assert math.isclose(ss.evaluate(ss.parse("sqrt(2) + log(exp(1))")), math.sqrt(2) + 1)
assert math.isclose(ss.evaluate(ss.parse("2*(3+4)")), 14)
assert math.isclose(ss.evaluate(ss.parse("log(-1*-1)")), 0.0)   # unary minus parses

ok = ss.classify("sqrt(2) + log(exp(1))")
assert ok["kind"] == "ok" and math.isclose(ok["value"], math.sqrt(2) + 1)

# well-formed grammar failures -> syntax_error (never evaluated)
for bad in ("log(", "sqrt(2)*", "2 @ 3", "(1+2", "foo(3)"):
    assert ss.classify(bad)["kind"] == "syntax_error", bad

# well-formed but MEANINGLESS -> semantic_error (the domain = the physics)
for undefined in ("log(-1)", "sqrt(3-5)", "1/(2-2)", "log(0)"):
    assert ss.classify(undefined)["kind"] == "semantic_error", undefined

# the exceptions are distinct types
try:
    ss.parse("log(")
    assert False
except ss.SyntaxErrorPL:
    pass
try:
    ss.evaluate(ss.parse("log(-1)"))
    assert False
except ss.SemanticError:
    pass

# 2. SEMANTICS = QM. transpose -> adjoint; Hermitian observable has real eigenvalues
A = np.array([[2, 1j], [-1j, 3]])                 # Hermitian
assert np.allclose(ss.dagger(ss.dagger(A)), A)    # (A-dagger)-dagger = A
assert np.allclose(ss.dagger(np.array([[1.0, 2], [3, 4]])), [[1, 3], [2, 4]])  # real: plain transpose
assert ss.is_hermitian(A) and not ss.is_hermitian(np.array([[0, 1], [0, 0]]))
real, eigs = ss.observable_eigenvalues_real(A)
assert real and np.allclose(eigs, [(5 - np.sqrt(5)) / 2, (5 + np.sqrt(5)) / 2])
try:
    ss.observable_eigenvalues_real(np.array([[0, 1], [0, 0]]))   # not Hermitian
    assert False
except ValueError:
    pass

# 3. sqrt -> normalization: Born probabilities sum to 1
psi = ss.normalize([3, 4j])
assert np.isclose(np.linalg.norm(psi), 1.0)
p = ss.born_probabilities([3, 4j])
assert np.isclose(p.sum(), 1.0) and np.allclose(p, [9/25, 16/25])

# 4. exp -> unitary time evolution, norm-preserving, group property
H = np.array([[1.0, 0.5], [0.5, -1.0]])
U = ss.time_evolution(H, t=0.7)
assert ss.is_unitary(U)
assert np.allclose(ss.time_evolution(H, 0.0), np.eye(2))          # U(0) = I
assert np.allclose(ss.time_evolution(H, 0.3) @ ss.time_evolution(H, 0.4), U)  # U(t1)U(t2)=U(t1+t2)
psi0 = ss.normalize([1, 1j])
assert np.isclose(np.linalg.norm(U @ psi0), 1.0)                  # probability conserved
# diagonal H gives pure phases
Ud = ss.time_evolution(np.diag([2.0, -1.0]), t=1.0)
assert np.allclose(np.diag(Ud), [np.exp(-2j), np.exp(1j)])
try:
    ss.time_evolution(np.array([[0, 1], [0, 0]]), 1.0)            # non-Hermitian H
    assert False
except ValueError:
    pass

# 5. log -> von Neumann entropy: 0 for pure, log(n) for maximally mixed
pure = np.array([[1, 0], [0, 0]], float)
assert abs(ss.von_neumann_entropy(pure)) < 1e-12
assert np.isclose(ss.von_neumann_entropy(np.eye(2) / 2), math.log(2))     # nats
assert np.isclose(ss.von_neumann_entropy(np.eye(2) / 2, base=2), 1.0)     # bits
assert np.isclose(ss.von_neumann_entropy(np.eye(4) / 4, base=2), 2.0)     # 2 qubits
try:
    ss.von_neumann_entropy(np.array([[2, 0], [0, 0]], float))     # trace != 1
    assert False
except ValueError:
    pass

# 6. quantum mechanics AND C code: the scalar ops match, if gcc is present
if ss.gcc_available():
    with tempfile.TemporaryDirectory() as d:
        c = ss.compile_and_run_c(d)
    assert np.isclose(c["sqrt"], math.sqrt(2))
    assert np.isclose(c["exp"], math.exp(2))
    assert np.isclose(c["log"], math.log(2))
    assert c["transpose"] == [1.0, 3.0, 2.0, 4.0]     # [[1,2],[3,4]]^T row-major
    print("test_syntax_semantics: all checks passed (incl. C cross-check)")
else:
    print("test_syntax_semantics: all checks passed (C cross-check skipped: no gcc)")
