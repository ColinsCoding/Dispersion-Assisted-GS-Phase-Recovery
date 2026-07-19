#!/usr/bin/env python
"""Builds notebooks/griffiths_problem_1_8_linear_algebra.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Griffiths Problem 1.8, in Linear Algebra / Operator Notation

**Problem.** Add a constant $V_0$ (independent of $x$ AND $t$) to the
potential. Classically this changes nothing. In quantum mechanics, show the
wavefunction picks up a phase $e^{-iV_0t/\hbar}$, and find the effect on
$\langle Q\rangle$ for any dynamical variable $Q$.

**The PDE, recast as a linear ODE on a vector space.** The time-dependent
Schrodinger equation

$$i\hbar\frac{\partial\Psi}{\partial t} = \hat H\Psi$$

is a PDE in $x,t$ -- but treat $\Psi(\cdot,t)$ as a single vector $|\Psi(t)\rangle$
in Hilbert space (a function of $x$ collapsed into "the state at time $t$"),
and it becomes EXACTLY the linear matrix ODE

$$\frac{d|\Psi\rangle}{dt} = -\frac{i}{\hbar}\hat H\,|\Psi\rangle,$$

the identical structure as $\dot{\vec p}=K\vec p$ solved by eigendecomposition
in `dgs/photosynthesis_energy_transfer.py` -- same linear-algebra machine,
applied to quantum state evolution instead of population transfer. We solve
Problem 1.8 in that language: with $\hat H$ as a finite Hermitian matrix
(the cleanest setting to do real linear algebra in), $V_0$ becomes
$V_0\hat I$ (a scalar times the identity OPERATOR), and the whole problem
reduces to one fact: **a scalar matrix commutes with everything.**
""")

code(r"""import sympy as sp
sp.init_printing()

t, V0, hbar = sp.symbols("t V_0 hbar", real=True, positive=True)
i = sp.I
print("SymPy", sp.__version__, "ready")
""")

md(r"""## 1. A concrete finite-dimensional $\hat H$ (3-level system, for explicit linear algebra)

Any Hermitian matrix stands in for $\hat H$ here -- the argument doesn't
care about the specific system, only that $\hat H$ is Hermitian and $V_0\hat I$
is a scalar matrix.
""")

code(r"""H = sp.Matrix([
    [2, 1, 0],
    [1, 3, 1],
    [0, 1, 1],
])
print("H (Hermitian, real symmetric here for simplicity) =")
sp.pprint(H)
print("\nH is Hermitian (H = H^dagger):", H == H.conjugate().T)

I3 = sp.eye(3)
H_prime = H + V0 * I3
print("\nH' = H + V0*I =")
sp.pprint(H_prime)
""")

md(r"""## 2. The key linear-algebra fact: $V_0\hat I$ commutes with $\hat H$

This is the entire content of the problem, stripped to one line of linear
algebra: a scalar multiple of the identity commutes with EVERY matrix.
""")

code(r"""commutator = sp.simplify(H * (V0 * I3) - (V0 * I3) * H)
print("[H, V0*I] = H(V0 I) - (V0 I)H =")
sp.pprint(commutator)
print("\ncommutes (zero matrix):", commutator == sp.zeros(3, 3))
""")

md(r"""## 3. The time-evolution operator factorizes: $U'(t) = e^{-iV_0t/\hbar}\,U(t)$

The formal solution of $\frac{d|\Psi\rangle}{dt}=-\frac{i}{\hbar}\hat H|\Psi\rangle$
is $|\Psi(t)\rangle = U(t)|\Psi(0)\rangle$ with $U(t)=e^{-i\hat Ht/\hbar}$.
Because $\hat H$ and $V_0\hat I$ commute, the matrix-exponential law
$e^{A+B}=e^Ae^B$ (which needs $[A,B]=0$ to hold for matrices) applies
directly:
""")

code(r"""# build the propagators via eigendecomposition (exact for a constant H,
# same method as the photosynthesis rate-matrix module): U(t) = V exp(diag(-i*lam*t/hbar)) V^-1
eigvecs, eigvals_diag = H.diagonalize()
lambdas = [eigvals_diag[k, k] for k in range(3)]
print("eigenvalues of H (this matrix happens to have messy cube-root roots --")
print("that's fine, the argument below never needs to simplify through them)")

U_diag = sp.diag(*[sp.exp(-i * lam * t / hbar) for lam in lambdas])
U_t = eigvecs * U_diag * eigvecs.inv()
print("\nU(t) = exp(-i H t/hbar) = V * diag(exp(-i*lambda_k*t/hbar)) * V^-1")

# H' = H + V0*I has the SAME eigenvectors (V), eigenvalues shifted by V0 --
# so the claim U'(t) = exp(-i V0 t/hbar) * U(t) only needs to be checked at
# the DIAGONAL level (trivial: exp(-i(lam+V0)t/hbar) = exp(-i V0 t/hbar) *
# exp(-i lam t/hbar) for each eigenvalue individually). Since both sides get
# conjugated by the IDENTICAL V * (...) * V^-1, equality of the diagonals
# guarantees equality of the full matrices -- no need to fight sp.simplify
# through cube-root eigenvector entries, which is what hung the first
# version of this cell for 180+ seconds and then gave a symbolic false
# negative on the second attempt.
lambdas_prime = [lam + V0 for lam in lambdas]
diag_lhs = [sp.exp(-i * lam_p * t / hbar) for lam_p in lambdas_prime]
diag_rhs = [sp.exp(-i * V0 * t / hbar) * sp.exp(-i * lam * t / hbar) for lam in lambdas]
diag_match = all(sp.simplify(a - b) == 0 for a, b in zip(diag_lhs, diag_rhs))
print("\ndiagonal entries of U'(t) match exp(-i V0 t/hbar) * [diagonal entries of U(t)]:", diag_match)
print("(since U'(t) and exp(-i V0 t/hbar) U(t) are both V * (this diagonal) * V^-1,")
print(" the full matrices are therefore equal too -- proven without ever simplifying")
print(" through the irrational eigenvector entries)")
""")

md(r"""## 4. Therefore $|\Psi'(t)\rangle = e^{-iV_0t/\hbar}|\Psi(t)\rangle$ -- exactly the claimed phase

$$|\Psi'(t)\rangle = U'(t)|\Psi(0)\rangle = e^{-iV_0t/\hbar}U(t)|\Psi(0)\rangle = e^{-iV_0t/\hbar}|\Psi(t)\rangle$$

The ENTIRE wavefunction (every component, at every $x$) picks up the same
global scalar phase $e^{-iV_0t/\hbar}$ -- not a different phase at different
points, which is exactly Griffiths' claimed result.
""")

md(r"""## 5. Effect on $\langle Q\rangle$: a global phase is invisible to any expectation value

$$\langle Q\rangle' = \langle\Psi'|\hat Q|\Psi'\rangle
= \left(e^{+iV_0t/\hbar}\langle\Psi|\right)\hat Q\left(e^{-iV_0t/\hbar}|\Psi\rangle\right)
= e^{+iV_0t/\hbar}e^{-iV_0t/\hbar}\langle\Psi|\hat Q|\Psi\rangle = \langle Q\rangle$$

because the phase factors are ordinary complex SCALARS (not operators) --
they pull straight out of the bra and ket and cancel. Verify this directly
on the finite matrix model: compute $\langle Q\rangle$ and $\langle Q\rangle'$
for an arbitrary Hermitian observable $\hat Q$ and an arbitrary state, and
confirm they're identical.
""")

code(r"""Q = sp.Matrix([
    [1, 0, 2],
    [0, 3, 0],
    [2, 0, 1],
])
print("a candidate observable Q (Hermitian) =")
sp.pprint(Q)

# U_t's eigenvectors involve nested radicals (irrational eigenvalues of H),
# so fully symbolic sp.simplify on <Q>(t) is computationally explosive --
# it hung for 180s+ on the first attempt. The general algebraic argument is
# already proven above (Sections 3-4); here, substitute concrete numeric
# values for t, hbar, V0 to make the <Q> vs <Q>' comparison a fast, concrete
# check instead, which is all this cell needs to demonstrate.
psi0 = sp.Matrix([1, 2, 1])   # an arbitrary (unnormalized) initial state, for illustration
numeric_vals = {t: sp.Rational(3, 2), hbar: 1, V0: sp.Rational(5, 2)}

U_t_num = U_t.subs(numeric_vals).evalf()
psi_t_num = U_t_num * psi0
phase_num = sp.exp(-i * numeric_vals[V0] * numeric_vals[t] / numeric_vals[hbar]).evalf()
psi_prime_t_num = phase_num * psi_t_num

Q_expectation = sp.simplify((psi_t_num.conjugate().T * Q * psi_t_num)[0])
Q_expectation_prime = sp.simplify((psi_prime_t_num.conjugate().T * Q * psi_prime_t_num)[0])

print(f"\nat t={numeric_vals[t]}, hbar={numeric_vals[hbar]}, V0={numeric_vals[V0]}:")
print("<Q>(t)  =", Q_expectation)
print("<Q>'(t) =", Q_expectation_prime)

print("\nidentical (the global phase has zero physical effect):",
      abs(complex(Q_expectation) - complex(Q_expectation_prime)) < 1e-9)
""")

md(r"""## Summary

Reframing the Schrodinger PDE as a linear ODE $\dot{|\Psi\rangle}=-\frac{i}{\hbar}\hat H|\Psi\rangle$
on a Hilbert space turns Problem 1.8 into one linear-algebra fact: $V_0\hat I$
is a scalar matrix, scalar matrices commute with everything, so the
propagator factorizes exactly as $U'(t)=e^{-iV_0t/\hbar}U(t)$ -- proving the
claimed phase directly from the structure of matrix exponentials rather than
from solving the PDE by separation of variables. Because that phase is a
genuine **scalar** (not an operator), it cancels exactly between the bra and
the ket in ANY expectation value $\langle Q\rangle$ -- confirmed both
algebraically and on the explicit 3-level matrix model above. Physically:
$V_0$ shifts every energy eigenvalue by the same amount and changes nothing
observable, the quantum-mechanical mirror of the classical fact that only
energy DIFFERENCES matter.
""")

nb["cells"] = cells

import os
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/griffiths_problem_1_8_linear_algebra.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/griffiths_problem_1_8_linear_algebra.ipynb with", len(cells), "cells")
