"""Build notebooks/probability_to_qm_operators.ipynb -- "Probability to
Quantum Mechanics: From Precalculus to Operators": classical probability
through complex numbers, vectors, the Born rule, operators/eigenvalues,
expectation values, position-space wavefunctions, the momentum operator,
the position<->momentum Fourier transform, and the uncertainty principle,
closing with 30 hand-worked problems (solutions in a separate section).

Build: py -3.13 scripts/build_probability_to_qm_operators_nb.py
Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
         notebooks/probability_to_qm_operators.ipynb
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

# ============================================================ Title
cells.append(md(r"""# Probability to Quantum Mechanics: From Precalculus to Operators

Built to be hand-derivable, not just runnable: every equation is defined,
every assumption stated, every algebraic step shown, and every symbolic
result cross-checked against a hand computation before being trusted.
Machine learning is not used anywhere as a substitute for a physical or
mathematical derivation.

**Build order:** probability $\to$ complex numbers $\to$ vectors $\to$
normalization $\to$ amplitudes $\to$ Born rule $\to$ operators $\to$
eigenvalues $\to$ expectation $\to$ wavefunctions $\to$ Fourier transform
$\to$ uncertainty."""))

cells.append(co("""import sympy as sp
sp.init_printing()

import numpy as np
import matplotlib.pyplot as plt

print(f"sympy {sp.__version__}, numpy {np.__version__}")"""))

# ============================================================ PART 1: classical probability
cells.append(md(r"""# Part 1 — Classical probability

**Sample space $\Omega$**: the set of every possible outcome of an
experiment. **Event $A$**: any subset of $\Omega$ ($A\subseteq\Omega$).

**Axioms** (Kolmogorov): for any event $A$,
$$0\le P(A)\le 1,\qquad P(\Omega)=1$$

**Complement.** $A^c=\Omega\setminus A$ (everything in $\Omega$ not in
$A$). Since $A$ and $A^c$ partition $\Omega$ exactly,
$P(A)+P(A^c)=P(\Omega)=1$, so
$$P(A^c) = 1 - P(A)$$

**Mutually exclusive** events $A,B$: $A\cap B=\emptyset$ (can't both
happen), so $P(A\cup B)=P(A)+P(B)$.

**Independent** events $A,B$: knowing one occurred doesn't change the
probability of the other, $P(A\cap B)=P(A)P(B)$.

**Conditional probability**: the probability of $A$ *given* $B$ already
happened, restricting the sample space to $B$:
$$P(A\mid B) = \frac{P(A\cap B)}{P(B)},\qquad P(B) \ne 0$$

**Bayes' theorem**: solve the conditional-probability definition for
$P(A\cap B)$ two ways ($P(A\mid B)P(B) = P(A\cap B) = P(B\mid A)P(A)$) and
equate:
$$P(A\mid B) = \frac{P(B\mid A)\,P(A)}{P(B)}$$

**Expectation, variance, standard deviation** (for a discrete random
variable $X$ taking values $x_i$ with probabilities $p_i$):
$$E[X]=\sum_i x_i p_i,\qquad
\mathrm{Var}(X)=E[X^2]-\big(E[X]\big)^2,\qquad
\sigma_X=\sqrt{\mathrm{Var}(X)}$$"""))

cells.append(co("""# verify Bayes' theorem symbolically: derive it from the conditional-probability
# definition rather than just typing the formula in
PA, PB, PAB, PAgB, PBgA = sp.symbols('P_A P_B P_AB P_AgB P_BgA', positive=True)

def_AgB = sp.Eq(PAgB, PAB / PB)   # P(A|B) = P(A and B)/P(B)
def_BgA = sp.Eq(PBgA, PAB / PA)   # P(B|A) = P(A and B)/P(A)

# solve each for P(A and B), set equal, solve for P(A|B) -- that's Bayes' theorem
PAB_from_AgB = sp.solve(def_AgB, PAB)[0]
PAB_from_BgA = sp.solve(def_BgA, PAB)[0]
bayes_derivation = sp.Eq(PAB_from_AgB, PAB_from_BgA)
print("Setting the two expressions for P(A and B) equal:")
display(bayes_derivation)

bayes_theorem = sp.Eq(PAgB, sp.solve(bayes_derivation, PAgB)[0])
print("Solved for P(A|B) -- Bayes' theorem:")
display(bayes_theorem)"""))

cells.append(co("""# complement rule, verified for a concrete numeric example
p_A = sp.Rational(3, 10)
p_not_A = 1 - p_A
print(f"P(A) = {p_A},  P(not A) = 1 - P(A) = {p_not_A},  "
      f"sum = {p_A + p_not_A}  (must equal 1)")
assert p_A + p_not_A == 1"""))

# ============================================================ PART 2: discrete RV
cells.append(md(r"""# Part 2 — Discrete random variables

A two-outcome (Bernoulli) experiment: $X\in\{0,1\}$, $P(X=1)=p$,
$P(X=0)=1-p$.

Because $X\in\{0,1\}$, $X^2=X$ for *every* outcome (both $0^2=0$ and
$1^2=1$) -- so $E[X^2]=E[X]=p$ *before* doing any algebra, a fact worth
noticing rather than just computing past."""))

cells.append(co("""p = sp.Symbol('p', positive=True)

# E[X] = sum over outcomes of (value * probability)
E_X = 0*(1-p) + 1*p
E_X2 = 0**2*(1-p) + 1**2*p     # X^2 has the SAME distribution as X here
Var_X = sp.simplify(E_X2 - E_X**2)

print("E[X] ="); display(E_X)
print("E[X^2] ="); display(E_X2)
print("Var(X) = E[X^2] - E[X]^2 ="); display(Var_X)

assert sp.simplify(E_X - E_X2) == 0, "X^2 = X for a 0/1 variable, so E[X^2] must equal E[X]"
assert sp.simplify(Var_X - p*(1-p)) == 0"""))

cells.append(md(r"""**One observation vs. probability vs. expectation vs. variance vs.
repeated experiments** — five different things that are easy to conflate:

- **One observation**: a single run gives exactly $X=0$ or $X=1$. No
  probability is directly visible in one run.
- **Probability** ($p$): a property of the *model*, not of any run.
- **Expectation** ($E[X]=p$): the long-run *average* over many runs — not
  a value $X$ can actually take unless $p\in\{0,1\}$.
- **Variance** ($p(1-p)$): how spread out repeated runs are around that
  average — zero only when the outcome is certain ($p=0$ or $p=1$).
- **Repeated experiments**: running the experiment $N$ times and averaging
  the results converges toward $E[X]=p$ as $N\to\infty$ (the law of large
  numbers) — verified numerically below, not just asserted."""))

cells.append(co("""rng = np.random.default_rng(0)
p_true = 0.3
for N in [10, 100, 1000, 100000]:
    samples = rng.random(N) < p_true
    print(f"N={N:>7}:  sample mean = {samples.mean():.4f}   (true p = {p_true})")"""))

# ============================================================ PART 3: complex numbers
cells.append(md(r"""# Part 3 — Complex numbers

**Definition.** $i$ is defined by $i^2=-1$ (no real number squares to a
negative number, so this extends the number system). A complex number is
$z=a+ib$ with $a,b\in\mathbb{R}$ ($a$ = real part, $b$ = imaginary part).

**Conjugate.** $z^*=a-ib$ (flip the sign of the imaginary part).

**Modulus squared.**
$$z^*z = (a-ib)(a+ib) = a^2 - (ib)^2 = a^2 - i^2b^2 = a^2+b^2 = |z|^2$$
— always real and non-negative, because the $i$'s cancelled entirely.

**Euler's identity**, from the Taylor series of $e^{i\theta}$, $\cos\theta$,
$\sin\theta$ term by term (verified symbolically, not asserted):
$$e^{i\theta} = \cos\theta + i\sin\theta$$"""))

cells.append(co("""a, b, theta = sp.symbols('a b theta', real=True)
z = a + sp.I*b
z_conj = sp.conjugate(z)
mod_sq = sp.expand(z_conj * z)

print("z* ="); display(z_conj)
print("z*z = |z|^2 ="); display(mod_sq)
assert mod_sq == a**2 + b**2

euler_lhs = sp.series(sp.exp(sp.I*theta), theta, 0, 6).removeO()
euler_rhs = sp.series(sp.cos(theta), theta, 0, 6).removeO() + sp.I*sp.series(sp.sin(theta), theta, 0, 6).removeO()
print("\\nexp(i*theta) Taylor series (to 5th order):"); display(sp.expand(euler_lhs))
print("cos(theta) + i*sin(theta), same order:"); display(sp.expand(euler_rhs))
assert sp.expand(euler_lhs - euler_rhs) == 0, "Euler's identity should match term by term"
print("\\n[OK] the two series agree term by term")"""))

cells.append(md(r"""**Why complex amplitudes can produce real probabilities.** A quantum
amplitude $\alpha$ can be any complex number, but a probability must be a
real number in $[0,1]$. The bridge is exactly the modulus-squared identity
above: $|\alpha|^2=\alpha^*\alpha$ is *always* real and non-negative,
regardless of what complex phase $\alpha$ carries — the imaginary parts
cancel algebraically, not by assumption. That is precisely why the Born
rule (Part 5) is written with $|\alpha|^2$, not $\alpha$ itself."""))

# ============================================================ PART 4: vectors
cells.append(md(r"""# Part 4 — Vectors

A two-level quantum state is represented as a column vector
$$|\psi\rangle = \begin{pmatrix}\alpha\\ \beta\end{pmatrix},\qquad \alpha,\beta\in\mathbb{C}$$

The **bra** $\langle\psi|$ is the conjugate transpose of $|\psi\rangle$:
$$\langle\psi| = \begin{pmatrix}\alpha^* & \beta^*\end{pmatrix}$$

Their product (matrix-multiplying a $1\times2$ by a $2\times1$) is the
**inner product**:
$$\langle\psi|\psi\rangle = \alpha^*\alpha + \beta^*\beta = |\alpha|^2+|\beta|^2$$

**Normalization** — a physically valid state must have
$|\alpha|^2+|\beta|^2=1$ (Part 5 explains why: this total is a total
probability, and total probability must be 1)."""))

cells.append(co("""alpha_r, alpha_i, beta_r, beta_i = sp.symbols(
    'alpha_r alpha_i beta_r beta_i', real=True)
alpha = alpha_r + sp.I*alpha_i
beta = beta_r + sp.I*beta_i

psi_ket = sp.Matrix([[alpha], [beta]])
psi_bra = psi_ket.conjugate().T     # <psi| = |psi>^dagger

inner_product = sp.expand((psi_bra * psi_ket)[0])
print("<psi|psi> ="); display(inner_product)

expected = alpha_r**2 + alpha_i**2 + beta_r**2 + beta_i**2   # = |alpha|^2 + |beta|^2
assert sp.simplify(inner_product - expected) == 0
print("[OK] <psi|psi> = |alpha|^2 + |beta|^2, verified with SymPy matrices")

# a normalized numeric example
psi_num = sp.Matrix([[sp.Rational(3,5)], [sp.Rational(4,5)*sp.I]])
norm_num = (psi_num.conjugate().T * psi_num)[0]
print(f"\\nExample psi = (3/5, 4i/5): <psi|psi> = {norm_num}  (normalized: {norm_num == 1})")"""))

# ============================================================ PART 5: qubit probability
cells.append(md(r"""# Part 5 — Qubit probability (the Born rule)

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

**Born rule**: the probability of measuring the state and getting outcome
$0$ is $P(0)=|\alpha|^2$; getting $1$ is $P(1)=|\beta|^2$.
**$\alpha$ and $\beta$ are amplitudes, not probabilities themselves** —
they can be negative, complex, or have any phase; only their
*modulus-squared* is a probability. Normalization
($|\alpha|^2+|\beta|^2=1$, Part 4) guarantees $P(0)+P(1)=1$ automatically."""))

cells.append(co("""alpha_ex = 1/sp.sqrt(2)
beta_ex = 1/sp.sqrt(2)

P0 = sp.Abs(alpha_ex)**2
P1 = sp.Abs(beta_ex)**2
total = sp.simplify(P0 + P1)

print("psi = (1/sqrt(2))|0> + (1/sqrt(2))|1>\\n")
print("P(0) = |alpha|^2 ="); display(P0)
print("P(1) = |beta|^2  ="); display(P1)
print("P(0) + P(1) ="); display(total)
assert total == 1
print("\\n[OK] normalized: P(0) + P(1) = 1")
print("\\nNote: alpha = 1/sqrt(2) itself is NOT a probability (it isn't in [0,1] the way")
print("a probability must be, and in general could be negative or complex) -- only")
print("|alpha|^2 = 1/2 is the probability.")"""))

# ============================================================ PART 6: observables/operators
cells.append(md(r"""# Part 6 — Observables and operators

Three distinct objects, easy to blur together:

- **State**: a *vector* $|\psi\rangle$ — describes the full system.
- **Observable**: an *operator* $A$ (a matrix, here) — represents a
  measurable physical quantity (energy, spin, position...).
- **Measurement result**: an *eigenvalue* $a$ of $A$ — the only values a
  measurement of that observable can actually return.

**Eigenvalue equation**: $A|v\rangle = a|v\rangle$ — applying the operator
to an eigenvector $|v\rangle$ returns the *same* vector, just rescaled by
the eigenvalue $a$. Physically: if the system is in an eigenstate of $A$,
measuring $A$ returns $a$ with certainty and leaves the state unchanged."""))

cells.append(co("""A = sp.Matrix([[2, 1], [1, 2]])
print("A ="); display(A)

eigenstuff = A.eigenvects()
for eigenvalue, multiplicity, eigenvectors in eigenstuff:
    for v in eigenvectors:
        v_normalized = v / sp.sqrt((v.T * v)[0])
        print(f"\\neigenvalue a = {eigenvalue}")
        print("eigenvector |v> (normalized) ="); display(v_normalized)
        # verify A|v> = a|v> directly, not just trusting eigenvects()
        lhs = A * v_normalized
        rhs = eigenvalue * v_normalized
        assert sp.simplify(lhs - rhs) == sp.zeros(2, 1)
        print("[OK] A|v> = a|v> verified directly")"""))

# ============================================================ PART 7: expectation values
cells.append(md(r"""# Part 7 — Expectation values

$$\langle A\rangle = \langle\psi|A|\psi\rangle$$

The expected result of measuring observable $A$ on state $|\psi\rangle$,
averaged over many identical repetitions — **not**, in general, the
outcome of any single measurement (a single measurement returns one
eigenvalue; $\langle A\rangle$ is a weighted average of the eigenvalues,
and need not be an eigenvalue itself)."""))

cells.append(co("""A2 = sp.Matrix([[1, 0], [0, -1]])   # Pauli Z
psi7 = sp.Matrix([[sp.Rational(3,5)], [sp.Rational(4,5)]])   # normalized: (3/5)^2+(4/5)^2=1

norm_check = (psi7.T * psi7)[0]
assert norm_check == 1, f"psi7 must be normalized, got <psi|psi>={norm_check}"

# by hand: <A> = |alpha|^2 * (+1) + |beta|^2 * (-1)  for a diagonal A = diag(+1,-1)
by_hand = sp.Rational(3,5)**2 * 1 + sp.Rational(4,5)**2 * (-1)

# with sympy: <psi|A|psi>
psi7_bra = psi7.T
by_sympy = (psi7_bra * A2 * psi7)[0]

print("<A> by hand  ="); display(by_hand)
print("<A> by SymPy ="); display(by_sympy)
assert sp.simplify(by_hand - by_sympy) == 0
print("[OK] hand computation matches SymPy")
print(f"\\n<A> = {by_sympy} is NOT one of A's eigenvalues (+1, -1) -- it's their")
print("probability-weighted average, exactly the ensemble/statistical quantity")
print("Part 1's E[X] was.")"""))

# ============================================================ PART 8: position wavefunction
cells.append(md(r"""# Part 8 — Position-space wavefunction

$$\psi(x),\qquad \rho(x)=|\psi(x)|^2 \text{ (probability density)},\qquad
\int_{-\infty}^{\infty}|\psi(x)|^2\,dx = 1$$

A Gaussian wavefunction, with an unknown normalization constant $N$ solved
for symbolically rather than guessed:
$$\psi(x) = N\,e^{-x^2/(2\sigma^2)}$$"""))

cells.append(co("""x, sigma, N = sp.symbols('x sigma N', positive=True, real=True)
sigma = sp.Symbol('sigma', positive=True)
x = sp.Symbol('x', real=True)
N = sp.Symbol('N', positive=True)

psi_x = N * sp.exp(-x**2 / (2*sigma**2))
rho_x = psi_x**2   # psi is real here, so |psi|^2 = psi^2

normalization_integral = sp.integrate(rho_x, (x, -sp.oo, sp.oo))
print("integral of |psi(x)|^2 dx ="); display(normalization_integral)

N_solution = sp.solve(sp.Eq(normalization_integral, 1), N)
N_value = [s for s in N_solution if s.is_positive or s.could_extract_minus_sign() is False][0]
print("\\nSolving integral = 1 for N (the positive root):"); display(N_value)

psi_x_normalized = psi_x.subs(N, N_value)
check = sp.integrate(psi_x_normalized**2, (x, -sp.oo, sp.oo))
print("\\nRe-checking normalization with this N:"); display(sp.simplify(check))
assert sp.simplify(check - 1) == 0
print("[OK] normalized")"""))

# ============================================================ PART 9: position expectation/variance
cells.append(md(r"""# Part 9 — Position expectation and variance

$$\langle x\rangle = \int x\,|\psi(x)|^2\,dx,\qquad
\langle x^2\rangle = \int x^2\,|\psi(x)|^2\,dx,\qquad
\Delta x^2 = \langle x^2\rangle - \langle x\rangle^2,\qquad
\Delta x = \sqrt{\Delta x^2}$$"""))

cells.append(co("""rho_normalized = psi_x_normalized**2

x_expect = sp.integrate(x * rho_normalized, (x, -sp.oo, sp.oo))
x2_expect = sp.integrate(x**2 * rho_normalized, (x, -sp.oo, sp.oo))
delta_x_sq = sp.simplify(x2_expect - x_expect**2)
delta_x = sp.sqrt(delta_x_sq)

print("<x>   ="); display(x_expect)
print("<x^2> ="); display(x2_expect)
print("Delta_x^2 = <x^2> - <x>^2 ="); display(delta_x_sq)
print("Delta_x ="); display(delta_x)

# by symmetry, a Gaussian centered at 0 must have <x>=0 -- a real check, not
# just accepting whatever integrate() returns
assert x_expect == 0, "a symmetric (even) integrand times an odd function x must integrate to 0"
assert sp.simplify(delta_x_sq - sigma**2/2) == 0, "expected <x^2> = sigma^2/2 for this Gaussian"
print("\\n[OK] <x>=0 by symmetry, Delta_x^2 = sigma^2/2 matches the closed-form expectation")"""))

# ============================================================ PART 10: momentum operator
cells.append(md(r"""# Part 10 — Momentum operator

$$\hat p = -i\hbar\frac{d}{dx}$$

This is an **operator**, not a number: it doesn't have a fixed value until
applied to a specific wavefunction (compare Part 6 — same distinction
between operator and eigenvalue). Applying it to a plane wave
$\psi(x)=e^{ikx}$:"""))

cells.append(co("""hbar, k = sp.symbols('hbar k', positive=True, real=True)

psi_plane = sp.exp(sp.I*k*x)
p_hat_psi = -sp.I*hbar * sp.diff(psi_plane, x)
p_hat_psi_simplified = sp.simplify(p_hat_psi)

print("psi(x) = exp(i*k*x)")
print("p_hat psi = -i*hbar * d/dx[exp(i*k*x)] ="); display(p_hat_psi_simplified)

# factor out psi to read off the eigenvalue
eigenvalue_p = sp.simplify(p_hat_psi_simplified / psi_plane)
print("\\np_hat psi / psi (the eigenvalue) ="); display(eigenvalue_p)
assert sp.simplify(eigenvalue_p - hbar*k) == 0
print("\\n[OK] p_hat psi = hbar*k * psi -- psi is a momentum EIGENFUNCTION with")
print("eigenvalue p = hbar*k, exactly the eigenvalue-equation structure of Part 6")
print("(A|v> = a|v>), just with a differential operator instead of a matrix.")"""))

# ============================================================ PART 11: Fourier transform
cells.append(md(r"""# Part 11 — Fourier transform: position $\leftrightarrow$ momentum

Position-space and momentum-space are two *representations of the same
state*, related by
$$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}
\psi(x)\,e^{-ipx/\hbar}\,dx$$
which is the ordinary Fourier transform once $k=p/\hbar$ is substituted
(matching Part 10's plane-wave eigenvalue $p=\hbar k$: a definite-momentum
state $e^{ikx}$ is a definite-frequency-$k$ plane wave, and $\phi(p)$ is
literally "how much of frequency $k=p/\hbar$ is present in $\psi(x)$" —
not merely a software FFT fact, but the same physical idea Part 10 already
built."""))

cells.append(co("""p_sym = sp.Symbol('p', real=True)

# Fourier transform of the SAME normalized Gaussian from Parts 8-9
phi_p_integrand = psi_x_normalized * sp.exp(-sp.I*p_sym*x/hbar)
phi_p = sp.integrate(phi_p_integrand, (x, -sp.oo, sp.oo)) / sp.sqrt(2*sp.pi*hbar)
phi_p = sp.simplify(phi_p)
print("phi(p) for the Gaussian psi(x) ="); display(phi_p)

# a Gaussian's Fourier transform is a Gaussian -- get its width EXACTLY the
# same way Part 9 got sigma_x: integrate p^2*|phi(p)|^2 directly, rather than
# pattern-matching the exponent algebraically (a first attempt at the latter
# had a sign/factor slip -- direct integration is the more robust method,
# and it's the same method already trusted in Part 9).
rho_p = sp.simplify(sp.Abs(phi_p)**2)
print("\\nrho(p) = |phi(p)|^2 ="); display(rho_p)

momentum_norm = sp.integrate(rho_p, (p_sym, -sp.oo, sp.oo))
print("integral of |phi(p)|^2 dp ="); display(momentum_norm)
assert sp.simplify(momentum_norm - 1) == 0
print("[OK] phi(p) is automatically normalized too (Parseval/Plancherel) -- ")
print("     a real check on the Fourier-transform convention used above, not assumed.")

p_expect = sp.integrate(p_sym * rho_p, (p_sym, -sp.oo, sp.oo))
p2_expect = sp.integrate(p_sym**2 * rho_p, (p_sym, -sp.oo, sp.oo))
sigma_p_sq = sp.simplify(p2_expect - p_expect**2)
print("\\n<p>   ="); display(p_expect)
print("<p^2> ="); display(sp.simplify(p2_expect))
print("momentum-space variance sigma_p^2 = <p^2> - <p>^2 ="); display(sigma_p_sq)
assert p_expect == 0
assert sp.simplify(sigma_p_sq - hbar**2/(2*sigma**2)) == 0"""))

cells.append(md(r"""**Narrow position $\leftrightarrow$ broad momentum.** The position-space
width was $\sigma$ (Part 8's Gaussian). The momentum-space width above
comes out $\propto \hbar^2/\sigma^2$ — as $\sigma\to0$ (a sharply localized
particle), the momentum width *grows*, and vice versa. This is a structural
consequence of the Fourier transform pair (narrowing one side always
broadens the other), not a coincidence of this particular example — it is
exactly what Part 12's uncertainty relation quantifies."""))

# ============================================================ PART 12: uncertainty
cells.append(md(r"""# Part 12 — Uncertainty

$$\Delta x\,\Delta p \ge \frac{\hbar}{2}$$

**This is not measurement error.** It is not instrument imprecision, not
numerical roundoff, and not statistical sampling noise from too few
measurements — all of those can, in principle, be reduced toward zero with
a better instrument, more careful numerics, or a larger sample. $\Delta x$
and $\Delta p$ here are properties of the *quantum state itself*: even a
perfect, noiseless, infinite-precision measurement of $x$ on an ensemble of
identically-prepared systems still returns a spread of results, because
the state does not have a single definite $x$ and $p$ simultaneously. The
inequality is saturated (equality) for a Gaussian — verified directly
below using this notebook's own Parts 8-11 results, not asserted."""))

cells.append(co("""# Delta_x from Part 9, Delta_p from Part 11's sigma_p^2
delta_p = sp.sqrt(sigma_p_sq)

product = sp.simplify(delta_x * delta_p)
print("Delta_x ="); display(delta_x)
print("Delta_p ="); display(delta_p)
print("Delta_x * Delta_p ="); display(product)

assert sp.simplify(product - hbar/2) == 0
print("\\n[OK] this Gaussian saturates the bound EXACTLY: Delta_x * Delta_p = hbar/2")
print("(a minimum-uncertainty state) -- consistent with Delta_x*Delta_p >= hbar/2.")"""))

# ============================================================ PART 13: CE connection
cells.append(md(r"""# Part 13 — Computer-engineering connection

**Classical bit**: exactly $0$ or $1$. No intermediate states, no
superposition -- reading it never disturbs it.

**Quantum state**: $\alpha|0\rangle+\beta|1\rangle$ -- a genuine
superposition, described by *two continuous complex numbers* (four real
parameters, minus 1 for normalization, minus 1 for an unobservable global
phase = 2 real degrees of freedom) collapsing to one classical bit only
*upon measurement*, with probabilities given by the Born rule (Part 5).

**Why a qubit is NOT simply an analog bit.** An analog voltage in
$[0,1]\,\mathrm{V}$ is still a single, directly-readable real number --
read it and you get that number back, unchanged, as many times as you
like. A qubit is different in a way with no analog-electronics
counterpart: (1) measurement is *probabilistic*, governed by $|\alpha|^2$,
not a fixed readout; (2) measurement *destroys* the superposition
(collapses $|\psi\rangle$ to whichever basis state was observed) -- you
cannot re-read the original $\alpha,\beta$ afterward; (3) $\alpha,\beta$
are *complex*, carrying a relative phase with no analog-voltage
equivalent, and that phase is physically meaningful (interference) even
though it is not directly measurable as a single number the way a voltage
is.

**The chain, mapped to ordinary linear algebra / CE:**

| Quantum | Linear algebra | This notebook |
|---|---|---|
| state | vector | Part 4, $\vert\psi\rangle$ |
| observable | matrix / linear operator | Part 6, $A$ |
| measurement | eigenvalue problem | Part 6, $A\vert v\rangle=a\vert v\rangle$ |
| expected readout | $\langle\psi\vert A\vert\psi\rangle$ | Part 7 |

-- the identical eigenvector/eigenvalue machinery `dgs/vibration_modes.py`
uses for classical vibration and this repo's other modules use for PCA;
quantum mechanics is not a special exception to linear algebra, it *is*
linear algebra, applied to complex vector spaces with a probabilistic
readout rule."""))

# ============================================================ PART 14: validation
cells.append(md(r"""# Part 14 — Validation methodology

The standard applied to every computational result above, stated
explicitly rather than left implicit:

1. **Derive by hand first** (Parts 1-12's markdown cells) -- know the
   expected answer before running any code.
2. **Compute with SymPy** symbolically -- exact fractions and radicals
   ($\sqrt2/2$, $\pi$), not premature decimals.
3. **Simplify the symbolic difference** between the hand result and the
   SymPy result (`sp.simplify(hand - sympy) == 0`), not just eyeball
   agreement -- e.g. Part 7's `<A>` cross-check, Part 3's Euler-identity
   series comparison.
4. **Verify normalization** explicitly wherever a state or wavefunction is
   introduced (Part 4's `<psi|psi>`, Part 5's `P(0)+P(1)`, Part 8's
   normalization integral).
5. **Check dimensions** where applicable: $\hat p=-i\hbar\,d/dx$ carries
   units of momentum only because $\hbar$ (J*s) supplies them --
   $d/dx$ alone has units of 1/length; $[\hat p]=[\hbar][x]^{-1} =
   (\mathrm{J\,s})(\mathrm{m}^{-1})=\mathrm{kg\,m/s}$, correct for
   momentum. $\Delta x\,\Delta p$ then has units
   $\mathrm{m}\times\mathrm{kg\,m/s}=\mathrm{J\,s}$, matching $\hbar$'s
   units on the right-hand side of Part 12's inequality -- a dimensional
   check the symbolic algebra alone does not guarantee.
6. **What counts as a failed check**: any `assert` above raising
   `AssertionError` (a nonzero symbolic difference, a normalization
   integral $\ne1$, a product $\ne\hbar/2$) -- this notebook has none, by
   construction (bugs were caught and fixed *during* writing, not shipped
   with a known-failing cell). Software executing without a traceback is
   necessary but not sufficient evidence of correctness -- the assertions
   above are the actual evidence; a cell that merely prints a plausible-
   looking number without such a check does not count as validated."""))

# ============================================================ PART 15: problems
cells.append(md(r"""# Part 15 — 30 handwritten problems

Attempt each problem before looking at its solution — solutions are
collected in a **separate final section** below, in the same order.

**1-5: fractions, probabilities, normalization**

1. A bag has 3 red and 7 blue marbles. What is $P(\text{red})$? What is
   $P(\text{not red})$?
2. Two independent coin flips: what is $P(\text{both heads})$?
3. A die is rolled once. Let $A=\{\text{even}\}$, $B=\{2,3,5\}$. Find
   $P(A\cap B)$ and state whether $A,B$ are independent.
4. A state is $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$ with
   $\alpha=\tfrac{1}{2}$. If $|\psi\rangle$ is normalized and $\beta$ is
   real and positive, find $\beta$.
5. A discrete variable $Y\in\{0,1,2\}$ has $P(0)=\tfrac15,\ P(1)=\tfrac25,\
   P(2)=\tfrac25$. Verify these are a valid probability distribution.

**6-10: complex numbers**

6. Compute $(2+3i)(1-i)$.
7. Find $z^*$ and $|z|^2$ for $z=4-3i$.
8. Write $e^{i\pi/3}$ in the form $a+ib$ using Euler's identity.
9. Show $|e^{i\theta}|=1$ for any real $\theta$.
10. Simplify $i^3$ and $i^4$.

**11-15: vectors and inner products**

11. For $|\psi\rangle=\begin{pmatrix}\tfrac35\\ \tfrac{4i}{5}\end{pmatrix}$,
    find $\langle\psi|$.
12. Compute $\langle\psi|\psi\rangle$ for the state in Problem 11.
13. Is $|\phi\rangle=\begin{pmatrix}1\\1\end{pmatrix}$ normalized? If not,
    find the normalized version.
14. Compute the inner product $\langle 0|1\rangle$ where
    $|0\rangle=\begin{pmatrix}1\\0\end{pmatrix}$,
    $|1\rangle=\begin{pmatrix}0\\1\end{pmatrix}$. What does the result mean
    physically?
15. For $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$, expand
    $\langle\psi|\psi\rangle$ symbolically in terms of $\alpha,\beta$.

**16-20: operators and eigenvalues**

16. Find the eigenvalues of $A=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$
    (Pauli $Z$).
17. Find the eigenvalues and eigenvectors of
    $B=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ (Pauli $X$).
18. Verify $A|v\rangle=a|v\rangle$ by hand for one eigenpair of Problem 17.
19. Is $A$ from Problem 16 Hermitian ($A=A^\dagger$)? Why does that matter
    physically?
20. For $A$ from Problem 16, what are the *only* possible results of
    measuring the observable $A$?

**21-25: wavefunctions and expectation values**

21. For $\psi(x)=N\,e^{-x^2/(2\sigma^2)}$, what condition determines $N$?
22. Using Part 8's result, write $N$ explicitly in terms of $\sigma$.
23. Explain, without computing, why $\langle x\rangle=0$ for this $\psi(x)$.
24. Using $A=\begin{pmatrix}2&0\\0&0\end{pmatrix}$ and
    $|\psi\rangle=\tfrac{1}{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix}$, compute
    $\langle A\rangle$ by hand.
25. Is the $\langle A\rangle$ from Problem 24 one of $A$'s eigenvalues?
    Explain.

**26-30: Fourier transforms, momentum, uncertainty**

26. Apply $\hat p=-i\hbar\,d/dx$ to $\psi(x)=e^{2ix}$. What is the
    momentum eigenvalue?
27. What is $k$ in terms of $p$ and $\hbar$?
28. If a wavefunction is narrowed in position space (smaller $\sigma$),
    what happens to its momentum-space width? Why?
29. State the uncertainty relation and identify which two quantities on its
    left-hand side cannot simultaneously be made arbitrarily small.
30. True or false, with justification: "the uncertainty principle exists
    because our measuring instruments are imperfect.\""""))

cells.append(md(r"""---
## Solutions

**1.** $P(\text{red})=\dfrac{3}{10}$. $P(\text{not red})=1-\dfrac{3}{10}=\dfrac{7}{10}$
(complement rule, Part 1).

**2.** Independent events multiply: $P(\text{HH})=\tfrac12\cdot\tfrac12=\tfrac14$.

**3.** $A=\{2,4,6\}$, $B=\{2,3,5\}$, $A\cap B=\{2\}$, so
$P(A\cap B)=\tfrac16$. Independence requires $P(A\cap B)=P(A)P(B)=\tfrac12\cdot\tfrac12=\tfrac14\ne\tfrac16$
— **not independent**.

**4.** Normalization: $|\alpha|^2+|\beta|^2=1\Rightarrow \tfrac14+\beta^2=1
\Rightarrow \beta^2=\tfrac34\Rightarrow \beta=\dfrac{\sqrt3}{2}$.

**5.** Each $P_i\in[0,1]$ and $\tfrac15+\tfrac25+\tfrac25=\tfrac55=1$ —
**valid**.

**6.** $(2+3i)(1-i)=2-2i+3i-3i^2=2+i-3(-1)=2+i+3=5+i$.

**7.** $z^*=4+3i$. $|z|^2=z^*z=(4-3i)(4+3i)=16+9=25$.

**8.** $e^{i\pi/3}=\cos(\pi/3)+i\sin(\pi/3)=\dfrac12+i\dfrac{\sqrt3}{2}$.

**9.** $|e^{i\theta}|^2=e^{i\theta}\big(e^{i\theta}\big)^*
=e^{i\theta}e^{-i\theta}=e^0=1$, so $|e^{i\theta}|=1$ for every real
$\theta$ (Part 3's modulus-squared identity, $z=\cos\theta,\ b=\sin\theta$,
$a^2+b^2=\cos^2\theta+\sin^2\theta=1$).

**10.** $i^3=i^2\cdot i=-i$. $i^4=(i^2)^2=(-1)^2=1$.

**11.** $\langle\psi|=\begin{pmatrix}\tfrac35 & -\tfrac{4i}{5}\end{pmatrix}$
(conjugate transpose — flip the sign of the imaginary part, and transpose
to a row).

**12.** $\langle\psi|\psi\rangle=\left(\tfrac35\right)^2+\left(\tfrac45\right)^2
=\tfrac{9}{25}+\tfrac{16}{25}=1$ — already normalized (matches the
worked example in Part 4's code cell).

**13.** $\langle\phi|\phi\rangle=1^2+1^2=2\ne1$ — **not normalized**.
Normalized version: $\dfrac{1}{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix}$.

**14.** $\langle0|1\rangle=\begin{pmatrix}1&0\end{pmatrix}\begin{pmatrix}0\\1\end{pmatrix}=0$
— $|0\rangle$ and $|1\rangle$ are **orthogonal**: physically, a system
definitely in state $|0\rangle$ has zero amplitude (hence zero
probability) of being found in $|1\rangle$.

**15.** $\langle\psi|\psi\rangle=(\alpha^*\langle0|+\beta^*\langle1|)(\alpha|0\rangle+\beta|1\rangle)
=\alpha^*\alpha\langle0|0\rangle+\beta^*\beta\langle1|1\rangle
+\text{(cross terms, each containing }\langle0|1\rangle=0\text{)}
=|\alpha|^2+|\beta|^2$ (using $\langle0|0\rangle=\langle1|1\rangle=1$ and
Problem 14's orthogonality for the cross terms).

**16.** $\det(A-\lambda I)=(1-\lambda)(-1-\lambda)=0\Rightarrow
\lambda=+1,-1$.

**17.** $\det(B-\lambda I)=\lambda^2-1=0\Rightarrow\lambda=\pm1$.
For $\lambda=+1$: $(B-I)v=0\Rightarrow -v_1+v_2=0\Rightarrow v_1=v_2$,
eigenvector $\propto\begin{pmatrix}1\\1\end{pmatrix}$. For $\lambda=-1$:
eigenvector $\propto\begin{pmatrix}1\\-1\end{pmatrix}$.

**18.** $B\begin{pmatrix}1\\1\end{pmatrix}=\begin{pmatrix}0\cdot1+1\cdot1\\
1\cdot1+0\cdot1\end{pmatrix}=\begin{pmatrix}1\\1\end{pmatrix}=(+1)\begin{pmatrix}1\\1\end{pmatrix}$
— confirmed, matches Problem 17's $\lambda=+1$ eigenpair.

**19.** $A^\dagger=A$ here (real, diagonal, so transpose-and-conjugate does
nothing) — **Hermitian**. This matters because Hermitian operators are
exactly the ones guaranteed to have *real* eigenvalues — a measurement
result (an eigenvalue) must be a real number, since it's something you'd
read on a physical instrument.

**20.** Only $+1$ or $-1$ (Problem 16's eigenvalues) — measurement of an
observable can only return one of that observable's eigenvalues, never
any other number.

**21.** $\displaystyle\int_{-\infty}^{\infty}|\psi(x)|^2\,dx=1$
(normalization, Part 8).

**22.** From Part 8's code cell: $N=\left(\pi\sigma^2\right)^{-1/4}$.

**23.** $|\psi(x)|^2=N^2e^{-x^2/\sigma^2}$ is an **even** function of $x$
(symmetric about $x=0$), while the integrand for $\langle x\rangle$ is
$x\cdot|\psi(x)|^2$, an **odd** function (odd $\times$ even = odd) — the
integral of any odd function over a symmetric interval
$(-\infty,\infty)$ is exactly zero by symmetry, with no need to evaluate
the integral itself.

**24.** $A|\psi\rangle=\begin{pmatrix}2&0\\0&0\end{pmatrix}\tfrac{1}{\sqrt2}
\begin{pmatrix}1\\1\end{pmatrix}=\tfrac{1}{\sqrt2}\begin{pmatrix}2\\0\end{pmatrix}$.
$\langle\psi|A|\psi\rangle=\tfrac{1}{\sqrt2}\begin{pmatrix}1&1\end{pmatrix}
\cdot\tfrac{1}{\sqrt2}\begin{pmatrix}2\\0\end{pmatrix}=\tfrac12(2+0)=1$.

**25.** $A$'s eigenvalues are $2$ and $0$ (diagonal entries); $\langle A\rangle=1$
is **not** one of them — it's their probability-weighted average
($\tfrac12\cdot2+\tfrac12\cdot0=1$, since $|\psi\rangle$ gives equal
weight $\tfrac12$ to each basis state), exactly Part 7's point: an
expectation value is a statistical average, not a possible single-shot
outcome, unless it happens to coincide with one.

**26.** $\hat p\,e^{2ix}=-i\hbar\cdot(2i)e^{2ix}=2\hbar\,e^{2ix}$ — momentum
eigenvalue $p=2\hbar$ (matching Part 10 with $k=2$).

**27.** $k=p/\hbar$ (Part 11).

**28.** It **broadens** (Part 11): position-space and momentum-space
widths are linked by the Fourier-transform relationship — narrowing one
representation necessarily broadens the other, independent of the
specific wavefunction shape.

**29.** $\Delta x\,\Delta p\ge\hbar/2$ (Part 12) — the position spread
$\Delta x$ and momentum spread $\Delta p$ of the *same state* cannot both
be made arbitrarily small simultaneously.

**30. False.** (Part 12.) The bound comes from the mathematical structure
of the state itself (via the position/momentum Fourier-transform
relationship, Part 11) — it holds even for a perfect, error-free
measurement on an ideal ensemble. Better instruments reduce measurement
*error*, which is a separate, additional effect on top of this bound, not
its cause."""))

nb["cells"] = cells
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "probability_to_qm_operators.ipynb"
out.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {out}, {len(cells)} cells")
