"""Generate notebooks/lagrange_multipliers_boltzmann.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Lagrange multipliers and the Boltzmann distribution

The previous notebooks *counted* microstates ($W=N!/\prod n_i!$) and *used* the exponential law
$f_{\rm MB}=A\,e^{-E/k_BT}$. This notebook *derives* that law from first principles. The physical claim
is: the equilibrium distribution is the one that maximizes the number of microstates $W$ (equivalently
the entropy $S=k_B\ln W$) subject to two conservation constraints,

$$\sum_i n_i = N \quad(\text{fixed particle number}),\qquad \sum_i n_i E_i = U \quad(\text{fixed energy}).$$

Maximizing a function subject to constraints is exactly the job of **Lagrange multipliers**. Carrying
it out yields $n_i = e^{-\alpha-\beta E_i}$, and thermodynamics identifies the second multiplier as
$\beta = 1/k_BT$. The same constrained-maximization is the maximum-entropy principle of statistics and
the **softmax** function of machine learning -- one piece of mathematics, three vocabularies.

Self-contained: NumPy, SymPy, Pandas, Matplotlib only."""),
setup_cell(),

md(r"""## The method of Lagrange multipliers

To maximize $f(\mathbf x)$ subject to $g(\mathbf x)=c$, introduce a multiplier $\lambda$ and solve
$\nabla f = \lambda\,\nabla g$ together with the constraint. Geometrically, at the optimum the level
surface of $f$ is tangent to the constraint surface. A minimal example: maximize the product $xy$ for
fixed sum $x+y=s$. SymPy returns the symmetric solution $x=y=s/2$."""),
co("""x, y, s, lam = sp.symbols('x y s lambda', positive=True)
f = x*y                      # objective
g = x + y - s                # constraint g = 0
sol = sp.solve([sp.diff(f - lam*g, x), sp.diff(f - lam*g, y), g], [x, y, lam], dict=True)[0]
print("maximize x*y with x+y=s  ->  x =", sol[x], ", y =", sol[y], ", lambda =", sol[lam])
assert sol[x] == s/2 and sol[y] == s/2"""),

md(r"""## Stirling's approximation

The count $\ln W=\ln N!-\sum_i\ln n_i!$ is intractable until we approximate the factorials. For large
$n$, $\ln n!\approx n\ln n-n$ (Stirling). Its derivative is the clean $\dfrac{d}{dn}\ln n!\approx\ln n$,
which is all the derivation needs."""),
co("""import math
def stirling(n): return n*np.log(n) - n
for n in (10, 100, 1000):
    exact = math.lgamma(n+1)                      # ln(n!) exactly
    print(f"n={n:5d}:  ln(n!) exact = {exact:12.3f},  Stirling = {stirling(n):12.3f},"
          f"  rel.err = {abs(stirling(n)-exact)/exact:.2e}")
# the derivative used below: d/dn [n ln n - n] = ln n
assert abs((stirling(1000.0001)-stirling(1000))/0.0001 - np.log(1000)) < 1e-3"""),

md(r"""## Deriving $n_i = e^{-\alpha-\beta E_i}$

Maximize $\ln W$ subject to the two constraints by extremizing the Lagrangian
$$\mathcal L=\ln W-\alpha\Big(\sum_i n_i-N\Big)-\beta\Big(\sum_i n_i E_i-U\Big),$$
with $\ln W\approx N\ln N-\sum_i(n_i\ln n_i-n_i)$ (Stirling). Setting $\partial\mathcal L/\partial n_i=0$
uses $\partial(\text -n_i\ln n_i+n_i)/\partial n_i=-\ln n_i$, giving $-\ln n_i-\alpha-\beta E_i=0$, hence
$$\boxed{\,n_i=e^{-\alpha-\beta E_i}=A\,e^{-\beta E_i}\,}, \qquad A\equiv e^{-\alpha}.$$
SymPy performs the per-level extremization symbolically."""),
co("""n, E, alpha, beta = sp.symbols('n_i E_i alpha beta', positive=True)
lnW_term = -(n*sp.log(n) - n)                     # this level's contribution to ln W (Stirling)
L_i = lnW_term - alpha*n - beta*E*n               # constraints enter linearly in n_i
n_star = sp.solve(sp.diff(L_i, n), n)[0]          # dL/dn_i = 0
print("dL/dn_i = 0  ->  n_i =", n_star)
assert sp.simplify(n_star - sp.exp(-alpha - beta*E)) == 0
print("=> n_i = e^{-alpha - beta E_i} = A e^{-beta E_i}, with A = e^{-alpha}")"""),

md(r"""## Identifying $\beta = 1/k_BT$

The multiplier $\alpha$ (through $A=e^{-\alpha}$) enforces normalization $\sum_i n_i=N$. The multiplier
$\beta$ enforces the energy constraint; comparing $dS=k_B\,d(\ln W)$ with the thermodynamic
$dS=dU/T$ identifies $\beta=1/k_BT$. Substituting recovers exactly the law the earlier notebooks
assumed:
$$n_i=A\,e^{-E_i/k_BT},\qquad A=\frac{N}{\sum_i e^{-E_i/k_BT}}=\frac{N}{Z},$$
where $Z=\sum_i e^{-E_i/k_BT}$ is the **partition function**."""),
co("""# partition function and probabilities for a small energy ladder
kT = 1.0                                          # work in units of k_B T (so beta = 1)
E = np.array([0.0, 1.0, 2.0, 3.0, 4.0])           # energy levels (units of kT)
Z = np.sum(np.exp(-E/kT))                          # partition function
p = np.exp(-E/kT) / Z                              # Boltzmann probabilities, normalized
print("partition function Z =", round(Z, 4))
print("probabilities p_i    =", np.round(p, 4), " sum =", round(p.sum(), 6))
assert np.isclose(p.sum(), 1.0)
# ratio of adjacent levels is a constant factor e^{-1/kT} (the exponential signature)
assert np.allclose(p[1:]/p[:-1], np.exp(-1.0/kT))"""),

md(r"""## Numerical check: maximum entropy reproduces the exponential

The derivation says the Boltzmann distribution is the *maximum-entropy* distribution at fixed mean
energy. We test this directly: given a target mean $\langle E\rangle$, solve
$\sum_i(E_i-\langle E\rangle)e^{-\beta E_i}=0$ for $\beta$ (a one-dimensional root find, no external
solver needed), form $p_i\propto e^{-\beta E_i}$, and confirm no other distribution with the same mean
has higher entropy $S=-\sum_i p_i\ln p_i$."""),
co("""def boltzmann_for_mean(E, mean_E, lo=-5.0, hi=5.0):
    # bisection on beta so that <E> = sum E_i p_i matches the target
    def mean_at(b):
        w = np.exp(-b*E); w /= w.sum(); return (E*w).sum()
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if mean_at(mid) > mean_E: lo = mid          # larger beta lowers the mean
        else: hi = mid
    b = 0.5*(lo+hi); w = np.exp(-b*E); return b, w/w.sum()

E = np.arange(6.0)                                  # levels 0..5
target = 1.5
beta, p = boltzmann_for_mean(E, target)
S = -np.sum(p*np.log(p))
print(f"solved beta = {beta:.4f},  <E> = {(E*p).sum():.4f} (target {target}),  S = {S:.4f}")

# perturb toward a neighbour while keeping the mean fixed; entropy must not increase
rng = np.random.default_rng(0)
worse = 0
for _ in range(2000):
    q = p + 1e-2*rng.standard_normal(len(p))
    q = np.clip(q, 1e-9, None); q /= q.sum()
    if abs((E*q).sum() - target) < 1e-3:            # same mean
        if -np.sum(q*np.log(q)) > S + 1e-9: worse += 1
print("perturbations with the same mean that beat Boltzmann's entropy:", worse, "(expected 0)")
assert worse == 0"""),

md(r"""## The computer-engineering payoff: softmax is the Boltzmann distribution

Machine learning turns a vector of scores $\mathbf z$ into probabilities with the **softmax**
$\text{softmax}(\mathbf z)_i=e^{z_i}/\sum_j e^{z_j}$. Setting $z_i=-\beta E_i$ makes this *identically*
the Boltzmann distribution: the network's logits play the role of negative energies and the "inverse
temperature" $\beta$ is the softmax gain. Training with cross-entropy loss is maximum-likelihood
under this model, and the maximum-entropy principle derived above is why softmax is the natural,
least-biased choice. Constrained optimization is thus the same mathematics in a Hamiltonian, a
thermodynamic ensemble, and a neural network."""),
co("""def softmax(z):
    z = z - z.max()                                 # subtract max for numerical stability
    e = np.exp(z); return e/e.sum()

E = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
beta = 0.8
boltz = np.exp(-beta*E); boltz /= boltz.sum()
assert np.allclose(softmax(-beta*E), boltz)         # softmax(-beta E) == Boltzmann
print("softmax(-beta E) equals the Boltzmann distribution:", np.round(boltz, 4))

# temperature/gain sweep: high beta (low T) -> peaked (argmax); low beta (high T) -> uniform
rows = []
for b in (0.0, 0.5, 1.0, 3.0):
    q = softmax(-b*E)
    rows.append({"beta": b, "T (1/beta)": (np.inf if b==0 else round(1/b,3)),
                 "p(ground)": round(q[0],4), "entropy": round(-np.sum(q*np.log(q)),4)})
print(pd.DataFrame(rows).to_string(index=False))"""),

md(r"""## Plots"""),
co("""fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
E = np.arange(8.0)
for b, c in [(0.2, "#4C78A8"), (0.6, "#54A24B"), (1.2, "#E45756")]:
    q = softmax(-b*E)
    ax[0].plot(E, q, "o-", color=c, label=f"beta={b} (T={1/b:.1f})")
ax[0].set_xlabel("energy level $E_i$"); ax[0].set_ylabel("probability $p_i$")
ax[0].set_title("Boltzmann / softmax: colder = more peaked"); ax[0].legend()

betas = np.linspace(0.01, 3, 200)
Ss = [-(lambda q: np.sum(q*np.log(q)))(softmax(-b*E)) for b in betas]
ax[1].plot(1/betas, Ss, color="#4C78A8")
ax[1].set_xlabel(r"temperature $T = 1/\\beta$"); ax[1].set_ylabel("entropy $S$")
ax[1].set_title("entropy rises with temperature toward the uniform limit")
plt.tight_layout(); plt.show()"""),

md(r"""## A note on exponentials (half-life, decay, and the Boltzmann factor)

The exponential $e^{-\beta E}$ derived here is the same functional form that governs **radioactive
decay** $N(t)=N_0e^{-\lambda t}$ with half-life $t_{1/2}=\ln 2/\lambda$, optical absorption
$I(z)=I_0e^{-\alpha z}$, and $RC$ discharge $V(t)=V_0e^{-t/\tau}$. Each is the solution of a first-order
proportionality (a rate constant, an absorption coefficient, a Lagrange multiplier), which is why the
exponential appears wherever one quantity changes in proportion to itself or is weighted linearly in a
constraint."""),
co("""ln2 = np.log(2)
for half_life, label in [(5730.0, "carbon-14 (yr)"), (1.28e9, "potassium-40 (yr)")]:
    lam = ln2/half_life
    assert np.isclose(np.exp(-lam*half_life), 0.5)  # one half-life halves the population
    print(f"{label:22s} lambda = {lam:.3e} /yr,  N/N0 after one half-life = "
          f"{np.exp(-lam*half_life):.3f}")"""),

md(r"""## Summary

- Equilibrium maximizes the microstate count $W$ (entropy $S=k_B\ln W$) under fixed $N$ and $U$;
  **Lagrange multipliers** turn this constrained maximization into $\partial\mathcal L/\partial n_i=0$.
- The result is $n_i=e^{-\alpha-\beta E_i}=A\,e^{-E_i/k_BT}$: the exponential law the earlier notebooks
  assumed is now derived, with $\alpha$ setting normalization and $\beta=1/k_BT$ setting temperature.
- Numerically, the Boltzmann distribution is verified to be the maximum-entropy distribution at fixed
  mean energy, and it is **identically the softmax** with $z_i=-\beta E_i$ -- the bridge to machine
  learning and cross-entropy training.

**Roadmap pointer.** Radiation, mentioned in passing, has the same two-viewpoint structure as this
chapter: Griffiths' electrodynamics gives the deterministic Larmor power $P=\mu_0 q^2 a^2/6\pi c$ of a
*single accelerating charge*, while statistical mechanics gives the Planck blackbody spectrum of a
*thermal ensemble* -- microscopic law versus ensemble average, the recurring theme from microstates
onward."""),
]

write("lagrange_multipliers", "boltzmann", cells)
