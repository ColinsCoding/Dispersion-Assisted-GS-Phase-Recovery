"""Generate notebooks/log_factorial_stirling.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# log(factorial) and Stirling's approximation

Statistical mechanics cannot avoid the factorial: the multiplicity is $W=N!/\prod_i n_i!$ and the
entropy is $S=k_B\ln W=k_B\big(\ln N!-\sum_i\ln n_i!\big)$. But $N$ is Avogadro-sized,
$N\sim6\times10^{23}$, so $N!$ is a number with more than $10^{24}$ digits -- impossible to form. The
escape is to work with $\ln N!$ directly and approximate it. **Stirling's approximation**
$$\ln N! \;=\; N\ln N - N + \tfrac12\ln(2\pi N) + \frac{1}{12N} + \cdots$$
makes $\ln N!$ a cheap, accurate formula for any $N$, which is exactly what the microstate and
Boltzmann derivations used. This notebook derives it, measures its error, and shows why log-space is
the right place to compute.

Self-contained: NumPy, SymPy, Pandas, Matplotlib, and `math.lgamma` from the standard library."""),
setup_cell(),

md(r"""## Exact log-factorial two ways

$\ln n!=\sum_{k=1}^{n}\ln k$ by definition, and $\ln n!=\ln\Gamma(n+1)$ through the gamma function.
The standard-library `math.lgamma` gives the second, exact to machine precision and defined for huge
arguments."""),
co("""import math
def log_factorial_sum(n):
    return sum(math.log(k) for k in range(1, n+1))     # direct sum of logs
for n in (5, 20, 100):
    s = log_factorial_sum(n)
    g = math.lgamma(n+1)                                # ln Gamma(n+1) = ln n!
    print(f"n={n:3d}:  sum ln k = {s:.6f},  lgamma(n+1) = {g:.6f},  ln(n!) exact = {math.log(math.factorial(n)):.6f}")
    assert abs(s - g) < 1e-9"""),

md(r"""## Deriving the leading term (SymPy)

Approximate the sum $\sum_{k=1}^{n}\ln k$ by the integral $\int_1^n\ln x\,dx$. SymPy evaluates it to
$n\ln n-n+1$, giving the leading Stirling term $\ln n!\approx n\ln n-n$ (the $+1$ is negligible once
the sub-leading $\tfrac12\ln(2\pi n)$ term is included)."""),
co("""x, n = sp.symbols('x n', positive=True)
I = sp.integrate(sp.log(x), (x, 1, n))
print("integral of ln x from 1 to n =", sp.simplify(I))
assert sp.simplify(I - (n*sp.log(n) - n + 1)) == 0
# the derivative d/dn (n ln n - n) = ln n is what the Lagrange derivation used
assert sp.simplify(sp.diff(n*sp.log(n) - n, n) - sp.log(n)) == 0
print("=> ln n! ~ n ln n - n  (leading),  with d/dn[n ln n - n] = ln n")"""),

md(r"""## The three orders and their error

Compare, against the exact `lgamma`, three truncations:
- $S_0=n\ln n-n$ (leading),
- $S_1=S_0+\tfrac12\ln(2\pi n)$,
- $S_2=S_1+\dfrac{1}{12n}$.
The leading term alone has a large *absolute* error but a *relative* error that vanishes as
$1/n$; adding the half-log and the $1/12n$ correction drives the relative error down like $n^{-3}$."""),
co("""def S0(n): return n*np.log(n) - n
def S1(n): return S0(n) + 0.5*np.log(2*np.pi*n)
def S2(n): return S1(n) + 1.0/(12*n)

rows = []
for n in (2, 5, 10, 50, 100):
    exact = math.lgamma(n+1)
    rows.append({"n": n, "exact ln n!": round(exact, 4),
                 "S0 rel.err": f"{abs(S0(n)-exact)/exact:.2e}",
                 "S1 rel.err": f"{abs(S1(n)-exact)/exact:.2e}",
                 "S2 rel.err": f"{abs(S2(n)-exact)/exact:.2e}"})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
# S1 is already excellent by n=100
assert abs(S1(100)-math.lgamma(101))/math.lgamma(101) < 1e-4"""),

md(r"""## Why log-space: the overflow wall

`float64` cannot hold $171!$ (it exceeds $1.8\times10^{308}$), so any formula that forms the factorial
first overflows. But $\ln N!$ is a modest number for any $N$: Stirling evaluates it for
$N=6\times10^{23}$ instantly, which is the only reason entropy is computable for real matter."""),
co("""# 170! fits a float; 171! overflows it
print("float(170!) =", float(math.factorial(170)), " (fits)")
try:
    print(float(math.factorial(171)))
except OverflowError as e:
    print("float(171!) -> OverflowError:", e)
# but ln(171!) is fine, and Stirling matches lgamma
print("ln(171!): lgamma =", round(math.lgamma(172), 3), " Stirling S2 =", round(S2(171), 3))
# Avogadro-scale: N! is unrepresentable, ln N! is a single cheap number
N = 6.02214076e23
lnNfact = N*np.log(N) - N + 0.5*np.log(2*np.pi*N)
print(f"ln(N!) for N=Avogadro = {lnNfact:.3e}  (N! itself has ~{lnNfact/np.log(10):.2e} digits)")
assert abs(math.lgamma(172) - S2(171)) < 1e-6"""),

md(r"""## Back to entropy: Stirling makes $\ln W$ computable

For the microstate count $W=N!/\prod_i n_i!$, the log is $\ln W=\ln N!-\sum_i\ln n_i!$. With Stirling
and $p_i=n_i/N$ this collapses to the Gibbs entropy $\ln W\approx -N\sum_i p_i\ln p_i$. We verify the
identity numerically on a small occupation array (exact `lgamma`) and confirm the Stirling/Gibbs form
agrees for larger $N$."""),
co("""def lnW_exact(occ):
    N = sum(occ)
    return math.lgamma(N+1) - sum(math.lgamma(n+1) for n in occ)
def lnW_gibbs(occ):
    N = sum(occ); p = np.array(occ)/N
    p = p[p > 0]
    return -N*np.sum(p*np.log(p))

for occ in ([3, 2, 1], [30, 20, 10], [300, 200, 100]):
    e, g = lnW_exact(occ), lnW_gibbs(occ)
    print(f"occ={str(occ):16s} ln W exact = {e:9.3f},  Gibbs -N sum p ln p = {g:9.3f},"
          f"  rel.err = {abs(e-g)/e:.2e}")
# the small-N case is exactly 60 microstates -> ln 60
assert abs(lnW_exact([3,2,1]) - np.log(60)) < 1e-9
# Gibbs form converges to the exact log-multiplicity as N grows
assert abs(lnW_exact([300,200,100]) - lnW_gibbs([300,200,100]))/lnW_exact([300,200,100]) < 0.02"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
ns = np.arange(2, 200)
exact = np.array([math.lgamma(k+1) for k in ns])
for f, lab, c in [(S0, "S0: n ln n - n", "#4C78A8"),
                  (S1, "S1: + 1/2 ln(2 pi n)", "#54A24B"),
                  (S2, "S2: + 1/(12n)", "#E45756")]:
    ax[0].loglog(ns, np.abs(f(ns)-exact)/exact, color=c, label=lab)
ax[0].set_xlabel("n"); ax[0].set_ylabel("relative error in ln n!")
ax[0].set_title("Stirling: each order converges faster"); ax[0].legend(fontsize=8)
# the sum-of-logs vs the integral (area) picture at small n
k = np.arange(1, 11)
ax[1].bar(k, np.log(k), width=1.0, align="edge", alpha=0.35, color="#4C78A8", label="ln k (the sum)")
xx = np.linspace(1, 11, 300)
ax[1].plot(xx, np.log(xx), color="#E45756", lw=2, label="ln x (the integral)")
ax[1].set_xlabel("k"); ax[1].set_ylabel("ln k"); ax[1].legend()
ax[1].set_title("sum of ln k is approximated by the area under ln x")
plt.tight_layout(); plt.show()""" ),

md(r"""## Summary

- $\ln n!=\sum_{k\le n}\ln k=\ln\Gamma(n+1)$; `math.lgamma` gives it exactly for any $n$.
- Stirling: $\ln n!\approx n\ln n-n+\tfrac12\ln(2\pi n)+\tfrac{1}{12n}$, derived from
  $\int_1^n\ln x\,dx=n\ln n-n+1$; the relative error falls order by order and is already $<10^{-4}$ at
  $n=100$ with the half-log term.
- Working in **log-space** dodges the overflow wall ($171!$ exceeds a float, $\ln N!$ never does),
  which is the only reason entropy is computable for Avogadro-scale matter.
- With Stirling, $\ln W=\ln N!-\sum_i\ln n_i!$ becomes the Gibbs entropy $-N\sum_i p_i\ln p_i$ -- the
  bridge back to the microstate and Boltzmann notebooks.

The same log-space discipline reappears in computer engineering as the log-sum-exp trick and
log-likelihoods: compute with logarithms, add instead of multiply, and overflow never happens."""),
]

write("log_factorial", "stirling", cells)
