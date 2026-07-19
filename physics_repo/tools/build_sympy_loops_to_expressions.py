"""Generate notebooks/sympy_loops_to_expressions.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Breaking loops into expressions with SymPy

A loop that *accumulates* or *iterates* can often be replaced by a single closed-form expression -- faster,
exact, and directly code-generatable. SymPy gives three tools for it:

1. **Symbolic summation** turns a `for`-sum into a closed form: $\sum_{k=1}^{n}k^2=\tfrac{n(n+1)(2n+1)}{6}$.
2. **Recurrence solving** (`rsolve`) turns an iterative recurrence into an explicit formula (e.g. Fibonacci
   $\to$ Binet).
3. **Common-subexpression elimination + `lambdify`** turn one expression into **loop-free, vectorized**
   numeric code -- no Python `for` at run time.

The physics payoff here is the **N-slit diffraction grating**: the field is a *sum over slits*
$\sum_{k=0}^{N-1}e^{ik\phi}$, and breaking that loop gives the closed-form intensity
$I(\phi)=\dfrac{\sin^2(N\phi/2)}{\sin^2(\phi/2)}$ -- the grating equation, derived by collapsing a loop.

Every result is checked against the explicit loop. Self-contained: NumPy, SymPy, Matplotlib."""),
setup_cell(),

md(r"""## 1. A loop sum becomes a closed form -- the diffraction grating

The sum over $N$ slits, $A(\phi)=\sum_{k=0}^{N-1}e^{ik\phi}$, is geometric: SymPy sums it to
$(r^N-1)/(r-1)$ with $r=e^{i\phi}$, and $|A|^2$ is the grating intensity. We confirm the closed form equals
the explicit slit-by-slit loop."""),
co("""k, n = sp.symbols('k n', integer=True, positive=True)
print("sum k^2  =", sp.factor(sp.summation(k**2, (k, 1, n))))          # loop -> closed form
r = sp.symbols('r')
geo = sp.summation(r**k, (k, 0, n - 1))
print("sum r^k  =", sp.simplify(geo), "  (geometric: the N-slit sum)")

# N-slit grating: loop over slits vs the closed-form intensity
def grating_loop(N, phi):
    return np.abs(sum(np.exp(1j*kk*phi) for kk in range(N)))**2
def grating_closed(N, phi):
    return np.sin(N*phi/2)**2/np.sin(phi/2)**2
phis = np.linspace(0.05, 2*np.pi - 0.05, 400)
for N in (4, 8):
    loop = np.array([grating_loop(N, p) for p in phis])
    closed = grating_closed(N, phis)
    print(f"N={N}: max |loop - closed| = {np.max(np.abs(loop - closed)):.2e}")
    assert np.allclose(loop, closed, atol=1e-8)"""),

md(r"""## 2. A recurrence becomes an explicit formula (`rsolve`)

An iterative recurrence -- each term from the previous ones -- is a loop in disguise. `rsolve` returns the
closed form. The Fibonacci recurrence $f_m=f_{m-1}+f_{m-2}$ collapses to Binet's formula; the same tool
handles the linear recurrences behind multilayer-optics reflectivity (Chebyshev) and other physics ladders."""),
co("""f = sp.Function('f'); m = sp.symbols('m', integer=True)
binet = sp.simplify(sp.rsolve(f(m) - f(m-1) - f(m-2), f(m), {f(0): 0, f(1): 1}))
print("Fibonacci closed form (Binet):", binet)

def fib_loop(M):
    a, b = 0, 1
    for _ in range(M):
        a, b = b, a + b
    return a
binet_fn = sp.lambdify(m, binet, "numpy")
for M in (10, 20, 30):
    assert round(float(binet_fn(M))) == fib_loop(M)
print("rsolve closed form matches the loop for m = 10, 20, 30")"""),

md(r"""## 3. One expression, no loop: CSE + `lambdify`

`cse` factors repeated subexpressions (compute once, reuse) and `lambdify` compiles the whole thing to a
vectorized NumPy callable -- so evaluating over an array is a single call, not a Python loop. Both the
op-count reduction and the speed-up are shown."""),
co("""import time
x = sp.symbols('x')
# an expression with a repeated inner term (the kind CSE loves)
inner = sp.sin(x)**2 + sp.cos(x)
expr = inner**3 + inner**2 + sp.sqrt(inner + 2)
subs, reduced = sp.cse(expr)
print("cse extracted:", subs, "\\n reduced:", reduced[0])

f = sp.lambdify(x, expr, "numpy")
xs = np.linspace(0, 10, 200000)
t = time.perf_counter(); vec = f(xs); t_vec = time.perf_counter() - t             # loop-free
t = time.perf_counter(); loop = np.array([float(expr.subs(x, xv)) for xv in xs[:2000]]); t_loop = time.perf_counter() - t
assert np.allclose(vec[:2000], loop)
print(f"vectorized lambdify: {t_vec*1e3:.2f} ms for {xs.size} points")
print(f"python subs-loop:    {t_loop*1e3:.2f} ms for 2000 points  -> lambdify avoids the loop entirely")"""),

md(r"""## Plot: the grating intensity, loop points vs closed-form curve"""),
co(r"""fig, ax = plt.subplots(figsize=(7, 4))
N = 6
ax.plot(phis, grating_closed(N, phis), color="#4C78A8", lw=2, label="closed form  $\\sin^2(N\\phi/2)/\\sin^2(\\phi/2)$")
ax.plot(phis[::12], [grating_loop(N, p) for p in phis[::12]], "o", color="#E45756", ms=4, label="slit-by-slit loop")
ax.set_xlabel(r"$\phi$"); ax.set_ylabel("intensity"); ax.set_title(f"N={N}-slit grating: loop collapsed to an expression")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()"""),

md(r"""## Summary

- **Symbolic summation** collapses accumulation loops into closed forms ($\sum k^2$, and the geometric
  N-slit sum that *is* the diffraction-grating intensity -- verified against the explicit loop).
- **`rsolve`** turns an iterative recurrence into an explicit formula (Fibonacci $\to$ Binet), matching the
  loop term-for-term.
- **`cse` + `lambdify`** turn an expression into loop-free vectorized code -- the same step that feeds the C
  code generator; no Python `for` at run time.

Subject-verb-object: the sum becomes a formula; the recurrence becomes a formula; the formula vectorizes;
the loop disappears."""),
]

write("sympy_loops", "to_expressions", cells)
