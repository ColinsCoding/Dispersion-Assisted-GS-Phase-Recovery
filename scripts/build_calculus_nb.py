"""Build notebooks/calculus_for_college.ipynb -- Calc 1-2 worked out with SymPy."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Calculus, worked out -- for college (Calc 1 & 2)
### every rule shown with SymPy, and tied to real physics

The whole of intro calculus is two ideas: the **derivative** (instantaneous rate
of change) and the **integral** (accumulated total), and the **Fundamental
Theorem** says they undo each other. This notebook proves the rules, works
examples, and connects each one to where it shows up in physics/engineering.
You do not need to already know this -- it is built up line by line."""),

co("""import numpy as np, matplotlib.pyplot as plt
import sympy as sp
sp.init_printing()
x = sp.Symbol("x")
print("ready, sympy", sp.__version__)"""),

md("""## 1. The limit -- and the derivative as a limit (epsilon-delta)

The derivative is the **limit** of the slope of a shrinking secant line:
$$f'(x)=\\lim_{h\\to 0}\\frac{f(x+h)-f(x)}{h}.$$
Epsilon-delta makes "h -> 0" precise: *for any tolerance you want on the answer,
there is a small enough h that achieves it.*"""),
co("""f = lambda t: t**2
x0, true = 2.0, 4.0                       # f'(2) = 2x = 4 for f = x^2
print("difference quotient -> the derivative as h -> 0:")
for h in (0.1, 0.01, 0.001, 1e-5):
    q = (f(x0+h)-f(x0))/h
    print(f"  h={h:<7} slope={q:.5f}  error={abs(q-true):.1e}")
print("\\nSymPy limit:", sp.limit((( (x0+sp.Symbol('h'))**2 - x0**2)/sp.Symbol('h')), sp.Symbol('h'), 0))

# picture: secant lines collapsing onto the tangent
xs = np.linspace(0.5, 3.5, 200)
plt.figure(figsize=(7,4)); plt.plot(xs, xs**2, 'k', lw=2, label="f(x)=x^2")
for h, c in [(1.0,"#fdae61"),(0.5,"#abd9e9"),(0.01,"#2c7bb6")]:
    s = (f(x0+h)-f(x0))/h
    plt.plot(xs, f(x0)+s*(xs-x0), color=c, lw=1.2, label=f"secant h={h}")
plt.scatter([x0],[f(x0)], color="k", zorder=5); plt.legend(); plt.title("secant -> tangent (the derivative)")
plt.tight_layout(); plt.show()"""),

md("""## 2. The derivative rules (proved by SymPy)

- **power rule:** $\\frac{d}{dx}x^n = n x^{n-1}$
- **product rule:** $(fg)' = f'g + fg'$
- **chain rule:** $\\frac{d}{dx}f(g(x)) = f'(g)\\,g'$"""),
co("""n = sp.Symbol("n")
print("power rule:   d/dx x^n =", sp.diff(x**n, x))
f, g = sp.Function("f"), sp.Function("g")
print("product rule: d/dx f g =", sp.diff(f(x)*g(x), x))
print("chain rule:   d/dx f(g) =", sp.diff(f(g(x)), x))
# concrete: d/dx sin(x^2) = cos(x^2) * 2x  (chain rule)
print("example: d/dx sin(x^2) =", sp.diff(sp.sin(x**2), x))"""),

md("""## 3. Application -- optimization (max / min)

Set the derivative to zero to find critical points; the **second derivative**
tells you max (concave down, f'' < 0) vs min (f'' > 0). Example: a farmer with
40 m of fence makes a rectangular pen against a wall -- maximize the area."""),
co("""w = sp.Symbol("w", positive=True)             # width; the wall is one side
# fence used = 2*depth + width = 40  ->  depth = (40-w)/2 ; area = w * depth
area = w * (40 - w)/2
dA = sp.diff(area, w)
w_star = sp.solve(dA, w)[0]
print("area(w) =", sp.expand(area))
print("dA/dw = 0  at  w =", w_star, "  -> second deriv", sp.diff(area, w, 2), "(<0 => maximum)")
print("max area =", area.subs(w, w_star), "m^2  (w =", w_star, ", depth =", (40-w_star)/2, ")")"""),

md("""## 4. The integral and the Fundamental Theorem

The integral accumulates area under a curve. The **Fundamental Theorem of
Calculus** says differentiation and integration are inverses:
$$\\frac{d}{dx}\\int_a^x f(t)\\,dt = f(x).$$"""),
co("""F = sp.integrate(sp.cos(x), x)
print("integral of cos(x) =", F, "  and d/dx of that =", sp.diff(F, x), " (back to cos -> FTC)")
# definite integral = signed area
val = sp.integrate(x**2, (x, 0, 3))
print("area under x^2 from 0 to 3 =", val)
# u-substitution example: integral 2x e^{x^2} dx = e^{x^2}
print("substitution: integral 2x e^(x^2) dx =", sp.integrate(2*x*sp.exp(x**2), x))

xs = np.linspace(0, 3, 200)
plt.figure(figsize=(7,3.4)); plt.plot(xs, xs**2, 'k', lw=2)
plt.fill_between(xs, xs**2, alpha=0.3, color="#2c7bb6")
plt.title(f"definite integral = area under the curve = {val}"); plt.xlabel("x"); plt.tight_layout(); plt.show()"""),

md("""## 5. Taylor series (Calc 2) -- and the repo connection

A function near a point equals the sum of its derivatives:
$f(x)=\\sum f^{(k)}(a)/k!\\,(x-a)^k$. This is `dgs/taylor.py`, and the same expansion
of the propagation constant $\\beta(\\omega)$ gives the dispersion the receiver
inverts."""),
co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
from dgs import taylor
print("e^x to order 5 :", taylor.taylor_series(sp.exp(x), x, 0, 5))
print("sin x to order 5:", taylor.taylor_series(sp.sin(x), x, 0, 5))
# Euler's formula falls out of the series
print("e^(ix) = cos x + i sin x  (as series):",
      sp.simplify(taylor.taylor_series(sp.exp(sp.I*x), x, 0, 8)
                  - (taylor.taylor_series(sp.cos(x), x, 0, 8) + sp.I*taylor.taylor_series(sp.sin(x), x, 0, 8))) == 0)"""),

md("""## Where this lives in physics (so it sticks)

| calculus idea | physics it becomes |
|---|---|
| derivative = rate | velocity $dx/dt$, EMF $=-d\\Phi/dt$, current in a capacitor $C\\,dV/dt$ |
| chain rule | dispersion $\\beta_2=d^2\\beta/d\\omega^2$, group velocity $d\\omega/dk$ |
| integral = total | work $\\int F\\,dx$, charge $\\int I\\,dt$, energy under a curve |
| FTC (inverse ops) | why $d/dt$ in time is $\\times i\\omega$ in frequency (the receiver) |
| Taylor series | the dispersion operator $H(f)=e^{i\\pi D f^2}$, small-angle approximations |

Calculus is not a separate subject from your research -- it *is* your research,
written in symbols. Master these five and you have the whole toolkit. Civilian
education."""),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/calculus_for_college.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
