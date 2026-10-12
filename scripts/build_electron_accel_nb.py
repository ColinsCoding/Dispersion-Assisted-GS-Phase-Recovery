"""Build notebooks/electron_accel_10V_sympy.ipynb (electron through 10 V, SymPy + indexed loops)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md(r"""# Electron accelerated from rest through $|\Delta V| = 10.0$ V — SymPy with indexed functions

**Question.** An electron, initially at rest, accelerates through a potential difference of magnitude 10.0 V. What is its final speed?

**Physics (cite).** Energy conservation for a charge in an electrostatic field:
$$\Delta K + \Delta U = 0,\qquad \Delta U = q\,\Delta V \;\Rightarrow\; \tfrac12 m v_f^2 - 0 = |q|\,|\Delta V|$$
- Halliday, Resnick & Walker, *Fundamentals of Physics*, Ch. 24 "Electric Potential", Eq. 24-6 ($\Delta U = q\Delta V$) and the "conservation of mechanical energy" sample problems.
- Young & Freedman, *University Physics*, §23.2 (electric potential; Eq. 23.13 $W_{a\to b} = q(V_a - V_b)$).
- Constants: CODATA 2018 (NIST), $e = 1.602176634\times10^{-19}$ C (exact), $m_e = 9.1093837015\times10^{-31}$ kg.

Non-relativistic check: $K = 10$ eV $\ll m_ec^2 = 511$ keV, so $\tfrac12 mv^2$ is fine (we verify with the relativistic formula below)."""),
code(r"""import sympy as sp
from sympy import IndexedBase, Idx, Function, Eq, solve, sqrt, Rational

q, m, dV, v = sp.symbols('q m Delta_V v', positive=True)

# Energy conservation: K_i + U_i = K_f + U_f, K_i = 0
K = Function('K'); U = Function('U')
energy = Eq(K(0) + U(0), K(1) + U(1))
subs = {K(0): 0, K(1): Rational(1, 2)*m*v**2, U(0) - U(1): q*dV}
# rearrange: K(1) = U(0) - U(1) = q*dV
eq = Eq(Rational(1, 2)*m*v**2, q*dV)
v_sym = solve(eq, v)[0]
energy, eq, Eq(v, v_sym)"""),
md(r"""## Constants as an indexed sequence

Use `IndexedBase` for the CODATA constants and loop over the index — the "indexed function" approach."""),
code(r"""c_val = IndexedBase('c')                  # c[0]=e, c[1]=m_e, c[2]=speed of light
i = Idx('i', 3)
names  = ['e (C)', 'm_e (kg)', 'c (m/s)']
values = [sp.Float('1.602176634e-19', 15), sp.Float('9.1093837015e-31', 15), sp.Integer(299792458)]
const = {c_val[k]: values[k] for k in range(3)}
for k in range(3):
    print(f"c[{k}] = {names[k]:9s} = {const[c_val[k]]}")

v_expr = v_sym.subs({q: c_val[0], m: c_val[1], dV: 10})
v_f = v_expr.subs(const).evalf(6)
print(f"\nv_f = sqrt(2 e dV / m_e) = {v_f} m/s")"""),
md(r"""## Loop over the answer choices (indexed)

`a[j]` holds each multiple-choice value; we loop and flag the one within 1 % of the computed speed."""),
code(r"""a = IndexedBase('a')
choices = [sp.Float('4.69e16'), sp.Float('4.69e5'), sp.Float('1.88e6'), sp.Float('1.88e3'), sp.Float('4.69e4')]
A = {a[j]: choices[j] for j in range(len(choices))}
for j in range(len(choices)):
    rel = abs(A[a[j]] - v_f) / v_f
    print(f"a[{j}] = {float(A[a[j]]):.3g} m/s   rel. err = {float(rel):.3e}   {'<-- ANSWER' if rel < 0.01 else ''}")"""),
md(r"""## Sweep: indexed potential differences $\Delta V_n$

Loop over $\Delta V_n = 1,5,10,100,1000$ V and compare classical vs relativistic speed
$v_{\rm rel} = c\sqrt{1 - (1+K/mc^2)^{-2}}$."""),
code(r"""Vn = IndexedBase('V'); n = sp.symbols('n', integer=True)
Vlist = [1, 5, 10, 100, 1000]
e_, me_, c_ = values
for k in range(len(Vlist)):
    Kk = e_ * Vlist[k]
    v_cl  = sqrt(2*Kk/me_).evalf(6)
    gamma = 1 + Kk/(me_*c_**2)
    v_rel = (c_*sqrt(1 - 1/gamma**2)).evalf(6)
    print(f"V[{k}] = {Vlist[k]:5d} V   v_classical = {float(v_cl):.4e}   v_relativistic = {float(v_rel):.4e}   diff = {float((v_cl-v_rel)/v_rel):.1e}")"""),
md(r"""## Answer

$$v_f = \sqrt{\frac{2(1.602\times10^{-19}\,\text{C})(10.0\,\text{V})}{9.109\times10^{-31}\,\text{kg}}} \approx \boxed{1.88\times10^{6}\ \text{m/s}}$$

Relativistic correction at 10 V is ~$10^{-5}$ — negligible. Distractors: $4.69\times10^{5}$ is exactly $v/4$; $1.88\times10^{3}$ has the right digits but is off by $10^3$ (a unit slip)."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, 'notebooks/electron_accel_10V_sympy.ipynb')
print('wrote')
