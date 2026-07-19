"""Extend notebooks/coppinger1999_sympy.ipynb (already covers Eq 1-9) with:
  - sp.init_printing() in the imports cell
  - a precalculus warm-up (completing the square) before Section 1
  - a group-velocity derivation (v_g, beta_2 from k(omega)) after Section 1
  - a measurement/quantum-parallel section (I=|E|^2 loses phase) before Summary
Requested for a professor-facing version of the notebook. Extends the
existing, working notebook rather than duplicating it.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "coppinger1999_sympy.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# 1. add init_printing() to the imports cell
cells[1].source = cells[1].source.replace(
    'print("Imports OK")',
    'sp.init_printing()\nprint("Imports OK -- init_printing enabled for LaTeX rendering")'
)

precalc_md = md(r"""## Section 0 -- Precalculus Warm-Up: Completing the Square

Every "stretch factor" and "chirp" result in this paper falls out of one algebra
move from precalculus: **completing the square** on a quadratic exponent. This
section does it once on a plain toy example, so Section 4's Appendix derivation
later is just "the same trick, more symbols."

Given $ax^2+bx+c$, completing the square rewrites it as $a(x+b/2a)^2 + (c-b^2/4a)$ --
the exponent is still a parabola, but its **vertex shifts** and its **width scales
by $a$**. In the paper, $x$ is frequency $f$, $a$ packages the fiber lengths and
$\beta_2$, and "width scales by $a$" is literally the pulse-stretch factor $M$
that shows up in Eq. (8).""")

precalc_code = code(r"""x, a_c, b_c, c_c = sp.symbols('x a b c', real=True)
quadratic = a_c*x**2 + b_c*x + c_c
completed = a_c*(x + b_c/(2*a_c))**2 + (c_c - b_c**2/(4*a_c))

print("Original:")
sp.pprint(quadratic)
print("\nCompleted-square form:")
sp.pprint(completed)

assert sp.simplify(sp.expand(quadratic) - sp.expand(completed)) == 0
print("\nVerified: completing the square is an EXACT identity, not an approximation.")
print("The leading coefficient 'a' is exactly the width-scaling factor that becomes")
print("M in the paper's Appendix (Eq. A2-A6) once x -> f and a -> a function of L1, L2, beta_2.")""")

gv_md = md(r"""## Group Velocity: What $\beta_2$ Actually Means

The paper treats $\beta_2$ ("second derivative of the propagation constant") as a
given constant, but it comes from a real physical quantity: the **group velocity**
$v_g$, the speed the pulse *envelope* travels at (as opposed to $v_p=\omega/k$, the
speed of the carrier's phase fronts).

Taylor-expand the dispersion relation $k(\omega)$ around the carrier $\omega_0$:

$$k(\omega) = k_0 + \frac{1}{v_g}(\omega-\omega_0) + \frac{\beta_2}{2}(\omega-\omega_0)^2 + \dots$$

- 1st-order term: $dk/d\omega|_{\omega_0} = 1/v_g$ -- this is *why* $v_g \equiv d\omega/dk$.
- 2nd-order term: $d^2k/d\omega^2|_{\omega_0} = \beta_2$ -- this IS the paper's $\beta_2$:
  it measures how much $v_g$ itself changes with frequency, i.e. how much the pulse's
  different colors separate as they propagate. $\beta_2=0$ would mean every frequency
  travels at the same $v_g$ -- no chirp, no stretch, no $M$.""")

gv_code = code(r"""w, w0 = sp.symbols('omega omega_0', real=True, positive=True)
k_fn = sp.Function('k')(w)

print("Generic k(omega), 2nd-order Taylor series around the carrier omega_0:")
sp.pprint(k_fn.series(w, w0, 3).removeO())

print("\ndk/domega at omega_0 defines 1/v_g (envelope speed):")
sp.pprint(sp.Derivative(k_fn, w).subs(w, w0))

print("\nd^2k/domega^2 at omega_0 IS the paper's beta_2 (GVD):")
sp.pprint(sp.Derivative(k_fn, w, 2).subs(w, w0))

# concrete toy dispersion relation to see actual numbers fall out
c_sym = sp.symbols('c', positive=True)
n_w = 1.45 + sp.Rational(1, 100) * (w - 1) ** 2   # toy n(omega) around w=1 (normalized units)
k_w = w * n_w / c_sym
vg_expr = sp.simplify(1 / sp.diff(k_w, w))
beta2_expr = sp.simplify(sp.diff(k_w, w, 2))

print("\ntoy n(omega) = 1.45 + 0.01*(omega-1)^2:")
print("v_g(omega) =")
sp.pprint(vg_expr)
print("beta_2(omega) =")
sp.pprint(beta2_expr)
print("\nAt the carrier (omega=1): v_g =", vg_expr.subs(w, 1), "  beta_2 =", beta2_expr.subs(w, 1))
print("beta_2 != 0 here because n(omega) is NOT flat -- exactly the GVD this whole paper runs on.")""")

measure_md = md(r"""## What a Photodetector Can Actually See

Every equation above computes the complex field $E(t)$ -- amplitude AND phase.
But Eq. (7)'s $I(t)=|E(t)|^2$ is the ONLY thing a real photodetector measures:
**phase is invisible to intensity-only detection.** This isn't a limitation of this
particular experiment -- it's the same measurement problem you meet again in
quantum mechanics: a photodetector, like the Born rule, only ever reports
$|\text{amplitude}|^2$, never the amplitude itself.

That's exactly the gap this repo's `dgs.gs_core.retrieve_phase` exists to close:
given only $I_1=|E_1|^2$ and $I_2=|E_2|^2$ measured through two *different*
dispersions, Gerchberg-Saxton recovers the hidden phase by alternating
projections between what's measured (intensity) and what's physical (a field
consistent with both). Coppinger/Jalali's 1999 ADC never needed phase -- it
digitizes intensity directly. The dispersion-assisted GS receiver is what you
get when you insist on recovering the field anyway.""")

measure_code = code(r"""A_sym, phi_sym = sp.symbols('A phi', real=True, positive=True)
E_field = A_sym * sp.exp(sp.I * phi_sym)
I_measured = sp.simplify(sp.Abs(E_field) ** 2)

print("E(t) = A*exp(i*phi)")
print("I = |E|^2 =", I_measured, " <-- phi has completely disappeared")
print()
print("Given only I, phi is undetermined: infinitely many (A, phi) pairs give the same I.")
print("Same structure as Born's rule: p(x) = |psi(x)|^2 loses the phase of psi(x).")
print("GS phase retrieval needs a SECOND, differently-dispersed intensity measurement")
print("to break this degeneracy -- see dgs.gs_core.retrieve_phase demonstrated above.")""")

# locate insertion points by scanning markdown headers (robust to any prior manual edits)
def find_index(snippet):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and snippet in c.source:
            return i
    raise ValueError(f"could not find a markdown cell containing: {snippet!r}")

idx_section1 = find_index("Section 1")
idx_section2 = find_index("Section 2")
idx_summary = find_index("## Summary")

# insert from the END backward so earlier indices stay valid as we splice
cells.insert(idx_summary, measure_code)
cells.insert(idx_summary, measure_md)

cells.insert(idx_section2, gv_code)
cells.insert(idx_section2, gv_md)

cells.insert(idx_section1, precalc_code)
cells.insert(idx_section1, precalc_md)

nb.cells = cells
nbf.write(nb, str(NB_PATH))
print(f"wrote {NB_PATH} with {len(cells)} cells (was {len(cells)-6})")
