"""Build notebooks/maxwell_discrete_symmetries.ipynb

Extends notebooks/griffiths_1p10_pseudovectors.ipynb (which proves, in the
abstract, that a cross product of two polar vectors is a pseudovector) by
applying that exact result to the real E and B field laws, and adding
time-reversal symmetry (not covered there at all).

Research-partner notebook template: Theory -> Derivation -> SymPy ->
Numerical example -> Plots -> Parameter sweep -> Engineering interpretation
-> Research discussion -> Possible experiments -> Future improvements.

Engine: dgs/maxwell_discrete_symmetries.py (this session).
"""
import json, pathlib

cells = []

def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in src.splitlines()]})
def code(src): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s+"\n" for s in src.splitlines()]})

# ── Title ─────────────────────────────────────────────────────────────────────
md("""# Discrete Symmetries of Maxwell's Equations: Parity and Time Reversal

`notebooks/griffiths_1p10_pseudovectors.ipynb` already proves, in the
abstract, that a cross product of two polar vectors is a pseudovector. This
notebook applies that exact result to the REAL field laws -- Coulomb's law
for **E**, Biot-Savart's law for **B** -- to derive (not assert) that E is a
polar vector and B is axial, then adds **time reversal** (not covered
there at all), and checks both symmetries directly against all four of
Maxwell's equations. Engine: `dgs/maxwell_discrete_symmetries.py`.
""")

code("""%matplotlib inline
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('..').resolve()))

import sympy as sp
from dgs import maxwell_discrete_symmetries as sym

sp.init_printing(use_latex="mathjax")
print("Setup complete. sympy", sp.__version__)
""")

# ── 1. Theory ─────────────────────────────────────────────────────────────────
md("""## 1. Theory: Two Discrete Symmetries

**Parity** $P$: $\\mathbf r \\to -\\mathbf r$ (mirror the whole universe
through the origin). **Time reversal** $T$: $t\\to -t$ (run the film
backward). Both are symmetries a fundamental physical law can, but need
not, respect -- Maxwell's equations respect both individually (this
notebook checks that directly), while the weak interaction famously does
not respect $P$ alone (a fact outside this notebook's scope, noted only
for contrast).
""")

# ── 2. Derivation: E and B from their defining laws ──────────────────────────
md("""## 2. Derivation: E is Polar, B is Axial -- From the Field Laws Themselves

**Coulomb's law**: $\\mathbf E(\\mathbf r) = kq\\,\\mathbf r/|\\mathbf r|^3$.
Substituting $\\mathbf r\\to-\\mathbf r$ flips the numerator's sign exactly
once: $\\mathbf E(-\\mathbf r)=-\\mathbf E(\\mathbf r)$ -- **E is polar**.

**Biot-Savart's law**: $d\\mathbf B \\sim I\\,d\\boldsymbol\\ell\\times
\\mathbf r/|\\mathbf r|^3$. A current element $d\\boldsymbol\\ell$ is itself
a displacement (polar, same as $\\mathbf r$). Under parity BOTH
$d\\boldsymbol\\ell\\to-d\\boldsymbol\\ell$ and $\\mathbf r\\to-\\mathbf r$,
so their cross product picks up $(-1)(-1)=+1$:
$\\mathbf B(-\\mathbf r,-d\\boldsymbol\\ell)=+\\mathbf B(\\mathbf r,
d\\boldsymbol\\ell)$ -- **B is axial** (a pseudovector), the exact mechanism
`griffiths_1p10_pseudovectors.ipynb` proves in the abstract, applied here
to the concrete law.
""")

code("""ok_E = sym.coulomb_field_parity_check()
ok_B = sym.biot_savart_field_parity_check()
print(f"Coulomb's law:  E(-r) = -E(r)      (E is polar)?       {ok_E}")
print(f"Biot-Savart:    B(-r,-dl) = +B(r,dl) (B is axial)?      {ok_B}")
""")

# ── 3. SymPy: time reversal via the chain rule ───────────────────────────────
md("""## 3. SymPy: Time Reversal via the Chain Rule

For a trajectory $x(t)$, the time-reversed motion is $\\tilde x(t)=x(-t)$
(run the film backward). The chain rule gives $\\tilde v(t)=-v(-t)$ --
**velocity is T-odd** -- and $\\tilde a(t)=+a(-t)$ -- **acceleration is
T-even** (the two chain-rule sign flips from differentiating twice cancel,
consistent with $F=ma$ holding in both time directions). Verified below
with `sympy.Function`, not quoted.
""")

code("""t = sp.Symbol('t', real=True)
x = sp.Function('x')(t)
v, a = sp.diff(x, t), sp.diff(x, t, 2)
x_tilde = x.subs(t, -t)
v_tilde, a_tilde = sp.diff(x_tilde, t), sp.diff(x_tilde, t, 2)

print("v~(t)      =", v_tilde)
print("-v(-t)     =", sp.simplify(-v.subs(t, -t)))
print("match (velocity T-odd)?     ", sym.time_reversal_velocity_parity())
print()
print("a~(t)      =", a_tilde)
print("+a(-t)     =", a.subs(t, -t))
print("match (acceleration T-even)?", sym.time_reversal_acceleration_parity())
""")

md("""Combined with $F=ma$ (T-even, Newton's law must hold run forward or
backward) and the Lorentz force law $\\mathbf F=q(\\mathbf E+\\mathbf v
\\times\\mathbf B)$: since $\\mathbf v$ is T-odd and $\\mathbf F$ must stay
T-even, $\\mathbf E$ must be **T-even** and $\\mathbf B$ must be **T-odd**
(so that $\\mathbf v\\times\\mathbf B$, odd$\\times$odd, comes out even).
""")

# ── 4. Numerical/table example: the full symmetry table ─────────────────────
md("""## 4. The Full Symmetry Table

Combining sections 2-3: E is P-odd (polar) / T-even; B is P-even (axial) /
T-odd. Every quantity's type below is DERIVED above, not a new assumption.
""")

code("""from IPython.display import Markdown, display

rows = [
    ("charge density rho", "+1 (scalar)", "+1 (T-even)"),
    ("current density J",  "-1 (polar, like v)", "-1 (T-odd, like v)"),
    ("E",                  "-1 (polar)", "+1 (T-even)"),
    ("B",                  "+1 (axial)", "-1 (T-odd)"),
]
table = "| quantity | P-type | T-type |\\n|---|---|---|\\n"
for name, ptype, ttype in rows:
    table += f"| {name} | {ptype} | {ttype} |\\n"
display(Markdown(table))
""")

# ── 5. Parameter sweep: checking all 4 Maxwell equations both ways ──────────
md("""## 5. Checking All Four Maxwell Equations, Both Symmetries

Composition rule for **parity**: cross/dot products and the gradient
operator (P-type $-1$, transforming like $\\mathbf r$) combine by
MULTIPLYING P-types -- this is exactly the mechanism
`griffiths_1p10_pseudovectors.ipynb` already proved abstractly (Part c).

Composition rule for **time reversal**: div/curl (spatial, don't touch
$t$) leave T-type unchanged; $\\partial/\\partial t$ FLIPS T-type.

$\\nabla\\cdot\\mathbf B=0$ is trivially consistent under both (its RHS is
exactly 0, which carries no nonzero-parity constraint) -- reported as such
below, not claimed as a real check.
""")

code("""print("=== Parity self-consistency ===")
for name, ok in sym.maxwell_parity_consistency().items():
    print(f"  {name}: {'consistent' if ok else 'INCONSISTENT'}")

print("\\n=== Time-reversal self-consistency ===")
for name, ok in sym.maxwell_time_reversal_consistency().items():
    print(f"  {name}: {'consistent' if ok else 'INCONSISTENT'}")
""")

code("""pt = sym.combined_pt_type()
print("=== Combined PT ===")
for field, val in pt.items():
    print(f"  PT-type of {field}: {val:+d}  ({'odd' if val < 0 else 'even'})")
print()
print("Both E and B are PT-odd -- a direct consequence of sections 2-3,")
print("not a new assumption: P-odd*T-even=-1 for E, P-even*T-odd=-1 for B.")
""")

# ── 6. Engineering interpretation ─────────────────────────────────────────────
md("""## 6. Engineering Interpretation

- The P and T types derived here aren't textbook trivia -- they're WHY a
  bar magnet's field pattern looks the same in a mirror held along its
  axis (B axial) while an electric dipole's field pattern flips (E polar),
  and why running a video of an LC circuit's oscillation backward still
  looks like a valid LC oscillation (E, B individually respect T) while a
  video of a resistor dissipating heat run backward looks obviously wrong
  (irreversibility comes from statistical/entropic physics NOT captured by
  Maxwell's equations alone, outside this notebook's scope).
- This is the same style of check as `dgs/gs_verify.py`'s symbolic
  verification of `gs_core.py` and `dgs/paraxial_optics_abcd.py`'s
  unimodularity check: derive the physics from first principles in SymPy,
  then verify self-consistency, rather than asserting a textbook rule.
""")

# ── 7. Research discussion ────────────────────────────────────────────────────
md("""## 7. Research Discussion

- This notebook doesn't touch charge conjugation $C$ or the combined $CPT$
  theorem (a much deeper result from quantum field theory) -- an open
  extension if that's ever wanted, but a genuinely different (and harder)
  claim than the classical $P$/$T$ derivations here.
- Could the same `_P_TYPE`/`_T_TYPE` bookkeeping in
  `dgs/maxwell_discrete_symmetries.py` be extended to the polarization and
  magnetization fields $\\mathbf P$, $\\mathbf M$ (relevant to
  `dgs/connective_tissue_electrodynamics.py`'s D=eps*E treatment) for a
  consistency check across that module too?
- Real historical/personal-context aside, not physics: Keisuke Goda
  (University of Tokyo) is a real co-author on the original STEAM paper
  (Goda, Tsia, Jalali, *Nature* 458, 1145 (2009)) already cited throughout
  this repo -- the actual Japan side of the Jalali-lab collaboration this
  repo's physics descends from.
""")

# ── 8. Possible experiments ───────────────────────────────────────────────────
md("""## 8. Possible Experiments

1. Extend `_P_TYPE`/`_T_TYPE` to include the polarization/magnetization
   fields and re-check consistency against the macroscopic Maxwell
   equations used in `dgs/connective_tissue_electrodynamics.py`.
2. Add charge conjugation $C$ as a third discrete symmetry (E, B under
   $q\\to-q$) and see whether the same multiplicative bookkeeping approach
   extends cleanly, or needs a genuinely different treatment.
3. Numerically animate a point charge's field under literal spatial
   inversion and time reversal (matching
   `notebooks/griffiths_1p10_pseudovectors.ipynb`'s existing 3-D
   visualization style) to make the abstract P/T tables in this notebook
   visually concrete.
""")

# ── 9. Future improvements ────────────────────────────────────────────────────
md("""## 9. Future Improvements

- If `dgs/maxwell_discrete_symmetries.py` grows a $C$-symmetry treatment
  (per §8), rename/reorganize it to reflect $CPT$ rather than just $PT$.
- Cross-link this notebook from `notebooks/griffiths_1p10_pseudovectors.ipynb`
  itself (a "see also" pointer), since this notebook is explicitly built as
  an extension of it and a reader of one would likely want the other.
""")

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
out = pathlib.Path("notebooks/maxwell_discrete_symmetries.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written: {out}  ({len(cells)} cells)")
