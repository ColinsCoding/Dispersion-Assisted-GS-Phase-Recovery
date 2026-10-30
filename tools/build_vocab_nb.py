"""Build notebooks/modern_physics_vocab.ipynb and docs/modern_physics_vocab.md from dgs/modern_physics_vocab.py.

Run:  py -3.12 tools/build_vocab_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/modern_physics_vocab.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgs import modern_physics_vocab as V  # noqa: E402

OUT = ROOT / "notebooks" / "modern_physics_vocab.ipynb"
SHEET = ROOT / "docs" / "modern_physics_vocab.md"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Modern physics vocabulary: English, math, science, and decisions

Every term from Knight Ch. 36–42 (PHYS 106), in four columns:

| Column | Question it answers |
|---|---|
| **English** | What does the word mean, in plain words? |
| **Math** | What is the equation (if any)? Every equation here is **checked for dimensional consistency** by SymPy. |
| **Science** | Why does it matter? What does it say about nature? |
| **Decide** | When do I use it on a problem, and what's the trap? |

The **Decisions** section turns the Midterm 1 mistakes into rules: given → asked → use → avoid.
A printable copy of everything is generated at `docs/modern_physics_vocab.md`.
The definitions are written for this repo, not copied from the book. Your syllabus decides
which chapters are tested.
""")

code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import sympy as sp
sp.init_printing(use_unicode=True)
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from dgs import modern_physics_vocab as V
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 250)
print(f"{len(V.VOCAB)} terms, {sum(1 for t in V.VOCAB if t.equation)} with equations, {len(V.DECISIONS)} decision rules")
""")

md("## 1. The vocabulary, chapter by chapter")
code(r"""
def show_equation(t):
    if not t.equation:
        return ""
    syms = {k: sp.Symbol(k) for k in t.units}
    return sp.sympify(t.equation, locals={**syms, "Rational": sp.Rational})

for ch, name in V.CHAPTERS.items():
    terms = V.by_chapter(ch)
    display(Markdown(f"### {'Ch. ' + str(ch) + ': ' if ch else ''}{name} ({len(terms)} terms)"))
    display(pd.DataFrame([(t.term, t.english, t.science, t.decide) for t in terms],
                         columns=["term", "English", "science", "decide"]))
""")

md(r"""
## 2. The math, with dimension checks

Each equation is shown in SymPy, with the SI unit of every symbol, and checked: both sides
must have the same dimensions, terms added together must match, and the inside of exp, sin,
and log must be dimensionless.
""")
code(r"""
rows = []
for t in V.VOCAB:
    if t.equation:
        ok = V.check_dimensions(t)
        rows.append((t.chapter, t.term, sp.sstr(show_equation(t)), ", ".join(f"{k} [{v}]" for k, v in t.units.items()), "consistent" if ok else "FAIL"))
display(pd.DataFrame(rows, columns=["ch", "term", "equation", "units", "dimension check"]))
print("all consistent:", all(r[-1] == "consistent" for r in rows))
""")
code(r"""
for name in ("photon energy", "relativistic velocity addition", "penetration distance"):
    t = V.lookup(name)[0]
    display(Markdown(f"**{t.term}**: {t.english}")); display(show_equation(t))
""")

md(r"""
## 3. The checker catches real mistakes

These are the errors from Midterm 1 (and two classics), written as equations. Every one fails.
""")
code(r"""
planted = [
    ("Midterm Q2.2: E = h * lambda", "Eq(E, h*lam)", {"E": "J", "h": "J*s", "lam": "m"}),
    ("Midterm Q3.3: v from E + B", "Eq(v, Ef + B)", {"v": "m/s", "Ef": "V/m", "B": "T"}),
    ("units inside an exponential", "Eq(Ptun, exp(-2*w))", {"Ptun": "1", "w": "m"}),
    ("kinetic energy as m v", "Eq(K, m*v)", {"K": "J", "m": "kg", "v": "m/s"}),
]
rows = []
for label, eq, units in planted:
    try:
        V.check_dimensions(V.Term(label, 0, "", "", "", eq, units)); rows.append((label, "NOT caught"))
    except ValueError as err:
        rows.append((label, "caught: " + str(err).split(": ", 1)[1][:90]))
display(pd.DataFrame(rows, columns=["mistake", "result"]))
""")

md("## 4. Decisions: given → asked → use → avoid")
code(r"""
display(pd.DataFrame(V.DECISIONS).drop(columns="keywords"))
""")
code(r"""
questions = [
    "A photon has a wavelength of 178 nm. Find its energy in eV.",
    "Light hits a metal and the stopping potential is 5.7 V. Find the work function.",
    "Two eruptions are simultaneous on Earth. Are they simultaneous for the moving observer?",
    "A rock moves toward the ship at 0.4c; find the velocity of the rock relative to the ship.",
    "A 10 eV electron hits the atom. Use the energy diagram to find the emitted wavelengths.",
    "A charge passes undeflected through crossed electric and magnetic fields. Find v.",
]
for q in questions:
    best = V.decide(q)
    print(f"Q: {q}")
    if best:
        r = best[0][1]
        print(f"   use:   {r['use']}\n   avoid: {r['avoid']}\n")
""")

md(r"""
## 5. Flashcards

Read the term, say the English, math, science, and decision out loud, then run the next cell
to check yourself. Change `seed` for a new deck and `chapter` to focus on one chapter.
""")
code(r"""
deck = V.flashcards(chapter=None, n=8, seed=2026)
for i, card in enumerate(deck, 1):
    print(f"{i}. {card.term}   (Ch. {card.chapter})")
""")
code(r"""
for i, card in enumerate(deck, 1):
    eq = f"\n   math:    {sp.sstr(show_equation(card))}" if card.equation else ""
    print(f"{i}. {card.term}\n   english: {card.english}{eq}\n   science: {card.science}\n   decide:  {card.decide}\n")
""")

md("## 6. Coverage and the printable sheet")
code(r"""
counts = pd.Series({f"{ch} {name[:22]}" if ch else name: len(V.by_chapter(ch)) for ch, name in V.CHAPTERS.items()})
ax = counts.plot.barh(figsize=(7, 3.5), color="C0"); ax.set_xlabel("terms"); ax.invert_yaxis()
ax.set_title(f"{len(V.VOCAB)} terms by chapter"); plt.tight_layout(); plt.show()
sheet = Path.cwd().parent / "docs" / "modern_physics_vocab.md" if Path.cwd().name == "notebooks" else Path("docs/modern_physics_vocab.md")
print("printable study sheet:", sheet, "exists:", sheet.exists())
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
SHEET.write_text(V.to_markdown(), encoding="utf-8")
print("wrote", OUT, "with", len(cells), "cells, and", SHEET)
