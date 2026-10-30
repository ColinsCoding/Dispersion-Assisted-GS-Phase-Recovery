"""Build notebooks/phys106_frq_practice.ipynb from dgs/phys106_frq.py.

Run:  py -3.12 tools/build_frq_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/phys106_frq_practice.ipynb

Solution cells are collapsed (source and output hidden) so you can try each problem first.
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgs import phys106_frq as F  # noqa: E402

OUT = ROOT / "notebooks" / "phys106_frq_practice.ipynb"
cells = []
HIDDEN = {"jupyter": {"source_hidden": True, "outputs_hidden": True}}


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s, hidden=False):
    c = nbf.v4.new_code_cell(s.strip("\n"))
    if hidden:
        c.metadata.update(HIDDEN)
    cells.append(c)


md(f"""
# PHYS 106 free-response practice

{len(F.PROBLEMS)} original problems across relativity, quantization, wave functions, 1-D QM, and nuclear
decay, worth {sum(p.points for p in F.PROBLEMS.values())} points in total. They're written for this repo, not taken from Knight or any exam.

**How to use it**
1. Work each problem on paper first, in the 6-step write-up below. Written work is what earns
   free-response credit.
2. Type your numbers, **with units**, into the `answers` cell and run it. The grader converts units
   (nm vs m, eV vs J), names the classic mistakes, and gives 1 of 2 points when you're within 10%.
3. Only then expand the collapsed solution cell (click the `...`).

Conceptual parts are not auto-graded: compare your sentences with the model answer's key ideas.
""")
md("## The write-up every free-response answer should follow\n```\n" + F.TEMPLATE + "```")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import pandas as pd
from IPython.display import display
from dgs import phys106_frq as F
pd.set_option("display.max_colwidth", 110)

def check(pid, answers):
    score, best, rows = F.grade_problem(pid, answers)
    display(pd.DataFrame([(lab, ans or "-", f"{r['points']}/{r['of']}", r["feedback"]) for lab, ans, r in rows],
                         columns=["part", "your answer", "points", "feedback"]))
    print(f"numeric score: {score}/{best}")

def solution(pid):
    for p in F.PROBLEMS[pid].parts:
        print(f"({p.label}) {p.prompt}\n    {'answer: ' + p.answer if p.answer else 'model answer:'}\n    {p.solution}")
        if p.key_ideas:
            print("    credit for: " + "; ".join(p.key_ideas))
        print()
""")

for i, p in enumerate(F.PROBLEMS.values(), 1):
    parts = "\n".join(f"- **({q.label})** {q.prompt} *[{q.points} pts{'' if q.numeric else ', written'}]*"
                      for q in p.parts)
    md(f"## Problem {i}: {p.title}  (Ch. {p.chapter}, {p.points} pts)\n\n{p.setup}\n\n{parts}")
    numeric = [q.label for q in p.parts if q.numeric]
    demo = i == 2
    entries = ", ".join(f'"{lab}": "{"1.3 c" if demo else ""}"' for lab in numeric)
    code(("# example: a Galilean answer, to show the feedback. Replace it with yours.\n" if demo else "")
         + f'answers = {{{entries}}}\ncheck("{p.pid}", answers)')
    code(f'solution("{p.pid}")', hidden=True)

md(r"""
## Timed practice exam

Pick 4 random problems (change `seed` each time) and give yourself 12 minutes per problem.
Set `chapters=[40]` to drill one chapter.
""")
code(r"""
for p in F.practice_exam(4, seed=2026):
    print(f"Ch. {p.chapter}  {p.title}  ({p.pid}, {p.points} pts)")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells")
