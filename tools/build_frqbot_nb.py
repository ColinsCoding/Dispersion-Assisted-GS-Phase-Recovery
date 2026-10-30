"""Build notebooks/frqbot_timed_drill.ipynb from dgs/frqbot.py.

Run:  py -3.12 tools/build_frqbot_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/frqbot_timed_drill.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "frqbot_timed_drill.ipynb"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# A 15-minute free-response drill bot in ANSI C

`dgs/c/frqbot/frqbot.c` is one C89 file: `gcc -std=c89 -pedantic -Wall -Wextra -Werror`. It
asks original practice questions in **ANSI C, laser safety, AP Biology, AP Chemistry,
AP Psychology, AP Economics (supply and demand), and CompTIA**, and gives each question a
15-minute limit. A late answer scores 0, as on the exam.

**Use it in a terminal** (that's where the timer is real):
```
gcc -std=c89 -pedantic -Wall -Wextra -Werror dgs/c/frqbot/frqbot.c -o frqbot -lm
./frqbot --subject econ          # or c, laser, bio, chem, psych, comptia
./frqbot --n 4 --seed 7          # 4 random questions = one hour of practice
```
The notebook below feeds the bot answers automatically, to show how its grading works.
""")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import pandas as pd
from IPython.display import display
from dgs import frqbot as B
from dgs.eventsim import ToolUnavailable
pd.set_option("display.max_colwidth", 95)
try:
    display(pd.DataFrame(B.bank(), columns=["subject", "question"])); HAVE = True
except ToolUnavailable as err:
    HAVE = False; print("skipped:", err)
""")

md(r"""
## How it grades

- **Numbers:** a relative tolerance (1% for most). `2.00` and `2` both count; `25` for 20 doesn't.
- **Written answers:** the key ideas must appear, and `a|b` means either word counts. The bot
  tells you which idea was **missing**. Keyword grading is crude. A real reader also checks that
  your reasoning connects the ideas, so write full sentences.
""")
code(r"""
if HAVE:
    def shaky(prompt):                      # some right, some half-right, some wrong
        if "Demand increases" in prompt: return "the price goes up"
        if "equilibrium price." in prompt: return "20"
        if "price ceiling" in prompt: return "25"
        return "not sure"
    res, (score, asked) = B.drill(shaky, subject="econ", seed=3)
    display(pd.DataFrame(res)[["prompt", "verdict", "detail", "why"]]); print(f"score {score}/{asked}")
""")

md("## The timer (here with a 1-second limit and a 2-second answer)")
code(r"""
if HAVE:
    res, (score, asked) = B.drill(B.model_answer, subject="comptia", n=1, limit=1, seed=1, delay_s=2.2)
    print(res[0]["detail"]); print("score", score, "/", asked)
""")

md("## Full run with model answers: every subject")
code(r"""
if HAVE:
    res, (score, asked) = B.drill(B.model_answer, seed=11)
    df = pd.DataFrame(res)
    display(df.groupby("subject")["verdict"].apply(lambda v: f"{(v == 'CORRECT').sum()}/{len(v)}").rename("correct"))
    print(f"total {score}/{asked}")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells")
