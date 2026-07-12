"""Shared helpers for generating repository notebooks.

Every builder in tools/build_NN_*.py imports this module so all notebooks share one structure:
a front-matter cell answering the six standard questions, a standard setup cell that loads physkit
and an optional PyTorch, and the section order required by the repository conventions.
"""

import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)

SECTIONS = ["English explanation", "Mathematical derivation", "Dimensions and SI units",
            "SymPy derivation", "NumPy implementation", "Pandas tables", "Matplotlib plots",
            "PyTorch (optional)", "Exercises", "Engineering applications"]


def frontmatter(number, title, questions):
    """A title cell and a table answering the six standard questions.

    `questions` is a dict with keys physics, mathematics, algorithm, software, experiment,
    engineering.
    """
    q = questions
    return md(f"""# {number} -- {title}

| question | answer |
|---|---|
| What physics problem? | {q['physics']} |
| What mathematics? | {q['mathematics']} |
| What algorithm? | {q['algorithm']} |
| What software? | {q['software']} |
| What experiment? | {q['experiment']} |
| What engineering application? | {q['engineering']} |

This notebook follows the repository template: English explanation, mathematical derivation,
dimensions and SI units, SymPy derivation, NumPy implementation, Pandas tables, Matplotlib plots,
optional PyTorch, exercises, and engineering applications. It reuses the `physkit` package and the
results of earlier chapters.""")


def setup_cell():
    """Standard imports: the scientific-Python stack, physkit, and an optional PyTorch.

    The first lines make ``physkit`` importable whether or not it is pip-installed, by walking up
    from the notebook's working directory to the repository ``src/`` folder. This lets the notebook
    run under any kernel that has the scientific-Python stack, without ``pip install -e src``.
    """
    return co("""import sys, pathlib
for _p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
    if (_p / "src" / "physkit" / "__init__.py").exists():
        sys.path.insert(0, str(_p / "src")); break     # locate physkit without installation
import numpy as np, pandas as pd, sympy as sp
import matplotlib.pyplot as plt
import physkit
from physkit import constants as C, units as U, linalg as la
from physkit.plotting import use_style
use_style()
torch = physkit.optional_torch()            # None if PyTorch is unavailable; NumPy stays authoritative
sp.init_printing()
print("physkit", physkit.__version__, "| numpy", np.__version__,
      "| torch:", "present" if torch is not None else "absent (optional)")""")


def section(name):
    """A level-2 heading cell for one of the standard sections."""
    return md(f"## {name}")


def write(number, slug, cells, outdir="notebooks"):
    """Assemble and write a notebook notebooks/NN_slug.ipynb with the python3 kernel."""
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    root = pathlib.Path(__file__).resolve().parents[1]
    out = root / outdir / f"{number}_{slug}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print("wrote", out)
    return out
