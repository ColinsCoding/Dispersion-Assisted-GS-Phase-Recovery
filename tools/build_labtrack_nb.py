"""Build notebooks/labtrack_sql_ansi_c.ipynb from dgs/labtrack.py.

Run:  py -3.12 tools/build_labtrack_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/labtrack_sql_ansi_c.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "labtrack_sql_ansi_c.ipynb"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Lab records in SQL and ANSI C: laser-safety training, journal club, and eyewear bounds

There are two different "ANSI"s in this notebook:
- **ANSI C** (C89) is the language standard. The program compiles with
  `gcc -std=c89 -pedantic -Wall -Wextra -Werror`.
- **ANSI Z136.1** is the US laser-safety standard. The database tracks who has finished training
  and checks eyewear with the same simplified visible-light MPE as `dgs/laser_eyewear.py`. It
  links to each standards body's site and doesn't reproduce any standard's text.

| File | Role |
|---|---|
| `dgs/c/labtrack/schema.sql` | tables, **bounds as CHECK constraints**, and a clearance view |
| `dgs/c/labtrack/seed.sql` | synthetic people, real papers (with DOI URLs), standards bodies by country |
| `dgs/c/labtrack/labtrack.c` | C89 + SQLite: run SQL files, print the report, insert with **prepared statements** |

*Educational only: the institution's Laser Safety Officer decides who works at the bench.*
""")
code(r"""
import sys, sqlite3, tempfile
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import pandas as pd
from IPython.display import display
from dgs import labtrack as T
from dgs.eventsim import ToolUnavailable
pd.set_option("display.max_colwidth", 70)
DB = Path(tempfile.gettempdir()) / "dgs_labtrack" / "notebook.db"
try:
    T.fresh_db(DB); HAVE_C = True
except ToolUnavailable as err:
    HAVE_C = False; print("skipped (no gcc):", err)
""")

md("## 1. The report printed by the C program")
code(r"""
if HAVE_C:
    print(T.report(DB, "2026-10-29"))
""")

md(r"""
## 2. SQL you should be able to write

These are the queries the C program runs, written in SQL: a JOIN, a GROUP BY with a
conditional aggregate, and a correlated subquery.
""")
code(r"""
if HAVE_C:
    con = sqlite3.connect(DB)
    queries = {
        "standards per country (GROUP BY)": "SELECT country, COUNT(*) AS n, group_concat(document, '; ') AS documents FROM standard GROUP BY country",
        "journal club with presenter (JOIN)": "SELECT j.held_on, m.name AS presenter, j.topic, j.doi_url FROM journal_club j JOIN member m ON m.id = j.presenter_id ORDER BY j.held_on",
        "attendance per session (subquery)": "SELECT j.title, (SELECT COUNT(*) FROM attendance a WHERE a.session_id = j.id) AS attended FROM journal_club j",
        "clearance view": "SELECT * FROM clearance",
    }
    for title, q in queries.items():
        print(title); print("   ", q); display(pd.read_sql(q, con))
""")

md(r"""
## 3. Bounds: the database refuses impossible data

Each row below breaks one CHECK or foreign-key constraint. The database rejects it, whichever
program tries to write it: Python here, C in the next section.
""")
code(r"""
bad = [
    ("wavelength below 180 nm", "INSERT INTO laser (name, wavelength_nm, power_w, beam_diameter_mm, hazard_class) VALUES ('uv', 100, 1, 1, '4')"),
    ("negative power", "INSERT INTO laser (name, wavelength_nm, power_w, beam_diameter_mm, hazard_class) VALUES ('n', 532, -1, 1, '2')"),
    ("laser class 5 doesn't exist", "INSERT INTO laser (name, wavelength_nm, power_w, beam_diameter_mm, hazard_class) VALUES ('c', 532, 1, 1, '5')"),
    ("OD above 10", "INSERT INTO eyewear (label, laser_id, od, lo_nm, hi_nm) VALUES ('x', 1, 11, 500, 560)"),
    ("eyewear for a laser that doesn't exist", "INSERT INTO eyewear (label, laser_id, od, lo_nm, hi_nm) VALUES ('x', 99, 3, 500, 560)"),
    ("non-https URL", "INSERT INTO standard (country, body, document, scope, url) VALUES ('X','Y','Z','s','http://x')"),
    ("training expires before completion", "INSERT INTO training (member_id, topic, completed, expires) VALUES (1, 'laser safety', '2026-10-01', '2026-09-01')"),
]
if HAVE_C:
    con.execute("PRAGMA foreign_keys = ON")
    rows = []
    for why, q in bad:
        try:
            con.execute(q); rows.append((why, "ACCEPTED (bug!)"))
        except sqlite3.IntegrityError as err:
            rows.append((why, f"rejected: {err}"))
    display(pd.DataFrame(rows, columns=["bad row", "result"]))
""")

md(r"""
## 4. ANSI C: prepared statements stop SQL injection

`add-laser` binds each argument with `sqlite3_bind_*` instead of pasting text into the SQL.
A name like `x'); DROP TABLE laser; --` is stored as ordinary text, and the bounds still apply.
""")
code(r"""
if HAVE_C:
    print(T.run(DB, "add-laser", "uv", 100, 1, 1, "4")[2])
    rc, out, err = T.run(DB, "add-laser", "x'); DROP TABLE laser; --", 532, 0.001, 1, "2")
    print(out); display(pd.read_sql("SELECT id, name, wavelength_nm, hazard_class FROM laser", con))
""")

md(r"""
## 5. The eyewear check, cross-checked against Python

The C program computes the exit-beam irradiance E = 4P/(πd²) and the 0.25 s visible MPE, then
OD_needed = log₁₀(E/MPE). The same numbers come from `dgs/laser_eyewear.py`. A 5 mW green pointer
with a 1 mm beam needs about **OD 2.4** at the aperture, so "OD 2" goggles **fail** for direct
viewing. A HeNe with green goggles fails for a different reason: the goggles aren't rated for
that wavelength.
""")
code(r"""
if HAVE_C:
    from dgs import laser_eyewear as LS
    rows = T.eyewear_rows(T.report(DB, "2026-10-29"))
    mpe = LS.mpe_visible(LS.BLINK_S)
    for r in rows:
        if r["need_od"] is not None:
            r["python need_od"] = round(LS.required_od(r["E_W_cm2"], mpe), 2)
    display(pd.DataFrame(rows))
    con.close()
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells")
