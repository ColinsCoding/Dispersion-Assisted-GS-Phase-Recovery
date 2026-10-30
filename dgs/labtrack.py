"""Build and drive the ANSI C (C89) + SQLite lab tracker in dgs/c/labtrack from Python.

    schema.sql   tables with CHECK-constraint bounds, and a clearance view
    seed.sql     synthetic members, standards by country, trainings, lasers, eyewear, journal club
    labtrack.c   C89 program: exec a SQL file, print a report, add a laser with a prepared statement

The C program is built with ``gcc -std=c89 -pedantic -Wall -Wextra -Werror ... -lsqlite3``.
Educational: not a substitute for an institution's Laser Safety Officer or ANSI Z136.1.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from dgs.eventsim import ToolUnavailable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dgs" / "c" / "labtrack"
CACHE = Path(tempfile.gettempdir()) / "dgs_labtrack"
EXE = ".exe" if os.name == "nt" else ""


def _gcc():
    gcc = shutil.which("gcc") or next((p for p in [r"C:\msys64\mingw64\bin\gcc.exe"] if os.path.exists(p)), None)
    if gcc is None:
        raise ToolUnavailable("gcc not found (MinGW-w64 at C:\\msys64\\mingw64\\bin, or build-essential)")
    env = dict(os.environ, PATH=str(Path(gcc).parent) + os.pathsep + os.environ.get("PATH", ""))
    return gcc, env


def build():
    """Compile labtrack.c (cached). Returns (exe path, env with the compiler's bin on PATH)."""
    gcc, env = _gcc()
    exe = CACHE / f"labtrack{EXE}"
    src = SRC / "labtrack.c"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        CACHE.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run([gcc, "-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", str(src),
                             "-o", str(exe), "-lsqlite3", "-lm"], capture_output=True, text=True, env=env)
        if cp.returncode:
            raise RuntimeError("gcc failed (is SQLite installed? MinGW: pacman -S mingw-w64-x86_64-sqlite3):\n"
                               + cp.stderr[-2000:])
    return exe, env


def run(db, *args):
    """Run labtrack on ``db``. Returns (exit code, stdout, stderr)."""
    exe, env = build()
    cp = subprocess.run([str(exe), str(db), *map(str, args)], capture_output=True, text=True, env=env)
    return cp.returncode, cp.stdout, cp.stderr


def fresh_db(path):
    """Create a new database from schema.sql + seed.sql (replacing any old file)."""
    path = Path(path)
    path.unlink(missing_ok=True)
    for f in ("schema.sql", "seed.sql"):
        rc, out, err = run(path, "exec", SRC / f)
        if rc:
            raise RuntimeError(err)
    return path


def report(db, date):
    rc, out, err = run(db, "report", date)
    if rc:
        raise RuntimeError(err)
    return out


_EYE = re.compile(r"^(.*?)\s+on (.*?)\s+E=(\S+) W/cm2\s+need OD\s+(\S+)\s+have (\S+)\s+(.*)$")


def eyewear_rows(report_text):
    """Parse the eyewear section into dicts (label, laser, E, need_od or None, od, verdict)."""
    rows, on = [], False
    for line in report_text.splitlines():
        if line.startswith("== Eyewear"):
            on = True
            continue
        m = _EYE.match(line) if on else None
        if m:
            label, laser, E, need, od, verdict = m.groups()
            rows.append({"label": label, "laser": laser, "E_W_cm2": float(E),
                         "need_od": None if need == "n/a" else float(need), "od": float(od), "verdict": verdict})
    return rows
