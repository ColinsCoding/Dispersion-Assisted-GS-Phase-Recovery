"""Build and drive the ANSI C (C89) timed free-response bot in dgs/c/frqbot from Python.

    frqbot.c   question bank (ANSI C, laser safety, AP Bio/Chem/Psych/Econ, CompTIA),
               15-minute default limit per question, numeric + keyword grading

Interactive use is in a terminal (``frqbot --subject econ``); this wrapper feeds answers on stdin
so the bot can be tested and used from a notebook.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dgs.eventsim import ToolUnavailable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dgs" / "c" / "frqbot" / "frqbot.c"
CACHE = Path(tempfile.gettempdir()) / "dgs_frqbot"
EXE = ".exe" if os.name == "nt" else ""
SUBJECTS = ("c", "laser", "bio", "chem", "psych", "econ", "comptia")

# A model answer per question, keyed by a phrase from its prompt (used by tests and the notebook).
MODEL_ANSWERS = {
    "a / b * b + a % b": "7",
    "if (x = 0)": "= assigns zero to x, and 0 is false",
    "OD 4": "1e-4",
    "minimum OD": "4.30",
    "Hardy-Weinberg": "0.48",
    "enzyme": "it denatures: weak bonds break and the active site loses its shape",
    "pH of": "3.0",
    "moles": "2.00",
    "Pavlov": "the conditioned stimulus",
    "negative reinforcement": "removing an unpleasant stimulus, which increases the behavior",
    "equilibrium price.": "20",
    "price ceiling": "20",
    "Demand increases": "price and quantity both rise",
    "/26": "62",
    "HTTPS": "443",
    "CIA triad": "confidentiality, integrity, availability",
}


def build():
    gcc = shutil.which("gcc") or next((p for p in [r"C:\msys64\mingw64\bin\gcc.exe"] if os.path.exists(p)), None)
    if gcc is None:
        raise ToolUnavailable("gcc not found (MinGW-w64 at C:\\msys64\\mingw64\\bin, or build-essential)")
    env = dict(os.environ, PATH=str(Path(gcc).parent) + os.pathsep + os.environ.get("PATH", ""))
    exe = CACHE / f"frqbot{EXE}"
    if not exe.exists() or exe.stat().st_mtime < SRC.stat().st_mtime:
        CACHE.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run([gcc, "-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", str(SRC), "-o", str(exe), "-lm"],
                            capture_output=True, text=True, env=env)
        if cp.returncode:
            raise RuntimeError("gcc failed:\n" + cp.stderr[-2000:])
    return exe, env


def bank():
    """[(subject, prompt)] from ``frqbot --list``."""
    exe, env = build()
    out = subprocess.run([str(exe), "--list"], capture_output=True, text=True, env=env, check=True).stdout
    return [tuple(line.split(None, 1)) for line in out.splitlines()]


def model_answer(prompt):
    for key, ans in MODEL_ANSWERS.items():
        if key in prompt:
            return ans
    raise KeyError(prompt)


_Q = re.compile(r"^\[(\d+)/(\d+)\] \((\w+)\) (.*)$")


def _parse(out):
    results = []
    for line in out.splitlines():
        m = _Q.match(line)
        if m:
            results.append({"subject": m.group(3), "prompt": m.group(4)})
        elif line.startswith("> ") and results:
            verdict = line[2:]
            results[-1]["verdict"] = "TIME" if verdict.startswith("TIME:") else verdict.split(" (")[0]
            results[-1]["detail"] = verdict
        elif line.startswith("why: ") and results:
            results[-1]["why"] = line[5:]
    score = re.search(r"score=(\d+)/(\d+)", out)
    return results, (int(score.group(1)), int(score.group(2))) if score else (0, 0)


def drill(answer_fn, subject=None, n=None, limit=900, seed=1, delay_s=0.0):
    """Run the bot, answering each question with ``answer_fn(prompt)``.

    Returns (results, (score, asked)). ``delay_s`` waits before each answer (to test the timer).
    """
    if subject is not None and subject not in SUBJECTS:
        raise ValueError(f"subject must be one of {SUBJECTS}")
    exe, env = build()
    args = [str(exe), "--limit", str(limit), "--seed", str(seed)]
    if subject:
        args += ["--subject", subject]
    if n:
        args += ["--n", str(n)]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, bufsize=1)
    out, buf = [], ""
    while True:
        ch = p.stdout.read(1)
        if not ch:
            break
        buf += ch
        if buf.endswith("\n> "):                      # the answer prompt (not "->" in an explanation)
            prompt = [l for l in buf.splitlines() if _Q.match(l)][-1]
            if delay_s:
                time.sleep(delay_s)
            p.stdin.write(answer_fn(_Q.match(prompt).group(4)) + "\n")
            p.stdin.flush()
            out.append(buf)
            buf = ""
    out.append(buf)
    p.wait()
    if p.returncode:
        raise ValueError(p.stderr.read().strip())
    return _parse("".join(out))
