"""
Lean 4 cells inside a Jupyter notebook, next to the SymPy cells they back up.

Usage in a notebook
-------------------
    %load_ext dgs.lean_magic

    %%lean header
    import Mathlib.Tactic.Ring
    namespace Demo

    %%lean energy_gap
    theorem gap (E1 : Real) : E1 * 6 - E1 * 3 = 3 * E1 := by ring

    %lean_check            # compiles every collected cell in ONE `lake env lean` run

Why one run: loading Mathlib takes a minute or two, so compiling each cell separately would
be painfully slow. The cells are joined into one file in order; any error or warning is mapped
back to the cell (and the line inside it) that caused it.

Honesty rules built in:
* A theorem counts as VERIFIED only if `lake env lean` exits 0 with no errors **and** no
  `sorry` (Lean accepts `sorry` with only a warning, so exit 0 alone is not proof).
* If Lean or a Mathlib project isn't available, the status is UNVERIFIED and says why.

The Lake project is $PHYS106_LEAN_PROJECT, or ~/lean/NavierStokesAndEuler by default (it must
list mathlib in lake-manifest.json).
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

MSG_RE = re.compile(r"^(?P<path>.*?):(?P<line>\d+):(?P<col>\d+):\s*(?P<sev>error|warning|info)(?:\([^)]*\))?:\s*(?P<text>.*)$")


def find_lake():
    exe = "lake.exe" if os.name == "nt" else "lake"
    return shutil.which("lake") or next((str(p) for p in [Path.home() / ".elan" / "bin" / exe] if p.exists()), None)


def find_project():
    proj = Path(os.environ.get("PHYS106_LEAN_PROJECT", Path.home() / "lean" / "NavierStokesAndEuler"))
    manifest = proj / "lake-manifest.json"
    if manifest.exists() and "mathlib" in manifest.read_text(encoding="utf-8", errors="ignore"):
        return proj
    return None


@dataclass
class Message:
    cell: str
    line: int          # line inside the cell (1-based)
    severity: str
    text: str


@dataclass
class CheckResult:
    status: str                    # VERIFIED / FAILED / UNVERIFIED
    returncode: object = None
    seconds: float = 0.0
    messages: list = field(default_factory=list)
    theorems: list = field(default_factory=list)
    reason: str = ""

    @property
    def verified(self):
        return self.status == "VERIFIED"


class LeanSession:
    """Ordered collection of named Lean snippets that compile as one file."""

    def __init__(self):
        self.cells = {}            # name -> source, insertion-ordered; re-running a cell replaces it
        self.last = None

    def add(self, name, source):
        if not name or not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
            raise ValueError(f"cell name must be letters, digits, _ . - (got {name!r})")
        self.cells[name] = source.rstrip() + "\n"

    def reset(self):
        self.cells.clear()
        self.last = None

    def assemble(self):
        """Joined source and a line map [(first_line, last_line, cell_name)] (1-based, inclusive)."""
        text, spans, line = "", [], 1
        for name, src in self.cells.items():
            n = src.count("\n")
            spans.append((line, line + n - 1, name))
            text += src
            line += n
        return text, spans

    @staticmethod
    def locate(spans, line):
        for lo, hi, name in spans:
            if lo <= line <= hi:
                return name, line - lo + 1
        return "(outside cells)", line

    def parse(self, output, spans):
        """Turn Lean's `file:line:col: severity: text` lines into Messages mapped to cells."""
        msgs, current = [], None
        for raw in output.splitlines():
            m = MSG_RE.match(raw.strip())
            if m:
                cell, ln = self.locate(spans, int(m["line"]))
                current = Message(cell, ln, m["sev"], m["text"])
                msgs.append(current)
            elif current is not None and raw.strip():
                current.text += "\n" + raw.rstrip()
        return msgs

    def theorems(self):
        text, _ = self.assemble()
        return re.findall(r"^\s*(?:theorem|lemma)\s+(\S+)", text, flags=re.M)

    def check(self, timeout=1800):
        """Compile every cell in one `lake env lean` run and classify the result."""
        thms = self.theorems()
        lake, proj = find_lake(), find_project()
        if not self.cells:
            self.last = CheckResult("UNVERIFIED", reason="no %%lean cells collected")
            return self.last
        if lake is None or proj is None:
            why = "lake not found" if lake is None else "no Lake project with Mathlib (set PHYS106_LEAN_PROJECT)"
            self.last = CheckResult("UNVERIFIED", theorems=thms, reason=why)
            return self.last
        text, spans = self.assemble()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "NotebookCells.lean"
            f.write_text(text, encoding="utf-8")
            t0 = time.perf_counter()
            r = subprocess.run([lake, "env", "lean", str(f)], cwd=proj, capture_output=True, text=True, timeout=timeout)
            secs = time.perf_counter() - t0
        msgs = self.parse(r.stdout + r.stderr, spans)
        errors = [m for m in msgs if m.severity == "error"]
        sorries = [m for m in msgs if "sorry" in m.text]
        if r.returncode == 0 and not errors and not sorries:
            status, reason = "VERIFIED", "lake env lean exit 0, no errors, no sorry"
        else:
            status = "FAILED"
            reason = f"exit {r.returncode}, {len(errors)} error(s), {len(sorries)} sorry"
        self.last = CheckResult(status, r.returncode, secs, msgs, thms, reason)
        return self.last


SESSION = LeanSession()


def _render(src, name):
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"**Lean cell `{name}`** (collected; compiled by `%lean_check`)\n```lean\n{src.rstrip()}\n```"))
    except ImportError:
        print(f"[lean cell {name}]\n{src}")


def lean_cell(line, cell):
    """%%lean NAME : collect a Lean snippet under NAME (re-running replaces it)."""
    name = line.strip() or f"cell{len(SESSION.cells) + 1}"
    SESSION.add(name, cell)
    _render(cell, name)


def lean_check(line=""):
    """%lean_check [timeout_s] : compile all collected cells once and report per-cell messages."""
    timeout = int(line.strip()) if line.strip() else 1800
    res = SESSION.check(timeout=timeout)
    print(f"Lean status: {res.status}  ({res.reason})" + (f", {res.seconds:.0f} s" if res.seconds else ""))
    for m in res.messages:
        print(f"  [{m.severity}] cell '{m.cell}', line {m.line}: {m.text}")
    return res


def lean_reset(line=""):
    """%lean_reset : forget all collected cells."""
    SESSION.reset()
    print("Lean cells cleared.")


def load_ipython_extension(ip):
    ip.register_magic_function(lean_cell, magic_kind="cell", magic_name="lean")
    ip.register_magic_function(lean_check, magic_kind="line", magic_name="lean_check")
    ip.register_magic_function(lean_reset, magic_kind="line", magic_name="lean_reset")
