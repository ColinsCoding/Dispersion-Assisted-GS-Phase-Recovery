"""
py_to_notebook.py
=================
Convert any _repl_*.py file to a Jupyter notebook (.ipynb).

Usage:
    py -3.12 repl/py_to_notebook.py repl/_repl_calc_qec_bessel.py
    py -3.12 repl/py_to_notebook.py --all          # convert all _repl_*.py

Strategy:
  1. Parse the .py source into logical sections using the S1/S2/... markers
     and triple-quoted print blocks.
  2. Each section header becomes a Markdown cell.
  3. Each print(triple-quoted) block becomes a Markdown cell (formatted text).
  4. Executable code (imports, SymPy, numpy, matplotlib) becomes Code cells.
  5. Prepend: %matplotlib inline + sp.init_printing() cell.
  6. Write .ipynb via nbformat.

Cell splitting heuristic:
  - Lines starting with print triple-quote -> text cell candidate
  - Comment blocks (## SECTION ...) -> markdown header cells
  - Everything else -> code cell
  - Consecutive code lines are merged into one code cell per section.
"""

import nbformat
import re
import sys
import os
from glob import glob


def parse_py_to_cells(py_source: str) -> list[dict]:
    """
    Parse Python source into a list of cells:
      {'type': 'code'|'markdown', 'source': str}
    """
    cells = []

    # Always prepend setup cell
    setup = (
        "import sympy as sp\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n"
        "sp.init_printing(use_latex='mathjax')  # LaTeX in Jupyter\n"
        "from IPython.display import display\n"
        "import os\n"
        "# __file__ is undefined in Jupyter -- fall back to cwd\n"
        "try:\n"
        "    _REPL_DIR = os.path.dirname(os.path.abspath(__file__))\n"
        "except NameError:\n"
        "    _REPL_DIR = os.getcwd()\n"
        "# Patch: redirect any OUT = os.path.join(os.path.dirname(__file__), ...)\n"
        "# by replacing __file__ references below with _REPL_DIR"
    )
    cells.append({"type": "code", "source": setup})

    lines = py_source.split("\n")
    i = 0
    current_code_lines = []
    current_section_title = None

    def flush_code():
        nonlocal current_code_lines
        src = "\n".join(current_code_lines).strip()
        if src:
            cells.append({"type": "code", "source": src})
        current_code_lines = []

    def flush_section(title):
        nonlocal current_section_title
        flush_code()
        if title:
            md = f"## {title}"
            cells.append({"type": "markdown", "source": md})
        current_section_title = title

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- Section headers from SEP + SECTION N: ... pattern ----
        if (stripped.startswith("print(SEP)") or
                stripped == 'print(SEP)'):
            # Look ahead for SECTION line
            if i+1 < len(lines) and "SECTION" in lines[i+1]:
                sec_match = re.search(r'print\(["\'](.+?)["\']\)', lines[i+1])
                if sec_match:
                    title = sec_match.group(1).strip()
                    flush_section(title)
                    i += 2
                    # Skip the trailing print(SEP) if present
                    if i < len(lines) and lines[i].strip() == "print(SEP)":
                        i += 1
                    continue
            i += 1
            continue

        # ---- Module docstring -> markdown ----
        if stripped.startswith('"""') and i == 0:
            doc_lines = []
            i += 1
            while i < len(lines):
                if lines[i].strip() == '"""':
                    i += 1
                    break
                doc_lines.append(lines[i])
                i += 1
            flush_code()
            md = "\n".join(doc_lines).strip()
            if md:
                cells.append({"type": "markdown", "source": md})
            continue

        # ---- print(""" ... """) -> markdown text cell ----
        if re.match(r'\s*print\((?:f?""")', line):
            flush_code()
            text_lines = []
            # Find the closing triple-quote
            # The opening line might have content after print("""
            content_after = re.sub(r'\s*print\(f?"""', "", line)
            if content_after.rstrip().endswith('"""'):
                # Single-line: print("""text""")
                content = content_after.rstrip()[:-3].strip()
                if content:
                    text_lines.append(content)
            else:
                if content_after.strip():
                    text_lines.append(content_after.rstrip())
                i += 1
                while i < len(lines):
                    l2 = lines[i]
                    if '"""' in l2:
                        # End of triple-quote block
                        before = l2[:l2.index('"""')]
                        if before.strip():
                            text_lines.append(before.rstrip())
                        i += 1
                        break
                    text_lines.append(l2.rstrip())
                    i += 1

            md_text = "\n".join(text_lines)
            # Remove leading 4-space indent (artifact of Python indentation)
            md_text = re.sub(r"^    ", "", md_text, flags=re.MULTILINE)
            if md_text.strip():
                cells.append({"type": "markdown", "source": "```\n" + md_text + "\n```"})
            continue

        # ---- print(f"...") with section-like content -> keep as code ----
        # ---- Skip bare print(SEP) lines ----
        if stripped in ('print(SEP)', 'print(f"\\n{SEP}")',
                        'print(f"\\n{SEP}")', 'print(SEP)'):
            i += 1
            continue

        # ---- "BUILDING FIGURE" print -> start new section ----
        if "BUILDING FIGURE" in stripped:
            flush_section("Figure")
            i += 1
            continue

        # ---- plt.savefig / plt.close -> keep in code but add display ----
        if "plt.savefig" in stripped:
            current_code_lines.append(line)
            current_code_lines.append("plt.show()")
            i += 1
            continue

        if stripped == "plt.close()":
            # skip; show() already added
            i += 1
            continue

        # ---- Everything else: code cell accumulator ----
        current_code_lines.append(line)
        i += 1

    flush_code()
    return cells


def _patch_file_refs(source: str) -> str:
    """Replace os.path.dirname(__file__) with _REPL_DIR for Jupyter compat."""
    import re
    source = re.sub(r'os\.path\.dirname\(__file__\)', '_REPL_DIR', source)
    source = re.sub(r'os\.path\.abspath\(__file__\)', '_REPL_DIR', source)
    return source


def cells_to_notebook(cells: list[dict]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb_cells = []
    for cell in cells:
        src = cell["source"]
        if not src.strip():
            continue
        if cell["type"] == "markdown":
            nb_cells.append(nbformat.v4.new_markdown_cell(src))
        else:
            src = _patch_file_refs(src)
            nb_cells.append(nbformat.v4.new_code_cell(src))
    nb.cells = nb_cells
    # Kernel metadata
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {
        "name": "python",
        "version": "3.12",
    }
    return nb


def convert(py_path: str) -> str:
    """Convert a .py file to .ipynb. Returns output path."""
    with open(py_path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    cells = parse_py_to_cells(source)
    nb    = cells_to_notebook(cells)

    # Output path: same dir, same name, .ipynb extension
    base    = os.path.splitext(py_path)[0]
    out_path = base + ".ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_md   = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(f"  {os.path.basename(py_path):45s} -> {os.path.basename(out_path)}"
          f"  [{n_code} code, {n_md} markdown cells]")
    return out_path


def main():
    args = sys.argv[1:]

    if not args or "--help" in args:
        print(__doc__)
        print("Usage: py py_to_notebook.py [--all] [file1.py file2.py ...]")
        return

    repl_dir = os.path.dirname(os.path.abspath(__file__))

    if "--all" in args:
        py_files = sorted(glob(os.path.join(repl_dir, "_repl_*.py")))
        if not py_files:
            print("No _repl_*.py files found.")
            return
        print(f"Converting {len(py_files)} files:\n")
        converted = []
        for pf in py_files:
            try:
                out = convert(pf)
                converted.append(out)
            except Exception as e:
                print(f"  ERROR converting {pf}: {e}")
        print(f"\nDone. {len(converted)} notebooks written.")
    else:
        for arg in args:
            pf = arg if os.path.isabs(arg) else os.path.join(os.getcwd(), arg)
            if not os.path.exists(pf):
                print(f"File not found: {pf}")
                continue
            convert(pf)


if __name__ == "__main__":
    main()
