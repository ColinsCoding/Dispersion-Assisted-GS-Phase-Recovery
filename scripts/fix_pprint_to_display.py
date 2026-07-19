"""One-time fix across this session's SymPy notebooks: sp.pprint() always
prints ASCII/unicode text to stdout, completely bypassing the rich LaTeX
rendering that sp.init_printing() sets up -- only display() (or a bare
trailing expression) actually engages it. Replaces every sp.pprint(X) with
display(X), and ensures `from IPython.display import display` is present.
"""
import pathlib
import re
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "matter_waves_chapter5_sympy_torch.ipynb",
    "jalali_lab_calculus_problems.ipynb",
    "coppinger1999_sympy.ipynb",
    "griffiths_notation_glossary_sympy.ipynb",
]

PPRINT_RE = re.compile(r"\bsp\.pprint\(")

for name in NOTEBOOKS:
    path = ROOT / "notebooks" / name
    nb = nbf.read(str(path), as_version=4)
    n_fixed = 0
    has_display_import = False

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "from IPython.display import display" in cell.source:
            has_display_import = True
        if PPRINT_RE.search(cell.source):
            n_fixed += len(PPRINT_RE.findall(cell.source))
            cell.source = PPRINT_RE.sub("display(", cell.source)

    if not has_display_import:
        for cell in nb.cells:
            if cell.cell_type == "code" and "init_printing" in cell.source:
                cell.source = "from IPython.display import display\n" + cell.source
                break

    nbf.write(nb, str(path))
    print(f"{name}: fixed {n_fixed} sp.pprint -> display, "
          f"display import {'already present' if has_display_import else 'added'}")
