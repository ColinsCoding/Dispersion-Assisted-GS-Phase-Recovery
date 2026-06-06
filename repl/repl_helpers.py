"""
repl_helpers.py — shared utilities for all repl notebooks.

Import with:
    from repl_helpers import hdr, show, chk

show(expr)         → LaTeX in Jupyter, pretty ASCII in terminal
chk(val, ref, ...) → PASS/FAIL verification with relative or absolute tolerance
hdr(s)             → section header bar
"""
import sys
import sympy as sp

# ── LaTeX display ──────────────────────────────────────────────────────────────
try:
    from IPython.display import display as _ipy_display, Latex as _Latex
    _IN_NOTEBOOK = True
except ImportError:
    _IN_NOTEBOOK = False

def show(expr, label=None):
    """
    Display a SymPy expression.
    - In Jupyter: renders as LaTeX via init_printing / MathJax.
    - In terminal: falls back to sp.pretty (unicode art).
    """
    if label:
        print(f'  {label}:')
    if _IN_NOTEBOOK:
        _ipy_display(expr)
    else:
        print('  ' + sp.pretty(expr, use_unicode=True))

def show_eq(lhs_str, expr, label=None):
    """Show  'lhs = expr'  as LaTeX."""
    if label:
        print(f'  {label}:')
    combined = sp.Eq(sp.Symbol(lhs_str), expr) if isinstance(lhs_str, str) else sp.Eq(lhs_str, expr)
    show(combined)

# ── Section header ─────────────────────────────────────────────────────────────
def hdr(s, width=64):
    bar = '─' * width
    print(f'\n{bar}\n  {s}\n{bar}')

# ── Numerical check ────────────────────────────────────────────────────────────
def chk(val, ref, label, tol=1e-9, absolute=False):
    """
    PASS/FAIL check.
    absolute=True or ref==0: uses absolute error |val-ref|.
    Otherwise: relative error |val-ref|/|ref|.
    """
    try:
        v, r = float(val), float(ref)
    except Exception:
        print(f'  [FAIL]  {label}  (could not convert to float)')
        return
    err = abs(v - r) if (absolute or r == 0) else abs(v - r) / (abs(r) + 1e-30)
    s = 'PASS' if err < tol else 'FAIL'
    print(f'  [{s}]  {label}  got={v:.8g}  ref={r:.8g}')

# ── Stdout UTF-8 fix for Windows terminals ────────────────────────────────────
def fix_stdout():
    """Call at top of script to prevent UnicodeEncodeError on cp1252 terminals."""
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                      encoding='utf-8', errors='replace')
