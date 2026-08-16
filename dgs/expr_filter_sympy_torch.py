"""expr_filter_sympy_torch.py -- a stdin/stdout SymPy+Torch expression
filter, meant to be piped from a text editor's "run selection through an
external command" feature (Vim `:'<,'>!`, Emacs shell-command-on-region,
VS Code tasks, etc.): select some math expressions, pipe them through this
script, get simplified forms + a cross-checked derivative back.

INPUT FORMAT (one item per line, read from stdin):
  VAR = value        -- sets a numeric substitution value for a symbol
  any other line     -- a SymPy-parseable expression to evaluate

Assignment lines apply to every expression line that FOLLOWS them (not
lines before), so `x = 2` then `x**2 + 1` uses x=2. Any free symbol with no
prior assignment defaults to 1.0 (documented, not silently arbitrary).

FOR EACH EXPRESSION, THIS PRINTS:
  - the SymPy-simplified form
  - its numeric value at the current substitution values
  - for each free symbol: the partial derivative, computed TWO independent
    ways -- SymPy's symbolic sp.diff (evaluated numerically) and PyTorch's
    autograd (via sp.lambdify(..., modules='torch') + .backward()) -- and
    whether they agree. This is the actual point of using both libraries
    here: SymPy and Torch computing the same derivative by genuinely
    different mechanisms (symbolic differentiation vs. reverse-mode
    autodiff) is a real correctness cross-check, not decoration.

Requires torch (py 3.12 here, matching this repo's existing convention).

Run:
  py -3.12 -m dgs.expr_filter_sympy_torch < input.txt
  echo "x**2 + sin(x)" | py -3.12 -m dgs.expr_filter_sympy_torch
"""

from __future__ import annotations
import sys
import sympy as sp
import torch
from typing import Dict, List, Tuple, Optional

DEFAULT_VALUE = 1.0


# ── 1. Parsing stdin lines into assignments + expressions ───────────────────

def parse_input_lines(lines: List[str]) -> List[Tuple[str, Optional[Tuple[str, float]]]]:
    """Classify each non-blank line as either an assignment
    ('VAR', value) or a plain expression string. Returns a list of
    (raw_line, assignment_or_None) preserving order, so main() can apply
    assignments sequentially as it encounters them."""
    out = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and not any(op in line for op in ("==", "<=", ">=")):
            name, _, value_str = line.partition("=")
            name = name.strip()
            if name.isidentifier():
                try:
                    value = float(value_str.strip())
                    out.append((line, (name, value)))
                    continue
                except ValueError:
                    pass  # not a numeric assignment -- fall through, treat as expression
        out.append((line, None))
    return out


# ── 2. Evaluating one expression against the current assignments ────────────

def evaluate_expression(expr_str: str, assignments: Dict[str, float]) -> Dict:
    """Parse expr_str with SymPy, simplify it, evaluate it numerically, and
    compute each free symbol's partial derivative two independent ways
    (SymPy symbolic diff vs. Torch autograd), cross-checked against each
    other. Any symbol not in `assignments` defaults to DEFAULT_VALUE.

    Returns a dict; on a parse error, returns {"error": str(e)} instead of
    raising -- a text-editor filter should report a bad line, not crash
    the whole pipe.
    """
    try:
        expr = sp.sympify(expr_str)
    except (sp.SympifyError, TypeError, SyntaxError) as e:
        return {"expr_str": expr_str, "error": str(e)}

    free_syms = sorted(expr.free_symbols, key=lambda s: s.name)
    values = {s.name: assignments.get(s.name, DEFAULT_VALUE) for s in free_syms}

    simplified = sp.simplify(expr)
    numeric_value = float(simplified.subs({sp.Symbol(n): v for n, v in values.items()})) \
        if free_syms else float(simplified)

    sympy_grad: Dict[str, float] = {}
    torch_grad: Dict[str, float] = {}
    grad_match: Dict[str, bool] = {}

    if free_syms:
        torch_inputs = {s.name: torch.tensor(values[s.name], dtype=torch.float64, requires_grad=True)
                         for s in free_syms}
        f = sp.lambdify(free_syms, expr, modules="torch")
        torch_val = f(*[torch_inputs[s.name] for s in free_syms])
        torch_val.backward()
        for s in free_syms:
            torch_grad[s.name] = float(torch_inputs[s.name].grad.item())
            d_sym = sp.diff(expr, s)
            sympy_grad[s.name] = float(d_sym.subs({sp.Symbol(n): v for n, v in values.items()}))
            grad_match[s.name] = abs(sympy_grad[s.name] - torch_grad[s.name]) < 1e-6

    return {
        "expr_str": expr_str, "simplified": str(simplified), "value": numeric_value,
        "free_symbols": [s.name for s in free_syms], "substitution_values": values,
        "sympy_grad": sympy_grad, "torch_grad": torch_grad, "grad_match": grad_match,
    }


# ── 3. Formatting a result for stdout ────────────────────────────────────────

def format_result(result: Dict) -> str:
    """Render one evaluate_expression() result as a few human-readable
    lines, suitable for a text editor to display inline."""
    if "error" in result:
        return f">>> {result['expr_str']}\n    ERROR: {result['error']}"
    lines = [f">>> {result['expr_str']}",
             f"    simplified: {result['simplified']}",
             f"    value:      {result['value']:.6g}  "
             f"(at {', '.join(f'{k}={v:g}' for k, v in result['substitution_values'].items()) or 'no free symbols'})"]
    for name in result["free_symbols"]:
        sg, tg, match = result["sympy_grad"][name], result["torch_grad"][name], result["grad_match"][name]
        lines.append(f"    d/d{name}:     sympy={sg:.6g}  torch={tg:.6g}  "
                      f"{'match' if match else 'MISMATCH'}")
    return "\n".join(lines)


# ── 4. The stdin/stdout filter itself ────────────────────────────────────────

def run_filter(input_lines: List[str]) -> str:
    """Process a full block of input lines (as main() would receive from
    stdin) and return the full formatted output block -- pure function,
    testable without actual stdin/stdout."""
    assignments: Dict[str, float] = {}
    blocks = []
    for line, assignment in parse_input_lines(input_lines):
        if assignment is not None:
            name, value = assignment
            assignments[name] = value
            blocks.append(f">>> {name} = {value:g}   (assignment noted)")
        else:
            result = evaluate_expression(line, assignments)
            blocks.append(format_result(result))
    return "\n".join(blocks)


def main():
    text = sys.stdin.read()
    output = run_filter(text.splitlines())
    print(output)


if __name__ == "__main__":
    main()
