"""Theory-to-code generator: SymPy expression -> optimized C / Fortran / JS.

Purpose:
    Finalize a symbolic expression and emit portable source. Generated code lives
    separately from handwritten code (write into `src/` and `include/`). Common-
    subexpression elimination runs before C generation so shared terms compute once.

References:
    - SymPy code-printers (`ccode`, `fcode`, `jscode`), `cse`.
Assumptions:
    - Scalar, real-valued expression of the ordered `SymbolicExpression.symbols`.
Limitations:
    - Single scalar return; extend to vector outputs via `codegen` if needed.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sympy as sp

from physics.symbolic import SymbolicExpression

__all__ = [
    "CodegenResult",
    "generate_c",
    "generate_fortran",
    "generate_js",
    "write_c",
    "compile_and_run_c",
]


@dataclass(frozen=True)
class CodegenResult:
    """A generated translation unit: source, header, and a runnable example."""

    name: str
    language: str
    source: str
    header: str
    example: str


def _signature_args(symbols: Sequence[sp.Symbol], ctype: str = "double") -> str:
    return ", ".join(f"{ctype} {s.name}" for s in symbols)


def generate_c(sym: SymbolicExpression) -> CodegenResult:
    """Generate an optimized C function (with CSE) plus header and example `main`."""
    replacements, reduced = sym.cse()
    body = [f"    const double {s} = {sp.ccode(e)};" for s, e in replacements]
    body.append(f"    return {sp.ccode(reduced)};")
    args = _signature_args(sym.symbols)
    signature = f"double {sym.name}({args})"

    source = "#include <math.h>\n" f'#include "{sym.name}.h"\n\n' + signature + " {\n" + "\n".join(body) + "\n}\n"

    guard = f"{sym.name.upper()}_H"
    header = (
        f"#ifndef {guard}\n#define {guard}\n\n"
        f"/* Auto-generated from SymPy. Do not edit by hand. */\n"
        f"{signature};\n\n#endif /* {guard} */\n"
    )

    call_args = ", ".join("1.0" for _ in sym.symbols)
    example = (
        "#include <stdio.h>\n" f'#include "{sym.name}.h"\n\n'
        "int main(void) {\n"
        f"    printf(\"%.12f\\n\", {sym.name}({call_args}));\n"
        "    return 0;\n}\n"
    )
    return CodegenResult(sym.name, "c", source, header, example)


def generate_fortran(sym: SymbolicExpression) -> str:
    """Generate a Fortran function body via SymPy `fcode`."""
    return sp.fcode(sym.expr, assign_to=sym.name, standard=95)


def generate_js(sym: SymbolicExpression) -> str:
    """Generate a JavaScript expression via SymPy `jscode`."""
    args = ", ".join(s.name for s in sym.symbols)
    return f"function {sym.name}({args}) {{\n  return {sp.jscode(sym.expr)};\n}}\n"


def write_c(result: CodegenResult, src_dir: str | Path, include_dir: str | Path) -> dict[str, Path]:
    """Write the C source/header/example to disk; return the created paths."""
    src_dir, include_dir = Path(src_dir), Path(include_dir)
    src_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": src_dir / f"{result.name}.c",
        "header": include_dir / f"{result.name}.h",
        "example": src_dir / f"{result.name}_example.c",
    }
    paths["source"].write_text(result.source, encoding="utf-8")
    paths["header"].write_text(result.header, encoding="utf-8")
    paths["example"].write_text(result.example, encoding="utf-8")
    return paths


def compile_and_run_c(sym: SymbolicExpression, args: Sequence[float], cc: str | None = None) -> float:
    """Compile the generated C for `sym`, run it with `args`, and return the result.

    Raises RuntimeError if no C compiler is available.
    """
    cc = cc or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError("No C compiler (gcc/cc) found on PATH.")

    result = generate_c(sym)
    call = ", ".join(repr(float(a)) for a in args)
    main = (
        "#include <math.h>\n#include <stdio.h>\n\n"
        + result.source.replace(f'#include "{sym.name}.h"\n', "")
        + "\nint main(void) {\n"
        f"    printf(\"%.12f\\n\", {sym.name}({call}));\n"
        "    return 0;\n}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        c_file = tmp_path / f"{sym.name}.c"
        exe = tmp_path / f"{sym.name}.exe"
        c_file.write_text(main, encoding="utf-8")
        subprocess.run([cc, "-O2", str(c_file), "-o", str(exe), "-lm"], check=True, capture_output=True)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    return float(out.stdout.strip())
