"""Code-generation stage: SymPy expression -> C / Fortran / JS, with CSE."""
from c_codegen.generator import (
    CodegenResult,
    compile_and_run_c,
    generate_c,
    generate_fortran,
    generate_js,
    write_c,
)

__all__ = [
    "CodegenResult",
    "generate_c",
    "generate_fortran",
    "generate_js",
    "write_c",
    "compile_and_run_c",
]
