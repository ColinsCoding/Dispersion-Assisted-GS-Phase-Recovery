"""Auto-generate Markdown API docs from module docstrings.

Walks the package folders, extracts each module's docstring and public callables, and
writes `docs/api.md`.

Usage:
    python scripts/gen_docs.py
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGES = ["physics", "optics", "photonics", "feature_extraction", "ml", "c_codegen", "visualization"]


def _document_module(name: str) -> list[str]:
    module = importlib.import_module(name)
    lines = [f"### `{name}`", ""]
    doc = inspect.getdoc(module)
    if doc:
        lines += [doc.split("\n\n")[0], ""]
    public = getattr(module, "__all__", [])
    for attr in public:
        obj = getattr(module, attr, None)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            sig = ""
            try:
                sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                pass
            summary = (inspect.getdoc(obj) or "").split("\n")[0]
            lines.append(f"- **`{attr}{sig}`** -- {summary}")
    lines.append("")
    return lines


def main() -> Path:
    """Write docs/api.md and return its path."""
    out_lines = ["# API reference", "", "Auto-generated from module docstrings.", ""]
    for pkg in PACKAGES:
        package = importlib.import_module(pkg)
        for _, mod_name, _ in pkgutil.iter_modules(package.__path__, prefix=f"{pkg}."):
            out_lines += _document_module(mod_name)
    out_path = ROOT / "docs" / "api.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
