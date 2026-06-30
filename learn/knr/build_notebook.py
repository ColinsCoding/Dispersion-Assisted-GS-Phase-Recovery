"""Build knr_exercises.ipynb — K&R Chapter 1 exercises 1-20 through 1-24.

Each exercise gets: problem statement cell, source code cell, live demo cell
showing the program running on example input via subprocess.

Run with:  py -3.13 build_notebook.py
           py -3.13 -m jupyter nbconvert --to notebook --execute knr_exercises.ipynb
"""

import json
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------

def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip().splitlines(True)}


# ---------------------------------------------------------------------------
# Exercise definitions
# ---------------------------------------------------------------------------

EXERCISES = [
    {
        "num": "1-20",
        "title": "detab — expand tabs to spaces",
        "statement": """**Exercise 1-20.** Write a program `detab` that replaces tabs in the
input with the proper number of blanks to space to the next tab stop.
Assume a fixed set of tab stops, say every *n* columns.
Should *n* be a variable or a symbolic parameter?

**Answer:** `n` is a `#define TABSTOP 8` — tab stops are a compile-time
convention (like ASCII), not a runtime input.
The formula: `spaces = TABSTOP - (col % TABSTOP)`.""",
        "src_file": "detab.c",
        "demo": r'''import subprocess, os
here = r"{HERE}"
result = subprocess.run(
    [os.path.join(here, "detab.exe")],
    input="abc\thello\tworld\n1234567\t8\n",
    capture_output=True, text=True
)
print("Input (→ = tab):")
print("  abc→hello→world")
print("  1234567→8")
print()
print("Output (tabs expanded to TABSTOP=8):")
for line in result.stdout.splitlines():
    print(f"  [{line}]  (len={len(line)})")
'''.replace("{HERE}", HERE.replace("\\", "\\\\")),
    },
    {
        "num": "1-21",
        "title": "entab — replace blanks with minimum tabs",
        "statement": """**Exercise 1-21.** Write a program `entab` that replaces strings of
blanks by the minimum number of tabs and blanks to achieve the same spacing.
When either a tab or a single blank would suffice to reach a tab stop,
which should be given preference?

**Answer:** Prefer the **blank**. A tab saves nothing when only one column
remains to the stop (same byte count), but a tab is visually ambiguous and
editor-dependent. Emit a tab only when it covers ≥ 2 columns of blanks.""",
        "src_file": "entab.c",
        "demo": r'''import subprocess, os
here = r"{HERE}"
expanded = "abc     hello   world\n"    # result of detab on abc\thello\tworld
result = subprocess.run(
    [os.path.join(here, "entab.exe")],
    input=expanded,
    capture_output=True, text=True
)
print("Input (spaces only):")
print(f"  [{expanded.rstrip()}]")
print("entab output (bytes shown):")
for line in result.stdout.splitlines():
    raw = [hex(ord(c)) for c in line]
    print(f"  {raw}")
    print(f"  (0x09 = tab, 0x20 = space)")
'''.replace("{HERE}", HERE.replace("\\", "\\\\")),
    },
    {
        "num": "1-22",
        "title": "fold — break long lines",
        "statement": """**Exercise 1-22.** Write a program to "fold" long input lines into two
or more shorter lines after the last non-blank character that occurs before
the *n*-th column of input. Make sure your program does something
intelligent with very long lines, and if there are no blanks or tabs
before the specified column.

**Design decisions:**
- `FOLDCOL = 60` is a `#define` (same argument as TABSTOP)
- Break at the last space/tab before column 60
- If no whitespace exists in the window → hard fold at column 60 (no infinite growth)""",
        "src_file": "fold.c",
        "demo": r'''import subprocess, os
here = r"{HERE}"
long_line = "the quick brown fox jumps over the lazy dog and then keeps running past the finish line\n"
no_space  = "A" * 90 + "\n"
result = subprocess.run(
    [os.path.join(here, "fold.exe")],
    input=long_line + no_space,
    capture_output=True, text=True
)
print("Input line 1 (88 chars with spaces):")
print(f"  {long_line.strip()!r}")
print("\nInput line 2 (90 'A's, no spaces):")
print(f"  {'A'*90!r}")
print("\nfolded output:")
for line in result.stdout.splitlines():
    print(f"  [{line}]  len={len(line)}")
'''.replace("{HERE}", HERE.replace("\\", "\\\\")),
    },
    {
        "num": "1-23",
        "title": "strip_comments — remove C comments",
        "statement": """**Exercise 1-23.** Write a program to remove all comments from a C
program. Don't forget to handle quoted strings and character constants
properly. C comments don't nest.

**State machine:** `CODE → IN_STRING / IN_CHAR / IN_LINE_COMMENT / IN_BLOCK_COMMENT`

Key bug fixed: when `*/` closes a block comment, zero out `c` so the closing
`/` doesn't look like a comment opener to the next character.""",
        "src_file": "strip_comments.c",
        "demo": r'''import subprocess, os
here = r"{HERE}"
src = r"""int x = 0; /* block comment */
// line comment
char *s = "/* not a comment */";
int y = x + 1; // another
char c = '\''; /* escaped quote in char const */
"""
result = subprocess.run(
    [os.path.join(here, "strip_comments.exe")],
    input=src,
    capture_output=True, text=True
)
print("Input:")
for line in src.splitlines():
    print(f"  {line}")
print("\nAfter strip_comments:")
for line in result.stdout.splitlines():
    print(f"  {line}")
'''.replace("{HERE}", HERE.replace("\\", "\\\\")),
    },
    {
        "num": "1-24",
        "title": "syntax_check — bracket balance checker",
        "statement": """**Exercise 1-24.** Write a program to check a C program for rudimentary
syntax errors like unmatched parentheses, brackets and braces. Don't
forget about quotes, both single and double, escape sequences, and
comments.

**Implementation:** push `(`, `[`, `{` onto a stack with their line numbers.
On `)`, `]`, `}` verify the top of stack matches. Report unclosed openers at EOF.
Exit code = number of errors (usable as a build tool).""",
        "src_file": "syntax_check.c",
        "demo": r'''import subprocess, os
here = r"{HERE}"
cases = [
    ("balanced",       "int f(int x) { return (x + 1); }"),
    ("unmatched ]",    "int a[10]; a[0] = 1];"),
    ("unclosed {",     "int main() { return 0;"),
    ("string with ]",  'char *s = "a[b]c"; int x = 0;'),
]
for label, src in cases:
    result = subprocess.run(
        [os.path.join(here, "syntax_check.exe")],
        input=src + "\n",
        capture_output=True, text=True
    )
    out = result.stdout.strip() or result.stderr.strip()
    print(f"{label:<20} -> {out}")
'''.replace("{HERE}", HERE.replace("\\", "\\\\")),
    },
]


# ---------------------------------------------------------------------------
# Build notebook
# ---------------------------------------------------------------------------

def build() -> None:
    cells = [md("""# K&R Chapter 1: Exercises 1-20 through 1-24

Five C programs covering text processing fundamentals.
All compiled with `gcc -Wall -Wextra -std=c99 -O2`.
""")]

    for ex in EXERCISES:
        src_path = os.path.join(HERE, ex["src_file"])
        with open(src_path, encoding="utf-8") as f:
            src_text = f.read()

        cells.append(md(f"## Exercise {ex['num']}: {ex['title']}\n\n{ex['statement']}"))
        cells.append(code(f"# {ex['src_file']}\nsrc = '''\n{src_text}'''\nprint(src)"))
        cells.append(code(ex["demo"]))

    # Stack vs Heap explainer at the end
    cells.append(md("""## Appendix: Stack vs Heap in C

```
Stack (automatic storage)          Heap (dynamic storage)
──────────────────────────         ──────────────────────
int x = 5;                         int *p = malloc(N * sizeof(int));
char line[MAXLINE];                // ... use p ...
                                   free(p);

LIFO: grows down, shrinks up       Arbitrary lifetime
Fast: just move the stack pointer  Slower: bookkeeping + fragmentation
Size: limited (~1–8 MB default)    Size: limited only by RAM
Freed: automatically at scope end  Freed: only when you call free()
```

All five K&R programs above use **only stack storage** (`char line[]` arrays,
`int` locals). That is intentional for Chapter 1 — no dynamic allocation needed
until you need data structures of unknown size at compile time (linked lists,
hash tables, etc.).

**The bug that kills programs:** forgetting `free()` → memory leak.
**The bug that kills safety:** `free()` then use → use-after-free (undefined behaviour).
"""))

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }

    out = os.path.join(HERE, "knr_exercises.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out}")
    print("Execute with:")
    print(f"  py -3.13 -m jupyter nbconvert --to notebook --execute {out}")


if __name__ == "__main__":
    build()
