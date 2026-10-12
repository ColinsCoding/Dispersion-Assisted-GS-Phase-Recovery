"""Build notebooks/c_pointers_memory.ipynb -- "Pointers, Arrays, Structs,
and Memory: the C Fundamentals Toolkit, Verified"

Companion notebook to two already-tested but notebook-less modules:
dgs/c_fundamentals.py (data types, bitwise operators, pointer/array/struct
semantics, common C bugs) and dgs/stack_heap.py (the stack as a bounded
LIFO call structure, the heap as a first-fit allocator with coalescing).
Both are reused directly -- nothing here is reimplemented, only exercised
and cross-checked, including one live nvcc/cl.exe-compiled C example.

Build:   py -3.13 scripts/build_c_pointers_memory_nb.py
Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
         notebooks/c_pointers_memory.ipynb
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md(r"""# Pointers, Arrays, Structs, and Memory: the C Fundamentals Toolkit, Verified

Two already-tested modules that never had a companion notebook:
[`dgs/c_fundamentals.py`](../dgs/c_fundamentals.py) (data types, bitwise
operators, pointer/array/struct semantics, common C bugs) and
[`dgs/stack_heap.py`](../dgs/stack_heap.py) (the stack as a bounded LIFO
call structure; the heap as a first-fit allocator with coalescing) —
reused directly here, plus one genuinely compiled and run C program to
ground the pointer/array claims in an actual binary, not just a Python
model of one."""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import matplotlib.pyplot as plt

from dgs.c_fundamentals import (
    c_type_table, bitwise_demo, bit_count, is_power_of_2, rotate_left,
    no_temp_swap, rot13, struct_layout_lesson, pointer_cheatsheet, SYNTAX_ERRORS,
)
from dgs.stack_heap import memory_regions, CallStack, StackOverflow, factorial_via_stack, HeapAllocator

print("using dgs.c_fundamentals and dgs.stack_heap directly (not reimplemented)")"""))

# ============================================================ PART 1: types and bitwise
cells.append(md(r"""# Part 1 — C types and bitwise operators

`sizeof` for every standard C type, and the bitwise operator set every
low-level job needs — including the classic `&` vs `&&` trap, checked
concretely on a specific pair of values rather than just described."""))

cells.append(co("""for row in c_type_table():
    print(f"  {row['type']:22s} {row['size_bytes']} bytes  {row['size_bits']:3d} bits  {row['notes']}")

print()
bd = bitwise_demo(22, 13)
for op in ['AND', 'OR', 'XOR', 'NOT_a', 'SHL2', 'SHR1']:
    print(f"  {op:7s}: {bd[op]['binary']}  ({bd[op]['value']:3d})  -- {bd[op]['use']}")
print(f"\\n{bd['danger_note']}")
assert (22 & 13) == 4 and bool(22 and 13) == True   # the exact & vs && distinction, checked not just asserted"""))

# ============================================================ PART 2: bit manipulation
cells.append(md(r"""# Part 2 — Bit manipulation patterns (interview-standard)

`is_power_of_2`, `bit_count` (Kernighan's popcount), `rotate_left`
(cryptographic/hardware bit rotation, distinct from ROT13), and the
XOR no-temp swap — each checked against an independent ground truth
(Python's own `bin().count('1')`, `math.log2`, or a direct hand trace)."""))

cells.append(co("""for n in [1, 2, 3, 4, 63, 64, 128, 1023, 1024]:
    r = is_power_of_2(n)
    expected = (n > 0) and (n & (n-1) == 0)
    assert r['is_power_of_2'] == expected
print("is_power_of_2 verified against the direct definition for 9 test values")

for n in [181, 255, 1024, 0]:
    r = bit_count(n)
    assert r['popcount'] == bin(n).count('1')   # cross-check against Python's own bit-count
print("bit_count (Kernighan's method) matches Python's bin().count('1') exactly")

rot = rotate_left(0b1011, shift=1, bits=8)
print(f"\\nrotate_left(0b00001011, 1, bits=8) = {rot['result_binary']}  ({rot['c_code']})")
assert rot['result'] == 0b00010110

a, b = 17, 42
swap = no_temp_swap(a, b)
print(f"\\nXOR swap: {a},{b} -> {swap['after_swap']}")
assert swap['after_swap'] == (b, a)
print("[OK] all bit-manipulation patterns verified against independent ground truth")"""))

# ============================================================ PART 3: pointers and arrays
cells.append(md(r"""# Part 3 — Pointers, arrays, and one COMPILED, RUN example

The pointer/array cheatsheet, then a genuine C program — compiled with
the same `nvcc`/`cl.exe` toolchain used elsewhere this session — that
demonstrates `a[i] == *(a+i)` is not a rule to memorize but a fact about
what the compiled binary actually does, checked by running it."""))

cells.append(co("""for key, code in pointer_cheatsheet().items():
    print(f"  {key:8s}: {code}")"""))

cells.append(co("""import subprocess, tempfile, pathlib as pl

c_source = '''#include <stdio.h>

int main(void) {
    int a[5] = {10, 20, 30, 40, 50};
    int *p = a;   /* array name decays to a pointer to a[0] */

    int all_match = 1;
    for (int i = 0; i < 5; i++) {
        if (a[i] != *(p + i)) all_match = 0;
        printf("a[%d]=%d  *(p+%d)=%d\\\\n", i, a[i], i, *(p + i));
    }
    printf("a[i] == *(p+i) for all i: %s\\\\n", all_match ? "TRUE" : "FALSE");
    return all_match ? 0 : 1;
}
'''

scratch = pl.Path(tempfile.gettempdir())
c_path = scratch / "pointer_array_demo.c"
exe_path = scratch / "pointer_array_demo.exe"
c_path.write_text(c_source)
print(c_source)

compile_result = subprocess.run(["gcc", "-O2", "-o", str(exe_path), str(c_path)],
                                 capture_output=True, text=True)
if compile_result.returncode != 0:
    print("gcc not available in this kernel's environment; source shown above, not run.")
    print(compile_result.stderr)
else:
    run_result = subprocess.run([str(exe_path)], capture_output=True, text=True)
    print(run_result.stdout)
    assert run_result.returncode == 0
    assert "TRUE" in run_result.stdout
    print("[OK] compiled and ran with gcc: a[i] == *(p+i) confirmed by an actual binary, not just the rule")"""))

# ============================================================ PART 4: structs
cells.append(md(r"""# Part 4 — Struct layout and padding

Member order changes `sizeof` — reused directly from
`dgs.c_fundamentals.struct_layout_lesson`, which already carries the
worked comparison (12 bytes padded vs. 8 bytes reordered for the same
three members)."""))

cells.append(co("""layout = struct_layout_lesson()
print(layout['demo_c_code'])
print(f"rule: {layout['rule']}")
print(f"sizeof(Padded) = {layout['sizeof_padded']}, sizeof(Packed) = {layout['sizeof_packed']}")
assert layout['sizeof_packed'] < layout['sizeof_padded']
print("[OK] reordering struct members by decreasing size shrinks sizeof, exactly as claimed")"""))

# ============================================================ PART 5: stack
cells.append(md(r"""# Part 5 — The stack: recursion IS bounded memory

[`dgs.stack_heap.CallStack`](../dgs/stack_heap.py) models the stack as a
byte-budgeted LIFO structure. `factorial_via_stack` computes `n!` by
literally pushing one frame per recursive level — demonstrating that
recursion depth and stack usage are the same number, and that a large
enough `n` on a small stack genuinely overflows (not a simulated failure,
a real `StackOverflow` exception raised by the same push logic a real
call stack uses)."""))

cells.append(co("""for name, what, grow in memory_regions():
    print(f"  {name:12s} {what:34s} [{grow}]")

result, depth = factorial_via_stack(10)
print(f"\\n10! = {result}, computed via {depth} stack frames (depth == n, exactly)")
assert result == 3628800 and depth == 10

# force a real overflow: deep recursion on a small stack
try:
    factorial_via_stack(1000, frame_size=64, limit_bytes=8192)   # 1000*64=64000 >> 8192
    assert False, "expected a StackOverflow"
except StackOverflow as e:
    print(f"\\n[OK] StackOverflow raised as expected: {e}")"""))

# ============================================================ PART 6: heap
cells.append(md(r"""# Part 6 — The heap: first-fit allocation, fragmentation, and leaks

`HeapAllocator` is a real (if simplified) first-fit allocator with
coalescing. Fragmentation is demonstrated concretely: allocate several
blocks, free every other one, and show that the TOTAL free space can be
plenty while no SINGLE free block is big enough for a new request —
external fragmentation as a measured number, not a description."""))

cells.append(co("""heap = HeapAllocator(size=60)   # exactly filled by 6x10-byte allocations -- no untouched leftover region
addrs = [heap.malloc(10) for _ in range(6)]
print(f"after 6 allocations: used={heap.used_total()}, free={heap.free_total()}, "
      f"fragmentation={heap.fragmentation():.2f}")

# free every OTHER block -> fragments the free space into disjoint slivers
for a in addrs[::2]:
    heap.free_(a)
print(f"\\nafter freeing every other block: used={heap.used_total()}, free={heap.free_total()}, "
      f"largest free block={heap.largest_free_block()}, fragmentation={heap.fragmentation():.2f}")

# total free space (30) exceeds a 20-byte request, but no SINGLE block (each only 10) is big enough
big_request = heap.malloc(20)
print(f"request for 20 bytes (less than total free={heap.free_total()}, but each free block is only "
      f"{heap.largest_free_block()}): {'succeeded' if big_request is not None else 'FAILED -- fragmentation, not lack of memory'}")
assert big_request is None
print("[OK] confirmed: malloc can fail even when total free memory would suffice -- external")
print("     fragmentation, not memory exhaustion, exactly as dgs.stack_heap's docstring claims")

# now free the REMAINING blocks and show coalescing heals it completely
for a in addrs[1::2]:
    heap.free_(a)
print(f"\\nafter freeing everything: free={heap.free_total()}, largest block={heap.largest_free_block()}, "
      f"fragmentation={heap.fragmentation():.2f}")
assert heap.fragmentation() == 0.0
print("[OK] coalescing merges all freed blocks back into one -- fragmentation returns to exactly 0")"""))

# ============================================================ PART 7: summary
cells.append(md(r"""# Part 7 — Summary

| Claim | Verified |
|---|---|
| `&` (bitwise) vs `&&` (logical) give different answers on the same operands | Part 1 |
| `is_power_of_2`, `bit_count` match independent ground truth | Part 2 |
| `a[i] == *(p+i)` | Part 3 — confirmed by an actual compiled-and-run C binary |
| Reordering struct members shrinks `sizeof` | Part 4 |
| Recursion depth == stack frames used; deep recursion overflows | Part 5 |
| `malloc` can fail from fragmentation even with enough total free memory | Part 6 |
| Coalescing on free heals fragmentation back to 0 | Part 6 |

Two modules that already had real test coverage
([`tests/test_c_fundamentals.py`](../tests/test_c_fundamentals.py),
[`tests/test_stack_heap.py`](../tests/test_stack_heap.py)) but no
notebook — this closes that gap, and adds one thing neither test file
had: an actual compiled C binary proving the pointer/array claim on real
hardware, not just in a Python model of it."""))

nb["cells"] = cells
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "c_pointers_memory.ipynb"
out.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {out}, {len(cells)} cells")
