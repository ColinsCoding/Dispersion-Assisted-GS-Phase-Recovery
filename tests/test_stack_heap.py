"""Test dgs.stack_heap: the call stack (LIFO frames, recursion depth = stack use,
overflow) and a first-fit heap allocator (fragmentation makes a malloc fail even
with enough total free space, coalescing heals it, double-free is caught)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import stack_heap as sh

# 1. region map: stack is first (highest), heap is present
regions = sh.memory_regions()
assert regions[0][0] == "stack"
assert any(r[0] == "heap" for r in regions)

# 2. CallStack: LIFO frames, byte accounting, overflow
cs = sh.CallStack(limit_bytes=200)
cs.push("f", 64); cs.push("g", 64)
assert cs.depth() == 2 and cs.used_bytes() == 128
assert cs.pop() == ("g", 64)                     # last in, first out
assert cs.depth() == 1
try:
    cs.push("big", 200); assert False            # 64 + 200 > 200 -> overflow
except sh.StackOverflow:
    pass
# pop from empty raises
sh_empty = sh.CallStack()
try:
    sh_empty.pop(); assert False
except IndexError:
    pass

# 3. max recursion depth = limit // frame_size
assert sh.max_recursion_depth(64, 8192) == 128
assert sh.max_recursion_depth(256, 8192) == 32   # bigger frames -> shallower

# 4. recursion IS stack usage: n! uses depth n; deep enough overflows a small stack
res, depth = sh.factorial_via_stack(10)
assert res == 3628800 and depth == 10
assert sh.factorial_via_stack(0)[0] == 1
try:
    sh.factorial_via_stack(200, frame_size=64, limit_bytes=8192)  # 200*64 > 8192
    assert False
except sh.StackOverflow:
    pass

# 5. heap first-fit malloc: contiguous addresses, byte accounting
h = sh.HeapAllocator(100)
a, b, c = h.malloc(30), h.malloc(30), h.malloc(30)
assert (a, b, c) == (0, 30, 60)                  # first-fit packs from the front
assert h.used_total() == 90 and h.free_total() == 10
# out of memory: no single block big enough -> None (not an exception)
assert h.malloc(50) is None
assert h.malloc(10) == 90                         # exactly fills the tail

# 6. FRAGMENTATION: free the middle, and a 40-byte malloc fails despite 40 free
h2 = sh.HeapAllocator(100)
x, y, z = h2.malloc(30), h2.malloc(30), h2.malloc(30)
h2.free_(y)                                       # hole in the middle
assert h2.free_total() == 40                      # 10 (tail) + 30 (hole)
assert h2.largest_free_block() == 30
assert abs(h2.fragmentation() - 0.25) < 1e-9      # 1 - 30/40
assert h2.malloc(40) is None                      # enough free, but not contiguous
# first-fit REUSES the freed hole for a small request
assert h2.malloc(20) == 30

# 7. COALESCING: freeing neighbors merges free blocks back into one
h3 = sh.HeapAllocator(100)
p, q, r = h3.malloc(30), h3.malloc(30), h3.malloc(30)
for addr in (p, q, r):
    h3.free_(addr)
assert h3.free == [(0, 100)]                      # all coalesced to one block
assert h3.largest_free_block() == 100 and h3.fragmentation() == 0.0
assert h3.malloc(100) == 0                         # the whole heap is usable again

# 8. use-after-free / double-free guard, and leak reporting
h4 = sh.HeapAllocator(50)
addr = h4.malloc(20)
h4.free_(addr)
try:
    h4.free_(addr); assert False                   # double free
except ValueError:
    pass
h4.malloc(10)                                      # allocated and never freed
assert h4.leaks()                                  # a leak is reported

# 9. kwarg bounds
for bad in (lambda: sh.CallStack(0),
            lambda: sh.HeapAllocator(0),
            lambda: sh.HeapAllocator(10).malloc(0),
            lambda: sh.max_recursion_depth(0),
            lambda: sh.factorial_via_stack(-1)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_stack_heap: all checks passed")
