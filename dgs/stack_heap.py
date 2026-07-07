"""Stack vs heap: the two ways a program gets memory, modeled and broken.

A running program's address space has regions, but two do the real work:

  THE STACK -- automatic storage. Every function call pushes a FRAME (its locals,
  return address); returning pops it. Last-in-first-out, so allocation is a single
  pointer bump: fast, but bounded. Recurse too deep and you run off the end -- a
  STACK OVERFLOW. A local variable lives exactly as long as its frame, which is why
  returning a pointer to a local is a dangling-pointer bug.

  THE HEAP -- dynamic storage. malloc/new hand out blocks whose lifetime YOU manage
  with free/delete. Arbitrary order, arbitrary size -- flexible, but you pay for it:
  FRAGMENTATION (free memory split into unusable slivers), and, if you forget to
  free, a LEAK. This module includes a real first-fit allocator with coalescing so
  the fragmentation is demonstrable, not just described.

This is the runtime face of dgs.scope_and_linkage's storage classes: an `auto`
local has AUTOMATIC duration -> it lives on the stack; a `static` or file-scope
object has STATIC duration -> its own fixed region; a malloc'd object has DYNAMIC
duration -> the heap. The three canonical bugs -- stack overflow, use-after-free,
and leak -- are each just a lifetime mistake about which region a value lives in.

Pure Python models (no real pointers), so every mechanism is inspectable and
tested. py-3.13.
"""


def memory_regions():
    """The classic address-space layout, high address to low, with what lives in
    each region and how it grows. Stack and heap grow toward each other."""
    return [
        ("stack", "automatic locals, call frames", "grows down (toward heap)"),
        ("(gap)", "unused address space", "the two grow into this"),
        ("heap", "malloc/new dynamic objects", "grows up (toward stack)"),
        ("BSS / data", "static & global variables", "fixed size"),
        ("text", "the machine code", "read-only, fixed"),
    ]


# ----------------------------------------------------------------------
# THE STACK: call frames, LIFO, and overflow
# ----------------------------------------------------------------------

class StackOverflow(Exception):
    """Raised when pushing a frame would exceed the stack's size limit."""


class CallStack:
    """A LIFO call stack with a byte budget. push() a frame on call, pop() on
    return; exceeding the limit is a StackOverflow -- exactly what unbounded
    recursion causes."""

    def __init__(self, limit_bytes=8192):
        if limit_bytes <= 0:
            raise ValueError("limit_bytes must be positive")
        self.limit = limit_bytes
        self.frames = []           # list of (label, size_bytes)

    def used_bytes(self):
        return sum(sz for _, sz in self.frames)

    def depth(self):
        return len(self.frames)

    def push(self, label, size_bytes):
        """Enter a function: allocate its frame. Overflows if it would not fit."""
        if size_bytes <= 0:
            raise ValueError("frame size must be positive")
        if self.used_bytes() + size_bytes > self.limit:
            raise StackOverflow(
                f"stack overflow: {self.used_bytes()}+{size_bytes} > {self.limit} "
                f"at depth {self.depth()}")
        self.frames.append((label, size_bytes))

    def pop(self):
        """Return from a function: free its frame (a single pointer bump)."""
        if not self.frames:
            raise IndexError("pop from empty call stack")
        return self.frames.pop()


def max_recursion_depth(frame_size, limit_bytes=8192):
    """How many nested calls fit before overflow: limit / frame_size. Bigger
    frames (or a smaller stack) mean shallower safe recursion."""
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    return limit_bytes // frame_size


def factorial_via_stack(n, frame_size=64, limit_bytes=8192):
    """Compute n! by pushing one call frame per level (like the real recursive
    factorial), then unwinding. Returns (result, max_depth). Demonstrates that
    recursion depth IS stack usage: max_depth == n, and a big enough n on a small
    stack raises StackOverflow -- the reason deep recursion crashes."""
    if n < 0:
        raise ValueError("n must be >= 0")
    cs = CallStack(limit_bytes)
    for k in range(n, 0, -1):
        cs.push(k, frame_size)          # each recursive call = one frame
    max_depth = cs.depth()
    result = 1
    while cs.frames:
        k, _ = cs.pop()
        result *= k
    return result, max_depth


# ----------------------------------------------------------------------
# THE HEAP: a first-fit allocator with coalescing
# ----------------------------------------------------------------------

class HeapAllocator:
    """A fixed-size heap with a first-fit malloc and a coalescing free -- the
    core of a real allocator. Tracks the free list, so external fragmentation is
    a number you can read, and a malloc can fail (return None) even when the TOTAL
    free space would suffice, because no single block is big enough."""

    def __init__(self, size):
        if size <= 0:
            raise ValueError("heap size must be positive")
        self.size = size
        self.free = [(0, size)]        # sorted, disjoint (start, length) blocks
        self.alloc = {}                # start -> length of live allocations

    def malloc(self, n):
        """First-fit: return the start address of the first free block big enough
        for n, carving it; None if no single block fits (out of memory)."""
        if n <= 0:
            raise ValueError("allocation size must be positive")
        for i, (start, length) in enumerate(self.free):
            if length >= n:
                self.alloc[start] = n
                if length == n:
                    self.free.pop(i)
                else:
                    self.free[i] = (start + n, length - n)
                return start
        return None

    def free_(self, addr):
        """Return a block to the free list and COALESCE it with any adjacent free
        blocks (so freeing neighbors heals fragmentation). Double/invalid free
        raises -- the model's use-after-free guard."""
        if addr not in self.alloc:
            raise ValueError(f"invalid or double free at {addr}")
        length = self.alloc.pop(addr)
        self.free.append((addr, length))
        self.free.sort()
        merged = []
        for start, ln in self.free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + ln)
            else:
                merged.append((start, ln))
        self.free = merged

    def free_total(self):
        return sum(ln for _, ln in self.free)

    def used_total(self):
        return sum(self.alloc.values())

    def largest_free_block(self):
        return max((ln for _, ln in self.free), default=0)

    def fragmentation(self):
        """External fragmentation in [0,1): 1 - largest_free/total_free. 0 means
        all free memory is one block; near 1 means it is shattered into slivers."""
        ft = self.free_total()
        return 0.0 if ft == 0 else 1 - self.largest_free_block() / ft

    def leaks(self):
        """Blocks still allocated -- if the program ended now, these are leaks."""
        return dict(self.alloc)


if __name__ == "__main__":
    print("address-space regions (high -> low):")
    for name, what, grow in memory_regions():
        print(f"  {name:12s} {what:34s} [{grow}]")

    print("\nSTACK -- recursion is stack usage:")
    res, depth = factorial_via_stack(5)
    print(f"  5! = {res} using max stack depth {depth} (one frame per call)")
    print(f"  64-byte frames on an 8 KB stack overflow past "
          f"{max_recursion_depth(64)} deep")
    try:
        factorial_via_stack(1000, frame_size=64, limit_bytes=8192)
    except StackOverflow as e:
        print(f"  factorial_via_stack(1000): {e}")

    print("\nHEAP -- fragmentation from a first-fit allocator:")
    h = HeapAllocator(100)
    a, b, c = h.malloc(30), h.malloc(30), h.malloc(30)
    print(f"  malloc 30,30,30 -> addresses {a},{b},{c}; free_total={h.free_total()}")
    h.free_(b)      # free the middle block -> a hole between a and c
    print(f"  free middle: free_total={h.free_total()}, largest_block="
          f"{h.largest_free_block()}, fragmentation={h.fragmentation():.2f}")
    print(f"  malloc(40) now -> {h.malloc(40)}  (fails: 40 free, but not contiguous)")
    h.free_(a); h.free_(c)     # free neighbors -> coalesce into one big block
    print(f"  free the rest: largest_block={h.largest_free_block()}, "
          f"malloc(40) -> {h.malloc(40)}  (succeeds after coalescing)")
