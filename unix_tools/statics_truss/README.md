# statics_truss

A real, compiled Unix C program for 2D truss statics (method of joints) —
the same physics as [`dgs/truss.py`](../../dgs/truss.py)'s `solve_truss`
(NumPy) and [`dgs/truss_symbolic.py`](../../dgs/truss_symbolic.py)'s
`solve_truss_symbolic` (SymPy), independently implemented in C and
cross-validated against both.

## Unix philosophy

Does one thing — solves a statically determinate 2D truss — and reads a
plain-text description from stdin **or** a file argument, writes results
to stdout, and signals success/failure via exit code, so it composes with
pipes and redirection like any other filter:

```bash
./statics_truss < truss.txt
./statics_truss truss.txt
cat truss.txt | ./statics_truss | tee results.txt
```

Exit codes: `0` success, `1` malformed input, `2` not statically
determinate (wrong member/reaction count), `3` singular system (a
mechanism — unstable geometry even with the right unknown count).

## Input format

```
n_nodes n_members
x y fix_x fix_y load_x load_y      (one line per node, 0-indexed)
node_a node_b                      (one line per member)
```

`fix_x`/`fix_y` are 0/1 (matches `dgs.truss`'s supports convention: pin =
`1 1`, roller = `0 1`). `#` starts a comment to end of line. See
[`three_bar_truss.txt`](three_bar_truss.txt) for a worked example (the
same reference truss used throughout this repo's Python truss work) and
[`mechanism.txt`](mechanism.txt) for a deliberately non-determinate one.

## Build & verify

```
make build     # compile with gcc
make run       # build + solve the reference truss
make verify    # build + cross-check against dgs.truss AND dgs.truss_symbolic
make mechanism # demonstrate the exit-code-2 path
```

`make verify` runs [`verify_cross_language.py`](verify_cross_language.py),
which solves the same truss three independent ways — NumPy linear solve,
SymPy symbolic solve, and this program's from-scratch Gaussian
elimination in C — and asserts all three agree to within 1e-3 N.
