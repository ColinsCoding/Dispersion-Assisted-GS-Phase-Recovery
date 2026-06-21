"""Overnight Big-O benchmark -- run it while you sleep, wake up to the curves.

Sweeps input sizes over orders of magnitude, times each algorithm, and fits the
log-log slope to RECOVER the exponent empirically:
    time ~ C * n^p   =>   log(time) = log C + p * log(n)
so the slope p of log-time vs log-n is the Big-O exponent. A sort should give
p ~ 1 (n log n looks nearly linear on a log-log plot); a naive O(n^2) gives p ~ 2;
binary search gives a flat line (log n) while linear search slopes up at p ~ 1.

Results are written to results/algo_bench.csv INCREMENTALLY (one row per size, as
it finishes) so a partial overnight run still leaves usable data and a plot.

Usage:
    py -3.13 scripts/bench_algorithms.py            # default sweep (~minutes)
    py -3.13 scripts/bench_algorithms.py --overnight # big sweep (hours)
    py -3.13 scripts/bench_algorithms.py 1000 5000 20000 100000   # custom sizes
"""
import csv
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dgs import algorithms as alg

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "algo_bench.csv"


def _time(fn, repeats=3):
    """Best-of-N wall time (best, not mean -- least polluted by OS jitter)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(sizes):
    print(f"benchmarking {len(sizes)} sizes -> {CSV}")
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "merge_sort", "quicksort", "builtin_sorted",
                    "binary_search", "linear_search"])
        for n in sizes:
            data = [random.random() for _ in range(n)]
            srt = sorted(data)
            targets = [srt[random.randrange(n)] for _ in range(1000)]  # 1000 hits
            row = [
                n,
                _time(lambda: alg.merge_sort(data)),
                _time(lambda: alg.quicksort(data)),
                _time(lambda: sorted(data)),
                _time(lambda: [alg.binary_search(srt, t) for t in targets]),
                _time(lambda: [alg.linear_search(srt, t) for t in targets]),
            ]
            w.writerow(row); f.flush()                 # incremental: survive a Ctrl-C
            print(f"  n={n:<10} merge={row[1]:.4f}s quick={row[2]:.4f}s "
                  f"sorted={row[3]:.4f}s  bin={row[4]:.4f}s lin={row[5]:.4f}s")
    return CSV


def fit_and_plot():
    """Read the CSV, fit log-log slopes (the empirical Big-O exponent), plot."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")                              # headless: save, don't show
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(CSV)))
    if len(rows) < 2:
        print("not enough data to fit yet"); return
    n = np.array([float(r["n"]) for r in rows])
    cols = ["merge_sort", "quicksort", "builtin_sorted", "binary_search", "linear_search"]
    plt.figure(figsize=(8, 5))
    print("\nfitted exponents  (time ~ n^p):")
    for c in cols:
        t = np.array([float(r[c]) for r in rows])
        p = np.polyfit(np.log(n), np.log(np.maximum(t, 1e-9)), 1)[0]
        plt.loglog(n, t, "o-", label=f"{c}  (p={p:.2f})")
        print(f"  {c:<16} p = {p:.2f}")
    plt.xlabel("input size n"); plt.ylabel("time [s] (best of 3)")
    plt.title("empirical Big-O: slope of log-time vs log-n = the exponent")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    png = OUT / "algo_bench.png"
    plt.savefig(png, dpi=120)
    print(f"\nwrote {png}")
    print("expect: sorts p~1.0-1.1 (n log n), linear_search p~1 (O(n)), "
          "binary_search ~flat (O(log n))")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--overnight":
        sizes = [1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000]
    elif args:
        sizes = [int(a) for a in args]
    else:
        sizes = [1000, 3000, 10000, 30000, 100000, 300000]
    run(sizes)
    fit_and_plot()
