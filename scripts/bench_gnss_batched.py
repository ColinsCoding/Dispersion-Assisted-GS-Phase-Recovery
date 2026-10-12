"""Standalone benchmark script for dgs.torch.gnss_batched: a proper
mean+-std-over-repeated-trials scaling sweep across batch sizes, matching
dgs.photonic_pipeline_benchmark's discipline (that module exists BECAUSE
an earlier, unmeasured "~50x GPU speedup" guess elsewhere in this repo
got caught and replaced with real time.perf_counter() numbers -- this
script applies the same standard to the batched GNSS solve's own speedup
claim: measured, repeated, and reported with its actual variance, not a
single noisy timing.

Run:  py -3.13 scripts/bench_gnss_batched.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from dgs.gnss_positioning import C_LIGHT, _sats_from_directions
from dgs.torch.gnss_batched import DEVICE, scaling_sweep

import torch


def main():
    print(f"=== GNSS BATCHED SOLVE: SCALING BENCHMARK (torch device: {DEVICE}) ===\n")
    if DEVICE == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
    else:
        print("No CUDA device visible to this Python process -- CPU timings only.")
        print("(Note: this repo's Jupyter kernel has separately measured a CUDA device")
        print("on this same machine -- see notebooks/gnss_batched.ipynb. Command-line")
        print("`py -3.13` here has no CUDA-enabled torch build installed.)\n")

    receiver_true = np.array([6.371e6, 0.0, 0.0])
    bias_true = 100e-6 * C_LIGHT
    dirs = np.array([[1, 0, 0], [0.5, 0.8, 0.3], [0.5, -0.8, 0.3],
                     [0.4, 0.3, -0.85], [0.6, -0.2, 0.75], [0.2, 0.6, -0.77]])
    sats = _sats_from_directions(receiver_true, dirs, 26.56e6)

    rows = scaling_sweep(sats, receiver_true, bias_true, sigma_rho=5.0,
                          batch_sizes=(25, 50, 100, 250, 500, 1000, 2000), n_repeats=5)

    print(f"{'B':>6} {'loop (ms)':>16} {'batched (ms)':>16} {'speedup':>10} {'max err (m)':>12}")
    for r in rows:
        loop_str = f"{r['loop_mean_s']*1e3:.2f} +- {r['loop_std_s']*1e3:.2f}"
        batch_str = f"{r['batched_mean_s']*1e3:.2f} +- {r['batched_std_s']*1e3:.2f}"
        print(f"{r['n_realizations']:>6} {loop_str:>16} {batch_str:>16} "
              f"{r['speedup_mean']:>9.1f}x {r['max_disagreement_m']:>12.2e}")

    max_disagreement = max(r["max_disagreement_m"] for r in rows)
    print(f"\nmax disagreement vs. the existing NumPy solver, across all batch sizes: {max_disagreement:.2e} m")
    assert max_disagreement < 1e-6, "batched solve disagrees with the trusted NumPy solver beyond floating-point noise"

    grew_count = sum(1 for r in rows[1:] if r.get("speedup_grew"))
    print(f"speedup increased at {grew_count}/{len(rows)-1} successive batch-size steps "
          f"(expected: mostly yes, while fixed per-call overhead is still being amortized)")

    print(f"\nlargest measured speedup: {max(r['speedup_mean'] for r in rows):.1f}x "
          f"at B={rows[np.argmax([r['speedup_mean'] for r in rows])]['n_realizations']}, on {DEVICE}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
