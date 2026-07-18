"""Benchmark symbolic vs numeric vs Torch vs generated C for the beam-width model.

Usage:
    python scripts/benchmark.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from c_codegen.generator import compile_and_run_c
from config import get_logger
from physics.symbolic import gaussian_beam_width

LOGGER = get_logger("benchmark")


def _time(fn, repeats: int = 5) -> float:
    """Return the best wall-clock time of `fn` over `repeats` runs."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(n: int = 100_000) -> dict[str, float]:
    """Time each backend evaluating w(z) over `n` points; return {label: seconds}."""
    bw = gaussian_beam_width()
    z = np.linspace(-1000, 1000, n)
    w0, zr = 10.0, 200.0
    f_np = bw.lambdify("numpy")

    timings: dict[str, float] = {}
    timings["sympy_subs_1000"] = _time(
        lambda: [bw.evaluate(z=float(zz), w0=w0, zR=zr) for zz in z[:1000]]
    )
    timings["numpy_lambdify"] = _time(lambda: f_np(z, w0, zr))

    zt = torch.tensor(z)
    timings["torch"] = _time(lambda: w0 * torch.sqrt(1 + (zt / zr) ** 2))

    if shutil.which("gcc") or shutil.which("cc"):
        timings["generated_c_single_call"] = _time(lambda: compile_and_run_c(bw, (300.0, w0, zr)), repeats=1)

    for label, seconds in timings.items():
        LOGGER.info("%-28s %.6f s", label, seconds)
    return timings


if __name__ == "__main__":
    run()
