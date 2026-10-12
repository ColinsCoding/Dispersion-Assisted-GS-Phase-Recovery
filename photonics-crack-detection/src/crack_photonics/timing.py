"""Stage timing with ``time.perf_counter()``.

Why ``perf_counter``?
    It is a monotonic, high-resolution clock meant for measuring short
    durations.  ``time.time()`` can jump when the system clock is adjusted,
    so it is the wrong tool for benchmarks.

What the numbers mean
    They are *wall-clock* seconds on your machine, for one run.  They depend
    on the CPU, on what else is running, on disk caching, and on whether
    libraries were already imported.  Compare trends (how time grows with
    image size), not single measurements.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

# Stages that every pipeline run reports, in pipeline order.
STAGE_NAMES = (
    "image_loading",
    "preprocessing",
    "symmetry_analysis",
    "edge_detection",
    "metric_extraction",
)

TOTAL_KEY = "total_runtime_s"

# Keys guaranteed to be present in every timing report.
REQUIRED_TIMING_KEYS = tuple(f"{name}_s" for name in STAGE_NAMES) + (TOTAL_KEY,)


class StageTimer:
    """Accumulate wall-clock time per named stage.

    Example
    -------
    >>> timer = StageTimer()
    >>> with timer.stage("preprocessing"):
    ...     pass
    >>> report = timer.report()
    >>> "preprocessing_s" in report and "total_runtime_s" in report
    True

    The total runs from construction until :meth:`stop` (or until
    :meth:`report` if ``stop`` was never called), so it also contains work
    done *between* stages.  That remainder is reported as
    ``untimed_overhead_s``.
    """

    def __init__(self) -> None:
        self._stages: dict[str, float] = {name: 0.0 for name in STAGE_NAMES}
        self._start = time.perf_counter()
        self._stop: float | None = None

    @contextmanager
    def stage(self, name: str):
        """Time the enclosed block and add it to stage ``name``.

        Using the same name twice adds the durations.  If the block raises,
        the elapsed time is still recorded and the exception propagates
        unchanged -- errors are never swallowed here.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Stage name must be a non-empty string.")
        if self._stop is not None:
            raise RuntimeError("Timer already stopped; create a new StageTimer.")
        begin = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = self._stages.get(name, 0.0) + time.perf_counter() - begin

    def stop(self) -> None:
        """Freeze the total runtime.  Calling it again has no effect."""
        if self._stop is None:
            self._stop = time.perf_counter()

    def report(self) -> dict:
        """Return a flat JSON-friendly dict of seconds per stage.

        Contains every key in :data:`REQUIRED_TIMING_KEYS` (stages that never
        ran report 0.0), any extra stages, ``untimed_overhead_s`` and the
        resolution of the clock.
        """
        end = self._stop if self._stop is not None else time.perf_counter()
        total = end - self._start
        report = {f"{name}_s": seconds for name, seconds in self._stages.items()}
        report["untimed_overhead_s"] = max(total - sum(self._stages.values()), 0.0)
        report[TOTAL_KEY] = total
        report["clock"] = "time.perf_counter"
        report["clock_resolution_s"] = time.get_clock_info("perf_counter").resolution
        return report


def format_report(report: dict) -> str:
    """Render a timing report as an aligned text table for the terminal."""
    missing = [key for key in REQUIRED_TIMING_KEYS if key not in report]
    if missing:
        raise KeyError(f"Timing report is missing required keys: {missing}")

    total = report[TOTAL_KEY]
    lines = [f"{'stage':<22}{'seconds':>12}{'share':>9}"]
    for key, value in report.items():
        if not key.endswith("_s") or key in (TOTAL_KEY, "clock_resolution_s"):
            continue
        share = 100.0 * value / total if total > 0 else 0.0
        lines.append(f"{key[:-2]:<22}{value:>12.6f}{share:>8.1f}%")
    lines.append(f"{'total_runtime':<22}{total:>12.6f}{100.0:>8.1f}%")
    return "\n".join(lines)
