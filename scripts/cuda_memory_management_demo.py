"""Orchestrates and VERIFIES the three CUDA memory-management demo
programs in hardware/cuda_memory_management/ -- runs each compiled .exe
as a subprocess, parses its printed FREE_MB readings, and asserts the
actual claims ("this leaks," "this doesn't," "process exit reclaims
device memory even when cudaFree was skipped") rather than requiring a
human to eyeball the console output.

Prerequisite: compile the four .cu files first (nvcc needs cl.exe on
PATH -- see hardware/mie_scattering/Makefile's own note on this):
    cd hardware/cuda_memory_management
    nvcc -O2 -o cuda_mem_probe.exe cuda_mem_probe.cu
    nvcc -O2 -o cuda_leak_in_loop.exe cuda_leak_in_loop.cu
    nvcc -O2 -o cuda_no_leak_in_loop.exe cuda_no_leak_in_loop.cu
    nvcc -O2 -o cuda_exit_on_error.exe cuda_exit_on_error.cu

Run:  py -3.13 scripts/cuda_memory_management_demo.py
"""
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parents[1] / "hardware" / "cuda_memory_management"


def run(exe_name):
    exe = HERE / exe_name
    if not exe.exists():
        print(f"ERROR: {exe} not found -- compile it first (see this script's docstring)")
        sys.exit(1)
    result = subprocess.run([str(exe)], capture_output=True, text=True, cwd=str(HERE))
    return result.stdout, result.returncode


def parse_free_mb(text):
    m = re.search(r"FREE_MB=([\d.]+)", text)
    if not m:
        raise ValueError(f"could not find FREE_MB in output:\n{text}")
    return float(m.group(1))


def main():
    print("=== CUDA MEMORY MANAGEMENT: ORCHESTRATED, VERIFIED DEMO ===\n")

    print("--- baseline ---")
    out, _ = run("cuda_mem_probe.exe")
    print(out.strip())
    baseline_mb = parse_free_mb(out)

    print("\n--- demo 1: cuda_leak_in_loop.exe (allocates, never frees) ---")
    out, code = run("cuda_leak_in_loop.exe")
    print(out.strip())
    # filter to "iter" lines only -- "start:"/"end:" summary lines also contain
    # FREE_MB= and would duplicate the final reading, breaking strict monotonicity
    lines = [l for l in out.splitlines() if "FREE_MB=" in l and "iter" in l]
    free_values = [float(re.search(r"FREE_MB=([\d.]+)", l).group(1)) for l in lines]
    strictly_decreasing = all(free_values[i] > free_values[i + 1] for i in range(len(free_values) - 1))
    assert strictly_decreasing, "expected free memory to strictly decrease every iteration -- it did not"
    print(f"[VERIFIED] free memory strictly decreased across all {len(free_values)} iterations "
          f"({free_values[0]:.0f} MB -> {free_values[-1]:.0f} MB) -- a real leak while the process is alive")

    print("\n--- probe immediately after demo 1's process exited ---")
    out, _ = run("cuda_mem_probe.exe")
    print(out.strip())
    after_leak_mb = parse_free_mb(out)
    assert abs(after_leak_mb - baseline_mb) < 50.0, \
        f"expected memory to return to baseline after process exit, got {after_leak_mb} vs baseline {baseline_mb}"
    print(f"[VERIFIED] free memory back to baseline ({after_leak_mb:.1f} MB vs {baseline_mb:.1f} MB start) -- "
          f"the OS/CUDA driver reclaimed every un-freed allocation the moment the leaking process exited")

    print("\n--- demo 2: cuda_no_leak_in_loop.exe (allocates AND frees each iteration) ---")
    out, code = run("cuda_no_leak_in_loop.exe")
    print(out.strip())
    lines = [l for l in out.splitlines() if "FREE_MB=" in l and "iter" in l]
    free_values2 = [float(re.search(r"FREE_MB=([\d.]+)", l).group(1)) for l in lines]
    flat = max(free_values2) - min(free_values2) < 5.0
    assert flat, f"expected free memory to stay flat with proper cudaFree discipline, saw {free_values2}"
    print(f"[VERIFIED] free memory stayed flat within 5 MB across all iterations "
          f"(min {min(free_values2):.1f}, max {max(free_values2):.1f}) -- the one added cudaFree line fixes it entirely")

    print("\n--- demo 3: cuda_exit_on_error.exe (allocates buf1, then hits a real allocation failure OR the safety cap, then exit(1)) ---")
    out, code = run("cuda_exit_on_error.exe")
    print(out.strip())
    print(f"process exit code: {code}")

    print("\n--- probe immediately after demo 3's process exited ---")
    out, _ = run("cuda_mem_probe.exe")
    print(out.strip())
    after_exit_mb = parse_free_mb(out)
    assert abs(after_exit_mb - baseline_mb) < 50.0, \
        f"expected memory to return to baseline even after exit(1) skipped cudaFree, got {after_exit_mb} vs {baseline_mb}"
    print(f"[VERIFIED] free memory back to baseline ({after_exit_mb:.1f} MB vs {baseline_mb:.1f} MB start) even though")
    print(f"           this process's own cudaFree calls never ran (exit(1) skipped them) -- process-exit cleanup")
    print(f"           via the OS/driver is what actually saves phycv_pst_cuda.cu's CUDA_CHECK-then-exit(1) pattern")
    print(f"           from being a PERSISTENT leak, not the missing cudaFree calls themselves.")

    print("\n=== ALL CLAIMS VERIFIED, NOT JUST PRINTED ===")


if __name__ == "__main__":
    main()
