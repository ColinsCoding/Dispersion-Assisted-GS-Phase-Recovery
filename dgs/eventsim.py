"""
Build and run the C++ (gcc) and CUDA (nvcc) M/M/c event simulators from Python.

* ``run_cpu``: the event-list simulator in dgs/c/eventsim (one long run, optional event trace)
* ``run_gpu``: dgs/cuda/eventsim.cu (many independent replications at once)

Binaries are built once into a cache folder in the system temp directory and rebuilt when
the source changes. A missing compiler or GPU raises ``ToolUnavailable`` so callers (tests,
notebooks) can skip cleanly. On Windows, gcc is MinGW (C:\\msys64\\mingw64\\bin) and nvcc
needs MSVC's vcvars64.bat.
"""
import glob
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CPP_SRC = ROOT / "dgs" / "c" / "eventsim" / "eventsim_cli.cpp"
CPP_HDR = ROOT / "dgs" / "c" / "eventsim" / "eventsim.hpp"
CU_SRC = ROOT / "dgs" / "cuda" / "eventsim.cu"
CACHE = Path(tempfile.gettempdir()) / "dgs_eventsim"
EXE = ".exe" if os.name == "nt" else ""


class ToolUnavailable(RuntimeError):
    """A compiler, MSVC environment, or CUDA device is missing."""


def find_gxx():
    return shutil.which("g++") or next((p for p in [r"C:\msys64\mingw64\bin\g++.exe"] if os.path.exists(p)), None)


def find_nvcc():
    nvcc = shutil.which("nvcc") or (sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe")) or [None])[-1]
    vc = None
    if os.name == "nt":
        vc = (glob.glob(r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat")
              + glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat") or [None])[0]
    return nvcc, vc


def _stale(exe, *srcs):
    return not exe.exists() or any(exe.stat().st_mtime < s.stat().st_mtime for s in srcs)


def cpu_build_command():
    gxx = find_gxx()
    if gxx is None:
        raise ToolUnavailable("g++ not found (install MinGW-w64 or build-essential)")
    exe = CACHE / f"eventsim{EXE}"
    return [gxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-static", str(CPP_SRC), "-o", str(exe)], exe


def build_cpu():
    """Compile the C++ simulator with g++ if needed; returns the executable path."""
    cmd, exe = cpu_build_command()
    if _stale(exe, CPP_SRC, CPP_HDR):
        CACHE.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if cp.returncode != 0:
            raise RuntimeError("g++ failed:\n" + cp.stderr[-2000:])
    return exe


def run_cpu(lam, mu, c, n, warmup=10_000, seed=1, trace_events=0):
    """Run the event-list simulator. Returns a dict of results (+ 'trace' array if requested)."""
    exe = build_cpu()
    args = [str(exe), repr(float(lam)), repr(float(mu)), str(int(c)), str(int(n)), str(int(warmup)), str(int(seed))]
    trace_path = None
    if trace_events:
        trace_path = CACHE / f"trace_{os.getpid()}_{seed}.csv"
        args += [str(trace_path), str(int(trace_events))]
    t0 = time.perf_counter()
    cp = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    if cp.returncode != 0:
        raise ValueError(cp.stderr.strip())
    out = {k: float(v) for k, v in (line.split("=") for line in cp.stdout.split())}
    out["wall_s"] = wall
    if trace_path:
        out["trace"] = np.loadtxt(trace_path, delimiter=",", ndmin=2)
        trace_path.unlink(missing_ok=True)
    return out


def build_gpu():
    """Compile the CUDA simulator with nvcc if needed; returns the executable path."""
    nvcc, vc = find_nvcc()
    if nvcc is None or (os.name == "nt" and vc is None):
        raise ToolUnavailable("nvcc (and on Windows, MSVC vcvars64.bat) not found")
    exe = CACHE / f"eventsim_cuda{EXE}"
    if not _stale(exe, CU_SRC):
        return exe
    CACHE.mkdir(parents=True, exist_ok=True)
    # 20199: CUDA 13's own curand_poisson.h trips "unrecognized #pragma in device code" under MSVC
    build = f'"{nvcc}" -O2 -arch=sm_89 -diag-suppress 20199 "{CU_SRC}" -o "{exe}"'
    if os.name == "nt":
        bat = CACHE / "build_cuda.bat"
        bat.write_text(f'@echo off\ncall "{vc}" >nul 2>nul\n{build}\nexit /b %ERRORLEVEL%\n')
        cp = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True, timeout=900)
    else:
        cp = subprocess.run(build, shell=True, capture_output=True, text=True, timeout=900)
    if cp.returncode != 0:
        raise RuntimeError("nvcc failed:\n" + (cp.stdout + cp.stderr)[-2000:])
    return exe


def run_gpu(lam, mu, c, n, reps, warmup=2_000, seed=1):
    """Run `reps` independent replications on the GPU.

    Returns dict with arrays 'wq' and 'p_wait' (one entry per replication), 'kernel_ms', 'wall_s'.
    """
    exe = build_gpu()
    out_bin = CACHE / f"gpu_{os.getpid()}_{seed}.bin"
    args = [str(exe), str(out_bin), repr(float(lam)), repr(float(mu)), str(int(c)), str(int(n)),
            str(int(warmup)), str(int(reps)), str(int(seed))]
    t0 = time.perf_counter()
    cp = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    if cp.returncode == 3:
        raise ToolUnavailable("no usable CUDA device: " + cp.stderr.strip())
    if cp.returncode != 0:
        raise ValueError(cp.stderr.strip())
    data = np.fromfile(out_bin, dtype=np.float64)
    out_bin.unlink(missing_ok=True)
    kernel_ms = float(cp.stdout.split("=")[1])
    return {"wq": data[:reps], "p_wait": data[reps:], "kernel_ms": kernel_ms, "wall_s": wall}


def confidence_interval(samples, z=1.96):
    """(mean, half-width) of a normal-approximation confidence interval over replications."""
    x = np.asarray(samples, dtype=float)
    return float(x.mean()), float(z * x.std(ddof=1) / np.sqrt(len(x)))
