"""Build and run the air hockey puck C code (gcc) and CUDA renderer (nvcc) from Python.

    dgs/c/puck/puck.{h,c}     physics: statics (inertia, hover pressure), dynamics, rotation matrix
    dgs/c/puck/puck_test.c    CSC 60 style unit tests (exit code = number of failures)
    dgs/c/puck/puck_game.c    terminal mini game
    dgs/cuda/puck_render.cu   GPU ray-cast renderer, one thread per pixel

Builds are cached in %TEMP%/dgs_puck. Missing compilers raise ``ToolUnavailable`` so tests and
notebooks can skip cleanly.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from dgs.eventsim import ToolUnavailable, find_nvcc

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "dgs" / "c" / "puck"
CU_SRC = ROOT / "dgs" / "cuda" / "puck_render.cu"
CACHE = Path(tempfile.gettempdir()) / "dgs_puck"
EXE = ".exe" if os.name == "nt" else ""
CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-pedantic", "-Werror", "-O2"]


def find_gcc():
    import shutil
    return shutil.which("gcc") or next((p for p in [r"C:\msys64\mingw64\bin\gcc.exe"] if os.path.exists(p)), None)


def _gcc(sources, out):
    gcc = find_gcc()
    if gcc is None:
        raise ToolUnavailable("gcc not found (MinGW-w64 at C:\\msys64\\mingw64\\bin, or build-essential)")
    CACHE.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=str(Path(gcc).parent) + os.pathsep + os.environ.get("PATH", ""))
    cp = subprocess.run([gcc, *CFLAGS, *[str(C_DIR / s) for s in sources], "-o", str(out), "-lm"],
                        capture_output=True, text=True, env=env)
    if cp.returncode:
        raise RuntimeError("gcc failed:\n" + cp.stderr[-2000:])
    return out, env


def run_c_tests():
    """Compile and run puck_test.c. Returns (passed, total, stdout)."""
    exe, env = _gcc(["puck.c", "puck_test.c"], CACHE / f"puck_test{EXE}")
    cp = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
    passed, total = map(int, cp.stdout.strip().splitlines()[-1].split()[0].split("/"))
    return passed, total, cp.stdout


def play(angle_deg, speed, spin=0.0):
    """One shot of the mini game. Returns (scored, board_text, info dict)."""
    exe, env = _gcc(["puck.c", "puck_game.c"], CACHE / f"puck_game{EXE}")
    cp = subprocess.run([str(exe), repr(float(angle_deg)), repr(float(speed)), repr(float(spin))],
                        capture_output=True, text=True, env=env)
    if cp.returncode not in (0, 1):
        raise ValueError(cp.stderr.strip())
    *board, last = cp.stdout.rstrip().splitlines()
    info = dict(kv.split("=") for kv in last.split())
    return info["result"] == "GOAL", "\n".join(board), info


def build_renderer():
    nvcc, vc = find_nvcc()
    if nvcc is None or (os.name == "nt" and vc is None):
        raise ToolUnavailable("nvcc (and on Windows, MSVC vcvars64.bat) not found")
    exe = CACHE / f"puck_render{EXE}"
    if exe.exists() and exe.stat().st_mtime >= CU_SRC.stat().st_mtime:
        return exe
    CACHE.mkdir(parents=True, exist_ok=True)
    build = f'"{nvcc}" -O2 -arch=sm_89 "{CU_SRC}" -o "{exe}"'
    if os.name == "nt":
        bat = CACHE / "build_render.bat"
        bat.write_text(f'@echo off\ncall "{vc}" >nul 2>nul\n{build}\nexit /b %ERRORLEVEL%\n')
        cp = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True, timeout=900)
    else:
        cp = subprocess.run(build, shell=True, capture_output=True, text=True, timeout=900)
    if cp.returncode:
        raise RuntimeError("nvcc failed:\n" + (cp.stdout + cp.stderr)[-2000:])
    return exe


def read_ppm(path):
    data = Path(path).read_bytes()
    parts = data.split(maxsplit=4)
    if parts[0] != b"P6":
        raise ValueError("not a binary PPM")
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(parts[4][: w * h * 3], dtype=np.uint8).reshape(h, w, 3)


def render(tilt_deg=35.0, spin_deg=0.0, h_over_R=0.3, size=256):
    """Render one view on the GPU. Returns (image HxWx3 uint8, info dict)."""
    exe = build_renderer()
    out = CACHE / f"puck_{os.getpid()}.ppm"
    cp = subprocess.run([str(exe), str(out), str(size), str(size), repr(float(tilt_deg)),
                         repr(float(spin_deg)), repr(float(h_over_R))], capture_output=True, text=True)
    if cp.returncode == 3:
        raise ToolUnavailable("no CUDA device: " + cp.stderr.strip())
    if cp.returncode:
        raise ValueError(cp.stderr.strip())
    img = read_ppm(out)
    out.unlink(missing_ok=True)
    info = {k: float(v) for k, v in (kv.split("=") for kv in cp.stdout.split())}
    return img, info


def projected_area_fraction(tilt_deg, h_over_R, extent=1.6):
    """Expected fraction of the image covered by the puck (orthographic projection, R = 1).

    A cylinder tilted by t projects to an ellipse of area pi cos t plus a band of 2 h sin t
    (the side), so the area is pi |cos t| + 2 h |sin t|. Used to check the renderer.
    """
    t = math.radians(tilt_deg)
    return (math.pi * abs(math.cos(t)) + 2 * h_over_R * abs(math.sin(t))) / (2 * extent) ** 2
