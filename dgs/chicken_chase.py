"""Python driver for learn/chicken_chase/chicken_chase.c -- build it with gcc,
run it, read its 1-bit framebuffer images, disassemble it, and check its
physics against SymPy.

Pure pursuit (Bouguer, 1732): a chaser at speed v_p always heads straight at a
target moving in a straight line at speed v_c = k v_p, starting a distance d
away perpendicular to the target's path. The pursuit curve gives
    T = d v_p / (v_p^2 - v_c^2) = (d / v_p) / (1 - k^2)   for k < 1,  never for k >= 1.
"""
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

import numpy as np
import sympy as sp

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "learn" / "chicken_chase" / "chicken_chase.c"
W, H = 128, 96
MINGW = pathlib.Path(r"C:\msys64\mingw64\bin")


def find_tool(name):
    """Locate gcc/objdump on PATH or in the MSYS2 mingw64 folder; None if absent."""
    hit = shutil.which(name)
    if hit:
        return hit
    cand = MINGW / (name + ".exe")
    return str(cand) if cand.exists() else None


def _env():
    env = dict(os.environ)
    if MINGW.exists():
        env["PATH"] = str(MINGW) + os.pathsep + env.get("PATH", "")
    return env


def build(out_dir=None, opt="-O2"):
    """Compile chicken_chase.c; returns the executable path."""
    gcc = find_tool("gcc")
    if gcc is None:
        raise RuntimeError("gcc not found (install MSYS2 mingw64 or add gcc to PATH)")
    out_dir = pathlib.Path(out_dir or tempfile.mkdtemp(prefix="chicken_"))
    exe = out_dir / ("chicken_chase.exe" if os.name == "nt" else "chicken_chase")
    cmd = [gcc, "-std=c99", opt, "-Wall", "-Wextra", "-Werror", "-o", str(exe), str(SRC), "-lm"]
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=_env())
    return exe


def run_pursuit(exe, k, d=10.0):
    out = subprocess.run([str(exe), "pursuit", repr(float(k)), repr(float(d))],
                         check=True, capture_output=True, text=True, env=_env()).stdout
    t = float(out.strip())
    return None if t < 0 else t


def run_selftest(exe):
    out = subprocess.run([str(exe), "selftest"], check=True, capture_output=True,
                         text=True, env=_env()).stdout.split()
    return out


def run_sim(exe, seed=1, steps=600, every=50, out_dir=None):
    """Run the chase; returns (list of frame paths, list of (step, caught))."""
    out_dir = pathlib.Path(out_dir or tempfile.mkdtemp(prefix="chicken_frames_"))
    out = subprocess.run([str(exe), "sim", str(seed), str(steps), str(every), str(out_dir)],
                         check=True, capture_output=True, text=True, env=_env()).stdout
    catches = [tuple(map(int, line.split())) for line in out.split("\n") if line.strip()]
    return sorted(out_dir.glob("frame_*.pbm")), catches


def read_pbm(path):
    """Binary PBM (P4) -> uint8 array of 0/1, shape (H, W). Bits are MSB-first."""
    data = pathlib.Path(path).read_bytes()
    m = re.match(rb"P4\s+(\d+)\s+(\d+)\s", data)
    if not m:
        raise ValueError("not a binary PBM (P4) file")
    w, h = int(m.group(1)), int(m.group(2))
    bits = np.unpackbits(np.frombuffer(data[m.end():], dtype=np.uint8))
    return bits[: w * h].reshape(h, w) if w % 8 == 0 else None


def pixel_address(x, y, width=W):
    """(byte index, bit mask) exactly as px_set computes them."""
    i = y * width + x
    return i >> 3, 0x80 >> (i & 7)


def framebuffer_bytes(width=W, height=H, bits_per_pixel=1):
    return width * height * bits_per_pixel // 8


def disassemble_px_set(opt="-O2"):
    """gcc -c then objdump -d: return the assembly lines of px_set."""
    gcc, objdump = find_tool("gcc"), find_tool("objdump")
    if gcc is None or objdump is None:
        raise RuntimeError("gcc/objdump not found")
    with tempfile.TemporaryDirectory() as td:
        obj = pathlib.Path(td) / "cc.o"
        subprocess.run([gcc, "-std=c99", opt, "-c", "-o", str(obj), str(SRC)],
                       check=True, capture_output=True, env=_env())
        dis = subprocess.run([objdump, "-d", "--no-show-raw-insn", "-M", "intel", str(obj)],
                             check=True, capture_output=True, text=True, env=_env()).stdout
    block, on = [], False
    for line in dis.splitlines():
        if re.match(r"^[0-9a-f]+ <px_set>:", line):
            on = True
            continue
        if on:
            if not line.strip() or re.match(r"^[0-9a-f]+ <", line):
                break
            block.append(line.split(":", 1)[1].strip() if ":" in line else line.strip())
    return block


# ── SymPy: pursuit curve ─────────────────────────────────────────────

def capture_time_symbolic():
    d, vp, vc = sp.symbols("d v_p v_c", positive=True)
    return d * vp / (vp ** 2 - vc ** 2), (d, vp, vc)


def capture_time(k, d=10.0, vp=1.0):
    """Closed form; None (never) for k >= 1."""
    if k < 0 or d <= 0 or vp <= 0:
        raise ValueError("k >= 0, d > 0, vp > 0")
    return None if k >= 1 else d / vp / (1 - k * k)
