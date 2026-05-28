"""
run.py — build and launch tdgs_anneal on Windows or Linux.

Usage:
    python run.py
    python run.py --arch sm_75    # override GPU arch
    python run.py --dry-run       # print commands only
"""
import argparse, subprocess, sys, os, shutil

CUDA_ARCHES = {
    "Ampere (RTX 30xx / A-series)": "sm_86",
    "Ada Lovelace (RTX 40xx)":      "sm_89",
    "Turing (RTX 20xx / GTX 16xx)": "sm_75",
    "Volta (V100)":                  "sm_70",
    "Hopper (H100)":                 "sm_90",
}

def detect_arch():
    """Try to detect GPU arch via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True).strip().split("\n")[0]
        major, minor = out.split(".")
        return f"sm_{major}{minor}"
    except Exception:
        return "sm_86"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",    default=None, help="CUDA arch, e.g. sm_86")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    here  = os.path.dirname(os.path.abspath(__file__))
    src   = os.path.join(here, "tdgs_anneal.cu")
    exe   = os.path.join(here, "tdgs_anneal" + (".exe" if sys.platform == "win32" else ""))
    arch  = args.arch or detect_arch()

    print(f"GPU arch : {arch}")
    print(f"Source   : {src}")
    print(f"Binary   : {exe}\n")

    # ── build ────────────────────────────────────────────────────────
    nvcc = shutil.which("nvcc")
    if not nvcc:
        sys.exit("nvcc not found — install CUDA toolkit and add it to PATH")

    build_cmd = [nvcc, "-O3", f"-arch={arch}", "-lineinfo",
                 "-lcufft", src, "-o", exe]
    print("Building:", " ".join(build_cmd))
    if not args.dry_run:
        r = subprocess.run(build_cmd, cwd=here)
        if r.returncode != 0:
            sys.exit(f"Build failed (exit {r.returncode})")
        print("Build OK\n")

    # ── run ──────────────────────────────────────────────────────────
    print("Starting 40-minute annealing run …")
    if not args.dry_run:
        subprocess.run([exe], cwd=here)

if __name__ == "__main__":
    main()
