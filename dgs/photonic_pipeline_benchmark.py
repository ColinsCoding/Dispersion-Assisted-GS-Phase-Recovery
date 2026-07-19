"""Wall-clock timing benchmark for dgs.cuda_photonic_ai's GS phase-retrieval pipeline.

dgs/cuda_photonic_ai.py's gpu_gs_phase_retrieval() returns a field literally called
'gpu_speedup_estimate': '~50x vs CPU for N=4096, 50 iterations' -- that string is an
UNMEASURED GUESS, not a benchmark result. For SBIR/federal-funding experiment prep you
cannot cite a number nobody actually measured on real hardware. This module replaces
the guess with real time.perf_counter() timings: CPU/NumPy now (this environment is
py-3.13, torch is py-3.12-only -- same constraint as dgs.matmul_benchmark), and a real
GPU number the moment you rerun torch_benchmark_gs_pipeline() under py-3.12 with
torch+CUDA installed.

Also answers the actual question a funding reviewer asks: is this fast enough for
real-time operation at the ADC's raw sample rate, not just "is it fast."
"""

import time

import numpy as np

from dgs.cuda_photonic_ai import gpu_gs_phase_retrieval


def benchmark_gs_pipeline(n_pts=512, n_iter=50, n_trials=5, use_gpu=False, rng_seed=42):
    """Time gpu_gs_phase_retrieval() end-to-end, synthetic-data generation included
    (that's the real per-frame cost a caller pays, same scope as demo()/
    publishable_pipeline() actually exercise). Returns mean/std over n_trials repeats."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    times = []
    result = None
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result = gpu_gs_phase_retrieval(n_pts=n_pts, n_iter=n_iter, use_gpu=use_gpu, rng_seed=rng_seed)
        times.append(time.perf_counter() - t0)
    return {
        "n_pts": n_pts,
        "n_iter": n_iter,
        "n_trials": n_trials,
        "backend": result["backend"],
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "times": times,
    }


def torch_benchmark_gs_pipeline(n_pts=512, n_iter=50, n_trials=5, rng_seed=42):
    """Same benchmark, forcing the torch code path (use_gpu=True) inside
    gpu_gs_phase_retrieval, which itself auto-selects cuda if torch.cuda.is_available()
    else cpu. torch is py-3.12 ONLY here -- run this under `py -3.12` after
    `pip install torch --index-url https://download.pytorch.org/whl/cu121`. Reports
    which device actually ran (via the returned 'backend' string) rather than assuming."""
    import torch  # local import: only installed under py-3.12 in this repo

    times = []
    result = None
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result = gpu_gs_phase_retrieval(n_pts=n_pts, n_iter=n_iter, use_gpu=True, rng_seed=rng_seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        "n_pts": n_pts,
        "n_iter": n_iter,
        "n_trials": n_trials,
        "backend": result["backend"],
        "cuda_available": torch.cuda.is_available(),
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "times": times,
    }


def scaling_sweep(n_pts_list=(128, 256, 512, 1024, 2048), n_iter=50, n_trials=3):
    """Sweep n_pts and report measured timing. The GS loop is FFT-bound (two
    fft/ifft pairs per iteration), so time should scale like n_pts*log2(n_pts),
    not linearly -- each row after the first carries the measured ratio against
    that prediction, a sanity check rather than a bare number."""
    rows = [benchmark_gs_pipeline(n_pts=n, n_iter=n_iter, n_trials=n_trials) for n in n_pts_list]
    for i in range(1, len(rows)):
        n0, n1 = rows[i - 1]["n_pts"], rows[i]["n_pts"]
        t0, t1 = rows[i - 1]["mean_s"], rows[i]["mean_s"]
        predicted_ratio = (n1 * np.log2(n1)) / (n0 * np.log2(n0))
        measured_ratio = t1 / t0 if t0 > 0 else float("nan")
        rows[i]["measured_vs_predicted_scaling"] = (measured_ratio, float(predicted_ratio))
    return rows


def frame_rate_feasibility(mean_s, n_pts, raw_sample_rate_hz=1e9):
    """Real-time feasibility check: required_fps = raw_sample_rate_hz/n_pts (how
    often a fresh frame of n_pts raw ADC samples arrives); achieved_fps = 1/mean_s
    (how fast this pipeline actually processes one frame, measured above).
    real_time_feasible = achieved_fps >= required_fps -- the number a reviewer
    checking a time-stretch-ADC proposal actually wants, not a raw millisecond count."""
    if mean_s <= 0 or n_pts <= 0 or raw_sample_rate_hz <= 0:
        raise ValueError("mean_s, n_pts, raw_sample_rate_hz must all be positive")
    required_fps = raw_sample_rate_hz / n_pts
    achieved_fps = 1.0 / mean_s
    return {
        "required_fps": required_fps,
        "achieved_fps": achieved_fps,
        "real_time_feasible": achieved_fps >= required_fps,
        "headroom_factor": achieved_fps / required_fps,
    }


if __name__ == "__main__":
    print("=== dgs.cuda_photonic_ai GS pipeline: measured wall-clock timing ===\n")
    print("(this environment: py-3.13, torch not installed -- CPU/NumPy path only;")
    print(" see torch_benchmark_gs_pipeline() docstring to add real GPU numbers)\n")

    single = benchmark_gs_pipeline(n_pts=512, n_iter=50, n_trials=10)
    print(f"n_pts=512, n_iter=50, backend={single['backend']}: "
          f"{single['mean_s']*1e3:.3f} +- {single['std_s']*1e3:.3f} ms  (n={single['n_trials']} trials)")

    feas = frame_rate_feasibility(single["mean_s"], n_pts=512, raw_sample_rate_hz=1e9)
    print("\nreal-time check @ 1 Gsample/s raw ADC rate, n_pts=512 frames:")
    print(f"  required: {feas['required_fps']:.2e} frames/s   achieved: {feas['achieved_fps']:.2e} frames/s")
    print(f"  real_time_feasible = {feas['real_time_feasible']}  (headroom factor = {feas['headroom_factor']:.2e})")

    print("\n=== scaling sweep (n_pts vs measured time) ===")
    sweep = scaling_sweep(n_pts_list=(128, 256, 512, 1024), n_iter=50, n_trials=3)
    for row in sweep:
        line = f"  n_pts={row['n_pts']:5d}  mean={row['mean_s']*1e3:8.3f} ms"
        if "measured_vs_predicted_scaling" in row:
            meas, pred = row["measured_vs_predicted_scaling"]
            line += f"   measured_ratio={meas:.2f}x  predicted(N logN)_ratio={pred:.2f}x"
        print(line)

    print("\nNOTE: gpu_gs_phase_retrieval()'s 'gpu_speedup_estimate': '~50x vs CPU' field")
    print("      in dgs/cuda_photonic_ai.py is an UNMEASURED guess -- do not cite it in a")
    print("      funding proposal. Use this module's real numbers (or re-run under py-3.12")
    print("      with torch+CUDA via torch_benchmark_gs_pipeline) instead.")
