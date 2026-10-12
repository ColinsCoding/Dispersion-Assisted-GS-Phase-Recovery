"""GNSS positioning, batched on the GPU: the SAME Gauss-Newton pseudorange
solve as dgs.gnss_positioning (reused directly, not reimplemented -- its
pseudorange model and DOP formula are the ground truth here), but solving
B independent problems SIMULTANEOUSLY as one batched torch computation
instead of one Python-loop call per problem.

Why this is the natural CUDA-shaped version of GNSS positioning: a single
position fix is a tiny 4-unknown least-squares problem -- nothing to
parallelize inside it. But a real receiver (or a fleet of them, or a
Monte-Carlo uncertainty sweep over noise realizations, both implemented
below) needs MANY such independent fixes, and those ARE embarrassingly
parallel: no fix depends on any other. Stacking them into a batch
dimension and solving with torch.linalg.lstsq's batched support turns
"B independent tiny problems" into one throughput-bound tensor op --
exactly the shape of workload that justifies edge GPU compute (a Jetson
Nano, say) over a CPU loop for a real-time multi-epoch or multi-receiver
GNSS payload.

Verified two ways: the batched torch solve is checked against
dgs.gnss_positioning.solve_position's NumPy Gauss-Newton run one-by-one on
the same data (must agree to numerical precision -- same math, different
execution shape), and a Monte-Carlo sweep of noisy realizations is checked
against the closed-form DOP prediction sigma_position ~= PDOP * sigma_rho
(dgs.gnss_positioning.dilution_of_precision, not re-derived) -- simulation
matching theory, not just internally self-consistent.

Whether a CUDA device is visible is checked and reported honestly, not
assumed either way -- torch.cuda.is_available() differs even on this one
machine between this repo's command-line Python install (no CUDA-enabled
build) and its Jupyter kernel (which does have one). Everything below is
written device-parameterized, so it runs correctly on CPU where no GPU is
visible, and places tensors on a real GPU (e.g. a Jetson Nano) with no
code change beyond DEVICE wherever one is.
"""
import time

import numpy as np
import torch

from dgs.gnss_positioning import (
    C_LIGHT, pseudorange, solve_position, dilution_of_precision, _sats_from_directions,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def batched_geometry_matrix(sats, receiver_est):
    """G for a BATCH of receiver estimates at once. sats: (N,3) tensor
    (shared satellite constellation). receiver_est: (B,3) tensor. Returns
    (B,N,4): each row still [line-of-sight unit vector, 1], now stacked
    over a batch dimension instead of computed one estimate at a time."""
    d = receiver_est[:, None, :] - sats[None, :, :]           # (B,N,3)
    ranges = torch.linalg.norm(d, dim=-1)                      # (B,N)
    if torch.any(ranges < 1e-9):
        raise ValueError("an estimate coincides with a satellite")
    los = d / ranges[..., None]                                # (B,N,3)
    ones = torch.ones(*los.shape[:-1], 1, device=los.device, dtype=los.dtype)
    return torch.cat([los, ones], dim=-1)                      # (B,N,4)


def batched_solve_position(sats, pseudoranges, x0=None, max_iter=15, tol=1e-6, device=DEVICE):
    """Gauss-Newton position solve for B independent problems at once.
    sats: (N,3) array/tensor (one shared constellation). pseudoranges:
    (B,N) array/tensor. Returns dict of (B,3) positions, (B,) clock
    biases, and per-problem residual RMS -- the batched analogue of
    dgs.gnss_positioning.solve_position, verified to agree with it below."""
    sats_t = torch.as_tensor(sats, dtype=torch.float64, device=device)
    rho = torch.as_tensor(pseudoranges, dtype=torch.float64, device=device)
    B, N = rho.shape
    if N != sats_t.shape[0]:
        raise ValueError("pseudoranges' satellite dimension must match sats")
    if N < 4:
        raise ValueError("need at least 4 satellites (x, y, z, clock bias)")

    x = torch.zeros(B, 4, dtype=torch.float64, device=device) if x0 is None \
        else torch.as_tensor(x0, dtype=torch.float64, device=device).clone()

    for _ in range(max_iter):
        r, b = x[:, :3], x[:, 3]
        pred = torch.linalg.norm(r[:, None, :] - sats_t[None, :, :], dim=-1) + b[:, None]  # (B,N)
        G = batched_geometry_matrix(sats_t, r)                 # (B,N,4)
        residual = (rho - pred)[..., None]                     # (B,N,1)
        dx = torch.linalg.lstsq(G, residual).solution.squeeze(-1)  # (B,4)
        x = x + dx
        if torch.max(torch.linalg.norm(dx, dim=-1)) < tol:
            break

    r, b = x[:, :3], x[:, 3]
    pred_final = torch.linalg.norm(r[:, None, :] - sats_t[None, :, :], dim=-1) + b[:, None]
    residual_rms = torch.sqrt(torch.mean((rho - pred_final) ** 2, dim=-1))
    return {
        "position": r.cpu().numpy(),
        "clock_bias": b.cpu().numpy(),
        "residual_rms": residual_rms.cpu().numpy(),
    }


def monte_carlo_positioning_uncertainty(sats, receiver_true, bias_true, sigma_rho,
                                         n_realizations=2000, seed=0, device=DEVICE):
    """Draw n_realizations independent noisy pseudorange sets for the SAME
    true receiver, solve all of them in one batched torch call, and compare
    the empirical position-error RMS against the CLOSED-FORM DOP prediction
    RMS(||error||) ~= PDOP * sigma_rho -- theory vs. simulation, not just
    internal consistency.

    RMS, not std: PDOP*sigma_rho predicts sqrt(E[||error||^2]) (the DOP
    formula comes from Cov(dx) ~= sigma_rho^2 (G^T G)^-1, i.e. E[error_i^2]
    per axis), which is the RMS of the error norm -- NOT std(||error||)
    (variance of the norm about ITS OWN mean). Those differ because
    ||error|| is non-negative with a nonzero mean (like a chi-distributed
    variable), so std(||error||) < RMS(||error||). Using std here first
    gave a ratio of ~0.44 instead of ~1.0 -- not a solver bug, a wrong
    choice of statistic, caught by checking against the theory it was
    supposed to match rather than trusting a plausible-looking number."""
    sats = np.asarray(sats, dtype=float)
    receiver_true = np.asarray(receiver_true, dtype=float)
    rng = np.random.default_rng(seed)

    rho_clean = np.array([pseudorange(receiver_true, s, bias_true) for s in sats])
    noise = rng.normal(0.0, sigma_rho, size=(n_realizations, len(sats)))
    rho_batch = rho_clean[None, :] + noise

    x0 = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (n_realizations, 1))
    sol = batched_solve_position(sats, rho_batch, x0=x0, device=device)

    pos_errors = sol["position"] - receiver_true[None, :]       # (B,3)
    empirical_rms = np.sqrt(np.mean(np.sum(pos_errors ** 2, axis=1)))

    dop = dilution_of_precision(sats, receiver_true)
    predicted_rms = dop["PDOP"] * sigma_rho

    return {
        "empirical_rms_m": float(empirical_rms),
        "predicted_rms_m": float(predicted_rms),
        "PDOP": dop["PDOP"],
        "n_realizations": n_realizations,
        "mean_residual_rms": float(np.mean(sol["residual_rms"])),
    }


def benchmark_batched_vs_loop(sats, receiver_true, bias_true, sigma_rho, n_realizations=500, seed=1):
    """Honest timing comparison: the SAME n_realizations noisy solves, done
    (a) one at a time via dgs.gnss_positioning.solve_position in a Python
    loop, vs (b) one batched torch call. Measures wall-clock directly --
    reports whatever speedup actually happens on THIS machine's CPU, no
    assumed numbers, and separately reports whether a CUDA device was used."""
    sats = np.asarray(sats, dtype=float)
    receiver_true = np.asarray(receiver_true, dtype=float)
    rng = np.random.default_rng(seed)

    rho_clean = np.array([pseudorange(receiver_true, s, bias_true) for s in sats])
    noise = rng.normal(0.0, sigma_rho, size=(n_realizations, len(sats)))
    rho_batch = rho_clean[None, :] + noise

    t0 = time.perf_counter()
    loop_positions = np.array([solve_position(sats, rho_batch[i])["position"] for i in range(n_realizations)])
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    x0 = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (n_realizations, 1))
    batched_result = batched_solve_position(sats, rho_batch, x0=x0)
    t_batched = time.perf_counter() - t0

    max_disagreement = float(np.max(np.abs(loop_positions - batched_result["position"])))

    return {
        "n_realizations": n_realizations,
        "loop_seconds": t_loop,
        "batched_seconds": t_batched,
        "speedup": t_loop / t_batched if t_batched > 0 else float("inf"),
        "max_position_disagreement_m": max_disagreement,
        "device_used": DEVICE,
    }


def benchmark_repeated(sats, receiver_true, bias_true, sigma_rho, n_realizations, n_repeats=5, base_seed=100):
    """A single call to benchmark_batched_vs_loop is one noisy wall-clock
    sample -- not something to report as THE timing. Repeat it n_repeats
    times (different seeds, same problem size) and return mean/std for
    both methods, the same mean+std-over-trials discipline
    dgs.photonic_pipeline_benchmark.benchmark_gs_pipeline already uses for
    the GS pipeline elsewhere in this repo."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    loop_times, batched_times, disagreements = [], [], []
    for k in range(n_repeats):
        r = benchmark_batched_vs_loop(sats, receiver_true, bias_true, sigma_rho,
                                       n_realizations=n_realizations, seed=base_seed + k)
        loop_times.append(r["loop_seconds"])
        batched_times.append(r["batched_seconds"])
        disagreements.append(r["max_position_disagreement_m"])
    loop_times, batched_times = np.array(loop_times), np.array(batched_times)
    return {
        "n_realizations": n_realizations,
        "n_repeats": n_repeats,
        "loop_mean_s": float(loop_times.mean()), "loop_std_s": float(loop_times.std()),
        "batched_mean_s": float(batched_times.mean()), "batched_std_s": float(batched_times.std()),
        "speedup_mean": float(loop_times.mean() / batched_times.mean()) if batched_times.mean() > 0 else float("inf"),
        "max_disagreement_m": float(max(disagreements)),
        "device_used": DEVICE,
    }


def scaling_sweep(sats, receiver_true, bias_true, sigma_rho,
                   batch_sizes=(25, 50, 100, 250, 500, 1000, 2000), n_repeats=5):
    """Sweep batch size B and measure (mean +- std over n_repeats trials
    each) how loop time, batched time, and speedup actually scale. The
    loop is O(B) with per-call Python/NumPy overhead; the batched call
    does a fixed amount of tensor-op launch overhead plus O(B) compute --
    so speedup should GROW with B while overhead is not yet amortized,
    then approach a roughly constant ratio once compute dominates over
    launch overhead. A genuine, checkable prediction, not asserted."""
    rows = [benchmark_repeated(sats, receiver_true, bias_true, sigma_rho, n, n_repeats=n_repeats)
            for n in batch_sizes]
    for i in range(1, len(rows)):
        rows[i]["speedup_grew"] = rows[i]["speedup_mean"] >= rows[i - 1]["speedup_mean"]
    return rows


if __name__ == "__main__":
    print(f"=== GNSS POSITIONING, BATCHED (torch device: {DEVICE}) ===\n")
    if DEVICE == "cuda":
        print(f"CUDA device detected and in use: {torch.cuda.get_device_name(0)}\n")
    else:
        print("No CUDA device visible to this Python process (torch.cuda.is_available() == False).")
        print("Everything below runs correctly on CPU; the code is device-parameterized")
        print("and would place tensors on a real GPU (e.g. a Jetson Nano) with no changes")
        print("beyond DEVICE -- reported honestly, not assumed.\n")

    receiver_true = np.array([6.371e6, 0.0, 0.0])
    bias_true = 100e-6 * C_LIGHT
    dirs = np.array([[1, 0, 0], [0.5, 0.8, 0.3], [0.5, -0.8, 0.3],
                     [0.4, 0.3, -0.85], [0.6, -0.2, 0.75], [0.2, 0.6, -0.77]])
    sats = _sats_from_directions(receiver_true, dirs, 26.56e6)

    print("--- batched solve vs. the existing NumPy per-problem solve ---")
    bench = benchmark_batched_vs_loop(sats, receiver_true, bias_true, sigma_rho=5.0, n_realizations=500)
    print(f"  {bench['n_realizations']} independent noisy fixes:")
    print(f"  loop (existing NumPy solver):  {bench['loop_seconds']*1e3:.1f} ms")
    print(f"  batched (this module, torch):  {bench['batched_seconds']*1e3:.1f} ms")
    print(f"  speedup: {bench['speedup']:.1f}x on {bench['device_used']}")
    print(f"  max disagreement vs. the NumPy solver: {bench['max_position_disagreement_m']:.2e} m")

    print("\n--- Monte Carlo uncertainty vs. closed-form DOP prediction ---")
    mc = monte_carlo_positioning_uncertainty(sats, receiver_true, bias_true, sigma_rho=5.0, n_realizations=3000)
    print(f"  {mc['n_realizations']} noisy realizations, pseudorange noise sigma=5.0 m")
    print(f"  PDOP = {mc['PDOP']:.3f}")
    print(f"  empirical position-error RMS: {mc['empirical_rms_m']:.3f} m")
    print(f"  DOP-predicted RMS (PDOP*sigma_rho): {mc['predicted_rms_m']:.3f} m")
    print(f"  ratio: {mc['empirical_rms_m']/mc['predicted_rms_m']:.3f}  (should be close to 1.0)")

    print("\n=== DONE ===")
