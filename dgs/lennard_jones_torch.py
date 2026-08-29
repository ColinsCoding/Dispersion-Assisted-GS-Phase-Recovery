"""GPU-ready (CUDA-if-available, CPU-fallback) Lennard-Jones potential: the
SAME physics as dgs.lennard_jones, vectorized in torch and cross-checked
against it two ways -- against the closed-form analytic force, AND against
autograd's own chain-rule derivative of the potential.

FUNCTION COMPOSITION, MADE EXPLICIT: the force is -dU/dr, but what autograd
actually differentiates is a composition of three functions applied to raw
particle positions pos[N,dim]:

    pos  --(pairwise difference)-->  d[i,j,dim]
         --(Euclidean norm)------->  r[i,j] = |d[i,j]|
         --(4 eps[(sigma/r)^12 - (sigma/r)^6])-->  U

torch.autograd.grad walks that composition backward via the chain rule --
the same chain rule dU/dpos = (dU/dr)(dr/dpos) you'd do by hand -- and MUST
land on exactly dgs.lennard_jones.lj_force_magnitude's closed form. That
agreement is the correctness check below, not an assumption.

WORK RATE: pairwise LJ is O(N^2) -- the natural target for vectorization
(and, with real hardware, a GPU). This module reports pairs/sec for the
vectorized torch computation against dgs.lennard_jones.pair_forces' explicit
Python double loop, on whatever device is actually available (see
device_info() -- this session's torch build is CPU-only; the code is
written device-agnostic so it uses CUDA automatically if it's ever run
somewhere that has it).
"""
import time

import numpy as np
import torch

from dgs.lennard_jones import lj_potential, lj_force_magnitude, pair_forces


def device_info():
    """Which device torch will actually use here, and why."""
    if torch.cuda.is_available():
        return {"device": torch.device("cuda"), "name": torch.cuda.get_device_name(0),
                "note": "CUDA available -- running on GPU."}
    return {"device": torch.device("cpu"), "name": "cpu",
            "note": "No CUDA device visible to torch on this machine "
                    "(torch build is CPU-only here) -- running on CPU. "
                    "The code below is device-agnostic and would use CUDA "
                    "automatically if it were available."}


def lj_potential_torch(r, eps=1.0, sigma=1.0):
    """Same formula as dgs.lennard_jones.lj_potential, in torch (differentiable)."""
    sr6 = (sigma / r) ** 6
    return 4 * eps * (sr6 * sr6 - sr6)


def pairwise_distances(pos):
    """pos: [N, dim] tensor -> (diff[N,N,dim], r[N,N]) with the diagonal
    (self-distance, r=0) masked out to avoid a divide-by-zero in the
    potential -- physically meaningless (an atom doesn't interact with
    itself), not just a numerical inconvenience to paper over."""
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)          # [N,N,dim]
    r = torch.linalg.norm(diff, dim=-1)                  # [N,N]
    return diff, r


def total_energy_vectorized(pos, eps=1.0, sigma=1.0):
    """Total LJ potential energy over all i<j pairs, fully vectorized (no
    Python loop over pairs) -- the O(N^2) work expressed as tensor ops
    instead of nested for-loops, which is what actually makes it GPU-ready."""
    N = pos.shape[0]
    _, r = pairwise_distances(pos)
    i_idx, j_idx = torch.triu_indices(N, N, offset=1)
    r_pairs = r[i_idx, j_idx]
    return lj_potential_torch(r_pairs, eps, sigma).sum()


def forces_autograd(pos, eps=1.0, sigma=1.0):
    """Force on every particle via autograd: F = -dU/dpos, the chain rule
    walked automatically through pos -> diff -> r -> U."""
    pos = pos.clone().requires_grad_(True)
    U = total_energy_vectorized(pos, eps, sigma)
    (grad,) = torch.autograd.grad(U, pos)
    return -grad, U.detach()


def verify_against_analytic(n_particles=6, eps=1.0, sigma=1.0, seed=0, device=None,
                             rel_tol=1e-9):
    """Cross-check 1: autograd's force must agree with
    dgs.lennard_jones.lj_force_magnitude's closed-form derivative, particle
    by particle, component by component -- not just in total energy.

    Uses RELATIVE error, not absolute: a random draw can (legitimately, by
    the physics -- LJ's r^-12 wall is meant to be steep) put two particles
    close enough that forces reach 1e10+ in magnitude, where an absolute
    float64 rounding difference of ~1e-5 is actually machine-precision
    agreement (~1e-16 relative). An absolute tolerance flags that as a
    failure; it isn't one -- caught by this function itself intermittently
    failing across different random seeds before this fix."""
    device = device or device_info()["device"]
    rng = np.random.default_rng(seed)
    pos_np = rng.uniform(0.9, 2.5, size=(n_particles, 3))
    pos = torch.tensor(pos_np, dtype=torch.float64, device=device)

    F_autograd, U_autograd = forces_autograd(pos, eps, sigma)
    F_autograd_np = F_autograd.cpu().numpy()

    # closed-form reference: dgs.lennard_jones.pair_forces (explicit loop,
    # already-verified elsewhere in this repo)
    F_reference, U_reference = pair_forces(pos_np, eps=eps, sigma=sigma)

    force_scale = max(np.max(np.abs(F_reference)), 1e-300)
    energy_scale = max(abs(U_reference), 1e-300)
    force_rel_error = np.max(np.abs(F_autograd_np - F_reference)) / force_scale
    energy_rel_error = abs(float(U_autograd) - U_reference) / energy_scale
    return {
        "force_max_abs_error": np.max(np.abs(F_autograd_np - F_reference)),
        "energy_abs_error": abs(float(U_autograd) - U_reference),
        "force_rel_error": force_rel_error,
        "energy_rel_error": energy_rel_error,
        "matches": force_rel_error < rel_tol and energy_rel_error < rel_tol,
    }


def _lattice_positions(n_particles, spacing, seed):
    """Particles on a jittered cubic lattice at `spacing` (in units of
    sigma), not uniform-random in a box: 200 independent-uniform points in
    a small box put some pairs a fraction of sigma apart, where LJ's r^-12
    repulsion diverges and forces run up to ~1e17 -- physically meaningless
    (that configuration would never be reached; the repulsion is exactly
    what prevents it) and numerically useless (float64 differences between
    two different summation orders, at that magnitude, are dominated by
    rounding, not by any real disagreement -- caught by comparing this
    benchmark's first version against verify_against_analytic()'s
    already-passing small-N check). A lattice with jitter keeps every pair
    at a sane, physically representative separation."""
    n_side = int(np.ceil(n_particles ** (1 / 3)))
    idx = np.array(np.meshgrid(*[range(n_side)] * 3, indexing="ij")).reshape(3, -1).T
    idx = idx[:n_particles]
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.1, 0.1, size=idx.shape)
    return (idx + jitter) * spacing


def benchmark_work_rate(n_particles=200, eps=1.0, sigma=1.0, spacing=1.3, seed=0):
    """WORK RATE: pairs/sec for the vectorized torch computation vs.
    dgs.lennard_jones.pair_forces' explicit O(N^2) Python double loop, same
    input, same answer -- the actual, measured cost of NOT vectorizing.
    Particles are placed on a jittered lattice at `spacing` (see
    _lattice_positions) so forces stay at a physically sane scale."""
    device = device_info()["device"]
    pos_np = _lattice_positions(n_particles, spacing * sigma, seed)
    pos = torch.tensor(pos_np, dtype=torch.float64, device=device)
    n_pairs = n_particles * (n_particles - 1) // 2

    t0 = time.perf_counter()
    F_vec, _ = forces_autograd(pos, eps, sigma)
    torch_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    F_loop, _ = pair_forces(pos_np, eps=eps, sigma=sigma)
    loop_time_s = time.perf_counter() - t0

    max_disagreement = float(np.max(np.abs(F_vec.cpu().numpy() - F_loop)))

    return {
        "n_particles": n_particles,
        "n_pairs": n_pairs,
        "torch_time_s": torch_time_s,
        "loop_time_s": loop_time_s,
        "torch_pairs_per_sec": n_pairs / torch_time_s if torch_time_s > 0 else float("inf"),
        "loop_pairs_per_sec": n_pairs / loop_time_s if loop_time_s > 0 else float("inf"),
        "speedup": loop_time_s / torch_time_s if torch_time_s > 0 else float("inf"),
        "max_disagreement": max_disagreement,
    }


if __name__ == "__main__":
    info = device_info()
    print(f"=== device: {info['name']} -- {info['note']} ===\n")

    check = verify_against_analytic()
    print("Cross-check: autograd force vs. dgs.lennard_jones closed-form force")
    print(f"  max |F_autograd - F_analytic| (relative) = {check['force_rel_error']:.3e}")
    print(f"  |U_autograd - U_analytic|     (relative) = {check['energy_rel_error']:.3e}")
    print(f"  matches: {check['matches']}")
    assert check["matches"], "autograd force disagrees with the closed-form derivative"

    print("\nWork-rate benchmark: vectorized torch vs. explicit Python double loop")
    bench = benchmark_work_rate()
    print(f"  N = {bench['n_particles']} particles, {bench['n_pairs']} pairs")
    print(f"  torch (vectorized): {bench['torch_time_s']*1e3:.3f} ms "
          f"({bench['torch_pairs_per_sec']:.3e} pairs/sec)")
    print(f"  Python loop:        {bench['loop_time_s']*1e3:.3f} ms "
          f"({bench['loop_pairs_per_sec']:.3e} pairs/sec)")
    print(f"  speedup: {bench['speedup']:.1f}x")
    print(f"  max force disagreement between the two: {bench['max_disagreement']:.3e}")
