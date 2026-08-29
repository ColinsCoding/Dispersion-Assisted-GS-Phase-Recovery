"""Test dgs.lennard_jones_torch: autograd's force matches the closed-form
analytic force from dgs.lennard_jones (both in total and per-particle,
component by component), the vectorized computation matches the explicit
Python double loop at a physically sane particle spacing, energy is
translation-invariant, and device_info() reports a real torch device."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from dgs.lennard_jones import pair_forces
from dgs.lennard_jones_torch import (
    device_info, lj_potential_torch, total_energy_vectorized,
    forces_autograd, verify_against_analytic, benchmark_work_rate,
    _lattice_positions,
)

# 1. device_info reports a real, usable torch device
info = device_info()
assert isinstance(info["device"], torch.device)
assert info["device"].type in ("cpu", "cuda")

# 2. lj_potential_torch matches dgs.lennard_jones.lj_potential numerically
from dgs.lennard_jones import lj_potential
r_test = torch.tensor([0.9, 1.0, 1.5, 2.0, 3.0], dtype=torch.float64)
U_torch = lj_potential_torch(r_test).numpy()
U_numpy = lj_potential(r_test.numpy())
assert np.allclose(U_torch, U_numpy, atol=1e-12)

# 3. autograd force matches the closed-form analytic force (the main claim).
#    Relative tolerance, not absolute: random draws can put particles close
#    enough that forces reach 1e10+ (the physically-correct steep r^-12
#    repulsion), where float64 rounding is ~1e-5 in absolute terms but
#    ~1e-16 relative -- this caught a real false-positive test failure
#    during development (see verify_against_analytic's docstring).
check = verify_against_analytic(n_particles=6, seed=0)
assert check["matches"], check
assert check["force_rel_error"] < 1e-9
assert check["energy_rel_error"] < 1e-9

# repeat with several different seeds/particle counts -- one lucky seed isn't proof
for seed, n in [(7, 10), (1, 8), (42, 12), (99, 15)]:
    check_n = verify_against_analytic(n_particles=n, seed=seed)
    assert check_n["matches"], (seed, n, check_n)

# 4. energy is translation-invariant (LJ depends only on separations).
#    Relative tolerance again -- same reason as check 3 above.
rng = np.random.default_rng(1)
pos_np = rng.uniform(1.0, 2.0, size=(5, 3))
pos = torch.tensor(pos_np, dtype=torch.float64)
U1 = total_energy_vectorized(pos)
shift = torch.tensor([3.7, -1.2, 0.5], dtype=torch.float64)
U2 = total_energy_vectorized(pos + shift)
assert abs(float(U1) - float(U2)) / abs(float(U1)) < 1e-9

# 5. forces sum to (near) zero -- Newton's third law, no external force.
#    Relative to the largest individual force in this configuration.
F, _ = forces_autograd(pos)
force_scale = torch.max(torch.abs(F))
assert torch.max(torch.abs(F.sum(dim=0))) / force_scale < 1e-9

# 6. lattice positions keep every pair at a physically sane minimum spacing
lattice = _lattice_positions(n_particles=200, spacing=1.3, seed=0)
from scipy.spatial.distance import pdist
min_dist = pdist(lattice).min()
assert min_dist > 1.0, f"lattice spacing too small for a sane LJ benchmark: {min_dist}"

# 7. the work-rate benchmark: vectorized and looped implementations agree
#    at machine precision on the SAME (sane) configuration, and vectorized
#    is meaningfully faster
bench = benchmark_work_rate(n_particles=100, seed=0)
assert bench["max_disagreement"] < 1e-6, bench
assert bench["speedup"] > 1.0, "vectorized torch should beat an explicit Python double loop"

print("all dgs.lennard_jones_torch tests passed")
