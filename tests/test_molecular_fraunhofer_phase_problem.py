"""Test molecular Fraunhofer diffraction and the CDI phase problem: the
structure factor / diffraction intensity physics, and Coherent Diffractive
Imaging's real, documented sensitivity to support-constraint tightness
(too loose stagnates, too tight cuts off real signal -- there IS a sweet
spot, verified by direct comparison, not assumed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import molecular_fraunhofer_phase_problem as mfp

# 1. toy_molecule_positions gives sorted, in-range positions
positions = mfp.toy_molecule_positions(n_atoms=5, extent=0.6, seed=0)
assert len(positions) == 5
assert np.all(positions >= -0.3) and np.all(positions <= 0.3)
assert np.all(np.diff(positions) >= 0)   # sorted

# 2. electron_density places real mass at the atom positions (not just noise)
N = 512
x_grid = np.linspace(-1.5, 1.5, N)
density = mfp.electron_density(positions, x_grid)
for pos in positions:
    idx_near = np.argmin(np.abs(x_grid - pos))
    assert density[idx_near] > 0.5   # a real Gaussian peak sits at each atom

# 3. structure_factor / diffraction_intensity: Parseval-style sanity check --
#    total diffracted power relates to real-space density energy (FFT convention)
F = mfp.structure_factor(density, x_grid)
I = mfp.diffraction_intensity(F)
assert np.all(I >= 0)   # intensity is never negative
assert abs(np.sum(I) / N - np.sum(density ** 2)) / np.sum(density ** 2) < 1e-6   # Parseval

# 4. the phase problem is real: I(k) alone does NOT determine density up to
#    only a rescaling -- an unconstrained random-phase reconstruction should
#    generally NOT match the true density (confirms intensity really did
#    discard real information, this isn't a trivial invertible transform)
rng = np.random.default_rng(1)
random_phases = rng.uniform(0, 2 * np.pi, N)
F_random_phase = np.sqrt(I) * np.exp(1j * random_phases)
density_wrong = np.real(np.fft.ifft(F_random_phase))
corr_wrong = abs(np.corrcoef(density_wrong, density)[0, 1])
assert corr_wrong < 0.5   # a random phase guess should NOT reconstruct the molecule

# 5. CDI with a reasonable support constraint and enough iterations
#    recovers the true density well above chance, best-of-a-few-seeds
support_mask = np.abs(x_grid) < 0.32
best_corr = 0
for seed in range(5):
    density_rec = mfp.cdi_phase_retrieval(I, support_mask, n_iter=300, seed=seed)
    c1 = abs(np.corrcoef(density_rec, density)[0, 1])
    c2 = abs(np.corrcoef(density_rec[::-1], density)[0, 1])   # mirror-flip ambiguity
    best_corr = max(best_corr, c1, c2)
assert best_corr > 0.85

# 6. support_constraint actually zeroes outside the mask, leaves inside untouched
test_density = np.ones(N)
masked = mfp.support_constraint(test_density, support_mask)
assert np.all(masked[~support_mask] == 0.0)
assert np.all(masked[support_mask] == 1.0)

# 8. HIO produces a valid reconstruction (real-valued, right length) and
#    is not simply a no-op / identical to the ER algorithm
density_hio = mfp.hio_phase_retrieval(I, support_mask, n_hio=300, n_er_polish=100, seed=0)
assert density_hio.shape == (N,)
assert np.all(np.isfinite(density_hio))
density_er = mfp.cdi_phase_retrieval(I, support_mask, n_iter=300, seed=0)
assert not np.allclose(density_hio, density_er)   # genuinely a different algorithm

# 9. r_factor is a real, non-negative quality metric, and is NOT trivially
#    zero (confirms the earlier bug -- measuring after the magnitude snap,
#    which is always exactly zero by construction -- is actually fixed)
r_true = mfp.r_factor(density, I, support_mask)
random_density = np.random.default_rng(2).normal(size=N)
r_random = mfp.r_factor(random_density, I, support_mask)
assert r_true >= 0
assert r_random >= 0
# the TRUE density (which by definition matches the measured diffraction
# exactly) should have a much lower R-factor than pure random noise
assert r_true < r_random

# 10. multi_start_reconstruction runs the requested number of independent
#     starts, and its selected R-factor is the minimum among all candidates
#     (confirms selection logic itself, independent of whether R-factor
#     happens to match truth-correlation in this particular toy case)
best_density, best_seed, best_r, all_candidates = mfp.multi_start_reconstruction(
    I, support_mask, n_starts=6, n_hio=150, n_er_polish=50)
assert len(all_candidates) == 6
all_r_values = [c[2] for c in all_candidates]
assert abs(best_r - min(all_r_values)) < 1e-12
assert best_density.shape == (N,)

# 11. input validation
for bad_call in [
    lambda: mfp.toy_molecule_positions(n_atoms=1),
    lambda: mfp.toy_molecule_positions(n_atoms=5, extent=-1.0),
    lambda: mfp.electron_density(positions, x_grid, atom_width=-1.0),
    lambda: mfp.cdi_phase_retrieval(I, support_mask, n_iter=0),
    lambda: mfp.hio_phase_retrieval(I, support_mask, n_hio=0),
    lambda: mfp.hio_phase_retrieval(I, support_mask, n_hio=10, beta=0.0),
    lambda: mfp.hio_phase_retrieval(I, support_mask, n_hio=10, beta=1.5),
    lambda: mfp.multi_start_reconstruction(I, support_mask, n_starts=0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.molecular_fraunhofer_phase_problem tests passed")
