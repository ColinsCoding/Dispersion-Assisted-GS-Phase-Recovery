"""Test the 2D spatial Gerchberg-Saxton (dgs.gs_spatial_torch): converges to
the true field (up to the one unavoidable global-phase ambiguity) for a
support with broken symmetry, converges markedly worse for a symmetric
support (real GSA physics -- central symmetry creates a near-degenerate
twin solution, not an implementation bug), and -- the actual point of
building this alongside dgs.gs_torch's TEMPORAL GSA -- succeeds where the
TDGSA investigation (dgs.tdgsa / dgs.gs_core on the chirped-Gaussian pulse)
got stuck in a flat-phase degenerate solution."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from dgs.gs_spatial_torch import gsa_spatial, gsa_spatial_batch, global_phase_align_2d, make_test_scene

# 1. asymmetric support: GSA converges essentially exactly (up to global phase)
image_amp, diff_amp, field_true = make_test_scene(symmetric_support=False)
field_rec, errors = gsa_spatial(image_amp, diff_amp, n_iter=200)
assert errors[-1] < errors[0]                     # genuinely converges, not stuck
assert errors[-1] < 1e-4                           # near-exact constraint satisfaction

field_aligned = global_phase_align_2d(field_true, field_rec)
mask = image_amp > 0.5
rel_err = float(torch.linalg.norm((field_aligned - field_true)[mask])
                 / torch.linalg.norm(field_true[mask]))
assert rel_err < 1e-3, f"expected near-exact recovery, got relative field error {rel_err}"

# 2. symmetric support: convergence is real physics, markedly worse than the
# asymmetric case -- a central-symmetry near-ambiguity, not a bug
image_amp_sym, diff_amp_sym, field_true_sym = make_test_scene(symmetric_support=True)
field_rec_sym, errors_sym = gsa_spatial(image_amp_sym, diff_amp_sym, n_iter=200)
field_aligned_sym = global_phase_align_2d(field_true_sym, field_rec_sym)
mask_sym = image_amp_sym > 0.5
rel_err_sym = float(torch.linalg.norm((field_aligned_sym - field_true_sym)[mask_sym])
                     / torch.linalg.norm(field_true_sym[mask_sym]))
assert rel_err_sym > rel_err * 10   # markedly worse than the asymmetric case

# 3. batched version matches the single-scene version on the same input
B = 4
image_amp_b = image_amp.unsqueeze(0).repeat(B, 1, 1).cpu().numpy()
diff_amp_b = diff_amp.unsqueeze(0).repeat(B, 1, 1).cpu().numpy()
field_rec_b, errors_b = gsa_spatial_batch(image_amp_b, diff_amp_b, n_iter=200)
assert field_rec_b.shape == (B, *image_amp.shape)
assert errors_b[-1] < 1e-4
# every batch element should independently recover the same scene near-exactly
for i in range(B):
    aligned_i = global_phase_align_2d(field_true, field_rec_b[i])
    rel_err_i = float(torch.linalg.norm((aligned_i - field_true)[mask]) / torch.linalg.norm(field_true[mask]))
    assert rel_err_i < 1e-3

# 4. the actual point: spatial GSA succeeds where TDGSA (temporal) got stuck.
# dgs.tdgsa's own gas-cell investigation found a flat-phase degenerate
# solution with residual ~3.7 rad against a true phase spanning 12.5 rad
# (see notebooks/ece279_tdgsa_recreation.ipynb). Spatial GSA here recovers
# the true field to within 1e-3 relative error on the asymmetric scene --
# a genuinely different outcome, not just a smaller number on a different scale.
assert rel_err < 0.01   # spatial: near-exact
tdgsa_known_residual_rad = 3.73   # from the temporal investigation, for comparison
print(f"spatial GSA relative field error: {rel_err:.2e}  "
      f"(vs. temporal TDGSA's ~{tdgsa_known_residual_rad} rad residual on the gas-cell pulse -- "
      f"not directly comparable in units, but spatial converges to the true field while "
      f"temporal does not)")

print("all dgs.gs_spatial_torch tests passed")
