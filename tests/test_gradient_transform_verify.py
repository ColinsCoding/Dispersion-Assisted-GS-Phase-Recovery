"""Test Griffiths 1.14's gradient-transforms-as-a-vector result three
independent ways: SymPy (symbolic), torch autograd in the original frame
+ the transformation-law formula, and torch autograd applied DIRECTLY in
the rotated frame (no formula given to torch at all). Requires py-3.12
(torch is not available on py-3.13 in this environment)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs.torch import gradient_transform_verify as gtv

phi = 0.7
n_pts = 6
y0 = torch.linspace(0.3, 2.7, n_pts, dtype=torch.float64)
z0 = torch.linspace(-1.2, 1.8, n_pts, dtype=torch.float64)

# 1. torch autograd, original (y,z) frame, batched
fy_torch, fz_torch = gtv.torch_gradient_original_frame(y0, z0)
assert fy_torch.shape == (n_pts,)

# 2. rotate this batch of points into the barred frame's coordinates
c, s = np.cos(phi), np.sin(phi)
ybar0 = y0 * c + z0 * s
zbar0 = -y0 * s + z0 * c

# 3. torch autograd, rotated frame, computed DIRECTLY through the composed graph
fybar_direct, fzbar_direct = gtv.torch_gradient_rotated_frame_direct(ybar0, zbar0, phi)

# 4. the Griffiths 1.14 transformation-law prediction, built only from step 1
fybar_predicted = c * fy_torch + s * fz_torch
fzbar_predicted = -s * fy_torch + c * fz_torch

# torch never saw this formula -- its OWN autograd result through the rotated
# graph must match it to near machine precision, across the WHOLE batch
assert torch.max(torch.abs(fybar_direct - fybar_predicted)) < 1e-10
assert torch.max(torch.abs(fzbar_direct - fzbar_predicted)) < 1e-10

# 5. cross-check every point (not just one) against SymPy's independent
#    symbolic derivation
for i in range(n_pts):
    fy_sym, fz_sym, fybar_sym, fzbar_sym = gtv.sympy_gradient_and_transform(
        float(y0[i]), float(z0[i]), phi)
    assert abs(fy_sym - fy_torch[i].item()) < 1e-8
    assert abs(fz_sym - fz_torch[i].item()) < 1e-8
    assert abs(fybar_sym - fybar_direct[i].item()) < 1e-8
    assert abs(fzbar_sym - fzbar_direct[i].item()) < 1e-8

# 6. batching sanity check: each point's gradient must depend ONLY on its
#    own (y,z), not on neighboring points in the batch -- verify by comparing
#    the batched result to looping one point at a time
for i in range(n_pts):
    fy_single, fz_single = gtv.torch_gradient_original_frame(y0[i:i+1], z0[i:i+1])
    assert abs(fy_single.item() - fy_torch[i].item()) < 1e-12
    assert abs(fz_single.item() - fz_torch[i].item()) < 1e-12

print("all dgs.torch.gradient_transform_verify tests passed")
