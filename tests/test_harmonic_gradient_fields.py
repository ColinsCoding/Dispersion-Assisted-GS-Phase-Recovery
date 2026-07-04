"""Test Problem 1.20 formalized as linear algebra (Jacobian trace =
divergence, antisymmetric part = curl) via torch autograd, and the
constructive theorem that v=grad(f) has zero div+curl for any harmonic f.
Also confirms the real textbook-example discrepancy found in this session:
one transcribed example has nonzero divergence, caught independently by
both SymPy and torch. Requires py-3.12 (torch)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from dgs.torch import harmonic_gradient_fields as hgf

point = [1.3, -0.7, 0.5]

# 1. the three CORRECT textbook examples: zero div, zero curl
correct_examples = {
    "v=(y,x,0)": lambda p: torch.stack([p[1], p[0], torch.zeros_like(p[0])]),
    "v=(yz,xz,xy)": lambda p: torch.stack([p[1]*p[2], p[0]*p[2], p[0]*p[1]]),
    "v=(sin(x)cosh(y),-cos(x)sinh(y),0)": lambda p: torch.stack([
        torch.sin(p[0])*torch.cosh(p[1]), -torch.cos(p[0])*torch.sinh(p[1]), torch.zeros_like(p[0])]),
}
for name, v_func in correct_examples.items():
    div, curl = hgf.verify_vector_field_torch(v_func, point)
    assert abs(div.item()) < 1e-8, name
    assert torch.norm(curl).item() < 1e-8, name

# 2. the SUSPECT example: curl is genuinely zero, but divergence is NOT --
#    confirming the real discrepancy found via SymPy independently, via torch
suspect = lambda p: torch.stack([3*p[0]**2*p[2] - p[2]**3, 3*p[1], p[0]**3 - 3*p[0]*p[2]**2])
div_suspect, curl_suspect = hgf.verify_vector_field_torch(suspect, point)
assert torch.norm(curl_suspect).item() < 1e-8   # curl really is zero
assert abs(div_suspect.item() - 3.0) < 1e-8      # but divergence is exactly 3, not 0

# 3. the constructive theorem: v=grad(f) has zero div AND zero curl for
#    several DIFFERENT harmonic functions -- not a single lucky case
harmonic_fs = {
    "x^2-y^2": lambda p: p[0]**2 - p[1]**2,
    "x^3-3xy^2": lambda p: p[0]**3 - 3*p[0]*p[1]**2,
    "exp(x)sin(y)": lambda p: torch.exp(p[0])*torch.sin(p[1]),
    "xyz": lambda p: p[0]*p[1]*p[2],
}
for name, f_func in harmonic_fs.items():
    lap = hgf.laplacian_torch(f_func, point)
    assert abs(lap.item()) < 1e-8, name

    v_from_f = lambda p, f=f_func: hgf.gradient_field_from_scalar(f, p)
    div, curl = hgf.verify_vector_field_torch(v_from_f, point)
    assert abs(div.item()) < 1e-6, name
    assert torch.norm(curl).item() < 1e-6, name

# 4. a NON-harmonic scalar's gradient should have zero curl (always true)
#    but NONZERO divergence (since it's not harmonic) -- confirms the
#    theorem's harmonic condition is doing real work, not vacuous
non_harmonic_f = lambda p: p[0]**2 + p[1]**2 + p[2]**2   # laplacian = 6, not 0
lap_non_harmonic = hgf.laplacian_torch(non_harmonic_f, point)
assert abs(lap_non_harmonic.item() - 6.0) < 1e-8
v_non_harmonic = lambda p: hgf.gradient_field_from_scalar(non_harmonic_f, p)
div_nh, curl_nh = hgf.verify_vector_field_torch(v_non_harmonic, point)
assert torch.norm(curl_nh).item() < 1e-8         # curl(grad) is STILL zero (always true)
assert abs(div_nh.item() - 6.0) < 1e-6            # but divergence is nonzero here

print("all dgs.torch.harmonic_gradient_fields tests passed")
