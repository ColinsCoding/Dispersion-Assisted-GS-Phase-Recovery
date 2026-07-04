"""Griffiths Problem 1.14 (the gradient-transforms-as-a-vector proof),
verified THREE independent ways for a real, non-trivial f(y,z): SymPy
(symbolic, exact), and PyTorch autograd TWICE (once for the ordinary
gradient in the (y,z) frame, once by composing the rotation directly into
the computational graph and differentiating through it in one shot).

The key point this answers: torch is never TOLD the transformation law
(grad f)_ybar = cos(phi)*(grad f)_y + sin(phi)*(grad f)_z -- it only ever
sees y = ybar*cos(phi) - zbar*sin(phi) and z = ybar*sin(phi) + zbar*cos(phi)
as ordinary elementwise tensor operations, composed with f. Backprop's
chain rule re-derives the cos/sin coefficients on its own, as the LOCAL
JACOBIAN of those operations, and the result matches SymPy's symbolic
formula exactly. The cos/sin "come back" because they are literally
dy/dybar, dz/dybar etc. baked into the autograd graph -- not because
anyone reminded torch about Griffiths 1.14.

Everything here is ELEMENT-WISE and BATCHED: a whole array of (y,z) points
is evaluated in one vectorized pass (torch.sin, multiplication, powers all
act elementwise), and `f_vals.sum().backward()` recovers each point's OWN
gradient correctly -- valid because f is applied independently per point,
so d(sum_i f_i)/dy_j = df_j/dy_j with zero cross-contamination between
points, letting autograd batch N independent gradient computations without
a Python loop.
"""

import numpy as np
import sympy as sp
import torch


def f_numpy_sympy(y, z):
    """The test function, in SymPy-compatible form: f(y,z) = y^2*z + sin(y*z).
    Rich enough that (df/dy, df/dz) actually vary with position."""
    return y ** 2 * z + sp.sin(y * z)


def f_torch(y, z):
    """The SAME function, built from torch's elementwise ops, so it can be
    evaluated on a whole batched tensor of points at once."""
    return y ** 2 * z + torch.sin(y * z)


def sympy_gradient_and_transform(y0, z0, phi0):
    """Analytical ground truth: symbolic (df/dy, df/dz) at (y0,z0), and the
    symbolic transformation-law prediction for (df/dybar, df/dzbar)."""
    y, z = sp.symbols("y z", real=True)
    f = f_numpy_sympy(y, z)
    fy_expr, fz_expr = sp.diff(f, y), sp.diff(f, z)
    fy = float(fy_expr.subs({y: y0, z: z0}))
    fz = float(fz_expr.subs({y: y0, z: z0}))
    c, s = np.cos(phi0), np.sin(phi0)
    f_ybar_predicted = c * fy + s * fz
    f_zbar_predicted = -s * fy + c * fz
    return fy, fz, f_ybar_predicted, f_zbar_predicted


def torch_gradient_original_frame(y_pts, z_pts):
    """Ordinary (y,z)-frame gradient via autograd, BATCHED across N points
    in one vectorized pass: f_vals.sum().backward() gives each point its
    own df/dy, df/dz because f is elementwise (no point depends on another
    point's inputs)."""
    y = y_pts.clone().detach().requires_grad_(True)
    z = z_pts.clone().detach().requires_grad_(True)
    f_vals = f_torch(y, z)
    f_vals.sum().backward()
    return y.grad.clone(), z.grad.clone()


def torch_gradient_rotated_frame_direct(ybar_pts, zbar_pts, phi):
    """The rotated-frame gradient, computed by autograd DIRECTLY: compose
    the inverse-rotation formulas (ordinary elementwise tensor ops) with f,
    and differentiate the WHOLE graph at once. Torch is never given the
    trig transformation law -- only the coordinate formulas themselves;
    the cos/sin coefficients reappear automatically as the local Jacobian
    of those formulas, discovered by backprop's chain rule."""
    ybar = ybar_pts.clone().detach().requires_grad_(True)
    zbar = zbar_pts.clone().detach().requires_grad_(True)
    phi_t = torch.as_tensor(phi, dtype=ybar_pts.dtype)   # match precision, not float32 by default
    c, s = torch.cos(phi_t), torch.sin(phi_t)
    y = ybar * c - zbar * s     # y(ybar,zbar): plain elementwise ops, no chain-rule formula given
    z = ybar * s + zbar * c     # z(ybar,zbar): same
    f_vals = f_torch(y, z)
    f_vals.sum().backward()
    return ybar.grad.clone(), zbar.grad.clone()


if __name__ == "__main__":
    torch.manual_seed(0)
    phi = 0.7  # radians, an arbitrary rotation angle

    n_pts = 5
    y0 = torch.linspace(0.5, 2.5, n_pts, dtype=torch.float64)
    z0 = torch.linspace(-1.0, 1.5, n_pts, dtype=torch.float64)

    print(f"phi = {phi} rad, {n_pts} points, batched (no Python loop over points)\n")

    # 1. torch autograd, ORIGINAL frame
    fy_torch, fz_torch = torch_gradient_original_frame(y0, z0)

    # 2. this batch of (y,z) points, expressed in the ROTATED frame's coordinates
    c, s = np.cos(phi), np.sin(phi)
    ybar0 = y0 * c + z0 * s
    zbar0 = -y0 * s + z0 * c

    # 3. torch autograd, ROTATED frame, computed DIRECTLY (no formula given)
    fybar_torch_direct, fzbar_torch_direct = torch_gradient_rotated_frame_direct(ybar0, zbar0, phi)

    # 4. the transformation-law PREDICTION, built only from step 1's numbers
    fybar_predicted = c * fy_torch + s * fz_torch
    fzbar_predicted = -s * fy_torch + c * fz_torch

    print(f"{'y0':>8} {'z0':>8} | {'(gradf)_y':>12} {'(gradf)_z':>12} | "
          f"{'ybar-direct':>12} {'ybar-pred':>12} | {'zbar-direct':>12} {'zbar-pred':>12}")
    for i in range(n_pts):
        print(f"{y0[i]:8.3f} {z0[i]:8.3f} | {fy_torch[i]:12.6f} {fz_torch[i]:12.6f} | "
              f"{fybar_torch_direct[i]:12.6f} {fybar_predicted[i]:12.6f} | "
              f"{fzbar_torch_direct[i]:12.6f} {fzbar_predicted[i]:12.6f}")

    max_err_ybar = (fybar_torch_direct - fybar_predicted).abs().max().item()
    max_err_zbar = (fzbar_torch_direct - fzbar_predicted).abs().max().item()
    print(f"\nmax |direct - predicted| over the batch: ybar={max_err_ybar:.2e}, zbar={max_err_zbar:.2e}")

    # 5. cross-check the first point against SymPy's independent symbolic derivation
    fy_sym, fz_sym, fybar_sym, fzbar_sym = sympy_gradient_and_transform(
        float(y0[0]), float(z0[0]), phi)
    print(f"\nSymPy (point 0): (gradf)_y={fy_sym:.6f}, (gradf)_z={fz_sym:.6f}, "
          f"predicted ybar={fybar_sym:.6f}, zbar={fzbar_sym:.6f}")
    print(f"torch  (point 0): (gradf)_y={fy_torch[0]:.6f}, (gradf)_z={fz_torch[0]:.6f}, "
          f"direct   ybar={fybar_torch_direct[0]:.6f}, zbar={fzbar_torch_direct[0]:.6f}")
