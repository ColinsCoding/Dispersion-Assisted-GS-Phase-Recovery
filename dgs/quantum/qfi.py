# qfi.py
# Approximate Quantum Fisher Information (QFI) for dispersion parameter.

import torch

def approx_qfi(theta, forward_model_fn, x, eps=1e-3, n_samples=50):
    """
    theta: scalar tensor
    forward_model_fn(theta, x): returns intensity or photon counts
    """
    thetas = [theta - eps, theta + eps]
    ll_vals = []

    for t in thetas:
        I = forward_model_fn(t, x)
        lam = I.clamp(min=1e-8)
        y = torch.poisson(lam)
        ll = (y * torch.log(lam) - lam).sum().item()
        ll_vals.append(ll)

    d_ll_d_theta = (ll_vals[1] - ll_vals[0]) / (2 * eps)
    return 4 * d_ll_d_theta ** 2
