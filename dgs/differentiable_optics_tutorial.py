"""Differentiable optics, taught progressively in torch.

"Differentiable optics" means: write the forward optical model (propagation,
a phase mask, a detector) as ordinary torch tensor ops, and let autograd hand
you d(loss)/d(every parameter) for free -- then optimize with gradient
descent instead of (or alongside) a hand-derived algorithm like GS.

Four steps, each building on the last:

  1. Scalar autograd -- the mechanism, on a function simple enough to check
     the gradient by hand (y = x^2, dy/dx = 2x).
  2. A differentiable forward optical model -- a phase mask followed by
     Fraunhofer (FFT) propagation, exactly the operation gs_torch.py's GS
     algorithm also uses, but here just *executed forward* with grad tracking.
  3. Phase retrieval by gradient descent -- optimize a phase guess directly
     against a measured-intensity loss, the "differentiable optics" approach,
     as an alternative to the alternating-projections GS algorithm.
  4. A head-to-head comparison against dgs.gs_torch.retrieve_phase on the same
     synthetic problem -- two different algorithms solving the identical
     inverse problem, so their answers (and their failure modes) can be
     compared directly.
"""

import numpy as np
import torch


def step1_scalar_autograd():
    """y = x^2 at x=3: autograd should give dy/dx = 2x = 6 exactly."""
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    analytic = 2 * x.item()
    return {"x": x.item(), "y": y.item(), "dy_dx_autograd": x.grad.item(), "dy_dx_analytic": analytic}


def forward_optical_model(phase, amplitude, D):
    """Phase mask -> dispersive (Fraunhofer-style) propagation -> intensity.

    field = amplitude * exp(i*phase)        (apply the phase mask)
    out   = ifft( fft(field) * exp(i*pi*D*nu^2) )   (propagate through dispersion D)
    returns |out|^2 -- what a detector actually measures. Every op here is a
    standard differentiable torch op, so d(intensity)/d(phase) exists for free.
    """
    field = amplitude * torch.exp(1j * phase)
    N = field.shape[-1]
    nu = torch.fft.fftfreq(N, device=phase.device)
    H = torch.exp(1j * np.pi * D * nu ** 2)
    out = torch.fft.ifft(torch.fft.fft(field) * H)
    return torch.abs(out) ** 2


def step2_forward_model_is_differentiable(N=64, D=500.0, seed=0):
    """Confirm gradients flow end-to-end through the forward optical model.

    Note: the *total* output energy sum(intensity) is phase-invariant here --
    the phase mask and the dispersive propagator are both unitary (Parseval),
    so d(sum intensity)/d(phase) is identically ~0. That's correct physics,
    not a broken gradient, but it's a bad demo. Instead probe the gradient of
    intensity at ONE output point -- "how would nudging the phase mask change
    how much light lands at this pixel" is exactly the differentiable-optics
    question (hologram/lens design), and it has a genuinely nonzero gradient.
    """
    rng = np.random.default_rng(seed)
    amplitude = torch.ones(N)
    phase = torch.tensor(rng.uniform(-np.pi, np.pi, N), requires_grad=True)

    intensity = forward_optical_model(phase, amplitude, D)
    target_pixel = N // 2
    loss = intensity[target_pixel]
    loss.backward()

    total_energy = forward_optical_model(phase.detach(), amplitude, D).sum()

    return {
        "loss_at_target_pixel": loss.item(),
        "grad_norm": float(torch.linalg.norm(phase.grad)),
        "grad_is_finite": bool(torch.isfinite(phase.grad).all()),
        "total_energy_for_reference": total_energy.item(),
    }


def step2b_design_a_focusing_phase_mask(N=64, D=500.0, target_pixel=None, n_iter=300, lr=0.1, seed=0):
    """The actual differentiable-optics use case: optimize the phase mask
    itself (not recover one from data) so that the propagated intensity
    concentrates at `target_pixel` -- i.e. design a simple hologram/lens by
    gradient ascent on the intensity at that point. Returns the optimized
    phase and how much the target pixel's intensity share improved."""
    target_pixel = N // 2 if target_pixel is None else target_pixel
    rng = np.random.default_rng(seed)
    amplitude = torch.ones(N)
    phase = torch.tensor(rng.uniform(-np.pi, np.pi, N), dtype=torch.float32, requires_grad=True)

    intensity0 = forward_optical_model(phase.detach(), amplitude, D)
    initial_share = (intensity0[target_pixel] / intensity0.sum()).item()

    opt = torch.optim.Adam([phase], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        intensity = forward_optical_model(phase, amplitude, D)
        loss = -intensity[target_pixel] / intensity.sum()   # maximize the target's energy share
        loss.backward()
        opt.step()

    intensity_final = forward_optical_model(phase.detach(), amplitude, D)
    final_share = (intensity_final[target_pixel] / intensity_final.sum()).item()

    return {
        "initial_share_at_target": initial_share,
        "final_share_at_target": final_share,
        "phase": phase.detach().numpy(),
    }


def step3_gradient_descent_phase_retrieval(I1, I2, D1, D2, n_iter=500, lr=0.05, seed=0,
                                            device=None):
    """Recover phase from two measured intensities by directly minimizing

        loss(phase) = MSE(|model(phase, D1)|^2, I1) + MSE(|model(phase, D2)|^2, I2)

    with gradient descent on `phase` -- the "differentiable optics" approach:
    no alternating projections, just autograd + an optimizer. Returns the
    recovered phase and the loss history.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N = min(len(I1), len(I2))
    t1 = torch.tensor(I1[:N], dtype=torch.float32, device=device)
    t2 = torch.tensor(I2[:N], dtype=torch.float32, device=device)
    amplitude = torch.ones(N, device=device)

    rng = np.random.default_rng(seed)
    phase = torch.tensor(rng.uniform(-np.pi, np.pi, N), dtype=torch.float32,
                          device=device, requires_grad=True)

    opt = torch.optim.Adam([phase], lr=lr)
    history = []
    for _ in range(n_iter):
        opt.zero_grad()
        i1_est = forward_optical_model(phase, amplitude, D1)
        i2_est = forward_optical_model(phase, amplitude, D2)
        loss = torch.mean((i1_est - t1) ** 2) + torch.mean((i2_est - t2) ** 2)
        loss.backward()
        opt.step()
        history.append(loss.item())

    return phase.detach().cpu().numpy(), history


def step4_compare_to_gs(N=128, D1=0.0, D2=800.0, n_iter_gs=40, n_iter_gd=600, seed=0):
    """Generate a synthetic ground-truth field, measure it at two dispersions,
    recover the phase BOTH ways (GS and gradient descent), and compare each
    recovery's reconstructed intensity against the measurements (not against
    the hidden ground truth -- consistent with this repo's verification
    philosophy: judge by physical self-consistency)."""
    from dgs import gs_torch

    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, N)
    amp_true = np.exp(-x ** 2 / (2 * 0.3 ** 2))
    phase_true = 8.0 * x ** 2 + 0.5 * rng.standard_normal(N)
    E_true = amp_true * np.exp(1j * phase_true)

    nu = np.fft.fftfreq(N)
    H1 = np.exp(1j * np.pi * D1 * nu ** 2)
    H2 = np.exp(1j * np.pi * D2 * nu ** 2)
    I1 = np.abs(np.fft.ifft(np.fft.fft(E_true) * H1)) ** 2
    I2 = np.abs(np.fft.ifft(np.fft.fft(E_true) * H2)) ** 2

    phi_gs, errors_gs = gs_torch.retrieve_phase(I1, I2, D1, D2, n_iter=n_iter_gs)
    phi_gd, errors_gd = step3_gradient_descent_phase_retrieval(I1, I2, D1, D2, n_iter=n_iter_gd, seed=seed)

    def reconstruction_mse(phi):
        E_rec = amp_true * np.exp(1j * phi)
        I2_rec = np.abs(np.fft.ifft(np.fft.fft(E_rec) * H2)) ** 2
        return float(np.mean((I2_rec - I2) ** 2))

    return {
        "gs_final_error": errors_gs[-1],
        "gs_reconstruction_mse": reconstruction_mse(phi_gs),
        "gd_final_loss": errors_gd[-1],
        "gd_reconstruction_mse": reconstruction_mse(phi_gd),
        "phi_gs": phi_gs,
        "phi_gd": phi_gd,
        "I1": I1,
        "I2": I2,
    }


if __name__ == "__main__":
    print("Step 1 -- scalar autograd:", step1_scalar_autograd())
    print("Step 2 -- forward model differentiable:", step2_forward_model_is_differentiable())
    focus = step2b_design_a_focusing_phase_mask()
    print(f"Step 2b -- focusing mask: target pixel energy share {focus['initial_share_at_target']:.4f} -> {focus['final_share_at_target']:.4f}")
    cmp = step4_compare_to_gs()
    print(f"Step 4 -- GS reconstruction MSE:               {cmp['gs_reconstruction_mse']:.3e}")
    print(f"Step 4 -- gradient-descent reconstruction MSE: {cmp['gd_reconstruction_mse']:.3e}")
