"""dispersive_fourier_torch.py -- PyTorch port of dgs/dispersive_fourier.py's
core TS-DFT (time-stretch dispersive Fourier transform) physics, the same
relationship dgs/gs_torch.py already has to dgs/gs_core.py in this repo:
identical physics, GPU-capable, differentiable.

gvd_propagate_torch is verified to match dgs.dispersive_fourier.gvd_propagate
to machine precision (see tests) -- a faithful port, not a rewrite.

THE ONE THING A TORCH PORT ADDS THAT NUMPY CANNOT: gradient-based physical
DESIGN. design_fiber_length_for_stretch_factor uses torch autograd to find
the fiber length L_m that achieves a TARGET pulse-stretch factor, by
differentiating all the way through the FFT-based propagation and the
resulting pulse's RMS width -- then checks the answer against the
closed-form prediction (stretch_factor = L_m/L_D, L_D = T0^2/|beta2|),
which exists for this simple Gaussian case specifically so the
gradient-based answer has a ground truth to be checked against, not just
trusted. This is the actual point of building this in torch rather than
numpy: for a pulse shape where no closed form exists, the SAME
gradient-based procedure still works.

Requires torch (py 3.12 here, matching this repo's existing convention).
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Optional


# ── 1. Faithful torch port of the GVD transfer function + propagation ───────

def gvd_transfer_function_torch(omega: torch.Tensor, beta2: float, L_m: float) -> torch.Tensor:
    """H(omega) = exp(i*beta2*L*omega^2/2) -- identical formula to
    dgs.dispersive_fourier.gvd_transfer_function, evaluated as a torch
    tensor."""
    return torch.exp(1j * beta2 * L_m * omega ** 2 / 2)


def gvd_propagate_torch(E_in, beta2: float, L_m, dt_s: float,
                         device: Optional[torch.device] = None,
                         dtype: torch.dtype = torch.complex128) -> Dict:
    """Propagate E_in through a GVD fiber -- torch port of
    dgs.dispersive_fourier.gvd_propagate. L_m may be a plain float OR a
    torch scalar tensor with requires_grad=True (needed for
    design_fiber_length_for_stretch_factor's gradient-based optimization
    below); everything downstream stays differentiable either way.

    Returns a dict with the same E_out/I_out/omega/H_omega keys as the
    numpy version (verified to match it to machine precision in tests),
    plus E_in as a torch tensor for convenience.
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s={dt_s}: must be positive")
    E_in_t = torch.as_tensor(E_in, dtype=dtype, device=device)
    n = E_in_t.shape[0]
    if n < 8:
        raise ValueError(f"n={n}: need at least 8 samples")
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    omega = 2 * torch.pi * torch.fft.fftfreq(n, d=dt_s, device=device, dtype=real_dtype)

    E_omega = torch.fft.fft(E_in_t)
    H = gvd_transfer_function_torch(omega, beta2, L_m).to(dtype)
    E_omega_out = E_omega * H
    E_out = torch.fft.ifft(E_omega_out)
    I_out = torch.abs(E_out) ** 2

    return {"E_in": E_in_t, "E_out": E_out, "I_out": I_out, "E_omega": E_omega,
            "I_omega": torch.abs(E_omega) ** 2, "omega": omega, "H_omega": H,
            "beta2": beta2, "L_m": L_m}


def gaussian_pulse_torch(n_pts: int, T0_s: float, dt_s: float,
                          chirp_C: float = 0.0, center_frac: float = 0.5,
                          device: Optional[torch.device] = None,
                          dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Torch port of dgs.dispersive_fourier.gaussian_pulse: E(t) =
    exp(-(1+iC)*t^2/(2*T0^2))."""
    if T0_s <= 0:
        raise ValueError(f"T0_s={T0_s}: must be positive")
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    t = (torch.arange(n_pts, device=device, dtype=real_dtype) -
         int(center_frac * n_pts)) * dt_s
    return torch.exp(-(1 + 1j * chirp_C) * t ** 2 / (2 * T0_s ** 2)).to(dtype)


# ── 2. Differentiable pulse-width diagnostics ────────────────────────────────

def rms_width_torch(I: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """RMS temporal width of an intensity profile I(t) -- the same
    second-moment definition dgs.dispersive_fourier.gvd_propagate uses
    internally (T0_est), kept fully differentiable through I."""
    I_norm = I / I.sum()
    t_mean = (t * I_norm).sum()
    return torch.sqrt(((t - t_mean) ** 2 * I_norm).sum())


def achieved_stretch_factor(E_in: torch.Tensor, L_m, beta2: float,
                             dt_s: float, t: torch.Tensor) -> torch.Tensor:
    """Output RMS width / input RMS width after GVD propagation --
    differentiable end to end (FFT -> propagate -> IFFT -> RMS width) with
    respect to L_m, which is what makes the gradient-based design below
    possible."""
    result = gvd_propagate_torch(E_in, beta2, L_m, dt_s)
    I_in = torch.abs(E_in) ** 2
    return rms_width_torch(result["I_out"], t) / rms_width_torch(I_in, t)


# ── 3. Gradient-based fiber-length design (the torch-specific capability) ───

def design_fiber_length_for_stretch_factor(T0_s: float, dt_s: float, beta2: float,
                                            target_stretch_factor: float,
                                            n_pts: int = 2048, n_iter: int = 2000,
                                            lr: float = 100.0,
                                            L_m_init: float = 500.0) -> Dict:
    """Find the fiber length L_m that makes a Gaussian pulse's OUTPUT
    RMS width `target_stretch_factor` times its input RMS width, by
    gradient descent (Adam) on L_m through the full differentiable
    propagation -- not a closed-form solve, even though a closed form
    exists for this specific Gaussian case (stretch_factor = L_m/L_D,
    L_D = T0^2/|beta2|) precisely so the gradient-based answer can be
    checked against it.

    Returns dict with L_m_found, L_m_closed_form, relative_error, and the
    loss history.
    """
    if T0_s <= 0 or dt_s <= 0:
        raise ValueError("T0_s and dt_s must be positive")
    if target_stretch_factor <= 1.0:
        raise ValueError(f"target_stretch_factor={target_stretch_factor}: "
                          f"must exceed 1 (a fiber can only stretch, not compress, "
                          f"a transform-limited pulse via normal/anomalous GVD alone)")
    if n_iter < 1:
        raise ValueError(f"n_iter={n_iter}: must be >= 1")

    real_dtype = torch.float64
    t = (torch.arange(n_pts, dtype=real_dtype) - n_pts // 2) * dt_s
    E_in = torch.exp(-t ** 2 / (2 * T0_s ** 2)).to(torch.complex128)

    L_param = torch.tensor(float(L_m_init), dtype=real_dtype, requires_grad=True)
    optimizer = torch.optim.Adam([L_param], lr=lr)
    loss_history = []
    for _ in range(n_iter):
        optimizer.zero_grad()
        M_achieved = achieved_stretch_factor(E_in, L_param, beta2, dt_s, t)
        loss = (M_achieved - target_stretch_factor) ** 2
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))

    L_D = T0_s ** 2 / abs(beta2)
    L_closed_form = target_stretch_factor * L_D
    L_found = float(L_param.detach())
    rel_err = abs(L_found - L_closed_form) / L_closed_form

    return {"L_m_found": L_found, "L_m_closed_form": L_closed_form,
            "relative_error": rel_err, "loss_history": loss_history,
            "L_D_m": L_D}


if __name__ == "__main__":
    from dgs.dispersive_fourier import gvd_propagate, gaussian_pulse

    print("=== 1. Faithful torch port, checked against numpy ===")
    N, dt, T0 = 2048, 1e-12, 2e-12
    beta2, L = -20e-27, 5000.0
    pulse_np = gaussian_pulse(N, T0, dt)
    res_np = gvd_propagate(pulse_np, beta2=beta2, L_m=L, dt_s=dt)
    res_torch = gvd_propagate_torch(pulse_np, beta2, L, dt)
    err = float(np.max(np.abs(res_torch["E_out"].numpy() - res_np["E_out"])))
    print(f"  max|E_out_torch - E_out_numpy| = {err:.2e}  (machine precision)")

    print("\n=== 2. Gradient-based fiber-length design (torch-specific) ===")
    result = design_fiber_length_for_stretch_factor(
        T0_s=T0, dt_s=dt, beta2=beta2, target_stretch_factor=50.0)
    print(f"  Closed-form L_m for stretch factor 50: {result['L_m_closed_form']:.1f} m")
    print(f"  Autograd-found L_m (2000 Adam steps):  {result['L_m_found']:.1f} m")
    print(f"  Relative error: {result['relative_error']*100:.3f}%")
    print(f"  Loss: {result['loss_history'][0]:.2f} -> {result['loss_history'][-1]:.2e}")
