"""
gs_diff.py -- differentiable Gerchberg-Saxton in PyTorch
=========================================================

Standard GS uses np.angle() -- not differentiable at origin,
and alternating projections have no gradient flow between iterations.

This module provides three differentiable variants:

1. gs_unrolled()      : unroll N GS iterations as a torch compute graph
                        backprop flows through all N steps
                        use to jointly optimize D1, D2, or n_iter

2. gs_soft_proj()     : replace hard projection with soft (smooth) version
                        E / (|E| + eps) instead of exp(i*angle(E))

3. phase_loss_*()     : differentiable loss functions for phase recovery
                        wrapped_phase_loss: 2*(1-cos(phi_hat - phi_true))
                        intensity_loss:     ||H*E|^2 - I||^2

All functions run on GPU if available.
"""

import torch
import torch.nn.functional as F
import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

def disperse_torch(E_complex, D, N):
    """
    Apply GVD dispersion D to complex field E (torch, differentiable).

    E_complex : (batch, N, 2) real tensor [real, imag]  OR
                (batch, N)    complex tensor
    D         : float or scalar tensor
    Returns   : same shape as input
    """
    if not E_complex.is_complex():
        E = torch.view_as_complex(E_complex.contiguous())
    else:
        E = E_complex

    nu = torch.fft.fftfreq(N, device=E.device, dtype=torch.float32)
    H  = torch.exp(1j * np.pi * D * nu**2)
    out = torch.fft.ifft(torch.fft.fft(E, dim=-1) * H, dim=-1)

    if not E_complex.is_complex():
        return torch.view_as_real(out)
    return out


def soft_proj(E, eps=1e-6):
    """
    Differentiable unit-amplitude projection.
    Hard: exp(i*angle(E)) = E / |E|
    Soft: E / (|E| + eps)   <- smooth at origin
    """
    mag = torch.abs(E) + eps
    return E / mag


def hard_proj(E):
    """Standard GS projection (not differentiable at origin)."""
    return torch.exp(1j * torch.angle(E))


# ── differentiable GS variants ────────────────────────────────────────────────

def gs_unrolled(I1, I2, D1, D2, n_iter=10, unit_amplitude=True, soft=False):
    """
    Unrolled GS: n_iter iterations as a differentiable torch graph.

    Structure matches dgs.gs_core.gs_iteration/retrieve_phase exactly: the
    measured-intensity amplitude constraint (sqrt(I) * phase-of-estimate)
    is applied UNCONDITIONALLY in the dispersed domain every iteration;
    unit_amplitude only adds an EXTRA |E(t)|=1 projection in the
    undispersed domain afterward, for constant-envelope formats (QPSK/
    DPSK). A previous version of this function made the I1/I2 constraint
    itself conditional on unit_amplitude (skipping it entirely when True),
    which meant the measured data was never actually used for the common
    QPSK case -- confirmed via regression test against dgs.gs_core's
    known-good numpy retrieve_phase on identical synthetic measurements.

    Parameters
    ----------
    I1, I2   : (N,) float tensors, measured intensities
    D1, D2   : float, dispersion values
    n_iter   : int, number of unrolled iterations (keep <= 20 for memory)
    soft     : bool, use soft projection (differentiable everywhere)

    Returns
    -------
    E        : (N,) complex tensor with grad_fn
    errors   : list of per-iteration intensity errors
    """
    N   = I1.shape[-1]
    dev = I1.device

    # Initial guess matches dgs.gs_core.retrieve_phase: sqrt(I1) with ZERO
    # phase, undispersed through D1 -- not a random phase guess. This is
    # the "reasonable choice ... from a chirped pulse of the proper
    # bandwidth" the original Solli/Gupta/Jalali paper describes, and it
    # matters: GS is not fully init-robust, and a random start converges
    # slower (or to a worse local solution) than this physically-motivated one.
    E = disperse_torch(torch.sqrt(I1.clamp(min=0)).to(torch.complex64).to(dev), -D1, N)

    proj = soft_proj if soft else hard_proj
    errors = []

    for _ in range(n_iter):
        # constraint 1: disperse -> ALWAYS enforce I1 (sqrt(I1)*phase) -> undisperse -> optional unit-amplitude
        E1 = disperse_torch(E, D1, N)
        E1 = torch.sqrt(I1.clamp(min=0).to(torch.float32)).to(E1.dtype) * proj(E1)
        E = disperse_torch(E1, -D1, N)
        if unit_amplitude:
            E = proj(E)

        # constraint 2: disperse -> ALWAYS enforce I2 (sqrt(I2)*phase) -> undisperse -> optional unit-amplitude
        E2 = disperse_torch(E, D2, N)
        E2 = torch.sqrt(I2.clamp(min=0).to(torch.float32)).to(E2.dtype) * proj(E2)
        E = disperse_torch(E2, -D2, N)
        if unit_amplitude:
            E = proj(E)

        err = float(torch.mean(torch.abs(
            torch.abs(disperse_torch(E, D1, N))**2 - I1.to(torch.float32)
        )))
        errors.append(err)

    return E, errors


def intensity_loss(E, I_target, D):
    """
    Differentiable intensity constraint loss.
    L = mean( ||H*E|^2 - I_target|^2 )
    """
    N    = E.shape[-1]
    E_d  = disperse_torch(E, D, N)
    I_pred = torch.abs(E_d)**2
    return torch.mean((I_pred - I_target.to(I_pred.dtype))**2)


def wrapped_phase_loss(phi_hat, phi_true):
    """
    Differentiable wrapped phase loss.
    L = mean(2 * (1 - cos(phi_hat - phi_true)))
    Invariant under global phase shift. Range [0, 4].
    """
    return torch.mean(2 * (1 - torch.cos(phi_hat - phi_true)))


def gs_gradient_descent(I1, I2, D1, D2, n_iter=100, lr=0.05,
                         unit_amplitude=True, verbose=True):
    """
    Pure gradient descent on the GS intensity loss (no alternating projections).
    Optimizes phi directly via autograd.

    L(phi) = intensity_loss(exp(i*phi), I1, D1)
           + intensity_loss(exp(i*phi), I2, D2)

    Returns
    -------
    phi : (N,) tensor, recovered phase
    losses : list of loss values
    """
    N   = I1.shape[-1]
    dev = I1.device

    phi = torch.zeros(N, device=dev, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)

    losses = []
    for i in range(n_iter):
        opt.zero_grad()
        E    = torch.exp(1j * phi.to(torch.complex64))
        if not unit_amplitude:
            E = torch.sqrt(I1.clamp(min=0).to(torch.complex64)) * E
        loss = (intensity_loss(E, I1, D1) + intensity_loss(E, I2, D2))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

        if verbose and (i % 20 == 0 or i == n_iter-1):
            print(f'  iter {i+1:4d}/{n_iter}  loss={losses[-1]:.6f}', flush=True)

    return phi.detach(), losses


# ── comparison: GS alternating projections vs gradient descent ────────────────

if __name__ == '__main__':
    import time
    from dgs.gs_core import make_measurements

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print()

    # synthetic QPSK
    m = make_measurements('QPSK', n_symbols=64, sps=8, snr_db=35,
                           D1=-5000, D2=-5750)
    I1 = torch.tensor(m['I1'], dtype=torch.float32, device=device)
    I2 = torch.tensor(m['I2'], dtype=torch.float32, device=device)
    phi_true = torch.tensor(m['phi_true'], dtype=torch.float32, device=device)
    N = len(I1)

    # ── method 1: unrolled GS (hard projection) ──
    print("=== Unrolled GS (hard projection, not backprop-able) ===")
    t0 = time.perf_counter()
    E_gs, errs_gs = gs_unrolled(I1, I2, -5000, -5750, n_iter=50, soft=False)
    t_gs = (time.perf_counter()-t0)*1000
    phi_gs = torch.angle(E_gs).detach()
    offset = float(torch.angle(torch.mean(torch.exp(1j*(phi_true - phi_gs)))))
    rms_gs = float(torch.sqrt(torch.mean((phi_true - phi_gs - offset)**2)))
    print(f'  Time: {t_gs:.1f} ms  Final err: {errs_gs[-1]:.4f}  '
          f'RMS: {np.degrees(rms_gs):.1f} deg')
    print()

    # ── method 2: unrolled GS (soft projection, differentiable) ──
    print("=== Unrolled GS (soft projection, differentiable) ===")
    t0 = time.perf_counter()
    E_soft, errs_soft = gs_unrolled(I1, I2, -5000, -5750, n_iter=50, soft=True)
    t_soft = (time.perf_counter()-t0)*1000
    phi_soft = torch.angle(E_soft).detach()
    offset = float(torch.angle(torch.mean(torch.exp(1j*(phi_true - phi_soft)))))
    rms_soft = float(torch.sqrt(torch.mean((phi_true - phi_soft - offset)**2)))
    print(f'  Time: {t_soft:.1f} ms  Final err: {errs_soft[-1]:.4f}  '
          f'RMS: {np.degrees(rms_soft):.1f} deg')
    print()

    # ── method 3: pure gradient descent on phi ──
    print("=== Gradient descent on phi (Adam, 100 iter) ===")
    t0 = time.perf_counter()
    phi_gd, losses_gd = gs_gradient_descent(I1, I2, -5000, -5750,
                                             n_iter=100, lr=0.05, verbose=True)
    t_gd = (time.perf_counter()-t0)*1000
    offset = float(torch.angle(torch.mean(torch.exp(1j*(phi_true - phi_gd)))))
    rms_gd = float(torch.sqrt(torch.mean((phi_true - phi_gd - offset)**2)))
    print(f'  Time: {t_gd:.1f} ms  Final loss: {losses_gd[-1]:.6f}  '
          f'RMS: {np.degrees(rms_gd):.1f} deg')
    print()

    # ── summary ──
    print("=== Summary ===")
    print(f"{'Method':30s}  {'RMS (deg)':>10}  {'Time (ms)':>10}  {'Differentiable':>14}")
    print("-" * 70)
    print(f"{'GS hard projection':30s}  {np.degrees(rms_gs):>10.1f}  {t_gs:>10.1f}  {'no':>14}")
    print(f"{'GS soft projection':30s}  {np.degrees(rms_soft):>10.1f}  {t_soft:>10.1f}  {'YES':>14}")
    print(f"{'Gradient descent (Adam)':30s}  {np.degrees(rms_gd):>10.1f}  {t_gd:>10.1f}  {'YES':>14}")
    print()
    print("Soft projection enables backprop through GS iterations.")
    print("Use case: jointly learn D1, D2 from data.")
    print("Use case: meta-learn n_iter as a continuous parameter.")
