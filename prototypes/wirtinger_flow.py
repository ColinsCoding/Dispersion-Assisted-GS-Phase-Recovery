"""
Hilbert-space gradient descent for phase retrieval (Wirtinger-flow style).

The Gerchberg-Saxton algorithm projects alternately onto the two modulus
constraints |D1(u)(t)| = sqrt(I1(t)) and |D2(u)(t)| = sqrt(I2(t)), where
D1, D2 are dispersive propagators with beta2_L1, beta2_L2.  An equivalent
view: minimise the Wirtinger loss

    L(u) = ||  |D1 u| - sqrt(I1)  ||^2
          + ||  |D2 u| - sqrt(I2)  ||^2

which is the correct two-channel dispersion physics (not a plain FFT loss).

PyTorch handles the Wirtinger derivatives through complex autograd, so the
gradient step is one line.  Compared to GS:

    GS:   alternating projection, deterministic, no learning-rate knob
    Adam: gradient descent, momentum, often beats GS on noisy targets,
          parallelisable across restarts

Usage
-----
    python prototypes/wirtinger_flow.py [--N 512] [--iters 2000] [--lr 5e-2]
                                        [--beta1 -1e-22] [--beta2 -4e-22]
                                        [--noise 0.0] [--seed 0]
"""

from __future__ import annotations
import argparse
import math
import time

import numpy as np
import torch
import torch.fft


# ---------------------------------------------------------------------------
# Physics helpers (pure torch, no numpy round-trip inside gradient tape)
# ---------------------------------------------------------------------------

def _disperse_torch(u: torch.Tensor, N: int, dt: float, beta2_L: float) -> torch.Tensor:
    """Apply GVD transfer function in PyTorch (centred-FFT convention)."""
    domega = 2.0 * math.pi / (N * dt)
    idx = torch.arange(N, device=u.device, dtype=torch.float64)
    omega = (idx - N // 2) * domega
    H = torch.exp(-1j * 0.5 * beta2_L * omega ** 2).to(u.dtype)
    A_fft = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(u)))
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(A_fft * H)))


def dispersion_loss(
    u: torch.Tensor,
    amp1: torch.Tensor,
    amp2: torch.Tensor,
    N: int,
    dt: float,
    beta2_L1: float,
    beta2_L2: float,
) -> torch.Tensor:
    """Two-channel dispersion modulus-mismatch loss.

    L(u) = ||  |D(u, β₁)| - A₁  ||² + ||  |D(u, β₂)| - A₂  ||²
    """
    v1 = _disperse_torch(u, N, dt, beta2_L1)
    v2 = _disperse_torch(u, N, dt, beta2_L2)
    return ((v1.abs() - amp1) ** 2).sum() + ((v2.abs() - amp2) ** 2).sum()


# ---------------------------------------------------------------------------
def make_chirped_gaussian(
    N: int = 512, dt: float = 0.61e-12, sigma: float = 50e-12, beta_chirp: float = 1e-24
):
    """Chirped Gaussian pulse (Solli-style) on a physical time grid."""
    t = torch.arange(N, dtype=torch.float64) * dt
    t = t - t.mean()
    u = torch.exp(-t ** 2 / (2 * sigma ** 2)) * torch.exp(1j * beta_chirp * t ** 2)
    return t, u.to(torch.complex128)


# ---------------------------------------------------------------------------
def run(
    N: int = 512,
    dt: float = 0.61e-12,
    n_iter: int = 2000,
    lr: float = 5e-2,
    beta2_L1: float = -1e-22,
    beta2_L2: float = -4e-22,
    noise_frac: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
):
    """Run Wirtinger-flow phase retrieval.

    Parameters
    ----------
    N : int
        Grid size.
    dt : float
        Time sample spacing (seconds).
    n_iter : int
        Adam iterations.
    lr : float
        Adam learning rate.
    beta2_L1, beta2_L2 : float
        GVD × length for the two measurement channels (s²/rad).
        Ratio |beta2_L2/beta2_L1| should exceed 1.33 (Solli 2009).
    noise_frac : float
        Additive Gaussian noise on intensities as fraction of peak.
    seed : int
        RNG seed.
    device : str
        "cpu" or "cuda".
    verbose : bool
        Print progress.

    Returns
    -------
    u_rec : ndarray, complex, shape (N,)
    u_true : ndarray, complex, shape (N,)
    losses : ndarray, float, shape (n_iter,)
    """
    g = torch.Generator().manual_seed(seed)
    _, u_true = make_chirped_gaussian(N, dt)
    u_true = u_true.to(device)

    # Ground-truth dispersed intensities
    with torch.no_grad():
        v1_true = _disperse_torch(u_true, N, dt, beta2_L1)
        v2_true = _disperse_torch(u_true, N, dt, beta2_L2)
    I1 = v1_true.abs() ** 2
    I2 = v2_true.abs() ** 2

    if noise_frac > 0:
        I1 = I1 + noise_frac * I1.max() * torch.randn_like(I1.real).abs()
        I2 = I2 + noise_frac * I2.max() * torch.randn_like(I2.real).abs()
        I1 = I1.clamp_min(0)
        I2 = I2.clamp_min(0)

    amp1 = I1.sqrt()
    amp2 = I2.sqrt()

    # Initialise: correct modulus in time, random phase
    with torch.no_grad():
        u_mag = u_true.abs()
    phase = (torch.rand(N, generator=g, dtype=torch.float64) - 0.5) * 2 * math.pi
    u = (u_mag * torch.exp(1j * phase.to(device))).to(torch.complex128).detach().clone()
    u.requires_grad_(True)

    opt = torch.optim.Adam([u], lr=lr)
    losses = np.empty(n_iter, dtype=np.float64)

    t0 = time.perf_counter()
    for k in range(n_iter):
        opt.zero_grad()
        L = dispersion_loss(u, amp1, amp2, N, dt, beta2_L1, beta2_L2)
        L.backward()
        opt.step()
        losses[k] = L.item()
        if verbose and (k & 511) == 0:
            print(f"  iter {k:5d}   loss = {losses[k]:.5e}")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  done in {elapsed:.2f} s  ({elapsed / n_iter * 1e3:.2f} ms/iter) on {device}")

    return u.detach().cpu().numpy(), u_true.cpu().numpy(), losses


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Wirtinger-flow dispersion phase retrieval")
    ap.add_argument("--N",      type=int,   default=512,    help="Grid size")
    ap.add_argument("--dt",     type=float, default=0.61e-12, help="Sample spacing (s)")
    ap.add_argument("--iters",  type=int,   default=2000,   help="Adam iterations")
    ap.add_argument("--lr",     type=float, default=5e-2,   help="Learning rate")
    ap.add_argument("--beta1",  type=float, default=-1e-22, help="beta2_L1 (s²/rad)")
    ap.add_argument("--beta2",  type=float, default=-4e-22, help="beta2_L2 (s²/rad)")
    ap.add_argument("--noise",  type=float, default=0.0,    help="Intensity noise fraction")
    ap.add_argument("--seed",   type=int,   default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ratio = abs(args.beta2 / args.beta1)
    if ratio < 1.33:
        print(f"WARNING: |beta2/beta1| = {ratio:.2f} < 1.33 (Solli 2009 minimum for stable recovery)")

    u_rec, u_true, losses = run(
        N=args.N, dt=args.dt, n_iter=args.iters, lr=args.lr,
        beta2_L1=args.beta1, beta2_L2=args.beta2,
        noise_frac=args.noise, seed=args.seed,
        device=args.device, verbose=True,
    )

    # Global-phase-aligned RMSE
    inner = np.vdot(u_true, u_rec)
    if abs(inner) > 1e-30:
        u_rec_aligned = u_rec * (inner / abs(inner)).conj()
    else:
        u_rec_aligned = u_rec

    norm = np.sqrt(np.mean(np.abs(u_true) ** 2))
    rmse = np.sqrt(np.mean(np.abs(u_rec_aligned - u_true) ** 2)) / norm

    ph_rec  = np.unwrap(np.angle(u_rec_aligned));  ph_rec  -= ph_rec.mean()
    ph_true = np.unwrap(np.angle(u_true));          ph_true -= ph_true.mean()
    phase_rmse = np.sqrt(np.mean((ph_rec - ph_true) ** 2))

    print(f"\nfinal loss   = {losses[-1]:.4e}")
    print(f"field RMSE   = {rmse:.4e}  (normalised, global-phase aligned)")
    print(f"phase RMSE   = {phase_rmse:.4e} rad")


if __name__ == "__main__":
    main()
