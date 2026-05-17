"""
Hilbert-space gradient descent for phase retrieval (Wirtinger-flow style).

The Gerchberg-Saxton algorithm projects alternately onto the two modulus
constraints |u(t)|=sqrt(I_t) and |U(omega)|=sqrt(I_w).  An equivalent view is
to treat u as a vector in the complex Hilbert space C^N and minimise

    L(u) = sum_t (|u(t)|     - sqrt(I_t(t)))**2
         + sum_w (|F[u](w)|  - sqrt(I_w(w)))**2.

PyTorch's complex autograd handles the Wirtinger derivatives for us, so we
get a one-line gradient step.  Compared to GS:

    * GS:   alternating projection, deterministic, no learning rate
    * Adam: gradient descent, gives a knob (lr) and a momentum that often
            beats GS on noisy targets and benefits from GPU batching.

Usage
-----
    python prototypes/wirtinger_flow.py
"""

from __future__ import annotations
import argparse
import math
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
def make_chirped_gaussian(N: int = 512, sigma: float = 0.5, beta: float = 4.0):
    """Same toy ground truth as the notebook (Solli-style chirped pulse)."""
    t = torch.linspace(-5, 5, N)
    u = torch.exp(-t**2 / (2 * sigma**2)) * torch.exp(1j * beta * t**2)
    return t, u.to(torch.complex64)


def loss_fn(u: torch.Tensor, amp_t: torch.Tensor, amp_w: torch.Tensor) -> torch.Tensor:
    """Modulus-mismatch loss summed across both Hilbert-space bases."""
    U = torch.fft.fft(u)
    return ((u.abs() - amp_t) ** 2).sum() + ((U.abs() - amp_w) ** 2).sum()


# ---------------------------------------------------------------------------
def run(N=512, n_iter=2000, lr=5e-2, seed=0, device="cpu", verbose=True):
    g = torch.Generator().manual_seed(seed)
    t, u_true = make_chirped_gaussian(N)
    I_t = (u_true.abs() ** 2).to(device)
    I_w = (torch.fft.fft(u_true).abs() ** 2).to(device)
    amp_t = I_t.clamp_min(0).sqrt()
    amp_w = I_w.clamp_min(0).sqrt()

    # complex parameter: random phase, correct modulus
    phase = (torch.rand(N, generator=g) - 0.5) * 2 * math.pi
    u = (amp_t.cpu() * torch.exp(1j * phase)).to(device).clone()
    u.requires_grad_(True)

    opt = torch.optim.Adam([u], lr=lr)
    losses = np.empty(n_iter, dtype=np.float32)

    t0 = time.perf_counter()
    for k in range(n_iter):
        opt.zero_grad()
        L = loss_fn(u, amp_t, amp_w)
        L.backward()
        opt.step()
        losses[k] = L.item()
        if verbose and (k & 511) == 0:
            print(f"  iter {k:5d}   loss = {losses[k]:.5e}")
    dt = time.perf_counter() - t0
    if verbose:
        print(f"  done in {dt:.2f} s on {device}")
    return u.detach().cpu().numpy(), u_true.numpy(), losses


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",       type=int,   default=512)
    ap.add_argument("--iters",   type=int,   default=2000)
    ap.add_argument("--lr",      type=float, default=5e-2)
    ap.add_argument("--seed",    type=int,   default=0)
    ap.add_argument("--device",  type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    u_rec, u_true, losses = run(N=args.N, n_iter=args.iters,
                                lr=args.lr, seed=args.seed,
                                device=args.device, verbose=True)

    # Phases up to a global constant
    ph_rec  = np.unwrap(np.angle(u_rec));   ph_rec  -= ph_rec.mean()
    ph_true = np.unwrap(np.angle(u_true));  ph_true -= ph_true.mean()
    err_phase = np.sqrt(np.mean((ph_rec - ph_true) ** 2))
    print(f"\nfinal loss  = {losses[-1]:.4e}")
    print(f"phase RMSE  = {err_phase:.4e} rad  (modulo global offset)")


if __name__ == "__main__":
    main()
