"""
Learned illumination for phase recovery (the loop in Fig. 3 of Jalali et al., arXiv:2109.05959).

In plain words
--------------
A camera records only intensity |field|^2, so the phase of a sample is lost. Lighting the sample
with several different known patterns (masks) brings phase back, because each pattern gives a
different magnitude-only view. Fig. 3 of the paper closes a loop: reconstruct, measure how wrong
the reconstruction is, and push that error back (gradient descent) into the *illumination*.

Here, in 1-D:

    y_k = | FFT( m_k * x ) |^2        k = 1..K     (m_k = non-negative illumination mask)

* ``reconstruct``  unrolled multi-mask Gerchberg-Saxton, differentiable end to end
* ``fidelity``     phase-invariant match |<x_hat, x>|^2 / (|x_hat|^2 |x|^2)  (global phase is
                   unobservable, so it is divided out)
* ``train_masks``  masks = K*N*sigmoid(theta)/sum(...) so total light power is fixed, then gradient
                   descent on 1 - fidelity over a training set of random samples

The honest baseline is K random masks with the same power budget.
"""
from __future__ import annotations

import numpy as np


def random_samples(n_samples, n, seed):
    """Complex test objects: smooth-ish random amplitude and phase (low-pass filtered)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    lp = (np.abs(f) < 0.25).astype(float)
    def smooth(a): return np.fft.ifft(np.fft.fft(a, axis=-1) * lp, axis=-1).real
    amp = 1.0 + 0.5 * smooth(rng.standard_normal((n_samples, n)))
    ph = 3.0 * smooth(rng.standard_normal((n_samples, n)))
    return (np.abs(amp) * np.exp(1j * ph)).astype(np.complex64)


def _normalize(theta, torch):
    m = torch.sigmoid(theta)
    return m * (m.numel() * 0.5 / m.sum())          # mean illumination fixed at 0.5


def reconstruct(y, masks, n_iter=30, torch=None, seed=0):
    """Multi-mask GS. y: (B, K, N) intensities, masks: (K, N). Returns complex (B, N)."""
    B, K, N = y.shape
    mag = torch.sqrt(y)
    g = torch.Generator().manual_seed(seed)
    x = torch.polar(torch.ones(B, N), 0.1 * torch.randn(B, N, generator=g))
    denom = (masks ** 2).sum(0) + 1e-3
    for _ in range(n_iter):
        z = torch.fft.fft(masks[None] * x[:, None, :], dim=-1)
        z = mag * z / (z.abs() + 1e-8)               # keep phase, impose measured magnitude
        back = torch.fft.ifft(z, dim=-1)
        x = (masks[None] * back).sum(1) / denom
    return x


def fidelity(x_hat, x):
    num = (x_hat.conj() * x).sum(-1).abs() ** 2
    return num / ((x_hat.abs() ** 2).sum(-1) * (x.abs() ** 2).sum(-1) + 1e-12)


def _measure(x, masks, torch):
    return torch.fft.fft(masks[None] * x[:, None, :], dim=-1).abs() ** 2


def train_masks(n=32, K=4, steps=150, n_train=64, lr=0.1, n_iter=20, seed=0, snap_at=()):
    """Returns dict with learned masks, random masks, and held-out mean fidelity of each.

    snap_at: training steps at which to record the design (0 = the previous/initial design,
    as in Fig. 3 panel c). Result key 'snapshots' maps step -> (masks, held-out fidelity).
    """
    import torch
    if K < 1 or n < 4 or steps < 0:
        raise ValueError('need K >= 1, n >= 4, steps >= 0')
    torch.manual_seed(seed)
    xtr = torch.tensor(random_samples(n_train, n, seed + 1))
    xte = torch.tensor(random_samples(128, n, seed + 2))
    theta = (0.5 * torch.randn(K, n)).requires_grad_()
    rand_masks = _normalize(0.5 * torch.randn(K, n), torch).detach()   # same-distribution init
    opt = torch.optim.Adam([theta], lr=lr)
    def held_out(m): return float(fidelity(reconstruct(_measure(xte, m, torch), m, n_iter, torch), xte).mean())
    snaps = {}
    for step in range(steps + 1):
        if step in snap_at:
            ms = _normalize(theta, torch).detach()
            snaps[step] = (ms.numpy(), held_out(ms))
        if step == steps:
            break
        m = _normalize(theta, torch)
        loss = 1 - fidelity(reconstruct(_measure(xtr, m, torch), m, n_iter, torch), xtr).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    learned = _normalize(theta, torch).detach()
    return {'learned': learned.numpy(), 'random': rand_masks.numpy(), 'snapshots': snaps,
            'fid_learned': held_out(learned), 'fid_random': held_out(rand_masks)}


if __name__ == '__main__':
    r = train_masks()
    print(f"held-out fidelity  random masks: {r['fid_random']:.3f}   learned masks: {r['fid_learned']:.3f}")
