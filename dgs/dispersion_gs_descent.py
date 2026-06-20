"""Gradient-descent phase retrieval for the dispersion problem (PyTorch).

Civilian optical-metrology companion to dispersion_gs_prototype.py. Instead of
Gerchberg-Saxton alternating projections, make the forward model
  phi -> x = sqrt(I1) exp(i phi) -> disperse(x, D) -> |.|^2
fully differentiable and minimise || |disperse|^2 - I2 ||^2 by Adam. Two
advantages over GS on the ill-posed (low-diversity) case: a smoother
optimisation path, and the ability to add a *prior* (here, field smoothness)
that GS cannot express. For fibre metrology / silicon photonics / education.
Requires torch (py 3.12 here).
"""

import numpy as np


def _disperse_np(x, D):
    N = len(x)
    f = np.fft.fftfreq(N)
    return np.fft.ifft(np.fft.fft(x) * np.exp(1j * np.pi * D * f**2))


def torch_phase_retrieval(I1, I2, D, reg=0.0, n_iter=2500, lr=0.03, seed=0,
                          device=None):
    """Recover phi from I1 (=|x|^2, fixes the amplitude) and I2 (after dispersion D).

    Parametrises x = sqrt(I1) exp(i phi) so I1 is satisfied exactly, then descends
    on the I2 mismatch plus reg * (field-smoothness penalty). Returns
    (phi_recovered, loss_history).
    """
    import torch
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    if reg < 0:
        raise ValueError("reg must be >= 0")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N = len(I1)
    g = torch.Generator().manual_seed(seed)
    A1 = torch.tensor(np.sqrt(np.maximum(I1, 0)), dtype=torch.float32, device=dev)
    I2t = torch.tensor(np.asarray(I2, dtype=float), dtype=torch.float32, device=dev)
    f = torch.fft.fftfreq(N).to(dev)
    H = torch.exp(1j * np.pi * D * f**2).to(torch.complex64)
    phi = (0.1 * torch.randn(N, generator=g)).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    hist = []
    for _ in range(n_iter):
        opt.zero_grad()
        x = A1 * torch.exp(1j * phi.to(torch.complex64))
        x2 = torch.fft.ifft(torch.fft.fft(x) * H)
        data_loss = ((x2.abs()**2 - I2t)**2).mean()
        smooth = ((x[1:] - x[:-1]).abs()**2).mean()       # field-smoothness prior
        loss = data_loss + reg * smooth
        loss.backward()
        opt.step()
        hist.append(float(data_loss.item()))
    return phi.detach().cpu().numpy(), np.array(hist)


def compare_phase(phi_rec, phi_true, weight):
    """Amplitude-weighted RMS phase error, removing global-offset and twin
    ambiguities (same convention as dispersion_gs_prototype)."""
    best = None
    for sign in (+1, -1):
        d = phi_true - sign * phi_rec
        offset = np.angle(np.sum(weight * np.exp(1j * d)))
        aligned = sign * phi_rec + offset
        err = np.sqrt(np.sum(weight * np.angle(np.exp(1j * (phi_true - aligned)))**2)
                      / np.sum(weight))
        if best is None or err < best[0]:
            best = (err, aligned)
    return best
