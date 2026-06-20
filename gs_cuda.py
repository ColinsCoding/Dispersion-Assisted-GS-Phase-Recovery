"""CUDA-accelerated Gerchberg-Saxton phase recovery (PyTorch).

A GPU port of the alternating-projection loop in gs_core: the dispersion
operator H(f)=exp(i pi D nu^2) and the two amplitude projections run as batched
complex FFTs on the device. Bit-for-bit the same algorithm, just on the GPU --
the speedup grows with signal length and iteration count. Civilian optical
metrology. Requires torch (py 3.12 here).
"""

import numpy as np


def gerchberg_saxton_cuda(I1, I2, D1, D2, n_iter=80, unit_amplitude=True, device=None):
    """Recover phi(t) from intensities at dispersions D1, D2 on the GPU.

    Matches gs_core.retrieve_phase (same H = exp(i pi D nu^2), same projection
    order). Returns the recovered phase array.
    """
    import torch
    if D1 == D2:
        raise ValueError("D1 == D2: zero measurement diversity")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N = min(len(I1), len(I2))
    nu = torch.fft.fftfreq(N).to(dev)
    H1 = torch.exp(1j * np.pi * float(D1) * nu**2).to(torch.complex64)
    H2 = torch.exp(1j * np.pi * float(D2) * nu**2).to(torch.complex64)
    A1 = torch.tensor(np.sqrt(np.maximum(I1[:N], 0)), dtype=torch.float32, device=dev)
    A2 = torch.tensor(np.sqrt(np.maximum(I2[:N], 0)), dtype=torch.float32, device=dev)

    def disp(E, H):
        return torch.fft.ifft(torch.fft.fft(E) * H)

    E = disp(A1.to(torch.complex64), torch.conj(H1))      # init: undisperse sqrt(I1) by D1
    for _ in range(n_iter):
        Ed1 = disp(E, H1)
        Ed1 = A1 * torch.exp(1j * Ed1.angle())            # enforce I1
        E = disp(Ed1, torch.conj(H1))
        if unit_amplitude:
            E = torch.exp(1j * E.angle())
        Ed2 = disp(E, H2)
        Ed2 = A2 * torch.exp(1j * Ed2.angle())            # enforce I2
        E = disp(Ed2, torch.conj(H2))
        if unit_amplitude:
            E = torch.exp(1j * E.angle())
    return E.angle().detach().cpu().numpy()
