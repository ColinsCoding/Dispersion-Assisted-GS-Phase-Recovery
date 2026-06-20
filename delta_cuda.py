"""Visualize the Dirac delta on the GPU -- real part builds it, imaginary cancels.

The delta isn't a function, it's a *limit*. The cleanest limit is its Fourier
representation,

    delta(x) = (1/2pi) integral e^{ikx} dk   over all k,

which we truncate to |k| <= K and evaluate as a sum of complex exponentials on
the device (CUDA if available). Splitting e^{ikx} = cos(kx) + i sin(kx):

    * the REAL part (even cosines) piles up into a tall narrow spike,
      delta_K(x) = sin(Kx)/(pi x), peak K/pi -- a nascent delta,
    * the IMAGINARY part (odd sines) cancels to ~0 by symmetry.

So "real vs imaginary" is literally "the spike vs the part that cancels". Also
provides the Gaussian and Lorentzian nascent deltas for comparison. torch
(py 3.12 here); runs on GPU when present. Civilian physics education.
"""

import numpy as np


def _device(device=None):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def fourier_delta(x, K, n_k=4000, device=None):
    """Truncated Fourier representation (1/2pi) int_{-K}^{K} e^{ikx} dk, on device.

    Returns a COMPLEX tensor (moved to CPU): real -> sin(Kx)/(pi x), imag -> ~0.
    """
    import torch
    if K <= 0 or n_k < 2:
        raise ValueError("K > 0 and n_k >= 2 required")
    dev = _device(device)
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)
    k = torch.linspace(-K, K, n_k, dtype=torch.float64, device=dev)
    dk = (2.0 * K) / (n_k - 1)
    integrand = torch.exp(1j * torch.outer(xt, k))      # (Nx, Nk) complex128
    val = integrand.sum(dim=1) * dk / (2 * np.pi)
    return val.cpu()


def gaussian_delta(x, eps, device=None):
    """Nascent delta: a Gaussian of width eps, area 1.  -> delta as eps -> 0."""
    import torch
    if eps <= 0:
        raise ValueError("eps > 0 required")
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=_device(device))
    return (torch.exp(-xt**2 / (2 * eps**2)) / (eps * np.sqrt(2 * np.pi))).cpu()


def lorentzian_delta(x, eps, device=None):
    """Nascent delta: a Lorentzian of half-width eps, area 1 (the resonance line)."""
    import torch
    if eps <= 0:
        raise ValueError("eps > 0 required")
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=_device(device))
    return ((eps / np.pi) / (xt**2 + eps**2)).cpu()


def render(out="figures/delta_cuda_real_imag.png", show=False):
    """Render the GPU delta: real spike vs vanishing imaginary part + nascent deltas."""
    import matplotlib.pyplot as plt
    import torch
    dev = _device()
    x = np.linspace(-3, 3, 1600)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0d1117")
    # (a) real part for growing K -> taller, narrower spike
    for K in (15, 40, 100):
        d = fourier_delta(x, K, device=dev)
        ax[0].plot(x, d.real.numpy(), lw=1.3, label=f"K={K}")
    ax[0].set_title(f"REAL part  →  the delta builds\nδ_K(x)=sin(Kx)/(πx)   [{dev}]", color="#e6edf3")
    ax[0].legend(); ax[0].set_xlabel("x")

    # (b) imaginary part ~ 0 (machine zero) -- it cancels
    d = fourier_delta(x, 100, device=dev)
    ax[1].plot(x, d.imag.numpy(), color="#f72585", lw=1.3)
    ax[1].set_title(f"IMAGINARY part  →  cancels\nmax|Im| = {np.abs(d.imag.numpy()).max():.1e}", color="#e6edf3")
    ax[1].set_xlabel("x")

    # (c) Gaussian & Lorentzian nascent deltas shrinking to a spike
    for eps in (0.5, 0.2, 0.08):
        ax[2].plot(x, gaussian_delta(x, eps).numpy(), lw=1.2, label=f"Gauss ε={eps}")
    ax[2].plot(x, lorentzian_delta(x, 0.08).numpy(), "--", color="#4cc9f0", lw=1.2, label="Lorentz ε=0.08")
    ax[2].set_title("nascent deltas  →  area 1, ε→0", color="#e6edf3")
    ax[2].legend(); ax[2].set_xlabel("x")

    for a in ax:
        a.set_facecolor("#0d1117"); a.tick_params(colors="#9fb3c8")
        for sp in a.spines.values():
            sp.set_color("#1b2735")
        a.title.set_fontsize(10)
    import pathlib
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return out


if __name__ == "__main__":
    import torch
    print("device:", _device(), "| cuda available:", torch.cuda.is_available())
    print("wrote", render())
