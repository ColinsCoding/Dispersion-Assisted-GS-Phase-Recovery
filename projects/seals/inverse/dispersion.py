"""
dispersion.py -- a PyTorch, autograd-compatible dispersive operator.

EXTENSION, not part of the original SEALS MATLAB implementation or paper.
Nothing in SEALS_paper.pdf describes a dispersive-fiber measurement stage;
this exists purely as a possible measurement-diversity operator for
phase_retrieval.py (see its Sec. 4 discussion of replacing the OSA with
time-stretch dispersive Fourier transform as a *future* direction, not
something the paper actually implements).

Matches dgs.gs_core.disperse's exact transfer-function convention --
H(nu) = exp(i*pi*D*nu^2), normalized frequency nu = fftfreq(N) in [-0.5, 0.5)
-- reimplemented in PyTorch for autograd, rather than modifying gs_core.py
(which already has unrelated uncommitted local changes; see commit notes).
tests/test_seals_dispersion.py cross-checks this against the real
dgs.gs_core.disperse (imported read-only) to confirm they agree.
"""
import torch


def dispersive_operator(E_t: torch.Tensor, D: float) -> torch.Tensor:
    """
    Apply dispersion D to a time-domain field E_t, in dgs.gs_core's
    normalized-frequency convention: H(nu) = exp(i*pi*D*nu^2).

    Parameters
    ----------
    E_t : complex torch.Tensor, shape (N,)
    D   : float, dispersion parameter (same units/convention as gs_core.disperse's D)

    Returns
    -------
    E_d : complex torch.Tensor -- dispersed field
    """
    N = E_t.shape[-1]
    nu = torch.fft.fftfreq(N, dtype=E_t.real.dtype if torch.is_complex(E_t) else E_t.dtype)
    H = torch.exp(1j * torch.pi * D * nu ** 2)
    return torch.fft.ifft(torch.fft.fft(E_t) * H)


def is_all_pass(D: float, N: int, dtype=torch.float64) -> bool:
    """
    Sanity check used by the phase-retrieval demo: dispersion must not touch
    |E(omega)| -- it is a pure phase mask. Returns True if |H(nu)|=1 everywhere.
    """
    nu = torch.fft.fftfreq(N, dtype=dtype)
    H = torch.exp(1j * torch.pi * D * nu ** 2)
    return bool(torch.allclose(H.abs(), torch.ones(N, dtype=dtype)))
