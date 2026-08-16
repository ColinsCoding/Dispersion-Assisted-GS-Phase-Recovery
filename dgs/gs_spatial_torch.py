"""
gs_spatial_torch.py -- GPU-accelerated 2D (spatial) Gerchberg-Saxton, PyTorch.

The classic GSA from the ECE 279AS slides (image plane <-> diffraction plane
via 2D FFT), as distinct from gs_torch.py's TEMPORAL GSA (two 1D dispersion
planes via the fiber transfer function H(f)=exp(i*pi*D*f^2)). Same
alternating-projection idea, different physical domains:

  spatial:  image-plane amplitude  <-FFT2->  diffraction-plane amplitude
  temporal: dispersed-arm-1 amplitude <-H(D1),H(D2)-> dispersed-arm-2 amplitude

Block-diagram match to the slides (image amplitude x random phase -> FFT ->
enforce diffraction amplitude -> IFFT -> enforce image amplitude -> repeat).
"""

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gsa_spatial(image_amplitude, diffraction_amplitude, n_iter=200, seed=0,
                 device=DEVICE):
    """Classic 2D GSA: recover the complex field consistent with a known
    image-plane amplitude and a known diffraction-plane (Fourier) amplitude.

    Parameters
    ----------
    image_amplitude, diffraction_amplitude : (H, W) arrays/tensors
    n_iter : int -- GS iterations
    seed   : int -- random-phase initial guess (first iteration only, per
             the slides: "Random Phase (from -pi to pi), first iteration only")

    Returns
    -------
    field  : (H, W) complex tensor -- recovered image-plane field
    errors : list[float] -- ||FFT2(g)| - diffraction_amplitude|| / ||diffraction_amplitude||
             per iteration (the slides' own convergence criterion)
    """
    A = torch.as_tensor(image_amplitude, dtype=torch.float32, device=device)
    D = torch.as_tensor(diffraction_amplitude, dtype=torch.float32, device=device)
    if A.shape != D.shape:
        raise ValueError(f"image_amplitude {tuple(A.shape)} and diffraction_amplitude "
                          f"{tuple(D.shape)} must have the same shape")

    gen = torch.Generator(device=device).manual_seed(seed)
    rand_phase = torch.rand(A.shape, generator=gen, device=device) * 2 * torch.pi - torch.pi
    g = (A * torch.exp(1j * rand_phase)).to(torch.complex64)

    errors = []
    D_norm = torch.linalg.norm(D) + 1e-12
    for _ in range(n_iter):
        G = torch.fft.fft2(g)
        G2 = D * torch.exp(1j * torch.angle(G))              # enforce diffraction amplitude
        g2 = torch.fft.ifft2(G2)
        g = (A * torch.exp(1j * torch.angle(g2))).to(torch.complex64)   # enforce image amplitude

        err = float(torch.linalg.norm(torch.abs(torch.fft.fft2(g)) - D) / D_norm)
        errors.append(err)

    return g, errors


def gsa_spatial_batch(image_amplitude_batch, diffraction_amplitude_batch, n_iter=200,
                        seed=0, device=DEVICE):
    """Batched version: (B, H, W) amplitudes, one FFT2 call per batch per iteration."""
    A = torch.as_tensor(image_amplitude_batch, dtype=torch.float32, device=device)
    D = torch.as_tensor(diffraction_amplitude_batch, dtype=torch.float32, device=device)
    if A.shape != D.shape:
        raise ValueError(f"image_amplitude_batch {tuple(A.shape)} and "
                          f"diffraction_amplitude_batch {tuple(D.shape)} must match")

    gen = torch.Generator(device=device).manual_seed(seed)
    rand_phase = torch.rand(A.shape, generator=gen, device=device) * 2 * torch.pi - torch.pi
    g = (A * torch.exp(1j * rand_phase)).to(torch.complex64)

    errors = []
    D_norm = torch.linalg.norm(D.reshape(D.shape[0], -1), dim=-1) + 1e-12   # (B,)
    for _ in range(n_iter):
        G = torch.fft.fft2(g)
        G2 = D * torch.exp(1j * torch.angle(G))
        g2 = torch.fft.ifft2(G2)
        g = (A * torch.exp(1j * torch.angle(g2))).to(torch.complex64)

        diff = (torch.abs(torch.fft.fft2(g)) - D).reshape(D.shape[0], -1)
        err = torch.linalg.norm(diff, dim=-1) / D_norm                      # (B,)
        errors.append(float(err.mean()))

    return g, errors


def global_phase_align_2d(reference, field):
    """The one unavoidable ambiguity for two-amplitude-constraint spatial GSA:
    a global phase e^{i*phi} leaves BOTH the image-plane and diffraction-plane
    amplitudes unchanged. Rotate `field` by the phase that best matches
    `reference` (2D analog of dgs.phase_retrieval_ambiguities.global_phase_align)."""
    ref = torch.as_tensor(reference)
    fld = torch.as_tensor(field)
    phi = torch.angle(torch.vdot(fld.flatten(), ref.flatten()))
    return fld * torch.exp(1j * phi)


def make_test_scene(H=128, W=128, symmetric_support=False, device=DEVICE):
    """A synthetic 'image at the source' (slide 4's bright blob) with a known
    smooth aberration phase (defocus + tilt), for self-testing gsa_spatial.
    symmetric_support=True reproduces the WORSE case (a single centered disk
    has a central symmetry that creates a near-ambiguous twin solution --
    real GSA behavior, not a limitation of this implementation)."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device), indexing="ij")
    if symmetric_support:
        image_amplitude = (torch.sqrt(xx**2 + yy**2) < 0.3).float()
    else:
        image_amplitude = ((torch.sqrt((xx - 0.15)**2 + (yy - 0.05)**2) < 0.22).float() +
                            (torch.sqrt((xx + 0.35)**2 + (yy + 0.3)**2) < 0.08).float()).clamp(max=1.0)

    true_phase = 6.0 * (xx**2 + yy**2) + 3.0 * xx    # defocus + tilt
    field_true = (image_amplitude * torch.exp(1j * true_phase)).to(torch.complex64)
    diffraction_amplitude = torch.abs(torch.fft.fft2(field_true))
    return image_amplitude, diffraction_amplitude, field_true


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}\n")

    for label, symmetric in [("asymmetric support (two off-center blobs)", False),
                              ("symmetric support (single centered disk)", True)]:
        image_amplitude, diffraction_amplitude, field_true = make_test_scene(symmetric_support=symmetric)
        field_rec, errors = gsa_spatial(image_amplitude, diffraction_amplitude, n_iter=200)
        field_aligned = global_phase_align_2d(field_true, field_rec)
        mask = image_amplitude > 0.5
        rel_err = float(torch.linalg.norm((field_aligned - field_true)[mask])
                         / torch.linalg.norm(field_true[mask]))
        print(f"{label}:")
        print(f"  constraint error: first={errors[0]:.4f}  last={errors[-1]:.6f}")
        print(f"  relative field error within support (after global-phase align): {rel_err:.6f}\n")
