"""The Helmholtz decomposition: any sufficiently well-behaved vector field
F splits uniquely into an IRROTATIONAL part (curl-free, a pure gradient)
plus a SOLENOIDAL part (divergence-free, a pure curl) --

    F = F_irrotational + F_solenoidal,   curl(F_irrotational) = 0,   div(F_solenoidal) = 0

dgs.irrotational_solenoidal_polyglot built the two individual example
fields (point-charge direction, wire direction) that satisfy each half of
this theorem separately; this module does the actual DECOMPOSITION of a
generic field that starts out with neither property.

For a PERIODIC field on a grid, the decomposition has an exact closed
form in Fourier space: at each wavevector k, project the field's Fourier
component onto k (the LONGITUDINAL part, automatically curl-free since a
field pointing everywhere along k in Fourier space has zero curl -- the
same "gradient fields point along k, curl(gradient)=0" fact
dgs.irrotational_solenoidal_polyglot demonstrated in real space) and onto
the plane perpendicular to k (the TRANSVERSE part, automatically
divergence-free). This is EXACT for a band-limited periodic field -- no
finite-difference truncation error, the same reason this repo's dispersion
kernel H(f)=exp(i*pi*D*f^2) is applied exactly via FFT rather than
approximated.

Cross-checked by two independent implementations of the same
spectral-derivative machinery: NumPy (the reference) and PyTorch's own FFT
(torch.fft), continuing this session's polyglot cross-validation pattern
at a lighter weight (same algorithm, two tensor libraries, rather than two
different derivation strategies).
"""

from __future__ import annotations
import numpy as np


def synthetic_test_field(N: int = 24, L: float = 2 * np.pi, n_modes: int = 5, seed: int = 0) -> np.ndarray:
    """A smooth, periodic, generic 3-D vector field (shape (N,N,N,3)) built
    from a handful of low-order random Fourier modes -- generic enough
    that it has BOTH nonzero curl and nonzero divergence at typical
    points (checked in verify_decomposition), unlike
    dgs.irrotational_solenoidal_polyglot's two special-case example
    fields."""
    if N < 4 or n_modes < 1:
        raise ValueError(f"N={N} must be >= 4, n_modes={n_modes} must be >= 1")
    rng = np.random.default_rng(seed)
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    F = np.zeros((N, N, N, 3))
    for _ in range(n_modes):
        k = rng.integers(1, 4, size=3)
        phase = rng.uniform(0, 2 * np.pi, size=3)
        amp = rng.uniform(0.5, 1.5, size=3)
        arg = k[0] * X + k[1] * Y + k[2] * Z
        F[..., 0] += amp[0] * np.cos(arg + phase[0])
        F[..., 1] += amp[1] * np.sin(arg + phase[1])
        F[..., 2] += amp[2] * np.cos(arg + phase[2] + 0.7)
    return F


def _wavevectors(N: int, L: float):
    """Angular wavenumbers (kx,ky,kz) as (N,N,N) meshgrids, FFT convention."""
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    return KX, KY, KZ


def spectral_divergence(F: np.ndarray, L: float = 2 * np.pi) -> np.ndarray:
    """div(F), computed EXACTLY (for a band-limited periodic field) via
    FFT: divergence in Fourier space is i*k.F_hat, transformed back."""
    N = F.shape[0]
    KX, KY, KZ = _wavevectors(N, L)
    Fx_hat = np.fft.fftn(F[..., 0])
    Fy_hat = np.fft.fftn(F[..., 1])
    Fz_hat = np.fft.fftn(F[..., 2])
    div_hat = 1j * (KX * Fx_hat + KY * Fy_hat + KZ * Fz_hat)
    return np.real(np.fft.ifftn(div_hat))


def spectral_curl(F: np.ndarray, L: float = 2 * np.pi) -> np.ndarray:
    """curl(F), computed EXACTLY via FFT: curl in Fourier space is
    i*k x F_hat, transformed back."""
    N = F.shape[0]
    KX, KY, KZ = _wavevectors(N, L)
    Fx_hat = np.fft.fftn(F[..., 0])
    Fy_hat = np.fft.fftn(F[..., 1])
    Fz_hat = np.fft.fftn(F[..., 2])
    curl_x_hat = 1j * (KY * Fz_hat - KZ * Fy_hat)
    curl_y_hat = 1j * (KZ * Fx_hat - KX * Fz_hat)
    curl_z_hat = 1j * (KX * Fy_hat - KY * Fx_hat)
    curl = np.stack([np.real(np.fft.ifftn(curl_x_hat)),
                      np.real(np.fft.ifftn(curl_y_hat)),
                      np.real(np.fft.ifftn(curl_z_hat))], axis=-1)
    return curl


def helmholtz_decompose(F: np.ndarray, L: float = 2 * np.pi) -> dict:
    """Splits F into F_irrotational (longitudinal: parallel to k in
    Fourier space, curl-free) and F_solenoidal (transverse: perpendicular
    to k, divergence-free), exactly, mode by mode:

        F_hat(k) = (F_hat(k).khat) khat  +  [F_hat(k) - (F_hat(k).khat) khat]
                   \\_____ longitudinal ____/  \\_______ transverse _______/
                    = F_irrotational_hat(k)      = F_solenoidal_hat(k)

    The k=0 (DC/mean) mode has no well-defined direction khat; by
    convention (matching most textbook treatments) it's assigned entirely
    to the solenoidal part, since a spatially CONSTANT field trivially has
    both zero curl and zero divergence, so the assignment doesn't affect
    either verified property -- it only affects which of the two returned
    arrays carries the field's mean value."""
    N = F.shape[0]
    KX, KY, KZ = _wavevectors(N, L)
    k2 = KX**2 + KY**2 + KZ**2
    k2_safe = np.where(k2 == 0, 1.0, k2)   # avoid 0/0 at the DC mode

    Fx_hat = np.fft.fftn(F[..., 0])
    Fy_hat = np.fft.fftn(F[..., 1])
    Fz_hat = np.fft.fftn(F[..., 2])

    # F_hat . khat / |k| = (F_hat.k)/|k|^2 -- the longitudinal projection coefficient
    dot_k = (Fx_hat * KX + Fy_hat * KY + Fz_hat * KZ) / k2_safe
    long_x_hat = np.where(k2 == 0, 0.0, dot_k * KX)
    long_y_hat = np.where(k2 == 0, 0.0, dot_k * KY)
    long_z_hat = np.where(k2 == 0, 0.0, dot_k * KZ)

    trans_x_hat = Fx_hat - long_x_hat
    trans_y_hat = Fy_hat - long_y_hat
    trans_z_hat = Fz_hat - long_z_hat

    F_irrot = np.stack([np.real(np.fft.ifftn(long_x_hat)),
                         np.real(np.fft.ifftn(long_y_hat)),
                         np.real(np.fft.ifftn(long_z_hat))], axis=-1)
    F_sol = np.stack([np.real(np.fft.ifftn(trans_x_hat)),
                       np.real(np.fft.ifftn(trans_y_hat)),
                       np.real(np.fft.ifftn(trans_z_hat))], axis=-1)
    return {"F_irrotational": F_irrot, "F_solenoidal": F_sol}


def verify_decomposition(N: int = 24, L: float = 2 * np.pi, n_modes: int = 5, seed: int = 0) -> dict:
    """CHECKED, not assumed: builds a generic synthetic field (which has
    BOTH nonzero curl and nonzero divergence generically), decomposes it,
    and verifies (1) the two parts sum back to the original field, (2) the
    irrotational part's curl is ~0 everywhere, (3) the solenoidal part's
    divergence is ~0 everywhere -- all three to near machine precision,
    since the spectral derivative is exact for this band-limited field."""
    F = synthetic_test_field(N, L, n_modes, seed)
    original_div = spectral_divergence(F, L)
    original_curl = spectral_curl(F, L)

    parts = helmholtz_decompose(F, L)
    F_irrot, F_sol = parts["F_irrotational"], parts["F_solenoidal"]

    reconstruction_error = float(np.max(np.abs((F_irrot + F_sol) - F)))
    curl_of_irrot = spectral_curl(F_irrot, L)
    div_of_sol = spectral_divergence(F_sol, L)

    return {
        "original_max_abs_divergence": float(np.max(np.abs(original_div))),
        "original_max_abs_curl": float(np.max(np.abs(original_curl))),
        "reconstruction_error": reconstruction_error,
        "max_abs_curl_of_irrotational_part": float(np.max(np.abs(curl_of_irrot))),
        "max_abs_div_of_solenoidal_part": float(np.max(np.abs(div_of_sol))),
        "F": F, "F_irrotational": F_irrot, "F_solenoidal": F_sol,
    }


# ── PyTorch cross-check: same algorithm, independent tensor library ────────

def torch_verify_decomposition(N: int = 24, L: float = 2 * np.pi, n_modes: int = 5, seed: int = 0) -> dict:
    """Reimplements spectral_divergence/spectral_curl/helmholtz_decompose
    using torch.fft instead of numpy.fft -- an independent implementation
    of the same spectral algorithm (a different tensor library and FFT
    backend, not a numpy-wrapped call), cross-checked against the numpy
    version's results."""
    import torch
    F_np = synthetic_test_field(N, L, n_modes, seed)
    F = torch.as_tensor(F_np, dtype=torch.float64)

    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY, KZ = (torch.as_tensor(a, dtype=torch.float64)
                  for a in np.meshgrid(k1, k1, k1, indexing="ij"))
    k2 = KX**2 + KY**2 + KZ**2
    k2_safe = torch.where(k2 == 0, torch.ones_like(k2), k2)

    Fx_hat = torch.fft.fftn(F[..., 0])
    Fy_hat = torch.fft.fftn(F[..., 1])
    Fz_hat = torch.fft.fftn(F[..., 2])

    dot_k = (Fx_hat * KX + Fy_hat * KY + Fz_hat * KZ) / k2_safe
    zero_mask = (k2 == 0)
    long_x_hat = torch.where(zero_mask, torch.zeros_like(dot_k), dot_k * KX)
    long_y_hat = torch.where(zero_mask, torch.zeros_like(dot_k), dot_k * KY)
    long_z_hat = torch.where(zero_mask, torch.zeros_like(dot_k), dot_k * KZ)
    trans_x_hat, trans_y_hat, trans_z_hat = Fx_hat - long_x_hat, Fy_hat - long_y_hat, Fz_hat - long_z_hat

    F_irrot = torch.stack([torch.fft.ifftn(long_x_hat).real, torch.fft.ifftn(long_y_hat).real,
                            torch.fft.ifftn(long_z_hat).real], dim=-1)
    F_sol = torch.stack([torch.fft.ifftn(trans_x_hat).real, torch.fft.ifftn(trans_y_hat).real,
                          torch.fft.ifftn(trans_z_hat).real], dim=-1)

    def curl_of(field):
        fx_hat = torch.fft.fftn(field[..., 0]); fy_hat = torch.fft.fftn(field[..., 1]); fz_hat = torch.fft.fftn(field[..., 2])
        cx = torch.fft.ifftn(1j * (KY * fz_hat - KZ * fy_hat)).real
        cy = torch.fft.ifftn(1j * (KZ * fx_hat - KX * fz_hat)).real
        cz = torch.fft.ifftn(1j * (KX * fy_hat - KY * fx_hat)).real
        return torch.stack([cx, cy, cz], dim=-1)

    def div_of(field):
        fx_hat = torch.fft.fftn(field[..., 0]); fy_hat = torch.fft.fftn(field[..., 1]); fz_hat = torch.fft.fftn(field[..., 2])
        return torch.fft.ifftn(1j * (KX * fx_hat + KY * fy_hat + KZ * fz_hat)).real

    F_irrot_np, F_sol_np = F_irrot.numpy(), F_sol.numpy()
    numpy_result = helmholtz_decompose(F_np, L)
    max_diff_irrot = float(np.max(np.abs(F_irrot_np - numpy_result["F_irrotational"])))
    max_diff_sol = float(np.max(np.abs(F_sol_np - numpy_result["F_solenoidal"])))

    return {
        "max_abs_diff_numpy_vs_torch_irrotational": max_diff_irrot,
        "max_abs_diff_numpy_vs_torch_solenoidal": max_diff_sol,
        "torch_max_abs_curl_of_irrotational_part": float(torch.max(torch.abs(curl_of(F_irrot)))),
        "torch_max_abs_div_of_solenoidal_part": float(torch.max(torch.abs(div_of(F_sol)))),
    }


if __name__ == "__main__":
    print("=== Helmholtz decomposition of a generic synthetic field ===")
    check = verify_decomposition()
    print(f"  original field: max|div| = {check['original_max_abs_divergence']:.4f}, "
          f"max|curl| = {check['original_max_abs_curl']:.4f}   (both genuinely nonzero)")
    print(f"  reconstruction error (F_irrot + F_sol - F): {check['reconstruction_error']:.3e}")
    print(f"  max|curl(F_irrotational)|: {check['max_abs_curl_of_irrotational_part']:.3e}  (expect ~0)")
    print(f"  max|div(F_solenoidal)|:    {check['max_abs_div_of_solenoidal_part']:.3e}  (expect ~0)")

    try:
        print("\n=== Cross-check: independent torch.fft implementation ===")
        tcheck = torch_verify_decomposition()
        print(f"  max|numpy - torch| (irrotational part): {tcheck['max_abs_diff_numpy_vs_torch_irrotational']:.3e}")
        print(f"  max|numpy - torch| (solenoidal part):   {tcheck['max_abs_diff_numpy_vs_torch_solenoidal']:.3e}")
        print(f"  torch: max|curl(F_irrotational)| = {tcheck['torch_max_abs_curl_of_irrotational_part']:.3e}")
        print(f"  torch: max|div(F_solenoidal)|    = {tcheck['torch_max_abs_div_of_solenoidal_part']:.3e}")
    except ImportError:
        print("\ntorch not available in this interpreter -- skipped cross-check (run under py -3.12)")

    print("\nA generic field with real curl AND real divergence, split exactly into a piece")
    print("with zero curl and a piece with zero divergence -- the two example fields from")
    print("dgs.irrotational_solenoidal_polyglot are what EACH half of this split looks like.")
