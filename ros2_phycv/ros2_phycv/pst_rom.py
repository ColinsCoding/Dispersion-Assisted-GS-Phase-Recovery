"""ROM-quantized PST: the hardware-faithful path.

Purpose:
    Reproduce, in software, exactly what the FPGA block-RAM coefficient ROM computes:
    the isotropic PST phase kernel stored as a 1-D radial cos/sin table in signed
    fixed-point, then reconstructed for any frame by rho-address lookup. Publishing this
    from the ROS2 node makes the running system bit-match the hardware.

Equations (see physics_repo/notebooks/pst_page_phase_kernels_rom):
    C(rho) = L(rho) exp(i phi_r(rho)); stored as int rom_re/rom_im over rho in [0, R],
    R = sqrt(1/2) (the max |fftfreq| radius, size-independent). Reconstruct:
    kernel(u,v) = (rom_re[a] + i rom_im[a]) / (2^(bits-1) - 1),  a = round(rho/R (Nbin-1)).

Assumptions:
    - Isotropic PST kernel (radial symmetry is what makes the 1-D ROM valid).
Limitations:
    - Nearest-bin lookup (no interpolation), matching a plain ROM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ros2_phycv.pst_core import PstParams, PstResult, apply_phase_kernel

__all__ = ["RadialCoeffRom", "build_radial_rom", "phase_stretch_transform_rom"]

_RHO_MAX = float(np.sqrt(0.5))  # max radius of numpy.fft.fftfreq grid, independent of image size


@dataclass(frozen=True)
class RadialCoeffRom:
    """A quantized 1-D radial coefficient ROM (signed fixed-point real/imag tables)."""

    re: np.ndarray  # signed-int table of Re{C(rho)} * scale
    im: np.ndarray  # signed-int table of Im{C(rho)} * scale
    n_bits: int = 8

    @property
    def scale(self) -> int:
        """Fixed-point scale, 2^(n_bits-1) - 1."""
        return 2 ** (self.n_bits - 1) - 1

    @property
    def n_bins(self) -> int:
        """Number of radial ROM entries."""
        return int(self.re.shape[0])

    def address(self, shape: tuple[int, int]) -> np.ndarray:
        """ROM address (rho-bin index) for every frequency sample of a `shape` frame."""
        m, n = shape
        u = np.fft.fftfreq(m)[:, None]
        v = np.fft.fftfreq(n)[None, :]
        rho = np.sqrt(u**2 + v**2)
        return np.clip(np.round(rho / _RHO_MAX * (self.n_bins - 1)).astype(int), 0, self.n_bins - 1)

    def kernel(self, shape: tuple[int, int]) -> np.ndarray:
        """Reconstruct the 2-D complex phase kernel for a `shape` frame from the ROM."""
        a = self.address(shape)
        return (self.re[a].astype(float) + 1j * self.im[a].astype(float)) / self.scale


def build_radial_rom(params: PstParams | None = None, n_bins: int = 256, n_bits: int = 8) -> RadialCoeffRom:
    """Quantize the isotropic PST coefficient C(rho) into a signed fixed-point radial ROM."""
    params = params or PstParams()
    r = np.linspace(0.0, _RHO_MAX, n_bins)
    g = lambda x: x * np.arctan(x) - 0.5 * np.log1p(x**2)
    phi = params.strength * g(params.warp * r) / g(params.warp * _RHO_MAX)  # normalized so phi(R)=strength
    lpf = np.exp(-0.5 * (r / params.sigma_lpf) ** 2)
    coeff = lpf * np.exp(1j * phi)
    scale = 2 ** (n_bits - 1) - 1
    re = np.round(np.clip(coeff.real, -1.0, 1.0) * scale).astype(np.int16)
    im = np.round(np.clip(coeff.imag, -1.0, 1.0) * scale).astype(np.int16)
    return RadialCoeffRom(re=re, im=im, n_bits=n_bits)


def phase_stretch_transform_rom(
    gray: np.ndarray, params: PstParams | None = None, rom: RadialCoeffRom | None = None
) -> PstResult:
    """PST using the quantized coefficient ROM (bit-matches the FPGA data path)."""
    params = params or PstParams()
    gray = np.asarray(gray, dtype=float)
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {gray.shape}")
    rom = rom or build_radial_rom(params)
    return apply_phase_kernel(gray, rom.kernel(gray.shape), params.threshold)
