"""
ghost_imaging.py — Classical (thermal-light) ghost imaging simulation.

Ghost imaging reconstructs an object using two detectors:
  - Signal arm  : passes through object, single-pixel "bucket" detector (no spatial res)
  - Reference arm: never touches object, spatially resolved camera

The image emerges from the second-order intensity correlation:
    G²(x_r) = <I_r(x_r) · B_s> − <I_r(x_r)><B_s>

This is the classical (thermal speckle) version — no entangled photons needed.
Same math as the quantum case for intensity-correlation imaging.

Connection to TD-GS phase retrieval:
  Both problems recover hidden structure from intensity-only measurements.
  Ghost imaging: spatial domain, intensity correlations across shots.
  TD-GS:         spectral domain, intensity at two dispersion planes.

Usage:
    from ghost_imaging import GhostImager, make_object, OPTICAL_MATERIALS
    gi = GhostImager(pixel_size=64, n_shots=5000)
    obj = make_object('double_slit', pixel_size=64)
    result = gi.run(obj, seed=42)
    gi.plot(result)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Literal

# ── optical material table (refractive index at 1550 nm, thermo-optic dn/dT) ──
OPTICAL_MATERIALS: dict = {
    #  name              n_1550   dn_dT [1/K]    notes
    "fused_silica":    (1.4440,   1.10e-5,  "standard SMF-28 fiber"),
    "N-BK7":           (1.5009,   3.00e-6,  "borosilicate, common lens glass"),
    "PMMA":            (1.4760,  -8.50e-5,  "3D-printable acrylic"),
    "polycarbonate":   (1.5620,  -1.07e-4,  "3D-printable, impact-resistant"),
    "silicon":         (3.4757,   1.80e-4,  "mid-IR, photonic chips"),
    "germanium":       (4.0030,   3.96e-4,  "mid-IR, high-n GRIN possible"),
    "NOA61":           (1.5600,  -2.00e-4,  "UV-cure optical adhesive"),
    "air":             (1.0000,   0.00e+0,  "reference"),
}

def refractive_index(material: str, T_C: float = 20.0) -> float:
    """n(material, T) using first-order thermo-optic correction."""
    if material not in OPTICAL_MATERIALS:
        raise ValueError(f"Unknown material '{material}'. "
                         f"Options: {list(OPTICAL_MATERIALS)}")
    n0, dn_dT, _ = OPTICAL_MATERIALS[material]
    return n0 + dn_dT * (T_C - 20.0)


# ── object masks ───────────────────────────────────────────────────────────────

def make_object(shape: Literal['double_slit','letter_T','circle','random'],
                pixel_size: int = 64,
                **kw) -> np.ndarray:
    """
    Return a binary transmission mask T(x,y) in [0,1].
    pixel_size: number of pixels along each axis.
    """
    N = pixel_size
    T = np.zeros((N, N), dtype=float)
    cx, cy = N // 2, N // 2

    if shape == 'double_slit':
        w   = kw.get('slit_width', max(2, N // 16))
        gap = kw.get('gap',        max(4, N // 8))
        T[N//4 : 3*N//4, cx - gap//2 - w : cx - gap//2]     = 1.0
        T[N//4 : 3*N//4, cx + gap//2     : cx + gap//2 + w] = 1.0

    elif shape == 'letter_T':
        bar = max(2, N // 10)
        T[N//4      : N//4 + bar, N//4 : 3*N//4] = 1.0   # horizontal
        T[N//4 + bar: 3*N//4,    cx - bar//2 : cx + bar//2] = 1.0  # vertical

    elif shape == 'circle':
        r = kw.get('radius', N // 4)
        y, x = np.ogrid[:N, :N]
        T[(x - cx)**2 + (y - cy)**2 <= r**2] = 1.0

    elif shape == 'random':
        rng = np.random.default_rng(kw.get('seed', 0))
        T   = rng.choice([0.0, 1.0], size=(N, N), p=[0.6, 0.4])

    else:
        raise ValueError(f"Unknown shape '{shape}'")

    return T


# ── core simulator ─────────────────────────────────────────────────────────────

@dataclass
class GhostImager:
    """
    Simulate classical (thermal speckle) ghost imaging.

    Parameters
    ----------
    pixel_size : int
        Spatial pixels along each axis of the detector.
    n_shots : int
        Number of speckle realizations (more → lower noise floor).
    speckle_grain : float
        Speckle correlation length in pixels (sets spatial resolution limit).
    """
    pixel_size  : int   = 64
    n_shots     : int   = 3000
    speckle_grain: float = 4.0

    # internal — filled by run()
    _G2         : np.ndarray = field(default=None, init=False, repr=False)
    _direct     : np.ndarray = field(default=None, init=False, repr=False)

    # ── speckle generation ────────────────────────────────────────────
    def _speckle(self, rng: np.random.Generator) -> np.ndarray:
        """
        One thermal speckle realization: |IFFT(gaussian_envelope · random_phase)|²
        Grain size controlled by speckle_grain (pixels in freq domain).
        """
        N  = self.pixel_size
        fx = np.fft.fftfreq(N)
        fy = np.fft.fftfreq(N)
        FX, FY = np.meshgrid(fx, fy)
        # Gaussian envelope limits spatial bandwidth → sets grain size
        sigma_f = 1.0 / (2.0 * np.pi * self.speckle_grain)
        envelope = np.exp(-(FX**2 + FY**2) / (2.0 * sigma_f**2))
        # complex random field in frequency domain
        field_f  = envelope * (rng.standard_normal((N, N))
                               + 1j * rng.standard_normal((N, N)))
        field_t  = np.fft.ifft2(field_f)
        return np.abs(field_t)**2   # intensity

    # ── main loop ─────────────────────────────────────────────────────
    def run(self, obj: np.ndarray, seed: int = 0) -> dict:
        """
        Run ghost imaging simulation.

        Returns dict with keys:
          ghost    : reconstructed image via G²
          direct   : reference arm mean intensity (no object info)
          obj      : original object mask
          snr_db   : signal-to-noise ratio of reconstruction
          n_shots  : shots used
        """
        rng   = np.random.default_rng(seed)
        N     = self.pixel_size
        assert obj.shape == (N, N), f"Object must be ({N},{N}), got {obj.shape}"

        sum_IrB    = np.zeros((N, N))   # Σ I_ref(x) · B_sig
        sum_Ir     = np.zeros((N, N))   # Σ I_ref(x)
        sum_B      = 0.0                # Σ B_sig
        sum_Ir_sq  = np.zeros((N, N))   # for SNR

        for _ in range(self.n_shots):
            I_speckle = self._speckle(rng)          # shared illumination

            # reference arm: full spatial resolution, no object
            I_ref  = I_speckle

            # signal arm: object modulates speckle, bucket integrates
            B_sig  = np.sum(I_speckle * obj)        # scalar

            sum_IrB   += I_ref * B_sig
            sum_Ir    += I_ref
            sum_B     += B_sig
            sum_Ir_sq += I_ref**2

        n = self.n_shots
        # G²(x_r) = <I_r · B> − <I_r><B>
        G2      = sum_IrB / n - (sum_Ir / n) * (sum_B / n)
        # clip negative values (noise artifact)
        ghost   = np.clip(G2, 0, None)
        # normalise to [0,1]
        if ghost.max() > 0:
            ghost /= ghost.max()

        direct  = sum_Ir / n   # mean speckle pattern — contains no object info

        # SNR: signal power over noise power
        signal_mask = obj > 0.5
        noise_mask  = ~signal_mask
        if signal_mask.any() and noise_mask.any():
            sig_pow   = ghost[signal_mask].mean()
            noise_pow = ghost[noise_mask].std() + 1e-12
            snr_db    = 20 * np.log10(sig_pow / noise_pow)
        else:
            snr_db = float('nan')

        self._G2     = ghost
        self._direct = direct

        return {
            "ghost":   ghost,
            "direct":  direct,
            "obj":     obj,
            "snr_db":  snr_db,
            "n_shots": n,
        }

    # ── visualisation ─────────────────────────────────────────────────
    def plot(self, result: dict, title: str = "") -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(title or
                     f"Classical Ghost Imaging  |  {result['n_shots']} shots  |"
                     f"  SNR = {result['snr_db']:.1f} dB",
                     fontsize=12)

        im0 = axes[0].imshow(result["obj"],    cmap="gray")
        axes[0].set_title("Object (ground truth)")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(result["direct"], cmap="hot")
        axes[1].set_title("Reference arm\n(no object info)")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(result["ghost"],  cmap="inferno")
        axes[2].set_title(f"Ghost reconstruction G²\nSNR={result['snr_db']:.1f} dB")
        plt.colorbar(im2, ax=axes[2])

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        return fig


# ── SNR vs shots sweep ─────────────────────────────────────────────────────────

def snr_vs_shots(obj: np.ndarray,
                 shot_counts: list[int] | None = None,
                 pixel_size: int = 64,
                 seed: int = 0) -> dict:
    """
    Run ghost imaging at increasing shot counts and return SNR curve.
    Useful for showing quality improves with more measurements.
    """
    if shot_counts is None:
        shot_counts = [100, 300, 500, 1000, 2000, 5000]

    snrs = []
    for n in shot_counts:
        gi  = GhostImager(pixel_size=pixel_size, n_shots=n)
        res = gi.run(obj, seed=seed)
        snrs.append(res["snr_db"])
        print(f"  shots={n:5d}  SNR={res['snr_db']:+.1f} dB")

    return {"shot_counts": shot_counts, "snr_db": snrs}


def plot_snr_curve(sweep: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(sweep["shot_counts"], sweep["snr_db"],
                "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Number of shots (log scale)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Ghost imaging quality vs. measurement count")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="SNR=0 dB threshold")
    ax.legend()
    fig.tight_layout()
    return fig


# ── quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Optical materials at 20°C and 100°C:")
    print(f"{'Material':<18} {'n(20°C)':>10} {'n(100°C)':>10}  Notes")
    print("-" * 60)
    for mat, (n0, dn_dT, note) in OPTICAL_MATERIALS.items():
        n20  = refractive_index(mat, 20)
        n100 = refractive_index(mat, 100)
        print(f"{mat:<18} {n20:>10.4f} {n100:>10.4f}  {note}")

    print("\nRunning ghost imaging demo (double slit, 3000 shots)...")
    obj    = make_object('double_slit', pixel_size=64)
    gi     = GhostImager(pixel_size=64, n_shots=3000)
    result = gi.run(obj, seed=42)
    print(f"SNR = {result['snr_db']:.1f} dB")

    print("\nSNR vs shots sweep:")
    sweep = snr_vs_shots(obj, shot_counts=[200, 500, 1000, 3000], pixel_size=64)

    fig = gi.plot(result)
    plt.show()
    print("Done.")
