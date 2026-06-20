"""Neon domain-colouring of the carrier-less optical field.

A "square-law" (carrier-less) receiver throws the phase away -- it only records
|E|^2. This module is the opposite view: it paints the *full* complex field
E(t) = A(t) exp(i phi(t)) back into colour, the way the Gerchberg-Saxton
recovery hands the phase back to you.

The mapping is standard complex domain-colouring, in a neon palette:
    hue        <- phase  phi(t)        (the cyclic quantity -> the cyclic axis)
    brightness <- |E|    (intensity)   (gamma-lifted so weak light still glows)
    saturation <- 1                    (full neon)

Two renders are produced:
  (a) a neon phase-ribbon of the 1-D field E(t), and
  (b) the complex short-time Fourier transform painted as a 2-D neon image --
      the time-frequency "colour photo" of the pulse.

Civilian optical metrology / education. Not a weapon or directed-energy system.
"""

import pathlib

import numpy as np
from matplotlib.colors import hsv_to_rgb

import dispersion_gs_prototype as dg


# ── neon colour map: complex value -> RGB ────────────────────────────
def phase_to_neon_rgb(phase, intensity=None, hue_shift=0.5, gamma=0.6, floor=0.06):
    """Domain-colour a complex field in a neon palette.

    phase     : array of phases (rad), any range -- wrapped to a hue.
    intensity : non-negative array (|E|^2 or |E|); sets brightness. None -> flat.
    hue_shift : rotate the colour wheel so phi=0 lands on cyan-ish neon.
    gamma     : <1 lifts dark values so low-light regions still glow.
    floor     : minimum brightness, so the neon never goes fully black.
    Returns an RGB array with a trailing length-3 axis, values in [0, 1].
    """
    phase = np.asarray(phase, dtype=float)
    hue = (phase / (2 * np.pi) + hue_shift) % 1.0          # cyclic phase -> hue
    if intensity is None:
        val = np.ones_like(hue)
    else:
        I = np.asarray(intensity, dtype=float)
        if np.any(I < 0):
            raise ValueError("intensity must be non-negative")
        amp = np.sqrt(np.maximum(I, 0.0))                  # |E| from |E|^2
        peak = amp.max() or 1.0
        val = floor + (1.0 - floor) * (amp / peak) ** gamma
    sat = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv_to_rgb(hsv)


# ── tiny self-contained STFT (no scipy) ──────────────────────────────
def stft(x, win=128, hop=16):
    """Complex short-time Fourier transform with a Hann window.

    Returns (S, t_idx, f_idx): S has shape (n_freq, n_frame), complex.
    """
    x = np.asarray(x)
    if win > len(x):
        raise ValueError("window longer than signal")
    w = np.hanning(win)
    starts = range(0, len(x) - win + 1, hop)
    cols = [np.fft.fftshift(np.fft.fft(x[s:s + win] * w)) for s in starts]
    S = np.stack(cols, axis=1)
    t_idx = np.array([s + win / 2 for s in starts])
    f_idx = np.fft.fftshift(np.fft.fftfreq(win))
    return S, t_idx, f_idx


# ── the figure ───────────────────────────────────────────────────────
def render(N=2048, D=6000.0, seed=7, ribbon_h=200,
           out="figures/gallery_neon_carrierless.png", show=False):
    """Render the neon carrier-less gallery and save it. Returns the figure."""
    import matplotlib.pyplot as plt

    t, x, A, phi = dg.make_field(N=N, seed=seed)
    I = np.abs(x) ** 2

    # (a) neon phase-ribbon: one colour column per time sample, glow = intensity
    row = phase_to_neon_rgb(phi, I)                         # (N, 3)
    ribbon = np.broadcast_to(row, (ribbon_h, N, 3))

    # (b) the dispersed field, then its complex STFT, domain-coloured.
    # gamma is pushed low here: the chirp energy piles up near f=0, so a strong
    # lift is needed to make the faint dispersed arms glow instead of crushing.
    xd = dg.disperse(x, D)
    S, ti, fi = stft(xd, win=256, hop=8)
    img = phase_to_neon_rgb(np.angle(S), np.abs(S) ** 2, gamma=0.35)
    fband = np.abs(fi) <= 0.16                              # zoom to where the energy lives
    img, fi = img[fband], fi[fband]

    fig = plt.figure(figsize=(12, 7), facecolor="#05070d")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.4], hspace=0.32)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(ribbon, aspect="auto", extent=[t[0], t[-1], 0, 1], origin="lower")
    ax0.set_yticks([])
    ax0.set_xlabel("time  t", color="#9fb3c8")
    ax0.set_title("carrier-less field  E(t)=A(t)e^{iφ(t)}   —   hue = phase, glow = intensity",
                  color="#e6edf3", fontsize=12)

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(img, aspect="auto", origin="lower",
               extent=[ti[0], ti[-1], fi[0], fi[-1]])
    ax1.set_xlabel("time", color="#9fb3c8")
    ax1.set_ylabel("frequency", color="#9fb3c8")
    ax1.set_title(f"complex STFT after dispersion D={D:.0f}  —  neon domain-colouring",
                  color="#e6edf3", fontsize=12)

    # neon phase-wheel legend (inset)
    axw = fig.add_axes([0.86, 0.78, 0.11, 0.18], projection="polar")
    th = np.linspace(0, 2 * np.pi, 256)
    rr = np.linspace(0.4, 1.0, 16)
    TH, RR = np.meshgrid(th, rr)
    wheel = phase_to_neon_rgb(TH, RR ** 2)
    axw.pcolormesh(th, rr, np.zeros_like(TH), color=wheel.reshape(-1, 3), shading="auto")
    axw.set_yticks([]); axw.set_xticks([])
    axw.set_title("phase", color="#9fb3c8", fontsize=9, pad=2)
    axw.set_facecolor("#05070d")

    for ax in (ax0, ax1):
        ax.tick_params(colors="#9fb3c8")
        for sp in ax.spines.values():
            sp.set_color("#1b2735")

    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


if __name__ == "__main__":
    p = render()
    print("wrote", p)
