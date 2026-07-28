"""Standalone script version of the plotting cells from
notebooks/phycv_pst_sympy_torch_image_and_audio.ipynb -- regenerates the
same two PNGs without needing to open Jupyter:

  notebooks/phycv_pst_blocky_image.png       -- procedural blocky test scene + PST
  notebooks/phycv_pst_audio_spectrogram.png  -- synthesized melody spectrogram + PST

No Minecraft assets, no recorded music: the "blocky scene" is a
procedurally generated grid of solid-color squares, and the "melody" is
three synthesized pure tones with a pluck-style envelope.

Run:
    python scripts/visualize_phycv_pst_results.py
"""

import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt

from phycv.pst import PST

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "notebooks"


def make_blocky_image(n_blocks=4, block=32, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((n_blocks * block, n_blocks * block), dtype=np.float32)
    shades = rng.uniform(0.2, 1.0, size=(n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            img[i * block:(i + 1) * block, j * block:(j + 1) * block] = shades[i, j]
    return img


def run_pst(img, S=0.5, W=15, sigma_LPF=0.1):
    pst = PST()
    pst.load_img(img_array=img)
    pst.init_kernel(S=S, W=W)
    pst.apply_kernel(sigma_LPF=sigma_LPF, thresh_min=None, thresh_max=None, morph_flag=0)
    return pst.pst_output


def plot_blocky_image():
    img = make_blocky_image()
    feat = run_pst(img)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Procedural blocky test scene\n(not Minecraft assets)")
    ax[1].imshow(feat, cmap="gray")
    ax[1].set_title("PhyCV PST output (CPU)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    out = OUT_DIR / "phycv_pst_blocky_image.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def make_melody(sr=8000, note_dur=0.3, freqs=(261.63, 329.63, 392.00)):
    t_note = np.arange(int(note_dur * sr)) / sr
    envelope = np.exp(-6 * t_note)
    melody = np.concatenate([envelope * np.sin(2 * np.pi * f * t_note) for f in freqs])
    return melody.astype(np.float32)


def spectrogram_image(sig, n_fft=256, hop=64):
    sig_t = torch.from_numpy(sig)
    stft = torch.stft(sig_t, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True)
    logmag = torch.log1p(stft.abs())
    span = logmag.max() - logmag.min()
    if span < 1e-8:
        return torch.zeros_like(logmag).numpy().astype(np.float32)
    return ((logmag - logmag.min()) / span).numpy().astype(np.float32)


def plot_audio_spectrogram():
    melody = make_melody()
    near_silence = (1e-6 * np.random.default_rng(0).standard_normal(len(melody))).astype(np.float32)

    img_melody = spectrogram_image(melody)
    img_silence = spectrogram_image(near_silence)
    feat_melody = run_pst(img_melody)
    feat_silence = run_pst(img_silence)

    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    ax[0, 0].imshow(img_melody, origin="lower", aspect="auto", cmap="magma")
    ax[0, 0].set_title("Spectrogram: 3 synthesized notes")
    ax[0, 1].imshow(feat_melody, origin="lower", aspect="auto", cmap="gray")
    ax[0, 1].set_title("PST output (melody)")
    ax[1, 0].imshow(img_silence, origin="lower", aspect="auto", cmap="magma")
    ax[1, 0].set_title("Spectrogram: near-silence")
    ax[1, 1].imshow(feat_silence, origin="lower", aspect="auto", cmap="gray")
    ax[1, 1].set_title("PST output (near-silence)")
    for a in ax.ravel():
        a.set_xlabel("time frame")
        a.set_ylabel("freq bin")
    plt.tight_layout()
    out = OUT_DIR / "phycv_pst_audio_spectrogram.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_blocky_image()
    plot_audio_spectrogram()
