#!/usr/bin/env python3
"""
Animate GS convergence for i1 and i2.

Saves:
  figures/gs_convergence_i1_i2.mp4
  figures/gs_convergence_i1_i2.gif
  figures/gs_convergence_i1_i2_frame.png

Usage:
  python tools/animate_gs_convergence.py [--frames N] [--recompute] [--fs FS] [--outfile PATH] [--gif PATH]
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

def safe_load_npz(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            try:
                return np.load(str(p), allow_pickle=True), p
            except Exception as e:
                print("Failed to load", p, ":", e)
    return None, None

def ensure_1d_numeric(arr, dtype=np.float64):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim > 1:
        a = a.ravel()
    try:
        return a.astype(dtype)
    except Exception:
        try:
            return np.array([float(x) for x in a.ravel()], dtype=dtype)
        except Exception:
            return None

def reconstruct_iterates_from_mag(mag, n_iter, pad_factor=1, seed=0):
    """Return iterates array shape (n_iter, N) using GS in time domain."""
    mag = ensure_1d_numeric(mag, dtype=np.float64)
    if mag is None:
        raise ValueError("measured magnitude invalid")
    N = mag.size
    np.random.seed(seed)
    phi = np.random.uniform(-np.pi, np.pi, size=N).astype(np.float64)
    f = (mag * np.exp(1j * phi)).astype(np.complex128)
    iterates = np.zeros((n_iter, N), dtype=np.complex128)
    for k in range(n_iter):
        F = np.fft.fft(f, n=N*pad_factor)
        f_back = np.fft.ifft(F, n=N*pad_factor)[:N].astype(np.complex128)
        iterates[k] = f_back
        f = (mag * np.exp(1j * np.angle(f_back))).astype(np.complex128)
    return iterates

def compute_freq_axis(nfft, fs):
    if fs is None:
        return np.fft.fftfreq(nfft, d=1.0)
    return np.fft.fftfreq(nfft, d=1.0/fs)

def load_measurements():
    # Try artifact first, then pipeline_input.npz in common locations
    candidates = [
        "figures/recon_artifact.npz",
        "pipeline_input.npz",
        "data/raw/pipeline_input.npz",
        "data/pipeline_input.npz"
    ]
    d, path = safe_load_npz(candidates)
    if d is None:
        raise FileNotFoundError("No recon_artifact.npz or pipeline_input.npz found in expected locations.")
    print("Loaded:", path)
    keys = list(d.files)
    print("Keys in file:", keys)
    # Extract arrays if present
    i1 = d.get("i1", None)
    i2 = d.get("i2", None)
    f_est = d.get("f_est", None)
    errs = d.get("errs", None)
    freq = d.get("freq", None)
    time_axis = d.get("time_axis", None)
    metadata = d.get("_metadata", None)
    return dict(i1=i1, i2=i2, f_est=f_est, errs=errs, freq=freq, time_axis=time_axis, metadata=metadata)

def infer_fs_from_metadata(metadata):
    if metadata is None:
        return None
    try:
        md = metadata.item() if isinstance(metadata, np.ndarray) and metadata.size==1 else metadata
        if isinstance(md, dict):
            for key in ("fs","Fs","sampling_rate","sample_rate"):
                if key in md:
                    return float(md[key])
            if "dt" in md:
                dt = float(md["dt"])
                if dt>0:
                    return 1.0/dt
    except Exception:
        pass
    return None

def prepare_iterates_for_signal(sig, f_est, errs, recompute, fs_override):
    sig_mag = np.abs(sig) if np.iscomplexobj(sig) else np.abs(ensure_1d_numeric(sig))
    if sig_mag is None:
        raise RuntimeError("Signal magnitude invalid")
    # If f_est is present and looks like iterates (n_iter, N), use it
    if f_est is not None:
        a = np.asarray(f_est)
        if a.ndim >= 2 and a.shape[-1] == sig_mag.size:
            # assume shape (n_iter, N) or (n_iter, N, 1)
            n_iter = a.shape[0]
            iterates = a.reshape((n_iter, sig_mag.size)).astype(np.complex128)
            used_reconstructed = False
            return iterates, n_iter, used_reconstructed
    # Otherwise, reconstruct if errs present or recompute requested
    if errs is not None or recompute:
        n_iter = int(errs.size) if errs is not None and errs.size>0 else 100
        iterates = reconstruct_iterates_from_mag(sig_mag, n_iter=n_iter, pad_factor=1, seed=0)
        used_reconstructed = True
        return iterates, n_iter, used_reconstructed
    # fallback: single-frame from measured magnitude (no iterates)
    iterates = np.atleast_2d(sig_mag.astype(np.complex128))
    n_iter = iterates.shape[0]
    used_reconstructed = False
    return iterates, n_iter, used_reconstructed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--outfile", type=str, default="figures/gs_convergence_i1_i2.mp4")
    p.add_argument("--gif", type=str, default="figures/gs_convergence_i1_i2.gif")
    args = p.parse_args()

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = load_measurements()
    except Exception as e:
        print("ERROR loading measurements:", e)
        sys.exit(2)

    i1 = ensure_1d_numeric(data.get("i1"))
    i2 = ensure_1d_numeric(data.get("i2"))
    if i1 is None and i2 is None:
        print("No i1 or i2 found in the data file.")
        sys.exit(3)
    if i1 is None:
        i1 = np.zeros_like(i2)
    if i2 is None:
        i2 = np.zeros_like(i1)

    f_est = data.get("f_est", None)
    errs = data.get("errs", None)
    freq = data.get("freq", None)
    time_axis = data.get("time_axis", None)
    metadata = data.get("metadata", data.get("_metadata", None))

    # infer fs
    fs = args.fs if args.fs else infer_fs_from_metadata(metadata)
    if fs:
        print("Using sampling frequency fs =", fs)
    else:
        print("No sampling frequency inferred; using sample indices for time/frequency axes.")

    # prepare iterates for i1 and i2
    iter_i1, niter1, rec1 = prepare_iterates_for_signal(i1, f_est, errs, args.recompute, fs)
    iter_i2, niter2, rec2 = prepare_iterates_for_signal(i2, f_est, errs, args.recompute, fs)
    n_iter = max(niter1, niter2)
    print(f"i1 shape: {i1.shape}, i2 shape: {i2.shape}")
    print(f"Iterates: i1 -> {iter_i1.shape}, reconstructed={rec1}; i2 -> {iter_i2.shape}, reconstructed={rec2}")
    print("Using n_iter =", n_iter)

    # ensure both iterates have same number of frames by padding last frame if needed
    def pad_iterates(it, target):
        if it.shape[0] >= target:
            return it
        pad = np.repeat(it[-1:,...], target - it.shape[0], axis=0)
        return np.concatenate([it, pad], axis=0)
    iter_i1 = pad_iterates(iter_i1, n_iter)
    iter_i2 = pad_iterates(iter_i2, n_iter)

    N = i1.size
    nfft = max(1, N)
    freq_axis = freq if freq is not None else compute_freq_axis(nfft, fs)

    # compute magnitudes for limits
    mag_i1 = np.abs(np.fft.fft(iter_i1, n=nfft, axis=1))
    mag_i2 = np.abs(np.fft.fft(iter_i2, n=nfft, axis=1))
    time_axis_plot = (np.arange(N)/fs) if fs else np.arange(N)

    # y-limits
    tmin, tmax = 0, N-1
    mag_time_max = max(np.max(np.abs(i1)), np.max(np.abs(i2)), np.max(np.abs(iter_i1)), np.max(np.abs(iter_i2)))
    mag_freq_max = max(mag_i1.max(), mag_i2.max()) if mag_i1.size and mag_i2.size else 1.0

    # frames sampling
    frames = min(args.frames, n_iter)
    if n_iter <= frames:
        frame_indices = np.arange(n_iter)
    else:
        frame_indices = np.linspace(0, n_iter-1, frames, dtype=int)

    # build figure with 3 columns x 2 rows
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    (ax_t1, ax_f1, ax_e1), (ax_t2, ax_f2, ax_e2) = axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]

    # top-left: i1 time
    ax_t1.plot(time_axis_plot, np.abs(i1), color='gray', lw=1, label='measured i1')
    line_t1, = ax_t1.plot([], [], color='tab:blue', lw=1.5, label='recon i1')
    ax_t1.set_xlim(time_axis_plot.min(), time_axis_plot.max())
    ax_t1.set_ylim(0, mag_time_max*1.05)
    ax_t1.set_title("i1 time-domain")
    ax_t1.legend(loc='upper right')

    # bottom-left: i2 time
    ax_t2.plot(time_axis_plot, np.abs(i2), color='gray', lw=1, label='measured i2')
    line_t2, = ax_t2.plot([], [], color='tab:orange', lw=1.5, label='recon i2')
    ax_t2.set_xlim(time_axis_plot.min(), time_axis_plot.max())
    ax_t2.set_ylim(0, mag_time_max*1.05)
    ax_t2.set_title("i2 time-domain")
    ax_t2.legend(loc='upper right')

    # middle column: frequency magnitude (linear)
    ax_f1.set_title("i1 frequency magnitude")
    line_f1, = ax_f1.plot([], [], color='tab:blue')
    ax_f1.set_xlim(freq_axis.min(), freq_axis.max())
    ax_f1.set_ylim(0, mag_freq_max*1.05)

    ax_f2.set_title("i2 frequency magnitude")
    line_f2, = ax_f2.plot([], [], color='tab:orange')
    ax_f2.set_xlim(freq_axis.min(), freq_axis.max())
    ax_f2.set_ylim(0, mag_freq_max*1.05)

    # right column: error history
    # compute simple error metric: MSE between magnitude of current iterate and measured magnitude
    def compute_errs(iterates, measured_mag):
        measured_mag = np.abs(measured_mag)
        errs = np.mean((np.abs(iterates) - measured_mag[None,:])**2, axis=1)
        return errs
    errs_i1 = compute_errs(iter_i1, np.abs(i1))
    errs_i2 = compute_errs(iter_i2, np.abs(i2))
    ax_e1.set_title("i1 GS error")
    ax_e1.set_xlim(0, n_iter-1)
    ax_e1.set_ylim(max(1e-16, min(errs_i1.min(), errs_i2.min())), max(errs_i1.max(), errs_i2.max())*1.05)
    line_e1, = ax_e1.plot([], [], color='tab:blue')
    marker_e1, = ax_e1.plot([], [], 'o', color='tab:blue')

    ax_e2.set_title("i2 GS error")
    ax_e2.set_xlim(0, n_iter-1)
    ax_e2.set_ylim(max(1e-16, min(errs_i1.min(), errs_i2.min())), max(errs_i1.max(), errs_i2.max())*1.05)
    line_e2, = ax_e2.plot([], [], color='tab:orange')
    marker_e2, = ax_e2.plot([], [], 'o', color='tab:orange')

    fig.suptitle("GS Convergence: i1 (top) vs i2 (bottom)")

    def init():
        line_t1.set_data([], [])
        line_t2.set_data([], [])
        line_f1.set_data([], [])
        line_f2.set_data([], [])
        line_e1.set_data([], [])
        marker_e1.set_data([], [])
        line_e2.set_data([], [])
        marker_e2.set_data([], [])
        return (line_t1, line_t2, line_f1, line_f2, line_e1, marker_e1, line_e2, marker_e2)

    def update(frame_idx):
        it = frame_indices[frame_idx]
        # time plots
        y_t1 = np.abs(iter_i1[it])
        y_t2 = np.abs(iter_i2[it])
        line_t1.set_data(time_axis_plot, y_t1)
        line_t2.set_data(time_axis_plot, y_t2)
        # freq plots
        Y1 = np.abs(np.fft.fft(iter_i1[it], n=nfft))
        Y2 = np.abs(np.fft.fft(iter_i2[it], n=nfft))
        line_f1.set_data(freq_axis, Y1)
        line_f2.set_data(freq_axis, Y2)
        # error plots
        line_e1.set_data(np.arange(it+1), errs_i1[:it+1])
        marker_e1.set_data([it], [errs_i1[it]])
        line_e2.set_data(np.arange(it+1), errs_i2[:it+1])
        marker_e2.set_data([it], [errs_i2[it]])
        # update titles with iteration
        ax_t1.set_xlabel(f"iteration {it+1}/{n_iter}")
        return (line_t1, line_t2, line_f1, line_f2, line_e1, marker_e1, line_e2, marker_e2)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), init_func=init, blit=False, interval=50)

    # save outputs
    mp4_path = Path(args.outfile)
    gif_path = Path(args.gif)
    png_path = out_dir / "gs_convergence_i1_i2_frame.png"

    try:
        # try ffmpeg writer for mp4
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='gs-anim'))
        ani.save(str(mp4_path), writer=writer, dpi=150)
        print("Saved MP4 to", mp4_path.resolve())
    except Exception as e:
        print("ffmpeg writer not available or failed:", e)
        try:
            ani.save(str(gif_path), writer='pillow', fps=15)
            print("Saved GIF to", gif_path.resolve())
        except Exception as e2:
            print("Failed to save GIF:", e2)

    # always save a representative frame (last frame)
    update(len(frame_indices)-1)
    fig.savefig(str(png_path), dpi=150)
    print("Saved representative PNG to", png_path.resolve())

    # ensure gif exists (try pillow if mp4 saved)
    if not gif_path.exists():
        try:
            ani.save(str(gif_path), writer='pillow', fps=15)
            print("Saved GIF to", gif_path.resolve())
        except Exception as e:
            print("Failed to save GIF:", e)

    print("Created files (if present):")
    for f in (mp4_path, gif_path, png_path):
        if f.exists():
            print("  -", f.resolve())
    print("Done.")

if __name__ == "__main__":
    import argparse
    main()
