"""
optical_dashboard/app.py — Jalali Lab Optical Data Dashboard
============================================================
Flask web frontend: upload optical data → numpy/matplotlib analysis → live plots

Usage:
    cd optical_dashboard
    python app.py
    # then open http://localhost:5000

Accepts: CSV (one or two columns: time, intensity), .npy (1D or 2D array)

Plots returned:
    1. Time-domain waveform + envelope
    2. Power spectrum (FFT)  with chirp rate estimate
    3. TD-GS phase retrieval preview (1 iteration to show concept)
    4. Spectrogram (STFT)
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import io, base64, traceback

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

# ── Jalali lab constants ──────────────────────────────────────────────────────
D1_PS2  = -600.0   # ps²   DCF1 dispersion
D2_PS2  = -1200.0  # ps²   DCF2 dispersion

# ── Helper: figure → base64 PNG ──────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

# ── Helper: dark axes ─────────────────────────────────────────────────────────
BG  = (0.06, 0.06, 0.14)
FG  = (0.04, 0.04, 0.10)

def dark(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#334466")
    ax.tick_params(colors="#99aabb", labelsize=8)
    if title: ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
    if xl:    ax.set_xlabel(xl, color="#99aabb", fontsize=8)
    if yl:    ax.set_ylabel(yl, color="#99aabb", fontsize=8)

# ── Load uploaded file ────────────────────────────────────────────────────────
def load_data(file_storage):
    filename = file_storage.filename.lower()
    raw = file_storage.read()

    if filename.endswith(".npy"):
        arr = np.load(io.BytesIO(raw))
        if arr.ndim == 2 and arr.shape[1] == 2:
            t, y = arr[:, 0], arr[:, 1]
        elif arr.ndim == 2:
            y = arr.ravel()
            t = np.arange(len(y))
        else:
            y = arr.ravel()
            t = np.arange(len(y))
    else:
        # CSV / TXT
        txt = raw.decode("utf-8", errors="ignore")
        rows = []
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
        arr = np.array(rows)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            t, y = arr[:, 0], arr[:, 1]
        elif arr.ndim == 2:
            y = arr.ravel()
            t = np.arange(len(y))
        else:
            y = arr.ravel()
            t = np.arange(len(y))

    # Normalise
    y = y.astype(float)
    if np.ptp(y) > 0:
        y = (y - y.min()) / np.ptp(y)
    return t.astype(float), y

# ── TD-GS one-iteration preview ───────────────────────────────────────────────
def tdgs_preview(y):
    """Single GS iteration: E → D1-plane → D2-plane, return phase."""
    N  = len(y)
    nu = np.fft.fftfreq(N)

    E_str = np.sqrt(np.maximum(y, 0)).astype(complex)
    H_fwd = np.exp(1j * np.pi * (D2_PS2 - D1_PS2) * nu**2)

    E2 = np.fft.ifft(np.fft.fft(E_str) * H_fwd)
    phi = np.angle(E2)
    return phi

# ── Main analysis function ────────────────────────────────────────────────────
def analyse(t, y):
    N    = len(y)
    dt   = float(np.diff(t).mean()) if len(t) > 1 else 1.0
    fs   = 1.0 / dt if dt != 0 else 1.0
    freqs = np.fft.rfftfreq(N, d=dt)
    Y    = np.fft.rfft(y)
    psd  = np.abs(Y)**2
    # Peak frequency
    f_peak = freqs[np.argmax(psd[1:]) + 1] if len(psd) > 1 else 0.0

    # Envelope via Hilbert
    analytic = signal.hilbert(y)
    envelope = np.abs(analytic)

    # Spectrogram
    nperseg = min(max(64, N // 32), 512)
    f_spec, t_spec, Sxx = signal.spectrogram(y, fs=fs, nperseg=nperseg,
                                              noverlap=nperseg//2)

    # Chirp rate: track peak frequency vs time in spectrogram
    peak_f_vs_t = f_spec[np.argmax(Sxx, axis=0)]

    # TD-GS phase preview
    phi_preview = tdgs_preview(y)

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(FG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # Panel A: waveform + envelope
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, y,        color="#50d8ff", lw=0.8, alpha=0.7, label="signal")
    ax0.plot(t, envelope, color="#ffd040", lw=1.4, label="envelope")
    ax0.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax0, f"Time Domain  (N={N})", "t", "amplitude")

    # Panel B: power spectrum
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.semilogy(freqs, np.maximum(psd, 1e-12), color="#00ff9f", lw=1.0)
    ax1.axvline(f_peak, color="#ff3278", lw=1, ls="--",
                label=f"f_peak={f_peak:.3g}")
    ax1.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax1, "Power Spectrum |FFT|²", "frequency", "PSD")

    # Panel C: spectrogram
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.pcolormesh(t_spec, f_spec,
                        10 * np.log10(np.maximum(Sxx, 1e-20)),
                        cmap="inferno", shading="auto")
    ax2.plot(t_spec, peak_f_vs_t, color="white", lw=0.8, ls="--", label="peak f")
    fig.colorbar(im, ax=ax2, label="dB").ax.tick_params(colors="#99aabb", labelsize=7)
    ax2.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax2, "STFT Spectrogram", "t", "frequency")

    # Panel D: TD-GS phase preview
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, phi_preview, color="#cc88ff", lw=0.9)
    dark(ax3, "TD-GS Phase Preview (1 GS iter)\nD1=-600ps²  D2=-1200ps²",
         "t", "φ [rad]")

    # Panel E: instantaneous frequency
    ax4 = fig.add_subplot(gs[1, 1])
    f_inst = np.diff(np.unwrap(np.angle(analytic))) / (2 * np.pi * dt)
    ax4.plot(t[:-1], f_inst, color="#ffd040", lw=0.8)
    dark(ax4, "Instantaneous Frequency\n(Hilbert)", "t", "f_inst")

    # Panel F: autocorrelation
    ax5 = fig.add_subplot(gs[1, 2])
    lags = signal.correlation_lags(len(y), len(y), mode="full")
    acor = np.correlate(y - y.mean(), y - y.mean(), mode="full")
    acor /= acor.max() + 1e-12
    center = len(acor) // 2
    half   = min(N // 4, 500)
    ax5.plot(lags[center - half: center + half] * dt,
             acor[center - half: center + half],
             color="#50d8ff", lw=0.8)
    ax5.axhline(0, color="gray", lw=0.5, ls=":")
    dark(ax5, "Autocorrelation", "lag", "R(τ)")

    fig.suptitle("Jalali Lab Optical Data Dashboard",
                 color="white", fontsize=11, fontweight="bold")

    stats = {
        "N":          int(N),
        "dt":         f"{dt:.4g}",
        "fs":         f"{fs:.4g}",
        "peak_freq":  f"{f_peak:.4g}",
        "rms":        f"{float(np.sqrt(np.mean(y**2))):.4f}",
        "dyn_range":  f"{float(20*np.log10(np.maximum(y.max()/np.maximum(y[y>0].min(),1e-9),1))):.1f} dB",
    }
    return fig_to_b64(fig), stats

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        t, y = load_data(f)
        if len(y) < 16:
            return jsonify({"error": f"Too few samples: {len(y)}"}), 400

        plot_b64, stats = analyse(t, y)
        return jsonify({"plot": plot_b64, "stats": stats})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/demo")
def demo():
    """Generate synthetic STEAM-like chirped pulse for demo."""
    N  = 4096
    t  = np.linspace(-200, 200, N)   # ps
    # Stretched Gaussian pulse (dispersion D1=-600 ps²)
    sigma_t = 30.0    # ps stretched width
    E_str   = np.exp(-t**2 / (2 * sigma_t**2))
    # Add fake cell phase modulation
    phi_cell = 0.8 * np.exp(-((t - 10)/15)**2) * np.sin(2*np.pi*t/60)
    y = E_str**2 * (1 + 0.3 * np.cos(phi_cell))
    y += 0.02 * np.random.default_rng(42).normal(size=N)
    y = np.clip(y, 0, None)
    plot_b64, stats = analyse(t, y)
    return jsonify({"plot": plot_b64, "stats": stats, "source": "synthetic STEAM pulse"})

if __name__ == "__main__":
    print("Jalali Lab Optical Dashboard → http://localhost:5000")
    print("Demo endpoint → http://localhost:5000/demo")
    app.run(debug=True, port=5000)
