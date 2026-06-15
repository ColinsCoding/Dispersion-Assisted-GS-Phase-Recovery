"""Dispersion-assisted Gerchberg-Saxton phase recovery -- a NumPy prototype.

Civilian optical-measurement context: recover the phase phi(t) of a complex
optical field from two intensity-only ("square-law detector") measurements
taken before and after a known fibre dispersion -- the carrier-less coherent
receiver idea. For fibre metrology, silicon photonics, detector modeling, and
teaching. Not a weapon or directed-energy system.

Pipeline:
  hidden field   x(t) = A(t) exp(i phi(t))
  dispersion     H(f) = exp(i pi D f^2)
  measure        I1 = |x|^2  (before),  I2 = |disperse(x, D)|^2  (after)
  recover phi    by Gerchberg-Saxton alternating projections between the planes.

Run as a script to generate synthetic data, export CSV + NPZ, run recovery, and
save plots; or import the functions.
"""

import csv
import pathlib

import numpy as np


# ── 1. synthetic hidden field ───────────────────────────────────────
def make_field(N=2048, seed=0):
    """Hidden complex field x(t) = A(t) exp(i phi(t)).

    A(t): a Gaussian main pulse + a weak satellite pulse.
    phi(t): quadratic chirp + sinusoidal ripple + small band-limited random texture.
    Returns (t, x, A, phi).
    """
    if N < 16:
        raise ValueError("N must be >= 16")
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.0, 1.0, N)

    # amplitude: main Gaussian + weak satellite
    A = np.exp(-(t + 0.05)**2 / (2 * 0.18**2))
    A += 0.35 * np.exp(-(t - 0.45)**2 / (2 * 0.07**2))   # weak satellite pulse

    # phase: quadratic chirp + sinusoidal ripple + small smooth random texture
    chirp = 18.0 * t**2
    ripple = 0.8 * np.sin(2 * np.pi * 3.0 * t + 0.6)
    texture = np.zeros(N)
    for k in (5, 7, 11):                                  # band-limited, smooth
        texture += rng.uniform(-1, 1) * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
    texture *= 0.15
    phi = chirp + ripple + texture

    x = A * np.exp(1j * phi)
    return t, x, A, phi


# ── 2. dispersion propagation ────────────────────────────────────────
def disperse(x, D):
    """Apply dispersion D: H(f) = exp(i pi D f^2), f = normalized FFT frequency."""
    N = len(x)
    f = np.fft.fftfreq(N)
    H = np.exp(1j * np.pi * D * f**2)
    return np.fft.ifft(np.fft.fft(x) * H)


# ── 3. measurements (clean + noisy) ──────────────────────────────────
def add_noise(I, snr_db, rng):
    """Add Gaussian detector noise at the given SNR; clip to non-negative."""
    noise_power = np.mean(I) * 10 ** (-snr_db / 10)
    return np.maximum(I + rng.normal(0, np.sqrt(noise_power), I.shape), 0.0)


def make_measurements(N=2048, D=6000.0, snr_db=30.0, seed=0):
    """Generate the two intensity planes, clean and noisy.

    Returns a dict: t, x, A, phi (hidden truth), D, and
    I1_clean, I2_clean, I1_noisy, I2_noisy.
    """
    t, x, A, phi = make_field(N, seed=seed)
    x2 = disperse(x, D)
    I1_clean = np.abs(x)**2
    I2_clean = np.abs(x2)**2
    rng = np.random.default_rng(seed + 1)
    return {
        "t": t, "x": x, "A": A, "phi": phi, "D": float(D),
        "I1_clean": I1_clean, "I2_clean": I2_clean,
        "I1_noisy": add_noise(I1_clean, snr_db, rng),
        "I2_noisy": add_noise(I2_clean, snr_db, rng),
    }


# ── 4/5. exports ─────────────────────────────────────────────────────
def export_csv(path, data):
    """Write t, intensities, and hidden truth to CSV."""
    path = pathlib.Path(path)
    cols = ["t", "I1_clean", "I2_clean", "I1_noisy", "I2_noisy", "A_true", "phi_true"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(data["t"])):
            w.writerow([data["t"][i], data["I1_clean"][i], data["I2_clean"][i],
                        data["I1_noisy"][i], data["I2_noisy"][i],
                        data["A"][i], data["phi"][i]])
    return path


def export_npz(path, data):
    """Write all arrays + D to a compressed .npz."""
    path = pathlib.Path(path)
    np.savez_compressed(
        path, t=data["t"], D=data["D"], A_true=data["A"], phi_true=data["phi"],
        I1_clean=data["I1_clean"], I2_clean=data["I2_clean"],
        I1_noisy=data["I1_noisy"], I2_noisy=data["I2_noisy"])
    return path


# ── 6. Gerchberg-Saxton recovery ─────────────────────────────────────
def gerchberg_saxton(I1, I2, D, n_iter=300, seed=0):
    """Recover the plane-1 field from intensities I1 (before) and I2 (after D).

    init x from sqrt(I1) with random phase; then iterate:
      propagate to plane 2 (apply D) -> replace amplitude with sqrt(I2)
      backpropagate (apply -D)       -> replace amplitude with sqrt(I1).
    Returns (x_recovered, errors) where errors is the per-iteration mismatch
    ||  |disperse(x, D)| - sqrt(I2)  || / || sqrt(I2) ||.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    A1, A2 = np.sqrt(np.maximum(I1, 0)), np.sqrt(np.maximum(I2, 0))
    rng = np.random.default_rng(seed)
    x = A1 * np.exp(1j * rng.uniform(-np.pi, np.pi, len(A1)))
    errors = []
    for _ in range(n_iter):
        x2 = disperse(x, D)                       # propagate to plane 2
        x2 = A2 * np.exp(1j * np.angle(x2))       # enforce measured I2
        x = disperse(x2, -D)                      # backpropagate to plane 1
        x = A1 * np.exp(1j * np.angle(x))         # enforce measured I1
        errors.append(float(np.linalg.norm(np.abs(disperse(x, D)) - A2)
                            / (np.linalg.norm(A2) + 1e-12)))
    return x, np.array(errors)


def compare_phase(phi_rec, phi_true, weight):
    """RMS phase error after removing the global-offset and conjugate (twin)
    ambiguities, weighted by `weight` (e.g. amplitude^2) so we score where there
    is signal. Returns (rms_error, aligned_phi)."""
    best = None
    for sign in (+1, -1):
        d = phi_true - sign * phi_rec
        offset = np.angle(np.sum(weight * np.exp(1j * d)))
        aligned = sign * phi_rec + offset
        err = np.sqrt(np.sum(weight * np.angle(np.exp(1j * (phi_true - aligned)))**2)
                      / np.sum(weight))
        if best is None or err < best[0]:
            best = (err, aligned)
    return best


# ── script entry point ───────────────────────────────────────────────
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = pathlib.Path(__file__).resolve().parent
    (root / "data").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    data = make_measurements(N=2048, D=6000.0, snr_db=30.0, seed=0)
    export_csv(root / "data" / "dispersion_gs_prototype.csv", data)
    export_npz(root / "data" / "dispersion_gs_prototype.npz", data)

    x_rec, errors = gerchberg_saxton(data["I1_noisy"], data["I2_noisy"], data["D"],
                                     n_iter=300, seed=0)
    rms, phi_aligned = compare_phase(np.angle(x_rec), data["phi"], data["A"]**2)
    print(f"recovered phase RMS error = {rms:.4f} rad   "
          f"(GS amplitude error {errors[0]:.3f} -> {errors[-1]:.3f})")

    t = data["t"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(t, data["I1_clean"], label="I1 clean")
    ax[0, 0].plot(t, data["I1_noisy"], ".", ms=2, alpha=0.4, label="I1 noisy")
    ax[0, 0].set_title("plane 1 (before dispersion)"); ax[0, 0].legend()
    ax[0, 1].plot(t, data["I2_clean"], "C1", label="I2 clean")
    ax[0, 1].plot(t, data["I2_noisy"], ".", ms=2, alpha=0.4, label="I2 noisy")
    ax[0, 1].set_title("plane 2 (after dispersion D)"); ax[0, 1].legend()
    ax[1, 0].plot(t, data["phi"], "k", lw=2, label="true phi")
    ax[1, 0].plot(t, phi_aligned, "C3--", lw=1, label=f"recovered (RMS {rms:.3f})")
    ax[1, 0].set_title("phase recovery"); ax[1, 0].legend()
    ax[1, 1].semilogy(errors, "C2"); ax[1, 1].set_title("GS convergence")
    ax[1, 1].set_xlabel("iteration"); ax[1, 1].set_ylabel("amplitude error")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "dispersion_gs_prototype.png", dpi=110)
    print("wrote data/dispersion_gs_prototype.{csv,npz} and "
          "figures/dispersion_gs_prototype.png")


if __name__ == "__main__":
    main()
