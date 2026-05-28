#!/usr/bin/env python3
"""Real-time-ish phase-recovery visualizer from phase_recovery_v8_final.ipynb.

This is CPU/NumPy, intended for exploration before porting kernels to CUDA.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

C_LIGHT, LAMBDA_0 = 2.99792458e8, 1550e-9
GHZ, NS, PS = 1e9, 1e-9, 1e-12


def phi2_from_D(D_ps_per_nm, lam=LAMBDA_0):
    return -(D_ps_per_nm * 1e-12 / 1e-9) * lam**2 / (2*np.pi*C_LIGHT)


def fftc(x):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def ifftc(X):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X)))


def h_gvd(omega, phi2):
    return np.exp(1j * 0.5 * phi2 * omega**2)


def disperse(et, omega, phi2):
    return ifftc(fftc(et) * h_gvd(omega, phi2))


def intensity(x):
    return np.abs(x)**2


def npeak(x, eps=1e-15):
    return x / max(np.max(np.abs(x)), eps)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))


def make_grid(N, df):
    f_hz = (np.arange(N) - N//2) * df
    w_rad = 2*np.pi*f_hz
    dt = 1.0/(N*df)
    t_s = (np.arange(N) - N//2) * dt
    return t_s, f_hz, w_rad


def envelope(f, sig=25*GHZ):
    return np.exp(-f**2/(2*sig**2))


def make_3line(f, centers=(35,40,45), widths=(1.7,1.5,1.7), depths=(0.72,0.88,0.72)):
    S = envelope(f).copy()
    for c, w, d in zip(centers, widths, depths):
        S *= (1 - d*np.exp(-((f - c*GHZ)/(w*GHZ))**2))
    return np.sqrt(np.clip(S, 0, None)).astype(complex)


def bits(t, br=10e9, n=64, C=0.0, seed=2):
    r = np.random.default_rng(seed)
    Tb = 1/br
    sig = 0.3*Tb
    centers = (np.arange(n) - n/2)*Tb
    b = r.integers(0, 2, size=n)
    out = np.zeros_like(t, dtype=complex)
    for bit, tc in zip(b, centers):
        if bit:
            out += np.exp(-(1+1j*C) * (t - tc)**2 / (2*sig**2))
    return out


def make_waveform(name, t_s, f_hz):
    if name == "gaussian":
        return np.exp(-t_s**2 / (2*(50*PS)**2)).astype(complex)
    if name == "chirped_gaussian":
        return np.exp(-(1+3j) * t_s**2 / (2*(50*PS)**2))
    if name == "gas3":
        return ifftc(make_3line(f_hz))
    if name == "chirped_nrz":
        return bits(t_s, C=2.0, seed=2)
    raise ValueError(f"unknown waveform: {name}")


def td_gs_live(I1, I2, omega, f_hz, p1, p2, true_E0, n_iter=250, support=None, seed=0, live=False):
    r = np.random.default_rng(seed)
    m1, m2 = np.sqrt(np.maximum(I1, 0)), np.sqrt(np.maximum(I2, 0))
    e1 = m1 * np.exp(1j * r.uniform(-np.pi, np.pi, size=m1.shape))
    H_back = np.exp(-1j * 0.5 * p1 * omega**2)
    hist = []

    if live:
        plt.ion()
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig = axs = None

    for it in range(n_iter):
        e2_hat = ifftc(fftc(e1) * h_gvd(omega, p2 - p1))
        e2 = m2 * np.exp(1j * np.angle(e2_hat))
        e1_hat = ifftc(fftc(e2) * h_gvd(omega, p1 - p2))
        e1 = m1 * np.exp(1j * np.angle(e1_hat))

        if support is not None:
            E0 = fftc(e1) * H_back * support
            e1 = m1 * np.exp(1j * np.angle(ifftc(E0 * np.exp(1j*0.5*p1*omega**2))))
        else:
            E0 = fftc(e1) * H_back

        hist.append(rmse(np.abs(e1_hat), m1))

        if live and (it % 5 == 0 or it == n_iter - 1):
            rec_S = npeak(np.abs(E0)**2)
            true_S = npeak(np.abs(true_E0)**2)
            axs[0].cla(); axs[1].cla()
            axs[0].plot(f_hz/GHZ, true_S, "k--", lw=2, label="true")
            axs[0].plot(f_hz/GHZ, rec_S, "r", lw=1, label="recovered")
            axs[0].set_xlim(-70, 70); axs[0].set_ylim(-0.05, 1.1)
            axs[0].set_title(f"Spectrum, iter {it}, RMSE={rmse(true_S, rec_S):.4f}")
            axs[0].set_xlabel("f (GHz)"); axs[0].legend()
            axs[1].semilogy(hist)
            axs[1].set_title("Plane-1 magnitude residual")
            axs[1].set_xlabel("iteration")
            plt.pause(0.001)

    if live:
        plt.ioff()
    return npeak(np.abs(E0)**2), E0, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waveform", default="gas3", choices=["gaussian", "chirped_gaussian", "gas3", "chirped_nrz"])
    ap.add_argument("--N", type=int, default=2**14)
    ap.add_argument("--df-ghz", type=float, default=0.10)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    t_s, f_hz, w_rad = make_grid(args.N, args.df_ghz * GHZ)
    phi2_1, phi2_2 = phi2_from_D(300), phi2_from_D(1200)
    et0 = make_waveform(args.waveform, t_s, f_hz)
    E0_true = fftc(et0)
    I1 = intensity(disperse(et0, w_rad, phi2_1))
    I2 = intensity(disperse(et0, w_rad, phi2_2))
    support = ((f_hz >= -60*GHZ) & (f_hz <= 60*GHZ)).astype(float)

    best = None
    for seed in range(1, args.restarts + 1):
        rec_S, E0, hist = td_gs_live(I1, I2, w_rad, f_hz, phi2_1, phi2_2, E0_true,
                                     n_iter=args.iters, support=support, seed=seed,
                                     live=args.live and seed == 1)
        true_S = npeak(np.abs(E0_true)**2)
        score = rmse(true_S, rec_S)
        print(f"restart={seed} spectrum_rmse={score:.5f} residual={hist[-1]:.5f}")
        if best is None or score < best[0]:
            best = (score, rec_S, E0, hist)

    score, rec_S, E0, hist = best
    true_S = npeak(np.abs(E0_true)**2)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(f_hz/GHZ, true_S, "k--", lw=2, label="true |E0|^2")
    axs[0].plot(f_hz/GHZ, rec_S, "r", lw=1, label="recovered")
    axs[0].set_xlim(-70, 70); axs[0].set_title(f"Best recovered spectrum, RMSE={score:.4f}")
    axs[0].set_xlabel("f (GHz)"); axs[0].legend()
    axs[1].semilogy(hist); axs[1].set_title("Best residual"); axs[1].set_xlabel("iteration")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
