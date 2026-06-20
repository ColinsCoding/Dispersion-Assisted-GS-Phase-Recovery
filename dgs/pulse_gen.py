"""Text-based one-shot pulse generator + carrier-less measurement (I1, I2).

Make one chirped Gaussian pulse, send it through a known fibre dispersion, and
"measure" the two square-law intensities a carrier-less receiver actually sees:
I1 (before) and I2 (after). One shot -> one pulse -> one (I1, I2) pair, printed
as ASCII sparklines and optionally written to CSV. These are exactly the inputs
the Gerchberg-Saxton receiver inverts to recover the phase.

The map pulse -> I2 is the paraxial / dispersive PDE
    i dA/dz = (beta_2/2) d^2A/dt^2        (a Schrodinger equation in disguise),
realized as H(f) = exp(i pi D f^2). One of the "famous equations that model
reality"; here it is something you can run. Civilian optical metrology.

Examples:
    py -3.13 pulse_gen.py --chirp 18 --D2 6000 --ascii
    py -3.13 pulse_gen.py --chirp 30 --D2 8000 --photons 200 --csv shot.csv
"""

import argparse
import sys

import numpy as np

from dgs import dispersion_gs_prototype as dg

# the sparklines use box glyphs; force UTF-8 so they print on a cp1252 console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

_BLOCKS = " ▁▂▃▄▅▆▇█"


def generate_pulse(N=512, chirp=18.0, width=0.18, satellite=0.35, ripple=0.8, seed=0):
    """One chirped Gaussian pulse x(t)=A(t)e^{i phi(t)} with an engineered phase.

    A: main Gaussian (+ optional weak satellite).  phi: quadratic chirp + ripple.
    Returns (t, x, A, phi).
    """
    if N < 16:
        raise ValueError("N must be >= 16")
    if width <= 0:
        raise ValueError("width must be > 0")
    if not 0 <= satellite <= 5:
        raise ValueError("satellite must be in [0, 5]")
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.0, 1.0, N)
    A = np.exp(-(t + 0.05)**2 / (2 * width**2))
    if satellite > 0:
        A += satellite * np.exp(-(t - 0.45)**2 / (2 * 0.07**2))
    phi = chirp * t**2 + ripple * np.sin(2 * np.pi * 3.0 * t + 0.6)
    phi += 0.1 * rng.standard_normal()                 # tiny shot-to-shot jitter
    return t, x_from(A, phi), A, phi


def x_from(A, phi):
    return A * np.exp(1j * phi)


def measure(x, D1=0.0, D2=6000.0, photons=None, snr_db=None, seed=0):
    """Square-law measurement at two dispersions: I1 = |disperse(x,D1)|^2, etc.

    Optionally add low-light Poisson shot noise (`photons`) or Gaussian noise
    (`snr_db`). Returns (I1, I2, corr) where corr is the measurement diversity.
    """
    if abs(D2 - D1) < 100:
        raise ValueError(f"|D2-D1|={abs(D2-D1):.0f} < 100: too little diversity "
                         "(I1 ~ I2, phase unrecoverable)")
    I1 = np.abs(dg.disperse(x, D1))**2
    I2 = np.abs(dg.disperse(x, D2))**2
    rng = np.random.default_rng(seed)
    if photons is not None:
        I1, I2 = dg.photon_shot_noise(I1, photons, rng), dg.photon_shot_noise(I2, photons, rng)
    elif snr_db is not None:
        I1, I2 = dg.add_noise(I1, snr_db, rng), dg.add_noise(I2, snr_db, rng)
    corr = float(np.corrcoef(I1, I2)[0, 1])
    return I1, I2, corr


def sparkline(y, width=70):
    """ASCII sparkline of a non-negative array, resampled to `width` columns."""
    y = np.asarray(y, dtype=float)
    idx = np.linspace(0, len(y) - 1, width).astype(int)
    s = y[idx] - y[idx].min()
    if s.max() > 0:
        s = s / s.max()
    return "".join(_BLOCKS[int(round(v * (len(_BLOCKS) - 1)))] for v in s)


def write_csv(path, t, I1, I2):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "I1", "I2"])
        w.writerows(zip(t, I1, I2))


def one_shot(N=512, chirp=18.0, D1=0.0, D2=6000.0, satellite=0.35,
             photons=None, snr_db=None, seed=0):
    """Generate one pulse and its (I1, I2). Returns a result dict."""
    t, x, A, phi = generate_pulse(N=N, chirp=chirp, satellite=satellite, seed=seed)
    I1, I2, corr = measure(x, D1, D2, photons, snr_db, seed)
    return {"t": t, "x": x, "A": A, "phi": phi, "I1": I1, "I2": I2, "corr": corr,
            "D1": D1, "D2": D2}


def main(argv=None):
    p = argparse.ArgumentParser(description="One-shot chirped-pulse generator + I1/I2 measurement")
    p.add_argument("--N", type=int, default=512, help="samples (>=16)")
    p.add_argument("--chirp", type=float, default=18.0, help="quadratic chirp coefficient")
    p.add_argument("--D1", type=float, default=0.0, help="dispersion of plane 1")
    p.add_argument("--D2", type=float, default=6000.0, help="dispersion of plane 2 (|D2-D1|>=100)")
    p.add_argument("--satellite", type=float, default=0.35, help="weak satellite-pulse amplitude")
    p.add_argument("--photons", type=float, default=None, help="mean photons/sample (low-light Poisson)")
    p.add_argument("--snr", type=float, default=None, help="Gaussian SNR in dB (alt. to --photons)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=str, default=None, help="write t,I1,I2 to this CSV")
    p.add_argument("--ascii", action="store_true", help="print ASCII sparklines")
    a = p.parse_args(argv)
    if a.photons is not None and a.photons <= 0:
        p.error("--photons must be > 0")

    r = one_shot(a.N, a.chirp, a.D1, a.D2, a.satellite, a.photons, a.snr, a.seed)
    noise = (f"{a.photons:g} photons/sample" if a.photons is not None
             else f"{a.snr:g} dB SNR" if a.snr is not None else "noiseless")
    print(f"one-shot pulse: N={a.N}, chirp={a.chirp:g}, D1={a.D1:g} -> D2={a.D2:g}, {noise}")
    print(f"measurement diversity  corr(I1,I2) = {r['corr']:.3f}  "
          f"({'GOOD' if r['corr'] < 0.9 else 'LOW -- raise |D2-D1|'})")
    if a.ascii:
        print(f"I1 |{sparkline(r['I1'])}|")
        print(f"I2 |{sparkline(r['I2'])}|")
    if a.csv:
        write_csv(a.csv, r["t"], r["I1"], r["I2"])
        print(f"wrote {a.csv}")
    return r


if __name__ == "__main__":
    main()
