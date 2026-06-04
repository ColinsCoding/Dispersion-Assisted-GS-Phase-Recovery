#!/usr/bin/env python3
"""
gsrecover -- optical phase recovery from intensity measurements
===============================================================

Public CLI tool. Given two dispersed intensity traces, recovers the
optical phase using the Gerchberg-Saxton algorithm.

Install:
    pip install numpy scipy matplotlib

Usage:
    python gsrecover.py --demo                        # run on synthetic data
    python gsrecover.py --i1 I1.npy --i2 I2.npy      # run on real data
    python gsrecover.py --i1 I1.npy --i2 I2.npy --d1 -695 --d2 -800 --plot

Based on:
    Solli, Gupta & Jalali, APL 95, 231108 (2009)
    https://github.com/ColinsCoding/Dispersion-Assisted-GS-Phase-Recovery
"""

import argparse
import sys
import numpy as np

# ── core GS (self-contained, no gs_core dependency for public users) ──────────

def disperse(E, D):
    N  = len(E)
    nu = np.fft.rfftfreq(N)
    H  = np.exp(1j * np.pi * D * nu**2)
    return np.fft.irfft(np.fft.rfft(E) * H, n=N)

def recover_phase(I1, I2, D1, D2, n_iter=50, unit_amplitude=True, verbose=True):
    """
    Recover optical phase from two intensity measurements.

    Parameters
    ----------
    I1, I2       : 1D arrays, measured intensities after dispersion D1, D2
    D1, D2       : float, normalized dispersion values (|D| >= 5000 recommended)
    n_iter       : int, number of GS iterations (default 50)
    unit_amplitude: bool, True for constant-envelope signals (QPSK, DPSK)
    verbose      : bool, print progress

    Returns
    -------
    phi : 1D array, recovered phase in radians
    errors : list of per-iteration error values
    """
    I1 = np.maximum(np.asarray(I1, dtype=float), 0.0)
    I2 = np.maximum(np.asarray(I2, dtype=float), 0.0)
    N  = len(I1)

    # initialise with sqrt(I1) * random phase
    rng = np.random.default_rng(0)
    E   = np.sqrt(I1) * np.exp(1j * rng.uniform(0, 2*np.pi, N))

    errors = []
    for i in range(n_iter):
        # ── constraint set 1 (dispersion D1) ──
        E1 = disperse(E, D1)
        if unit_amplitude:
            E1 = np.exp(1j * np.angle(E1))
        else:
            E1 = np.sqrt(I1) * np.exp(1j * np.angle(E1))
        E  = disperse(E1, -D1)

        # ── constraint set 2 (dispersion D2) ──
        E2 = disperse(E, D2)
        if unit_amplitude:
            E2 = np.exp(1j * np.angle(E2))
        else:
            E2 = np.sqrt(I2) * np.exp(1j * np.angle(E2))
        E  = disperse(E2, -D2)

        # error: how well do we satisfy I1 constraint
        err = np.mean(np.abs(np.abs(disperse(E, D1))**2 - I1))
        errors.append(float(err))

        if verbose and (i % 10 == 0 or i == n_iter-1):
            print(f'  iter {i+1:3d}/{n_iter}  err={err:.4f}', flush=True)

    return np.angle(E), errors


def plot_results(I1, I2, phi, errors, outfile=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed -- skipping plot (pip install matplotlib)')
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GS Phase Recovery Results', fontsize=14, fontweight='bold')

    t = np.arange(len(I1))

    axes[0,0].plot(t, I1, 'b-', lw=0.8, label='I1 (dispersion 1)')
    axes[0,0].plot(t, I2, 'r-', lw=0.8, alpha=0.7, label='I2 (dispersion 2)')
    axes[0,0].set_title('Input Intensity Measurements')
    axes[0,0].set_xlabel('Sample'); axes[0,0].set_ylabel('Intensity [a.u.]')
    axes[0,0].legend()

    axes[0,1].plot(t, np.degrees(phi), 'g-', lw=0.8)
    axes[0,1].set_title('Recovered Phase')
    axes[0,1].set_xlabel('Sample'); axes[0,1].set_ylabel('Phase [degrees]')

    axes[1,0].plot(errors, 'k-', lw=1.5)
    axes[1,0].set_title('GS Convergence')
    axes[1,0].set_xlabel('Iteration'); axes[1,0].set_ylabel('Error')
    axes[1,0].set_yscale('log')

    # phase histogram
    axes[1,1].hist(np.degrees(phi), bins=60, color='steelblue', edgecolor='none')
    axes[1,1].set_title('Phase Distribution')
    axes[1,1].set_xlabel('Phase [degrees]'); axes[1,1].set_ylabel('Count')

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Plot saved: {outfile}')
    else:
        plt.show()


def make_demo_data(N=512, modulation='QPSK', snr_db=35, D1=-5000, D2=-5750):
    """Generate synthetic demo data without requiring gs_core."""
    rng = np.random.default_rng(42)
    # simple QPSK-like signal
    n_sym = N // 8
    symbols = rng.choice([0, 1, 2, 3], size=n_sym)
    phases  = symbols * np.pi / 2
    # upsample
    phi_true = np.repeat(phases, 8)[:N]
    # smooth slightly
    from numpy import convolve
    phi_true = convolve(phi_true, np.ones(4)/4, mode='same')

    E_true = np.exp(1j * phi_true)
    noise_amp = 10 ** (-snr_db / 20)

    E1 = disperse(E_true, D1)
    E2 = disperse(E_true, D2)

    I1 = np.abs(E1)**2 + noise_amp * rng.standard_normal(N)
    I2 = np.abs(E2)**2 + noise_amp * rng.standard_normal(N)

    return np.maximum(I1, 0), np.maximum(I2, 0), phi_true


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Optical phase recovery via Gerchberg-Saxton algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--i1',     type=str,   default=None,   help='.npy file for I1(t)')
    p.add_argument('--i2',     type=str,   default=None,   help='.npy file for I2(t)')
    p.add_argument('--d1',     type=float, default=-5000,  help='Dispersion D1 (normalized, default -5000)')
    p.add_argument('--d2',     type=float, default=-5750,  help='Dispersion D2 (normalized, default -5750)')
    p.add_argument('--iter',   type=int,   default=50,     help='GS iterations (default 50)')
    p.add_argument('--demo',   action='store_true',        help='Run on synthetic QPSK data')
    p.add_argument('--plot',   action='store_true',        help='Show plots')
    p.add_argument('--save',   type=str,   default=None,   help='Save recovered phase to .npy')
    p.add_argument('--png',    type=str,   default=None,   help='Save plot to .png instead of showing')
    p.add_argument('--quiet',  action='store_true',        help='Suppress iteration output')
    args = p.parse_args()

    # ── load data ──
    if args.demo:
        print('Running on synthetic QPSK demo data (N=512, SNR=35 dB)...')
        I1, I2, phi_true = make_demo_data()
        print(f'  I1 shape: {I1.shape}  I2 shape: {I2.shape}')
    elif args.i1 and args.i2:
        print(f'Loading {args.i1} and {args.i2}...')
        I1 = np.load(args.i1).astype(float).ravel()
        I2 = np.load(args.i2).astype(float).ravel()
        phi_true = None
        if len(I1) != len(I2):
            print(f'ERROR: I1 length {len(I1)} != I2 length {len(I2)}')
            sys.exit(1)
        print(f'  Loaded: N={len(I1)} samples')
    else:
        p.print_help()
        print('\nTry:  python gsrecover.py --demo --plot')
        sys.exit(0)

    # ── validate dispersion ──
    if abs(args.d1) < 100 or abs(args.d2) < 100:
        print('WARNING: |D| < 100 -- dispersion may be too low for reliable recovery')
    if abs(args.d2 - args.d1) < 100:
        print('WARNING: |D2-D1| < 100 -- diversity too low, GS will not converge')

    print(f'\nRunning GS: D1={args.d1}  D2={args.d2}  iter={args.iter}')
    print('-' * 50)

    # ── recover ──
    phi, errors = recover_phase(
        I1, I2, D1=args.d1, D2=args.d2,
        n_iter=args.iter, verbose=not args.quiet
    )

    # ── report ──
    print('-' * 50)
    print(f'Convergence: {errors[0]:.4f} -> {errors[-1]:.4f}  '
          f'({(1-errors[-1]/errors[0])*100:.1f}% reduction)')

    if phi_true is not None:
        offset = np.angle(np.mean(np.exp(1j*(phi_true - phi))))
        phi_al = phi + offset
        rms = np.degrees(np.sqrt(np.mean((phi_true - phi_al)**2)))
        print(f'RMS vs ground truth: {rms:.2f} deg')

    print(f'Phase range: [{np.degrees(phi.min()):.1f}, {np.degrees(phi.max()):.1f}] deg')

    # ── save ──
    if args.save:
        np.save(args.save, phi)
        print(f'Saved: {args.save}')

    # ── plot ──
    if args.plot or args.png:
        plot_results(I1, I2, phi, errors, outfile=args.png)

    return phi


if __name__ == '__main__':
    main()
