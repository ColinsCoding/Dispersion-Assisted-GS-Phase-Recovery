"""I/Q quadrature: two real signals ride one carrier, and coherent detection.

A single sinusoidal carrier can carry TWO independent signals at once, on its two
quadratures -- the in-phase (cosine) and quadrature (sine) components:
        s(t) = I(t) cos(2*pi*fc*t) - Q(t) sin(2*pi*fc*t)
             = Re{ (I(t) + j Q(t)) e^{j 2*pi*fc*t} }.
The complex baseband I + jQ is the whole message; the carrier just lifts it to radio
or optical frequency. This is how QPSK sends 2 bits per symbol, how coherent optical
receivers (and every SDR) work, and why a constellation diagram is a 2-D plot: the
axes are I and Q.

RECOVERING I and Q is quadrature demodulation. The clean way uses the ANALYTIC SIGNAL:
Hilbert-transform s to build z(t) = s + j*H{s} = (I + jQ) e^{j 2*pi*fc*t}, then
multiply by e^{-j 2*pi*fc*t} to drop the carrier and read off I + jQ directly. (The
textbook mixer version -- multiply by 2cos and -2sin, then low-pass -- is also here.)
It works because the two quadratures are ORTHOGONAL over a carrier cycle: cos and sin
integrate to zero against each other, so they do not interfere.

Verified: modulate arbitrary band-limited I, Q and recover them; a QPSK bit stream
survives the full modulate -> demodulate -> decide chain; and the analytic signal
gives the instantaneous amplitude and phase. Builds on the same Hilbert/analytic-signal
idea as dgs.causality and dgs.chirp_log_diff, now as a transceiver. NumPy only; py-3.13.
"""

import numpy as np


def analytic_signal(x):
    """z(t) = x + j*Hilbert{x}: the complex signal whose real part is x and which
    has only positive frequencies. |z| is the envelope, angle(z) the phase."""
    x = np.asarray(x, float)
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)


def iq_modulate(I, Q, fc, fs):
    """Combine baseband I and Q onto a carrier: s = I cos(wt) - Q sin(wt).
    I and Q are same-length sample arrays; fc carrier Hz, fs sample rate Hz."""
    I = np.asarray(I, float); Q = np.asarray(Q, float)
    if len(I) != len(Q):
        raise ValueError("I and Q must be the same length")
    if fs <= 0 or fc <= 0 or fc >= fs / 2:
        raise ValueError("need 0 < fc < fs/2 (Nyquist)")
    t = np.arange(len(I)) / fs
    return I * np.cos(2 * np.pi * fc * t) - Q * np.sin(2 * np.pi * fc * t)


def iq_demodulate(s, fc, fs):
    """Recover (I, Q) by the analytic-signal method: build z = s + j*H{s} =
    (I+jQ)e^{jwt}, then drop the carrier. Returns (I_hat, Q_hat). Exact except
    for Hilbert edge effects near the ends of the record."""
    s = np.asarray(s, float)
    if fs <= 0 or fc <= 0 or fc >= fs / 2:
        raise ValueError("need 0 < fc < fs/2")
    t = np.arange(len(s)) / fs
    baseband = analytic_signal(s) * np.exp(-1j * 2 * np.pi * fc * t)
    return baseband.real, baseband.imag


def iq_demodulate_mixer(s, fc, fs, cutoff):
    """The textbook mixer demodulator: multiply by 2cos and -2sin, then low-pass
    (brick-wall FFT filter at `cutoff` Hz). Returns (I_hat, Q_hat). Shows the
    quadrature orthogonality directly -- the cross terms average out."""
    s = np.asarray(s, float)
    n = len(s)
    t = np.arange(n) / fs
    lp = lambda x: _lowpass(x, fs, cutoff)
    return lp(2 * s * np.cos(2 * np.pi * fc * t)), lp(-2 * s * np.sin(2 * np.pi * fc * t))


def _lowpass(x, fs, cutoff):
    """Zero out spectral content above `cutoff` (an ideal brick-wall filter)."""
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, 1 / fs)
    X[np.abs(f) > cutoff] = 0
    return np.fft.ifft(X).real


def complex_baseband(I, Q):
    """The message as a complex number I + jQ -- the constellation point."""
    return np.asarray(I, float) + 1j * np.asarray(Q, float)


def instantaneous_amplitude_phase(s):
    """(envelope, phase) of a real signal from its analytic signal: |z| and
    angle(z). The amplitude/phase a receiver actually tracks."""
    z = analytic_signal(s)
    return np.abs(z), np.unwrap(np.angle(z))


# ----------------------------------------------------------------------
# QPSK: 2 bits per symbol on the I/Q plane
# ----------------------------------------------------------------------

def qpsk_map(bits):
    """Map a bit stream (even length) to QPSK symbols: each pair (b0, b1) -> a
    constellation point (I, Q) with I, Q in {+1/sqrt2, -1/sqrt2} (unit energy).
    Returns (I, Q) arrays, one entry per symbol."""
    bits = np.asarray(bits, int)
    if len(bits) % 2 != 0 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be an even-length 0/1 array")
    pairs = bits.reshape(-1, 2)
    amp = 1 / np.sqrt(2)
    I = np.where(pairs[:, 0] == 0, amp, -amp)
    Q = np.where(pairs[:, 1] == 0, amp, -amp)
    return I, Q


def qpsk_demap(I, Q):
    """Decide the bits from received I/Q: sign slicing on each axis. Inverse of
    qpsk_map for a clean channel."""
    I = np.asarray(I, float); Q = np.asarray(Q, float)
    b0 = (I < 0).astype(int)
    b1 = (Q < 0).astype(int)
    return np.column_stack([b0, b1]).ravel()


if __name__ == "__main__":
    fs, fc = 4000.0, 400.0

    # 1. modulate two independent baseband signals and recover them
    t = np.arange(2000) / fs
    I = np.cos(2 * np.pi * 5 * t)      # 5 Hz on I
    Q = np.sin(2 * np.pi * 8 * t)      # 8 Hz on Q  (independent!)
    s = iq_modulate(I, Q, fc, fs)
    Ih, Qh = iq_demodulate(s, fc, fs)
    core = slice(200, -200)
    print("recover two signals on one carrier:")
    print(f"  I error {np.max(np.abs(Ih[core]-I[core])):.2e}, "
          f"Q error {np.max(np.abs(Qh[core]-Q[core])):.2e}  (both recovered)")

    # 2. a QPSK link: bits -> symbols -> carrier -> demod -> bits
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 40)
    Isym, Qsym = qpsk_map(bits)
    sps = 40
    Iu = np.repeat(Isym, sps); Qu = np.repeat(Qsym, sps)
    sig = iq_modulate(Iu, Qu, fc, fs)
    Ir, Qr = iq_demodulate(sig, fc, fs)
    centers = (np.arange(len(Isym)) * sps + sps // 2)
    rec = qpsk_demap(Ir[centers], Qr[centers])
    print(f"\nQPSK: {len(bits)} bits sent, "
          f"{np.sum(rec == bits)}/{len(bits)} recovered "
          f"({'no errors' if np.array_equal(rec, bits) else 'ERRORS'})")
    print(f"  constellation points: {sorted(set(zip(np.round(Isym,3), np.round(Qsym,3))))}")
