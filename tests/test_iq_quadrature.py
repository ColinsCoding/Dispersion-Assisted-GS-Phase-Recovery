"""Test dgs.iq_quadrature: the analytic signal, I/Q modulation onto a carrier and
quadrature demodulation recovering BOTH baseband signals (they're orthogonal, so
they don't interfere), instantaneous amplitude/phase, and a full QPSK bit stream
surviving modulate -> demodulate -> decide."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import iq_quadrature as iq

fs, fc = 4000.0, 400.0
core = slice(200, -200)                      # interior, away from Hilbert edges

# 1. analytic signal: Re(z)=x, and for a cosine z = e^{jwt}
t = np.arange(2000) / fs
x = np.cos(2 * np.pi * 50 * t)
z = iq.analytic_signal(x)
assert np.allclose(z.real, x, atol=1e-9)
assert np.allclose(np.abs(z)[core], 1.0, atol=1e-2)          # unit envelope
assert np.allclose(z.imag[core], np.sin(2*np.pi*50*t)[core], atol=1e-2)

# 2. modulation form s = I cos - Q sin, and Nyquist guard
I = 0.7 * np.ones(500); Q = -0.3 * np.ones(500)
s = iq.iq_modulate(I, Q, fc, fs)
tt = np.arange(500) / fs
assert np.allclose(s, 0.7*np.cos(2*np.pi*fc*tt) + 0.3*np.sin(2*np.pi*fc*tt))
for bad in (lambda: iq.iq_modulate([1, 2], [1], fc, fs),
            lambda: iq.iq_modulate(I, Q, fs, fs),        # fc = fs (> Nyquist)
            lambda: iq.iq_modulate(I, Q, fc, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

# 3. demodulation recovers BOTH independent baseband signals
I = np.cos(2 * np.pi * 5 * t)                # 5 Hz on I
Q = np.sin(2 * np.pi * 8 * t)                # 8 Hz on Q -- independent
s = iq.iq_modulate(I, Q, fc, fs)
Ih, Qh = iq.iq_demodulate(s, fc, fs)
assert np.max(np.abs(Ih[core] - I[core])) < 1e-2
assert np.max(np.abs(Qh[core] - Q[core])) < 1e-2
# the mixer demodulator works too (with a low-pass)
Im, Qm = iq.iq_demodulate_mixer(s, fc, fs, cutoff=50)
assert np.max(np.abs(Im[core] - I[core])) < 5e-2
assert np.max(np.abs(Qm[core] - Q[core])) < 5e-2

# 4. ORTHOGONALITY: a signal on I does not leak into Q
s_I_only = iq.iq_modulate(np.ones(2000), np.zeros(2000), fc, fs)
Ih, Qh = iq.iq_demodulate(s_I_only, fc, fs)
assert np.allclose(Ih[core], 1.0, atol=1e-2)
assert np.max(np.abs(Qh[core])) < 1e-2                      # no crosstalk into Q

# 5. complex baseband and instantaneous amplitude/phase
assert np.allclose(iq.complex_baseband([1, 0], [0, 1]), [1+0j, 0+1j])
amp, phase = iq.instantaneous_amplitude_phase(2.0 * np.cos(2*np.pi*30*t + 0.5))
assert np.allclose(amp[core], 2.0, atol=1e-2)
slope = np.polyfit(t[core], phase[core], 1)[0]
assert np.isclose(slope, 2*np.pi*30, rtol=1e-3)             # d(phase)/dt = w

# 6. QPSK map/demap: 4 unit-energy points, and a clean round-trip
bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
Isym, Qsym = iq.qpsk_map(bits)
assert len(Isym) == 4
assert np.allclose(Isym**2 + Qsym**2, 1.0)                  # unit energy per symbol
assert len(set(zip(np.round(Isym, 3), np.round(Qsym, 3)))) == 4   # 4 distinct points
assert np.array_equal(iq.qpsk_demap(Isym, Qsym), bits)     # inverse

# 7. FULL LINK: random bits through the carrier and back, no errors
rng = np.random.default_rng(1)
bits = rng.integers(0, 2, 60)
Isym, Qsym = iq.qpsk_map(bits)
sps = 40
sig = iq.iq_modulate(np.repeat(Isym, sps), np.repeat(Qsym, sps), fc, fs)
Ir, Qr = iq.iq_demodulate(sig, fc, fs)
centers = np.arange(len(Isym)) * sps + sps // 2
assert np.array_equal(iq.qpsk_demap(Ir[centers], Qr[centers]), bits)

# 8. kwarg bounds
for bad in (lambda: iq.qpsk_map([0, 1, 1]),                # odd length
            lambda: iq.qpsk_map([0, 2, 1, 0]),             # non-binary
            lambda: iq.iq_demodulate(s, fs, fs)):          # fc >= Nyquist
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_iq_quadrature: all checks passed")
