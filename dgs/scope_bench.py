"""A simulated oscilloscope you put on a bench: voltage versus time, with the scope's own flaws.

In plain words: a real scope does not show you the signal. It shows the signal after a low-pass
filter (its analog bandwidth), sampled at a finite rate, rounded to a few ADC bits, clipped to the
screen, and lined up on a trigger. This class does those five things in that order, so you can see
which of them is lying to you. dgs.oscilloscope has the closed-form rules (rise time 0.35/BW,
Nyquist, probe loading); this is the instrument built from them.

    scope = Scope(bandwidth=200e6, sample_rate=1e9, bits=8)
    scope.connect(1, "RF electrode", lambda t: electrode_voltage(t), volts_per_div=100)
    scope.trigger(channel=1, level=0.0, slope="rising")
    shot = scope.acquire(time_per_div=20e-9)
    scope.measure(shot, 1)          # frequency, Vpp, RMS
    scope.plot(shot)                # graticule, Okabe-Ito colours, plus a text description

Signals are functions of time in seconds returning volts (a probe can scale them, e.g. a 10x probe).
"""
from __future__ import annotations

import numpy as np

from dgs.oscilloscope import RISE_TIME_K

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]
LINESTYLES = ["-", "--", "-.", ":"]
DIVS_X, DIVS_Y = 10, 8


class Scope:
    def __init__(self, bandwidth=200e6, sample_rate=1e9, bits=8, noise_vrms=0.0, oversample=16, seed=0):
        self.bandwidth, self.sample_rate, self.bits = bandwidth, sample_rate, bits
        self.noise_vrms, self.oversample = noise_vrms, oversample
        self.channels = {}
        self.trig = dict(channel=1, level=0.0, slope="rising")
        self.rng = np.random.default_rng(seed)

    @property
    def rise_time(self):
        return RISE_TIME_K / self.bandwidth

    def connect(self, ch, name, signal, volts_per_div=1.0, offset=0.0, probe_gain=1.0):
        """Attach a signal (callable t -> volts) to channel ch with its vertical settings."""
        self.channels[ch] = dict(name=name, signal=signal, vdiv=volts_per_div, offset=offset, gain=probe_gain)

    def trigger(self, channel=1, level=0.0, slope="rising"):
        self.trig = dict(channel=channel, level=level, slope=slope)

    # ------------------------------------------------ acquisition pipeline
    def _analog(self, v, dt):
        """Single-pole analog front end at the scope bandwidth (rise time 0.35 / BW)."""
        tau = 1 / (2 * np.pi * self.bandwidth)
        alpha = dt / (tau + dt)
        out = np.empty_like(v)
        acc = v[0]
        for i, x in enumerate(v):
            acc += alpha * (x - acc)
            out[i] = acc
        return out

    def acquire(self, time_per_div, pre_trigger=0.5, search_divs=3.0):
        """One triggered acquisition. Returns dict(t, ch{n: volts}, trig_time, settings)."""
        window = time_per_div * DIVS_X
        settle = 10 / (2 * np.pi * self.bandwidth)        # let the front end forget its start-up (10 tau)
        span = window * (1 + search_divs) + settle
        fine_dt = 1 / (self.sample_rate * self.oversample)
        tf = np.arange(0.0, span, fine_dt)
        analog = {}
        for ch, c in self.channels.items():
            v = c["gain"] * np.asarray(c["signal"](tf), float)
            v = self._analog(v, fine_dt)
            analog[ch] = v + self.noise_vrms * self.rng.standard_normal(v.size)
        # trigger: first crossing after enough history for the pre-trigger part of the screen
        tc = self.trig["channel"]
        lvl = self.trig["level"]
        v = analog[tc]
        start = int((window * pre_trigger + settle) / fine_dt) + 1
        if self.trig["slope"] == "rising":
            hits = np.nonzero((v[start - 1:-1] < lvl) & (v[start:] >= lvl))[0]
        else:
            hits = np.nonzero((v[start - 1:-1] > lvl) & (v[start:] <= lvl))[0]
        k0 = start + (hits[0] if hits.size else 0)
        t_trig = tf[k0]
        # sample at the scope's rate (this is where aliasing happens) and quantise to the ADC
        step = self.oversample
        first = k0 - int(window * pre_trigger / fine_dt)
        idx = np.arange(first, first + int(window / fine_dt), step)
        idx = idx[(idx >= 0) & (idx < tf.size)]
        out = {}
        for ch, c in self.channels.items():
            full = c["vdiv"] * DIVS_Y / 2
            vs = np.clip(analog[ch][idx] - c["offset"], -full, full)         # off-screen clips
            lsb = 2 * full / 2 ** self.bits
            out[ch] = np.round(vs / lsb) * lsb + c["offset"]
        return dict(t=tf[idx] - t_trig, ch=out, trig_time=t_trig, time_per_div=time_per_div,
                    triggered=bool(hits.size), sample_rate=self.sample_rate)

    # ------------------------------------------------ measurements
    @staticmethod
    def measure(shot, ch):
        v = shot["ch"][ch]
        t = shot["t"]
        dt = t[1] - t[0]
        spec = np.abs(np.fft.rfft(v - v.mean()))
        freqs = np.fft.rfftfreq(v.size, dt)
        k = int(np.argmax(spec[1:]) + 1)
        if 1 <= k < spec.size - 1:          # parabolic peak interpolation
            y0, y1, y2 = np.log(spec[k - 1:k + 2] + 1e-30)
            k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        return dict(frequency=float(k * (freqs[1] - freqs[0])), vpp=float(v.max() - v.min()),
                    rms=float(np.sqrt(np.mean(v ** 2))), mean=float(v.mean()))

    # ------------------------------------------------ display
    def plot(self, shot, ax=None, title=None):
        """Draw the screen (10 x 8 graticule). Returns (ax, text description)."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))
        tdiv = shot["time_per_div"]
        t0 = shot["t"][0]
        ax.set_facecolor("#0b0f0c")
        for k in range(DIVS_X + 1):
            ax.axvline(t0 + k * tdiv, color="#26372c", lw=0.6)
        for k in range(DIVS_Y + 1):
            ax.axhline(-DIVS_Y / 2 + k, color="#26372c", lw=0.6)
        lines = []
        for n, (ch, v) in enumerate(sorted(shot["ch"].items())):
            c = self.channels[ch]
            ax.plot(shot["t"], (v - c["offset"]) / c["vdiv"], color=OKABE_ITO[n % 6], ls=LINESTYLES[n % 4], lw=1.4,
                    label=f"CH{ch} {c['name']}  ({c['vdiv']:g} V/div)")
            m = self.measure(shot, ch)
            lines.append(f"CH{ch} ({c['name']}, {LINESTYLES[n % 4]!r} line): {m['frequency'] / 1e6:.3f} MHz, "
                         f"{m['vpp']:.3g} Vpp")
        ax.set_xlim(t0, t0 + DIVS_X * tdiv)
        ax.set_ylim(-DIVS_Y / 2, DIVS_Y / 2)
        ax.set_yticklabels([])
        ax.set_xlabel(f"time ({tdiv * 1e9:g} ns/div)  |  BW {self.bandwidth / 1e6:g} MHz, "
                      f"{self.sample_rate / 1e9:g} GS/s, {self.bits}-bit")
        ax.legend(loc="upper right", fontsize=8, facecolor="#1b2420", labelcolor="white")
        if title:
            ax.set_title(title)
        desc = ("Oscilloscope screen, 10 x 8 divisions, trigger at t = 0 on CH"
                f"{self.trig['channel']} ({self.trig['slope']} through {self.trig['level']:g} V). " + "; ".join(lines) + ".")
        return ax, desc
