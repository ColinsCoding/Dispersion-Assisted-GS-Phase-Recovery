"""Analog-to-Digital and Digital-to-Analog conversion — educational simulation.

Covers the full ADC pipeline:
  1. Continuous analog signal (band-limited by Nyquist)
  2. Sampling at f_s  (aliasing when f_s < 2 f_max)
  3. Uniform quantization to N bits  (quantization noise = LSB²/12)
  4. DAC reconstruction via zero-order hold or sinc interpolation
  5. SQNR = Signal-to-Quantization-Noise Ratio = 6.02 N + 1.76 dB  (theory)

Independent variables you control:
  - n_bits  : ADC resolution (1..24)
  - fs      : sampling rate [Hz]
  - signal  : any callable t → float, or use the built-in helpers

Usage:
    from dgs.adc import ADC, sinusoidal, multi_tone
    adc = ADC(n_bits=8, fs=1000.0)
    t, analog = sinusoidal(f=100.0)
    samples, quant = adc.convert(t, analog)
    adc.report(t, analog, samples, quant)
    adc.plot(t, analog, samples, quant)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple


# ── signal generators ────────────────────────────────────────────────────────

def sinusoidal(f: float = 100.0, duration: float = 0.05,
               n_points: int = 10_000, amplitude: float = 1.0,
               phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Single-tone sine: x(t) = A sin(2π f t + φ)."""
    t = np.linspace(0, duration, n_points, endpoint=False)
    return t, amplitude * np.sin(2 * np.pi * f * t + phase)


def multi_tone(freqs=(100.0, 250.0, 370.0), duration: float = 0.05,
               n_points: int = 10_000) -> Tuple[np.ndarray, np.ndarray]:
    """Sum of sinusoids, each with amplitude 1/N (unit total power)."""
    t = np.linspace(0, duration, n_points, endpoint=False)
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return t, sig


def chirp_signal(f0: float = 50.0, f1: float = 450.0,
                 duration: float = 0.05, n_points: int = 10_000):
    """Linear chirp from f0 to f1 over duration."""
    t = np.linspace(0, duration, n_points, endpoint=False)
    k = (f1 - f0) / duration
    return t, np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))


# ── ADC core ─────────────────────────────────────────────────────────────────

class ADC:
    """Uniform mid-tread ADC with configurable resolution and sample rate.

    n_bits : number of quantization bits (1–24)
    fs     : sampling rate [Hz]
    v_range: (v_min, v_max) — full-scale input range; defaults to signal peak.
    """

    def __init__(self, n_bits: int = 8, fs: float = 1000.0,
                 v_range: Optional[Tuple[float, float]] = None):
        if not 1 <= n_bits <= 24:
            raise ValueError("n_bits must be 1..24")
        if fs <= 0:
            raise ValueError("fs must be positive")
        self.n_bits = n_bits
        self.fs = fs
        self.v_range = v_range
        self.n_levels = 2 ** n_bits

    @property
    def lsb(self) -> float:
        vmin, vmax = self._effective_range
        return (vmax - vmin) / self.n_levels

    @property
    def _effective_range(self) -> Tuple[float, float]:
        return self.v_range if self.v_range else self._stored_range

    def _sample(self, t: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ideal uniform sampling at fs. Returns (t_s, x_s)."""
        dt = 1.0 / self.fs
        t_s = np.arange(t[0], t[-1], dt)
        x_s = np.interp(t_s, t, signal)
        return t_s, x_s

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """Mid-tread uniform quantization, clipped at full-scale."""
        vmin, vmax = self._effective_range
        # map to [0, n_levels)
        normalized = (x - vmin) / (vmax - vmin) * self.n_levels
        codes = np.clip(np.floor(normalized).astype(int), 0, self.n_levels - 1)
        # reconstruct analog value at the centre of each bin
        reconstructed = (codes + 0.5) / self.n_levels * (vmax - vmin) + vmin
        return reconstructed

    def convert(self, t: np.ndarray, signal: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Full ADC pipeline: sample → quantize.

        Returns (t_samples, quantized_samples).
        Stores effective v_range from signal if not set by caller.
        """
        if self.v_range is None:
            peak = max(abs(signal.max()), abs(signal.min()))
            self._stored_range = (-peak, peak)
        t_s, x_s = self._sample(t, signal)
        q = self._quantize(x_s)
        return t_s, q

    def reconstruct_zoh(self, t: np.ndarray, t_s: np.ndarray,
                        q: np.ndarray) -> np.ndarray:
        """Zero-order hold reconstruction onto the original time grid."""
        idx = np.searchsorted(t_s, t, side='right') - 1
        idx = np.clip(idx, 0, len(q) - 1)
        return q[idx]

    def reconstruct_sinc(self, t: np.ndarray, t_s: np.ndarray,
                         q: np.ndarray) -> np.ndarray:
        """Ideal sinc (Whittaker-Shannon) interpolation.  O(N·M) — slow for large N."""
        T = 1.0 / self.fs
        result = np.zeros(len(t))
        for k, (tk, qk) in enumerate(zip(t_s, q)):
            result += qk * np.sinc((t - tk) / T)
        return result

    def sqnr_db(self, signal: np.ndarray, quantized_on_same_grid: np.ndarray) -> float:
        """Measured SQNR in dB from matched arrays."""
        noise = signal - quantized_on_same_grid
        ps = np.mean(signal**2)
        pn = np.mean(noise**2)
        if pn == 0:
            return float('inf')
        return 10 * np.log10(ps / pn)

    def sqnr_theory_db(self) -> float:
        """Theoretical SQNR for a full-scale sinusoid: 6.02 N + 1.76 dB."""
        return 6.02 * self.n_bits + 1.76

    def report(self, t: np.ndarray, signal: np.ndarray,
               t_s: np.ndarray, q: np.ndarray) -> None:
        """Print a summary report."""
        # SQNR measured at the sample points (pure quantization noise, no ZOH distortion)
        x_at_samples = np.interp(t_s, t, signal)
        sqnr_m = self.sqnr_db(x_at_samples, q)
        print(f"ADC Report")
        print(f"  Resolution    : {self.n_bits} bits  ({self.n_levels} levels)")
        print(f"  Sample rate   : {self.fs:.1f} Hz")
        print(f"  LSB           : {self.lsb:.6f} V")
        print(f"  SQNR theory   : {self.sqnr_theory_db():.2f} dB  (6.02*N + 1.76)")
        print(f"  SQNR measured : {sqnr_m:.2f} dB")
        print(f"  Quant. noise  : sigma = LSB/sqrt(12) = {self.lsb / np.sqrt(12):.6f}")

    def plot(self, t: np.ndarray, signal: np.ndarray,
             t_s: np.ndarray, q: np.ndarray,
             show_sinc: bool = False) -> None:
        """4-panel plot: analog, sampled+quantized, error, spectrum."""
        recon_zoh = self.reconstruct_zoh(t, t_s, q)
        error = signal - recon_zoh

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle(f"ADC/DAC  —  {self.n_bits}-bit, fs = {self.fs} Hz", fontsize=12)

        # panel 1: analog + samples
        ax = axes[0, 0]
        ax.plot(t * 1e3, signal, lw=1.2, color='steelblue', label='Analog x(t)')
        ax.step(t_s * 1e3, q, where='post', color='tomato', lw=1.5,
                label='Quantized (ZOH)', alpha=0.8)
        ax.scatter(t_s * 1e3, q, s=15, color='tomato', zorder=3)
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude")
        ax.set_title("Analog signal & quantized samples")
        ax.legend(fontsize=8)

        # panel 2: quantization error
        ax = axes[0, 1]
        ax.plot(t * 1e3, error, lw=0.8, color='purple')
        ax.axhline(self.lsb / 2, color='k', ls='--', lw=1, label='+LSB/2')
        ax.axhline(-self.lsb / 2, color='k', ls='--', lw=1, label='-LSB/2')
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Error (V)")
        ax.set_title(f"Quantization error  (LSB = {self.lsb:.4f})")
        ax.legend(fontsize=8)

        # panel 3: spectrum
        ax = axes[1, 0]
        N = len(signal)
        dt = t[1] - t[0]
        freqs = np.fft.rfftfreq(N, dt)
        S_analog = np.abs(np.fft.rfft(signal)) / N
        S_quant  = np.abs(np.fft.rfft(recon_zoh)) / N
        ax.semilogy(freqs, S_analog, color='steelblue', lw=1.2, label='Analog')
        ax.semilogy(freqs, S_quant,  color='tomato',    lw=1.0, label='Quantized', alpha=0.8)
        ax.axvline(self.fs / 2, color='k', ls=':', lw=1.2, label='Nyquist fs/2')
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("|X(f)|")
        ax.set_title("Spectrum: analog vs reconstructed")
        ax.legend(fontsize=8)

        # panel 4: SQNR vs bits
        ax = axes[1, 1]
        bits_range = np.arange(1, 17)
        sqnr_theory = 6.02 * bits_range + 1.76
        ax.plot(bits_range, sqnr_theory, 'o-', color='#4CAF50', lw=2,
                label='Theory: 6.02N + 1.76 dB')
        ax.axvline(self.n_bits, color='k', ls='--', lw=1.2,
                   label=f'This design ({self.n_bits}-bit)')
        ax.set_xlabel("ADC bits (N)"); ax.set_ylabel("SQNR (dB)")
        ax.set_title("SQNR vs resolution")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.savefig("adc_demo.png", dpi=120, bbox_inches='tight')
        print("Saved adc_demo.png")
        plt.show()


# ── Nyquist aliasing demo ────────────────────────────────────────────────────

def aliasing_demo():
    """Show what happens when fs < 2 f_signal (undersampling / aliasing)."""
    f_sig = 100.0
    t, analog = sinusoidal(f=f_sig, duration=0.05)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Nyquist–Shannon aliasing  (signal at {f_sig} Hz)", fontsize=12)

    for ax, (fs, label) in zip(axes, [
        (300.0, "fs = 3×f  (above Nyquist, OK)"),
        (200.0, "fs = 2×f  (exactly Nyquist)"),
        (130.0, "fs = 1.3×f  (aliasing!)"),
    ]):
        adc = ADC(n_bits=12, fs=fs)
        t_s, q = adc.convert(t, analog)
        recon = adc.reconstruct_zoh(t, t_s, q)
        ax.plot(t * 1e3, analog, color='steelblue', lw=1.2, label='Analog', alpha=0.7)
        ax.step(t_s * 1e3, q, where='post', color='tomato', lw=1.5, label='Sampled')
        ax.plot(t * 1e3, recon, color='purple', lw=1.0, ls='--', label='ZOH recon')
        ax.set_xlabel("Time (ms)"); ax.set_title(label)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("aliasing_demo.png", dpi=120, bbox_inches='tight')
    print("Saved aliasing_demo.png")
    plt.show()


if __name__ == "__main__":
    print("=== 8-bit ADC, 1 kHz sampling ===")
    t, analog = sinusoidal(f=100.0, duration=0.05)
    adc = ADC(n_bits=8, fs=1000.0)
    t_s, q = adc.convert(t, analog)
    adc.report(t, analog, t_s, q)

    print("\n=== SQNR vs bits ===")
    print(f"{'Bits':>5}  {'Theory SQNR':>13}  {'Formula'}")
    for n in (4, 8, 12, 16):
        print(f"{n:>5}  {6.02*n+1.76:>13.2f} dB  6.02×{n}+1.76")
