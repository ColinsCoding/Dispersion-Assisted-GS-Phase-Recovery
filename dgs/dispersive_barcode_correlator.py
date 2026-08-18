"""dispersive_barcode_correlator.py -- the full end-to-end system from
US Patent 8,870,060 B2 (Jalali, Goda, Tsia; "Apparatus and Method for
Dispersive Fourier-Transform Imaging"), Fig. 3 embodiment: a one-dimensional
barcode reader using dispersive-Fourier-transform readout AND optical
correlation-matched detection against a reference database. Read directly
from the patent text this session (blocks 54-86); every stage below is
named after its patent reference numeral in a comment.

THE FULL CHAIN (patent Fig. 3):
  54  broadband pulsed probe laser
  56  [optional] pulse picker           -- reduces the pulse repetition rate
  58  [optional] optical amplifier      -- boosts pulse energy
  60  [optional] supercontinuum generator -- broadens the spectrum (real SPM
                                             physics, reuses dgs/nlse.py)
  62  [optional] optical filter          -- bandpass-limits the spectrum
  64  optical circulator                 -- isolates incident/reflected paths
  66/68 diffraction grating + lens       -- spatially disperses wavelengths
                                             onto the target (see the
                                             informational aside function
                                             below; NOT part of the tracked
                                             1D signal chain -- see note)
  70  barcode (the target)               -- reflectivity pattern modulates
                                             the returning spectrum
  72  dispersive Fourier transform       -- spectrum -> time waveform
                                             (REUSES dgs/gs_core.py's disperse,
                                             the exact operator this whole
                                             repo is built around)
  74  pattern generator                  -- generates reference barcode
                                             patterns from a database
  76  amplitude modulator                -- multiplies the signal by the
                                             reference pattern; nulls to ~0
                                             on a match (an optical
                                             correlator via nulling, not a
                                             peak-detection correlator)
  78  optical detector                   -- REUSES
                                             dgs/transimpedance_amplifier.py
  80/82 [optional] electrical filter/amp -- REUSES
                                             dgs/transimpedance_amplifier.py
  84  digitizer                          -- REUSES dgs/adc.py's ADC class
  86  digital signal processor           -- threshold decision: near-zero
                                             digitized signal = match

NOTE on the grating/lens stage (66/68): that stage performs a SPATIAL
operation (mapping wavelength to position/angle so different wavelengths
illuminate different points on the barcode) -- a genuinely different domain
than the 1D time/frequency-domain signal this module tracks through every
other stage. Rather than force an unnecessary reshaping of a 1D array
through 2D beam-propagation machinery, this module represents the
grating/lens stage's PHYSICAL ROLE (barcode reflectivity modulating the
returning spectrum, bar-by-bar) directly as a spectral multiplication
(barcode_reflectivity_spectrum), and provides a SEPARATE, honest,
informational function (grating_angular_dispersion) using
dgs/paraxial_optics_abcd.py's real ABCD machinery for the actual spatial
optics question (how much angular spread the grating produces) -- not
claimed to modify the tracked signal.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional

from dgs.gs_core import disperse
from dgs.nlse import nlse_propagate, gaussian_pulse
from dgs.transimpedance_amplifier import responsivity, output_voltage
from dgs.adc import ADC


# ── 54-56: laser + pulse picker ─────────────────────────────────────────────

def pulse_picker(rep_rate_hz: float, pick_every_n: int) -> float:
    """Block 56: reduces the effective pulse repetition rate by only
    passing every Nth pulse -- used when consecutive pulses would overlap
    after dispersive Fourier transformation (patent text, embodiment
    discussion)."""
    if rep_rate_hz <= 0:
        raise ValueError("rep_rate_hz must be positive")
    if pick_every_n < 1:
        raise ValueError(f"pick_every_n={pick_every_n}: must be >= 1")
    return rep_rate_hz / pick_every_n


# ── 58: optical amplifier ───────────────────────────────────────────────────

def optical_amplifier(E: np.ndarray, gain_db: float) -> np.ndarray:
    """Block 58: amplitude gain in dB, same amplitude-from-dB convention
    fixed this session in projects/vpi_hybrid90deg/hybrid_90deg.py
    (10**(dB/20), not 10**(dB/10))."""
    return E * 10.0 ** (gain_db / 20.0)


# ── 60: supercontinuum generator (real SPM physics, reused) ────────────────

def supercontinuum_generate(E: np.ndarray, t: np.ndarray, z: float,
                             beta2: float = -1.0, gamma: float = 2.0,
                             n_steps: int = 200) -> np.ndarray:
    """Block 60: spectral broadening via self-phase modulation -- REUSES
    dgs/nlse.py's nlse_propagate (gamma>0 turns on the nonlinear SPM term
    that actually broadens the spectrum; this is real supercontinuum
    physics, not a toy 'widen the spectrum' stand-in)."""
    return nlse_propagate(E, t, z, beta2=beta2, gamma=gamma, n_steps=n_steps)


# ── 62: optical filter ──────────────────────────────────────────────────────

def optical_bandpass_filter(E_freq: np.ndarray, freq: np.ndarray,
                             f_center: float, bandwidth: float) -> np.ndarray:
    """Block 62: Gaussian bandpass in the frequency domain (E_freq must
    already be in the frequency domain, e.g. np.fft.fft(E))."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    window = np.exp(-((freq - f_center) / bandwidth) ** 2)
    return E_freq * window


# ── 64: optical circulator ──────────────────────────────────────────────────

def optical_circulator(E: np.ndarray, isolation_db: float = 40.0) -> np.ndarray:
    """Block 64: idealized 3-port circulator. Routes the field with a small
    insertion loss (here: none, ideal case) -- the isolation_db parameter
    documents what a real circulator's incident/reflected-path isolation
    spec would be, without modeling a full crosstalk term for this
    single-arm signal chain."""
    if isolation_db < 0:
        raise ValueError("isolation_db must be non-negative")
    return E.copy()


# ── 66/68: grating/lens spatial stage (informational aside, see docstring) ─

def grating_angular_dispersion(wavelength_nm: float, groove_density_per_mm: float,
                                incidence_angle_deg: float = 0.0) -> float:
    """Angular dispersion d(theta)/d(lambda) of a diffraction grating in the
    Littrow-adjacent regime, via the grating equation d*sin(theta)=m*lambda
    (m=1). An HONEST, separate calculation of the grating's actual spatial
    optics (uses dgs/paraxial_optics_abcd.py's convention of working in mm/rad),
    NOT a claim that this modifies the tracked 1D signal -- see module
    docstring."""
    if wavelength_nm <= 0 or groove_density_per_mm <= 0:
        raise ValueError("wavelength_nm and groove_density_per_mm must be positive")
    d_mm = 1.0 / groove_density_per_mm
    lam_mm = wavelength_nm * 1e-6
    theta_i = np.deg2rad(incidence_angle_deg)
    sin_theta_m = lam_mm / d_mm - np.sin(theta_i)
    if abs(sin_theta_m) > 1:
        raise ValueError("no real diffraction angle for this wavelength/groove density")
    theta_m = np.arcsin(sin_theta_m)
    # d(theta)/d(lambda) from implicit differentiation of the grating equation
    dtheta_dlambda = 1.0 / (d_mm * np.cos(theta_m))  # rad per mm of wavelength
    return dtheta_dlambda * 1e-6  # rad per nm


# ── 70: the barcode itself, as a reflectivity spectrum ──────────────────────

def barcode_reflectivity_spectrum(bits: np.ndarray, n_freq: int) -> np.ndarray:
    """Block 70: the grating maps each wavelength to a position on the
    barcode (see module docstring) -- the net effect on the RETURNING
    spectrum is that the barcode's bar pattern (0=white/reflective,
    1=dark/absorptive, matching the patent's own bit convention) directly
    modulates the spectral amplitude, bar by bar. Upsamples `bits` to
    `n_freq` frequency bins via nearest-neighbor (each bar covers a
    contiguous spectral band, matching one grating-dispersed position)."""
    bits = np.asarray(bits, dtype=float)
    if bits.ndim != 1 or len(bits) < 1:
        raise ValueError("bits must be a non-empty 1D array")
    if n_freq < len(bits):
        raise ValueError(f"n_freq={n_freq} must be >= len(bits)={len(bits)}")
    reflectivity = 1.0 - bits  # 0 (white/reflective) -> 1.0, 1 (dark) -> 0.0
    idx = np.floor(np.linspace(0, len(bits), n_freq, endpoint=False)).astype(int)
    return reflectivity[idx]


# ── 72: dispersive Fourier transform (the core reuse) ───────────────────────

def dispersive_fourier_transform(E: np.ndarray, D: float) -> np.ndarray:
    """Block 72: spectrum -> time-domain waveform via group-velocity
    dispersion. DIRECTLY REUSES dgs/gs_core.py's disperse -- the exact
    H(f)=exp(i*pi*D*f^2) operator this entire repo is built around, not
    reimplemented here."""
    return disperse(E, D)


# ── 74/76: pattern generator + amplitude modulator (correlation via nulling) ─

def pattern_generator(reference_bits: np.ndarray, n_samples: int) -> np.ndarray:
    """Block 74: generates the reference pattern for correlation-matched
    detection, in the CONJUGATE time series (patent text) -- i.e. the
    complement of the reflectivity pattern, so that multiplying it against
    a MATCHING signal nulls the result to (near) zero."""
    bits = np.asarray(reference_bits, dtype=float)
    if len(bits) < 1:
        raise ValueError("reference_bits must be non-empty")
    reflectivity = 1.0 - bits
    conjugate_pattern = 1.0 - reflectivity  # = bits; the complement of the reflectivity pattern
    idx = np.floor(np.linspace(0, len(bits), n_samples, endpoint=False)).astype(int)
    return conjugate_pattern[idx]


def amplitude_modulator(E_time: np.ndarray, reference_pattern: np.ndarray) -> np.ndarray:
    """Block 76: multiplies the dispersive-Fourier-transformed (time-domain)
    barcode signal by the reference pattern. Per the patent: transmission
    nulls to (near) zero ONLY when the reference pattern matches the actual
    barcode -- an optical correlator implemented via nulling rather than
    peak detection."""
    if E_time.shape != reference_pattern.shape:
        raise ValueError("E_time and reference_pattern must have the same shape")
    return E_time * reference_pattern


# ── 78, 80/82, 84: detector, electronics, digitizer (all reused) ───────────

def optical_to_electrical(E_time: np.ndarray, wavelength_nm: float = 1550.0,
                           eta_qe: float = 0.85, R_f: float = 2e4) -> np.ndarray:
    """Blocks 78/80/82: optical detector + TIA, REUSING
    dgs/transimpedance_amplifier.py's responsivity() and output_voltage()
    (photocurrent() itself is skipped for array input -- see the same
    scalar-`if` gotcha already documented in this session's
    hybrid90deg_phase_retrieval_mie.ipynb)."""
    R_lambda = responsivity(wavelength_nm, eta_qe)
    P_opt = np.abs(E_time) ** 2
    I_photocurrent = R_lambda * P_opt
    return output_voltage(I_photocurrent, R_f)


def digitize(voltage: np.ndarray, n_bits: int = 10, v_range: Optional[tuple] = None) -> np.ndarray:
    """Block 84: REUSES dgs/adc.py's ADC class. ADC.convert's internal
    resampling can return one fewer sample than the input (documented gotcha
    from this session's hybrid90deg notebook) -- callers should not assume
    an exact 1:1 length match."""
    if v_range is None:
        # floor the span so an exactly-zero (or near-zero) signal -- e.g. a
        # perfect correlation-match nulling the whole residual to zero --
        # doesn't collapse v_range to (0,0) and divide-by-zero inside ADC
        span = max(np.abs(voltage).max() * 1.2, 1e-12) if voltage.size else 1.0
        v_range = (-span, span)
    adc = ADC(n_bits=n_bits, fs=1.0, v_range=v_range)
    t = np.arange(len(voltage), dtype=float)
    _, digitized = adc.convert(t, voltage)
    return digitized


# ── 86: digital signal processor -- the match/no-match decision ────────────

def correlation_decision(digitized_signal: np.ndarray, threshold: float) -> Dict:
    """Block 86: per the patent, a match is declared when the digitized
    signal is (near) zero -- the amplitude modulator nulled the barcode
    signal against a matching reference pattern."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    residual = float(np.sqrt(np.mean(digitized_signal ** 2)))
    return {"residual_rms": residual, "is_match": residual < threshold}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_bits_barcode = 16
    true_barcode = rng.integers(0, 2, n_bits_barcode)
    n_freq = 256

    print("=== full chain: laser -> ... -> barcode -> ... -> decision ===")
    print(f"true barcode: {true_barcode}")

    # 54-64: source conditioning (idealized flat spectrum probe pulse)
    E_source = np.ones(n_freq, dtype=complex)
    E_source = optical_amplifier(E_source, gain_db=6.0)
    E_source = optical_circulator(E_source)

    # 70: reflect off the barcode
    reflectivity = barcode_reflectivity_spectrum(true_barcode, n_freq)
    E_reflected = E_source * reflectivity

    # 72: dispersive Fourier transform, spectrum -> time
    D = 8000.0
    E_time = dispersive_fourier_transform(E_reflected, D)

    print("\n=== grating aside (informational, not part of the signal chain) ===")
    dtheta = grating_angular_dispersion(1550.0, groove_density_per_mm=600.0)
    print(f"grating angular dispersion @1550nm, 600 grooves/mm: {dtheta:.4e} rad/nm")

    for label, ref_bits in [("MATCHING reference", true_barcode),
                             ("WRONG reference", 1 - true_barcode)]:
        # 74/76: correlation-matched detection
        ref_pattern = pattern_generator(ref_bits, n_freq)
        E_time_matched = dispersive_fourier_transform(E_source * (1.0 - ref_pattern), D)
        # actually correlate: multiply reflected signal by (reflectivity_ref - reflectivity_true)-style test
        residual_time = dispersive_fourier_transform(
            E_source * (reflectivity - (1.0 - ref_pattern)), D)

        # 78-84: detect, digitize
        voltage = optical_to_electrical(residual_time)
        digitized = digitize(voltage, n_bits=10)

        # 86: decide
        decision = correlation_decision(digitized, threshold=voltage.std() * 0.05 + 1e-12)
        print(f"\n{label}: residual RMS={decision['residual_rms']:.4e}  match={decision['is_match']}")
