"""Sensor model: illumination -> scattered light -> photodiode -> TIA -> ADC.

A behavioural model of the measurement chain. Every component value here is
a NOMINAL DESIGN ASSUMPTION for simulation, not a measured property of real
hardware. All numbers produced with it are synthetic.

    particle sample
      -> illumination (SEALS rainbow: one wavelength = one angle)
      -> Mie-scattered optical power on the detector        [W]
      -> photodiode            i = responsivity * P          [A]
      -> transimpedance amp    v = i * R_f                   [V]
      -> programmable gain     v * G
      -> anti-alias filter     single pole at f_c
      -> ADC                   code = round(v / LSB)         [counts]

Noise (all referred to the photodiode current, added in quadrature):
    shot      sqrt(2 q (i_signal + i_dark) B)     Poisson photon/electron statistics
    thermal   sqrt(4 k_B T B / R_f)               Johnson noise of the feedback resistor
    amplifier i_n * sqrt(B)                       amplifier input current noise
    ADC       LSB / sqrt(12) (in volts, after gain) quantization
with B = (pi/2) f_c the noise bandwidth of a single-pole filter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import physics as ph


@dataclass(frozen=True)
class OpticalBench:
    """Illumination and collection geometry (SEALS front end + sample)."""
    groove_spacing: float = ph.SEALS_DEFAULTS["groove_spacing"]        # d [m]
    grating_distance: float = ph.SEALS_DEFAULTS["grating_distance"]    # D [m]
    grating_angle: float = ph.SEALS_DEFAULTS["grating_angle"]          # alpha [rad]
    dcorr: float = ph.SEALS_DEFAULTS["dcorr"]                          # [m]
    pupil_diameter: float = ph.SEALS_DEFAULTS["pupil_diameter"]        # P [m]
    numerical_aperture: float = ph.SEALS_DEFAULTS["numerical_aperture"]
    measurement_angle: float = ph.SEALS_DEFAULTS["measurement_angle"]  # detector angle [rad]
    lam_min: float = ph.SEALS_DEFAULTS["lam_min"]                      # [m]
    lam_max: float = ph.SEALS_DEFAULTS["lam_max"]                      # [m]
    source_bandwidth: float = ph.SEALS_DEFAULTS["source_bandwidth"]    # [m]
    n_channels: int = 42            # resolvable angles in the 20 nm band (paper, Sec. 3)
    incident_irradiance: float = 1.0e6   # at the sample, at the spectral peak [W/m^2]
    detector_area: float = 1.0e-6        # [m^2]
    detector_distance: float = 0.05      # r [m]
    nmed: float = 1.33                   # medium (water)
    probe_volume: float = 1.0e-12        # illuminated AND viewed volume [m^3] (1 nL)
    beam_stop_angle: float = np.deg2rad(5.0)   # |theta| below this sees the direct beam
    equalization: tuple = ()             # optical equalization filter, transmission per channel

    def __post_init__(self):
        if self.n_channels < 4:
            raise ValueError("n_channels must be at least 4")
        if self.equalization:
            eq = np.asarray(self.equalization, dtype=float)
            if eq.shape != (self.n_channels,) or np.any(eq <= 0) or np.any(eq > 1):
                raise ValueError("equalization must hold one transmission in (0, 1] per channel")
        for name in ("incident_irradiance", "detector_area", "detector_distance",
                     "nmed", "probe_volume"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def wavelengths(self):
        """Channel wavelengths [m], increasing."""
        return np.linspace(self.lam_min, self.lam_max, self.n_channels)

    @property
    def lam_center(self):
        return 0.5 * (self.lam_min + self.lam_max)

    @property
    def solid_angle(self):
        """Detector solid angle seen from the sample [sr] (small-detector form A / r^2)."""
        return self.detector_area / self.detector_distance ** 2

    def scattering_angles(self):
        """Scattering angle of each channel [rad] = illumination angle + detector angle."""
        _, theta = ph.seals_wavelength_to_angle(
            self.groove_spacing, self.grating_distance, self.grating_angle, self.dcorr,
            self.pupil_diameter, self.numerical_aperture, self.wavelengths)
        return theta + self.measurement_angle

    def usable_channels(self):
        """Boolean mask: channels far enough from the forward direction to be
        free of unscattered illumination (which this model does not include)."""
        return np.abs(self.scattering_angles()) >= self.beam_stop_angle

    def source_spectrum(self):
        """Relative illumination power per channel: laser lineshape (peak = 1)
        times the equalization filter transmission, if one is fitted."""
        shape = ph.laser_lineshape(self.wavelengths, self.lam_center, self.source_bandwidth)
        return shape * np.asarray(self.equalization) if self.equalization else shape


def design_equalization(bench: OpticalBench, dia_ref, npar_ref, max_attenuation_db=30.0):
    """Return a copy of `bench` with an optical equalization filter fitted.

    Forward scattering is orders of magnitude brighter than side scattering,
    more than one ADC range can hold. The SEALS paper's answer is to shape
    the source spectrum so that wavelengths mapped to dim angles get more
    power. Here the filter transmission is chosen to flatten the pattern of
    a REFERENCE particle (dia_ref, npar_ref) on the usable channels, limited
    to `max_attenuation_db` and normalized so its largest transmission is 1.
    The filter is part of the known instrument, so the estimator's templates
    include it.
    """
    if max_attenuation_db <= 0:
        raise ValueError("max_attenuation_db must be positive")
    plain = replace(bench, equalization=())
    pattern = scattered_power_per_particle(dia_ref, npar_ref, plain)
    usable = plain.usable_channels()
    floor = 10 ** (-max_attenuation_db / 10)
    transmission = np.clip(pattern[usable].min() / pattern, floor, 1.0)
    transmission[~usable] = floor            # also dump the direct-beam channels
    return replace(bench, equalization=tuple(float(t) for t in transmission))


def scattered_power_per_particle(dia, npar, bench: OpticalBench):
    """Optical power on the detector from ONE particle, per channel [W].

        P_j = I0 * L(lam_j) * dsigma/dOmega(theta_j; lam_j) * solid_angle

    Each channel uses its own wavelength in the Mie size parameter (main.m
    used the centre wavelength for all channels). Unpolarized illumination,
    point detector, single scattering; no direct beam, no refraction at the
    sample window.
    """
    theta = bench.scattering_angles()
    weights = bench.source_spectrum()
    power = np.empty(bench.n_channels)
    for j, (lam, th) in enumerate(zip(bench.wavelengths, theta)):
        dsigma = ph.mie_differential_cross_section(npar, bench.nmed, dia, lam, [th])[0]
        power[j] = bench.incident_irradiance * weights[j] * dsigma * bench.solid_angle
    return power


def scattered_power(dia, npar, concentration, bench: OpticalBench):
    """Power per channel [W] from a dilute sample of identical particles.

    N = concentration * probe_volume particles scatter independently, so the
    powers add (incoherent sum). Valid only for dilute single scattering.
    """
    if concentration < 0:
        raise ValueError("concentration must be non-negative [particles/m^3]")
    return concentration * bench.probe_volume * scattered_power_per_particle(dia, npar, bench)


@dataclass(frozen=True)
class SensorConfig:
    """Electronics of the receive chain. Nominal design assumptions."""
    responsivity: float = 1.0            # photodiode [A/W] (InGaAs near 1.6 um, nominal)
    dark_current: float = 1.0e-9         # [A]
    tia_gain: float = 1.0e6              # transimpedance R_f [V/A = ohm]
    amplifier_current_noise: float = 1.0e-13   # [A/sqrt(Hz)]
    temperature: float = 293.15          # [K]
    pga_gains: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
    antialias_cutoff: float = 40.0e3     # f_c [Hz], single pole
    adc_bits: int = 12
    adc_vref: float = 1.0                # full scale [V], unipolar 0 .. vref
    sampling_rate: float = 100.0e3       # f_s [Hz]
    samples_per_channel: int = 16        # averaged per angle channel (power of two)

    def __post_init__(self):
        if self.antialias_cutoff >= self.sampling_rate / 2:
            raise ValueError("anti-alias cutoff must be below the Nyquist frequency fs/2")
        if not 4 <= self.adc_bits <= 24:
            raise ValueError("adc_bits must be between 4 and 24")
        n = self.samples_per_channel
        if n < 1 or n & (n - 1):
            raise ValueError("samples_per_channel must be a power of two")
        if min(self.responsivity, self.tia_gain, self.adc_vref, self.sampling_rate) <= 0:
            raise ValueError("responsivity, tia_gain, adc_vref, sampling_rate must be positive")

    @property
    def noise_bandwidth(self):
        """Equivalent noise bandwidth of a single-pole low-pass: (pi/2) f_c [Hz]."""
        return np.pi / 2 * self.antialias_cutoff

    @property
    def lsb(self):
        """ADC step [V]."""
        return self.adc_vref / 2 ** self.adc_bits

    @property
    def max_code(self):
        return 2 ** self.adc_bits - 1


def noise_current(signal_current, cfg: SensorConfig):
    """RMS noise per ADC sample, referred to the photodiode current [A]."""
    signal_current = np.asarray(signal_current, dtype=float)
    b = cfg.noise_bandwidth
    shot = np.sqrt(2 * ph.E_CHARGE * (signal_current + cfg.dark_current) * b)
    thermal = np.sqrt(4 * ph.K_BOLTZMANN * cfg.temperature * b / cfg.tia_gain)
    amplifier = cfg.amplifier_current_noise * np.sqrt(b)
    total = np.sqrt(shot ** 2 + thermal ** 2 + amplifier ** 2)
    return {"shot": shot, "thermal": float(thermal), "amplifier": float(amplifier),
            "total": total}


def noise_floor_power(cfg: SensorConfig):
    """Optical power giving SNR = 1 in one sample, with no signal shot noise [W]."""
    return float(noise_current(0.0, cfg)["total"]) / cfg.responsivity


def choose_pga_gain(peak_voltage, cfg: SensorConfig, headroom=0.8):
    """Largest programmable gain that keeps the peak below headroom * full scale."""
    if not 0 < headroom <= 1:
        raise ValueError("headroom must be in (0, 1]")
    usable = [g for g in cfg.pga_gains if peak_voltage * g <= headroom * cfg.adc_vref]
    return max(usable) if usable else min(cfg.pga_gains)


def acquire(power, cfg: SensorConfig, rng=None, gain=None):
    """Convert optical power per channel [W] to averaged ADC codes.

    Returns a dict:
        codes        mean ADC code per channel (float: sum of n samples / n)
        code_sums    integer sum of the n samples per channel (what hardware holds)
        gain         PGA gain used
        volts        codes converted back to TIA-output volts (gain removed)
        sigma_volts  expected RMS noise of `volts` per channel
        clipped      True where any sample hit the ADC limits
    rng=None gives the noiseless expectation (still quantized).
    """
    power = np.asarray(power, dtype=float)
    if power.ndim != 1 or np.any(power < 0):
        raise ValueError("power must be a 1-D array of non-negative watts")
    n = cfg.samples_per_channel
    current = cfg.responsivity * power + cfg.dark_current
    v_tia = current * cfg.tia_gain
    if gain is None:
        gain = choose_pga_gain(float(v_tia.max()), cfg)
    elif gain not in cfg.pga_gains:
        raise ValueError(f"gain {gain} is not one of the PGA settings {cfg.pga_gains}")

    sigma_i = noise_current(cfg.responsivity * power, cfg)["total"]
    samples = np.repeat(v_tia[:, None], n, axis=1)
    if rng is not None:
        samples = samples + rng.normal(0.0, 1.0, samples.shape) * (sigma_i * cfg.tia_gain)[:, None]
    raw = np.round(samples * gain / cfg.lsb)
    clipped = np.any((raw < 0) | (raw > cfg.max_code), axis=1)
    code_sums = np.clip(raw, 0, cfg.max_code).astype(np.int64).sum(axis=1)
    codes = code_sums / n

    sigma_sample = np.sqrt((sigma_i * cfg.tia_gain) ** 2 + (cfg.lsb / gain) ** 2 / 12)
    return {"codes": codes, "code_sums": code_sums, "gain": int(gain),
            "volts": codes * cfg.lsb / gain, "sigma_volts": sigma_sample / np.sqrt(n),
            "clipped": clipped}


def expected_volts(power, cfg: SensorConfig):
    """Noiseless TIA output including the dark-current offset [V]."""
    return (cfg.responsivity * np.asarray(power, dtype=float) + cfg.dark_current) * cfg.tia_gain


def photon_rate(power, lam):
    """Photons per second in an optical power P at wavelength lam: P / (h f)."""
    return np.asarray(power, dtype=float) / ph.photon_energy(lam)
