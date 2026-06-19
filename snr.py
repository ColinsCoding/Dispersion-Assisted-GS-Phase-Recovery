"""Signal-to-noise budget for the one-shot carrier-less receiver.

Every measurement in this repo is one shot of light hitting a detector, and the
question is always the same: how much of what you read is signal vs noise? This
collects the canonical SNR results for the three noise floors the receiver
actually fights:

  * shot (Poisson) noise   -- counting photons: SNR = sqrt(N), so SNR_dB = 10 log10 N
  * quantization noise     -- the ADC's finite bits: SQNR_dB = 6.02 B + 1.76
  * thermal (Johnson) noise -- random motion of the *free charge* in a resistor:
                              V_rms = sqrt(4 k_B T R B)

and the ultimate floor, machine precision (~1e-16 for float64). Higher SNR ->
cleaner I1, I2 -> better Gerchberg-Saxton phase recovery. NumPy only. Education.
"""

import numpy as np

KB = 1.380649e-23      # Boltzmann constant [J/K]


def snr_db(signal_power, noise_power):
    """SNR in decibels: 10 log10(P_signal / P_noise)."""
    if signal_power < 0 or noise_power <= 0:
        raise ValueError("signal_power >= 0 and noise_power > 0 required")
    return 10.0 * np.log10(signal_power / noise_power)


def snr_from_signal(clean, noisy):
    """Empirical SNR (dB) from a clean reference and its noisy measurement:
    10 log10( sum clean^2 / sum (noisy-clean)^2 ). The 'how much survived' metric."""
    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    sig = np.sum(clean**2)
    noise = np.sum((noisy - clean)**2)
    if noise == 0:
        return np.inf
    return 10.0 * np.log10(sig / noise)


def shot_noise_snr_db(n_photons):
    """Photon shot noise: count N photons, std is sqrt(N), so SNR = N/sqrt(N) = sqrt(N).
    In power dB that is 10 log10(N). Doubling the light adds ~3 dB; you are forever
    chasing sqrt(N), which is why low-light recovery is hard."""
    n_photons = np.asarray(n_photons, dtype=float)
    if np.any(n_photons <= 0):
        raise ValueError("n_photons must be > 0")
    return 10.0 * np.log10(n_photons)


def quantization_snr_db(n_bits):
    """Ideal ADC signal-to-quantization-noise ratio for a full-scale sinusoid:
    SQNR = 6.02 * B + 1.76 dB. Each extra bit buys ~6 dB -- the price of phase
    resolution in `phase_binary_quantization`."""
    if n_bits <= 0:
        raise ValueError("n_bits must be > 0")
    return 6.02 * n_bits + 1.76


def johnson_noise_voltage(R, T, bandwidth):
    """Thermal (Johnson-Nyquist) noise RMS voltage across a resistor:
    V_rms = sqrt(4 k_B T R B). It is the random thermal motion of the resistor's
    *free charge* (the rho_0 of charge relaxation) -- noise with no signal at all."""
    if R < 0 or T < 0 or bandwidth < 0:
        raise ValueError("R, T, bandwidth must be >= 0")
    return np.sqrt(4 * KB * T * R * bandwidth)


def effective_bits(snr_db_measured):
    """Invert the SQNR formula: ENOB = (SNR_dB - 1.76) / 6.02 (effective number of bits)."""
    return (snr_db_measured - 1.76) / 6.02


if __name__ == "__main__":
    print("shot-noise SNR:   100 photons -> %.1f dB,  10000 -> %.1f dB (sqrt(N) law)"
          % (shot_noise_snr_db(100), shot_noise_snr_db(10000)))
    print("ADC SQNR:         8-bit -> %.1f dB,  12-bit -> %.1f dB (6 dB/bit)"
          % (quantization_snr_db(8), quantization_snr_db(12)))
    print("Johnson noise:    1 kohm, 300 K, 1 MHz -> %.2e V rms"
          % johnson_noise_voltage(1e3, 300, 1e6))
