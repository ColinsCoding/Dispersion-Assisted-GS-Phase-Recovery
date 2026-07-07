"""Photonic time-stretch / dispersive Fourier transform: the spectrum, in time.

Jalali's time-stretch (STEAM, photonic time-stretch ADC) rests on one fact: send
an optical field through enough group-velocity dispersion and its TEMPORAL output
becomes a scaled copy of its SPECTRUM. A photodiode + oscilloscope then reads an
optical spectrum in real time -- which is how the lab captures single-shot,
non-repetitive events (rogue waves, fast transients) an optical spectrum analyzer
could never scan.

The physics, field-level (this module), underneath the ps/nm engineering formulas
in dgs.coppinger_jalali_1999:

  DISPERSION is an all-pass quadratic phase on the field's spectrum,
      E_out = IFFT{ FFT{E_in} * exp(i * phi2 * omega^2 / 2) },
  where phi2 = beta2 * L is the group-delay dispersion (units time^2). It changes
  NO spectral amplitude -- only phase -- yet reshapes the waveform in time.

  FREQUENCY-TO-TIME MAP. By stationary phase, the component at angular frequency
  omega emerges at time
      t(omega) = -phi2 * omega        (verified numerically here),
  so a spectral feature of width d_omega maps to a temporal width |phi2| d_omega:
  the scope's time axis IS a frequency axis, calibrated by phi2. Two spectral
  lines separated by d_omega land as two pulses separated by |phi2| d_omega.

  VALIDITY (far field). The mapping is faithful only when dispersion dominates the
  pulse's own duration: the dimensionless FAR-FIELD parameter |phi2| / T0^2 must be
  >> 1 (T0 = input pulse duration). Below that you are in the near field and the
  output is not yet the spectrum. This is the condition you check before trusting a
  time-stretch measurement.

  STRETCH FACTOR. In the two-arm STEAM configuration (dispersion phi2_pre before the
  modulator, phi2_post after), a modulated signal is slowed by
      M = 1 + phi2_post / phi2_pre,
  the field-level form of Coppinger-Jalali's M = (D1 L1 + D2 L2)/(D1 L1).

Everything is checkable against the analytic map, so a simulated scope trace can be
compared to a real one. NumPy only; py-3.13. (A CUDA version would parallelize the
FFTs; the math is identical and validated here on CPU.)
"""

import numpy as np

C_NM_PER_PS = 299792.458      # speed of light in nm/ps, for ps/nm <-> phi2


def gdd_transfer(omega, phi2):
    """The dispersion transfer function H(omega) = exp(i * phi2 * omega^2 / 2).
    |H| = 1 everywhere -- dispersion is ALL-PASS (pure phase)."""
    return np.exp(0.5j * phi2 * np.asarray(omega, float) ** 2)


def propagate(E, dt, phi2):
    """Propagate a complex field E sampled at spacing dt through group-delay
    dispersion phi2 (time^2). All-pass: conserves energy, reshapes in time."""
    E = np.asarray(E, complex)
    if dt <= 0:
        raise ValueError("dt must be positive")
    omega = 2 * np.pi * np.fft.fftfreq(len(E), dt)
    return np.fft.ifft(np.fft.fft(E) * gdd_transfer(omega, phi2))


def frequency_to_time(omega, phi2):
    """Stationary-phase map: the spectral component at omega exits at time
    t = -phi2 * omega. The calibration that turns the scope's time axis into a
    frequency (wavelength) axis."""
    if phi2 == 0:
        raise ValueError("phi2 must be nonzero for a frequency-to-time map")
    return -phi2 * np.asarray(omega, float)


def time_to_frequency(t, phi2):
    """Inverse map omega = -t / phi2: read which optical frequency a given
    arrival time corresponds to -- how a time-stretch trace becomes a spectrum."""
    if phi2 == 0:
        raise ValueError("phi2 must be nonzero")
    return -np.asarray(t, float) / phi2


def far_field_parameter(T0, phi2):
    """|phi2| / T0^2: the dimensionless dispersion strength. The dispersive
    Fourier transform (time trace == spectrum) is valid only when this is >> 1;
    of order 1 or less you are in the near field and the map is not yet faithful."""
    if T0 <= 0:
        raise ValueError("T0 (pulse duration) must be positive")
    return abs(phi2) / T0 ** 2


def dispersive_fourier_transform(E_in, dt, phi2):
    """Simulate a time-stretch/DFT measurement: propagate E_in through phi2 and
    return (t, intensity, omega_axis) -- the time grid, the detected intensity
    |E_out(t)|^2 (what the photodiode sees), and the optical-frequency axis
    omega = -t/phi2 that each time sample maps to. In the far field,
    intensity(t) is the input SPECTRUM |FFT{E_in}(omega)|^2 read out in time."""
    E_in = np.asarray(E_in, complex)
    n = len(E_in)
    # E_in is assumed sampled on the centered grid t = (i - n//2)*dt; propagate
    # preserves that index<->time correspondence, so no fftshift is applied
    t = (np.arange(n) - n // 2) * dt
    E_out = propagate(E_in, dt, phi2)
    intensity = np.abs(E_out) ** 2
    omega_axis = time_to_frequency(t, phi2)
    return t, intensity, omega_axis


def spectrum_fidelity(E_in, dt, phi2, floor=0.01):
    """How faithfully the time-stretch trace reproduces the input SPECTRUM:
    the correlation between the detected intensity(t) and the true power
    spectrum |FFT{E_in}|^2 mapped onto the time axis by omega = -t/phi2.
    Approaches 1 in the far field (|phi2|/T0^2 >> 1 AND the record covers the
    mapped span) and falls toward 0 in the near field. A direct 'is this
    measurement in the valid regime?' check."""
    E_in = np.asarray(E_in, complex)
    n = len(E_in)
    t, I, w_axis = dispersive_fourier_transform(E_in, dt, phi2)
    w = 2 * np.pi * np.fft.fftfreq(n, dt)
    S = np.abs(np.fft.fft(E_in)) ** 2
    ws, Ss = np.fft.fftshift(w), np.fft.fftshift(S)
    S_mapped = np.interp(w_axis, ws, Ss, left=0.0, right=0.0)
    mask = I > floor * I.max()
    return float(np.corrcoef(I[mask], S_mapped[mask])[0, 1])


def time_stretch_factor(phi2_pre, phi2_post):
    """STEAM two-arm stretch factor M = 1 + phi2_post/phi2_pre: how many times
    slower the modulated signal runs at the detector. Field-level form of
    dgs.coppinger_jalali_1999's M = (D1 L1 + D2 L2)/(D1 L1)."""
    if phi2_pre == 0:
        raise ValueError("phi2_pre must be nonzero")
    return 1 + phi2_post / phi2_pre


def gdd_from_dispersion(D_ps_per_nm, wavelength_nm=1550.0):
    """Convert a fiber's total dispersion D (ps/nm, i.e. D_param * length) to the
    group-delay dispersion phi2 (ps^2): phi2 = -D * lambda^2 / (2 pi c). Bridges
    the ps/nm engineering numbers of Coppinger-Jalali to this field model."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    return -D_ps_per_nm * wavelength_nm ** 2 / (2 * np.pi * C_NM_PER_PS)


def wavelength_time_calibration(delta_lambda_nm, D_ps_per_nm):
    """Lab calibration in practical units: a spectral span delta_lambda maps to a
    time span delta_t = D * delta_lambda [ps] on the scope (Coppinger-Jalali
    Eq. 1, tau = D L lambda). The ps-per-nm you read off a known spectral marker."""
    return D_ps_per_nm * np.asarray(delta_lambda_nm, float)


if __name__ == "__main__":
    N, dt, T0 = 16384, 0.05, 1.0
    t = (np.arange(N) - N // 2) * dt

    # frequency-to-time map: a pulse at carrier w0 arrives at t = -phi2*w0
    phi2 = 10.0
    print("frequency-to-time map (t = -phi2*omega):")
    for w0 in (2.0, 5.0, 10.0):
        E = np.exp(-t**2 / (2 * T0**2)) * np.exp(1j * w0 * t)
        tt, I, _ = dispersive_fourier_transform(E, dt, phi2)
        print(f"  omega0={w0:5.1f}: peak at t={tt[np.argmax(I)]:8.1f}  "
              f"predicted {frequency_to_time(w0, phi2):8.1f}")

    # two spectral lines -> two pulses; the scope reads the spectrum
    wa, wb = 4.0, 7.0
    E = np.exp(-t**2 / (2 * T0**2)) * (np.exp(1j*wa*t) + np.exp(1j*wb*t))
    tt, I, _ = dispersive_fourier_transform(E, dt, 20.0)
    peaks = tt[np.where((I[1:-1] > I[:-2]) & (I[1:-1] > I[2:]) &
                        (I[1:-1] > 0.2*I.max()))[0] + 1]
    print(f"\ntwo spectral lines d_omega={wb-wa}: temporal peaks at "
          f"{np.round(np.sort(peaks),1)}, separation {abs(np.ptp(peaks)):.1f} "
          f"(predicted |phi2|*d_omega = {20.0*(wb-wa):.1f})")

    print(f"\nfar-field parameter |phi2|/T0^2 = {far_field_parameter(T0, phi2):.0f} "
          f"(>>1 -> the time trace is the spectrum)")
    print(f"stretch factor (pre=1, post=9): M = {time_stretch_factor(1.0, 9.0):.0f}")
    D = 17.0 * 50.0     # 17 ps/nm/km over 50 km
    print(f"phi2 from D=850 ps/nm at 1550 nm: {gdd_from_dispersion(D):.1f} ps^2; "
          f"10 nm span -> {wavelength_time_calibration(10.0, D):.0f} ps on the scope")
