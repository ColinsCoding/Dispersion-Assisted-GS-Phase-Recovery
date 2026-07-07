"""Test dgs.time_stretch_dft: dispersion is all-pass (energy-conserving), the
frequency-to-time map t=-phi2*omega is exact, two spectral lines land as two
pulses separated by |phi2|*d_omega, the whole trace reproduces the input
spectrum in the far field (and not in the near field), and the stretch factor /
unit conversions. NumPy only."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import time_stretch_dft as ts

N, dt, T0 = 16384, 0.02, 1.0
t = (np.arange(N) - N // 2) * dt

# 1. dispersion is ALL-PASS: |H| = 1 and energy is conserved
omega = 2 * np.pi * np.fft.fftfreq(N, dt)
assert np.allclose(np.abs(ts.gdd_transfer(omega, 12.3)), 1.0)
E = np.exp(-t**2 / (2 * T0**2)) * np.exp(1j * 4 * t)
E_out = ts.propagate(E, dt, 30.0)
assert np.isclose(np.sum(np.abs(E_out)**2), np.sum(np.abs(E)**2), rtol=1e-10)

# 2. frequency-to-time map is EXACT: a pulse at carrier omega0 peaks at -phi2*omega0
phi2 = 10.0
for w0 in (2.0, 5.0, 10.0):
    Ew = np.exp(-t**2 / (2 * T0**2)) * np.exp(1j * w0 * t)
    tt, I, _ = ts.dispersive_fourier_transform(Ew, dt, phi2)
    assert abs(tt[np.argmax(I)] - ts.frequency_to_time(w0, phi2)) < 2 * dt
# the two maps are inverses
assert np.allclose(ts.time_to_frequency(ts.frequency_to_time(omega, 7.0), 7.0), omega)

# 3. far-field parameter |phi2|/T0^2
assert np.isclose(ts.far_field_parameter(2.0, 40.0), 10.0)
assert np.isclose(ts.far_field_parameter(1.0, 10.0), 10.0)

# 4. dispersive_fourier_transform: shapes and the omega axis = -t/phi2
tt, I, w_axis = ts.dispersive_fourier_transform(E, dt, phi2)
assert len(tt) == len(I) == len(w_axis) == N
assert np.allclose(w_axis, -tt / phi2)

# 5. TWO spectral lines -> two pulses separated by |phi2|*d_omega
wa, wb = 3.0, 8.0
phi2b = 20.0
E2 = np.exp(-t**2 / (2 * (2.0)**2)) * (np.exp(1j*wa*t) + np.exp(1j*wb*t))  # narrow lines
tt, I2, _ = ts.dispersive_fourier_transform(E2, dt, phi2b)
idx = np.where((I2[1:-1] > I2[:-2]) & (I2[1:-1] > I2[2:]) & (I2[1:-1] > 0.3*I2.max()))[0] + 1
peaks = np.sort(tt[idx])
assert len(peaks) == 2
assert np.isclose(peaks[-1] - peaks[0], phi2b * abs(wb - wa), rtol=0.05)  # |phi2|*d_omega

# 6. FIDELITY: the trace IS the spectrum in the far field, but not the near field
Estruct = np.exp(-t**2 / (2 * T0**2)) * (1 + 0.8*np.cos(6*t)) * np.exp(1j*3*t)
far = ts.spectrum_fidelity(Estruct, dt, 10.0)      # far-field param 10
near = ts.spectrum_fidelity(Estruct, dt, 0.1)      # near-field param 0.1
assert far > 0.99
assert near < 0.6
assert far > near
# fidelity improves monotonically as dispersion grows toward the far field
fids = [ts.spectrum_fidelity(Estruct, dt, p) for p in (0.3, 1.0, 5.0)]
assert fids[0] < fids[1] < fids[2]

# 7. STEAM stretch factor M = 1 + phi2_post/phi2_pre (Coppinger-Jalali M=10 case)
assert np.isclose(ts.time_stretch_factor(1.0, 9.0), 10.0)
assert np.isclose(ts.time_stretch_factor(5.0, 45.0), 10.0)      # D=17, L1=5, L2=45

# 8. unit bridges: ps/nm -> phi2 (ps^2), and the linear wavelength-time calibration
phi2_ps2 = ts.gdd_from_dispersion(850.0, wavelength_nm=1550.0)
assert phi2_ps2 < 0                                             # anomalous D -> phi2 < 0
assert np.isclose(phi2_ps2, -850.0 * 1550.0**2 / (2*np.pi*ts.C_NM_PER_PS))
assert np.isclose(ts.wavelength_time_calibration(10.0, 850.0), 8500.0)   # D*d_lambda
assert np.allclose(ts.wavelength_time_calibration(np.array([1., 2.]), 850.0), [850., 1700.])

# 9. kwarg bounds
for bad in (lambda: ts.propagate(E, 0, 10.0),
            lambda: ts.frequency_to_time(1.0, 0.0),
            lambda: ts.time_to_frequency(1.0, 0.0),
            lambda: ts.far_field_parameter(0, 10.0),
            lambda: ts.time_stretch_factor(0, 5.0),
            lambda: ts.gdd_from_dispersion(100.0, wavelength_nm=0)):
    try:
        bad()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("test_time_stretch_dft: all checks passed")
