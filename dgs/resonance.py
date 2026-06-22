"""Resonance -- finding an RLC's resonant frequency and Q by a frequency sweep.

Drive a series R-L-C with an AC source and watch the voltage across R:
    H(omega) = R / (R + j(omega L - 1/(omega C))).
The inductor's reactance (+omega L) and the capacitor's (-1/omega C) cancel at
    omega_0 = 1 / sqrt(LC)   (the RESONANT angular frequency),
where |H| peaks at 1, the current is maximal, and the phase crosses zero. How SHARP
that peak is is the quality factor
    Q = (1/R) sqrt(L/C) = omega_0 L / R,
and the -3 dB bandwidth is Delta_omega = R/L = omega_0/Q. High Q = a narrow, selective
peak (a radio tuner); low Q = broad.

The "experiment": sweep omega, record |H|, and read omega_0 (the peak) and Q (peak
over -3 dB width) straight off the curve -- exactly how you find a resonance on a
bench. Connects to dgs.ac_circuits / dgs.spice. NumPy. Education.
"""

import numpy as np


def rlc_response(omega, R, L, C):
    """Series-RLC bandpass response H(omega) = R/(R + j(omega L - 1/(omega C))): the
    voltage across R as a fraction of the drive. |H| = 1 at resonance (reactances
    cancel), falling on either side."""
    omega = np.asarray(omega, float)
    Z = R + 1j * (omega * L - 1.0 / (omega * C))
    return R / Z


def resonant_frequency(L, C):
    """Resonant angular frequency omega_0 = 1/sqrt(LC) [rad/s]."""
    return 1.0 / np.sqrt(L * C)


def quality_factor(R, L, C):
    """Q = (1/R) sqrt(L/C) = omega_0 L/R -- the sharpness/selectivity of the resonance."""
    return (1.0 / R) * np.sqrt(L / C)


def bandwidth(R, L):
    """-3 dB bandwidth Delta_omega = R/L [rad/s] = omega_0/Q (where |H| >= 1/sqrt(2))."""
    return R / L


def frequency_sweep(R, L, C, n=4000, span=8.0):
    """The experiment: sweep omega around omega_0 and return (omega, H). `span` sets the
    range as omega_0/span .. omega_0*span (log-spaced)."""
    w0 = resonant_frequency(L, C)
    omega = np.logspace(np.log10(w0 / span), np.log10(w0 * span), n)
    return omega, rlc_response(omega, R, L, C)


def find_resonance(omega, response):
    """Read (omega_peak, Q_measured) off a measured |H(omega)| sweep: the peak frequency
    and Q = omega_peak / (-3 dB bandwidth). This is resonance found by experiment."""
    mag = np.abs(response)
    ipk = int(np.argmax(mag))
    omega_pk, peak = omega[ipk], mag[ipk]
    above = np.where(mag >= peak / np.sqrt(2))[0]
    bw = omega[above[-1]] - omega[above[0]] if len(above) > 1 else 0.0
    return omega_pk, (omega_pk / bw if bw > 0 else np.inf)


def find_resonance_torch(R, L, C, steps=800, lr=0.05):
    """Find omega_0 by OPTIMIZATION instead of the formula: gradient-descend the
    impedance magnitude |Z(omega)| = sqrt(R^2 + (omega L - 1/(omega C))^2) with torch
    autograd, starting off-peak. Minimizing |Z| maximizes the current (resonance).
    Returns (omega_found [rad/s], trajectory). Should land on resonant_frequency(L, C)."""
    import torch
    w0 = resonant_frequency(L, C)
    logw = torch.tensor(float(np.log(w0 * 0.4)), requires_grad=True)   # start below the peak
    opt = torch.optim.Adam([logw], lr=lr)
    traj = []
    for _ in range(steps):
        opt.zero_grad()
        w = torch.exp(logw)
        absZ = torch.sqrt(torch.tensor(float(R)) ** 2 + (w * L - 1.0 / (w * C)) ** 2)
        absZ.backward()                      # minimize |Z| -> maximize current
        opt.step()
        traj.append(float(torch.exp(logw).item()))
    return traj[-1], np.array(traj)


if __name__ == "__main__":
    R, L, C = 5.0, 1e-3, 1e-6
    w0 = resonant_frequency(L, C)
    print(f"omega_0 = 1/sqrt(LC) = {w0:.1f} rad/s  (f0 = {w0/2/np.pi:.1f} Hz)")
    print(f"Q (formula) = {quality_factor(R, L, C):.2f},  bandwidth = {bandwidth(R, L):.1f} rad/s")
    omega, H = frequency_sweep(R, L, C)
    w_pk, Q_meas = find_resonance(omega, H)
    print(f"experiment: peak at {w_pk:.1f} rad/s, Q_measured = {Q_meas:.2f}")
    print(f"at resonance: |H| = {abs(rlc_response(w0, R, L, C)):.4f}, "
          f"phase = {np.degrees(np.angle(rlc_response(w0, R, L, C))):.2f} deg")
