"""EM-wave dispersion -- Griffiths E&M Ch 9 straight into the optics research.

Chapter 9.4 (Absorption and Dispersion) IS the physics this repo's receiver lives on.
A Lorentz oscillator gives a frequency-dependent permittivity
    eps(omega) = 1 + wp^2 / (omega0^2 - omega^2 - i gamma omega),
so the refractive index n = sqrt(eps) is COMPLEX: its real part is dispersion (the
phase speed depends on frequency) and its imaginary part is absorption. Dispersion
means a short pulse, made of many frequencies, spreads as it propagates -- governed by
the group-velocity dispersion beta_2 = d^2 k/d omega^2. Propagating a distance L
multiplies the spectrum by the DISPERSION OPERATOR

    H(omega) = exp( i beta_2 omega^2 L / 2 ),

and the dispersion-assisted Gerchberg-Saxton receiver INVERTS exactly this to recover
the phase a detector cannot see. (Chapter 10's retarded potentials are the causality
that makes absorption and dispersion Kramers-Kronig partners -- see dgs.causality.)
NumPy. Education.
"""

import numpy as np

C = 299792458.0          # speed of light [m/s]


def lorentz_index(omega, omega0, gamma, wp):
    """Complex refractive index n(omega) = sqrt(eps) from one Lorentz resonance
    (Griffiths 9.4.3). Re(n) is dispersion; Im(n) is absorption (peaks at omega0)."""
    eps = 1.0 + wp ** 2 / (omega0 ** 2 - omega ** 2 - 1j * gamma * omega)
    return np.sqrt(eps)


def phase_velocity(omega, n_real):
    """v_p = c / n -- the speed of a single frequency's wave crests."""
    return C / np.asarray(n_real, float)


def group_velocity(omega, n_real):
    """v_g = d omega/dk = c / (n + omega dn/domega) -- the speed of a pulse envelope.
    In normal dispersion (dn/domega > 0) v_g < v_p. Numerical derivative on the grid."""
    omega = np.asarray(omega, float)
    k = omega * np.asarray(n_real, float) / C
    return 1.0 / np.gradient(k, omega)


def gvd_beta2(omega, n_real):
    """Group-velocity dispersion beta_2 = d^2 k/d omega^2 [s^2/m]: the curvature of k(omega)
    that spreads a pulse. >0 normal, <0 anomalous (near a resonance)."""
    omega = np.asarray(omega, float)
    k = omega * np.asarray(n_real, float) / C
    return np.gradient(np.gradient(k, omega), omega)


def disperse_pulse(field, t, beta2, L):
    """Propagate a pulse distance L through a medium of GVD beta_2: multiply the spectrum
    by H(omega) = exp(i beta_2 omega^2 L/2) and inverse-transform. Spreads (and chirps) a
    transform-limited pulse -- the exact operation the GS receiver inverts."""
    t = np.asarray(t, float)
    omega = 2 * np.pi * np.fft.fftfreq(len(t), t[1] - t[0])
    F = np.fft.fft(np.asarray(field, complex))
    return np.fft.ifft(F * np.exp(1j * beta2 * omega ** 2 * L / 2))


def pulse_width(t, field):
    """RMS temporal width of the pulse intensity |field|^2."""
    t = np.asarray(t, float)
    I = np.abs(field) ** 2; I = I / I.sum()
    tbar = np.sum(t * I)
    return float(np.sqrt(np.sum((t - tbar) ** 2 * I)))


if __name__ == "__main__":
    # dispersion + absorption from the Lorentz model
    w = np.linspace(0.2, 3.0, 2000); w0, g, wp = 1.0, 0.05, 0.4
    n = lorentz_index(w, w0, g, wp)
    print(f"refractive index away from resonance: n(0.5) = {lorentz_index(0.5,w0,g,wp).real:.4f} (real, n>1)")
    print(f"absorption peaks near omega0=1: Im n(1.0) = {lorentz_index(1.0,w0,g,wp).imag:.4f}")
    # a transform-limited Gaussian spreads under GVD
    t = np.linspace(-50, 50, 8192); pulse = np.exp(-t**2 / (2 * 2.0**2))
    for beta2L in (0, 20, 80):
        out = disperse_pulse(pulse, t, beta2L, 1.0)
        print(f"  beta2*L = {beta2L:3d}: pulse width {pulse_width(t, pulse):.2f} -> {pulse_width(t, out):.2f}")
    # the receiver's move: disperse then UN-disperse recovers the pulse
    there = disperse_pulse(pulse, t, 80, 1.0)
    back = disperse_pulse(there, t, -80, 1.0)
    print(f"disperse(+L) then (-L) recovers pulse: max err {np.max(np.abs(back - pulse)):.2e}")
