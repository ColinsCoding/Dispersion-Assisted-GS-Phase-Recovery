"""Phase-modulation sidebands via the Jacobi-Anger expansion -- the Bessel-
function math underneath every electro-optic modulator's (EOM) optical
spectrum.

A pure sinusoidal phase modulator (as opposed to the MZM intensity
modulator already in dgs/lnbo3.py) imposes E(t) = E0*exp(j*beta*sin(2*pi
*fm*t)) on the field, where beta is the modulation index. The Jacobi-
Anger expansion says this is EXACTLY a sum of discrete tones spaced by
fm, with amplitudes given by Bessel functions of the first kind:

    exp(j*beta*sin(x)) = sum_{n=-inf}^{inf} J_n(beta) * exp(j*n*x)

so an optical spectrum analyzer sees a carrier plus symmetric sidebands
at f0 +/- n*fm, with sideband POWER J_n(beta)^2 -- this IS what a real
EOM produces, and this module reuses dgs.lnbo3 (Sellmeier / V_pi) to tie
the abstract beta directly to a real LiNbO3 device's drive voltage.
"""
import numpy as np
from scipy.special import jv
from scipy.optimize import brentq


def phase_modulated_field(t, E0, beta, fm):
    """E(t) = E0 * exp(j*beta*sin(2*pi*fm*t)) -- pure sinusoidal PM, direct form."""
    t = np.asarray(t, float)
    return E0 * np.exp(1j * beta * np.sin(2 * np.pi * fm * t))


def jacobi_anger_sidebands(beta, n_max=10):
    """Sideband complex amplitudes c_n = J_n(beta), n=-n_max..n_max, from the
    Jacobi-Anger expansion exp(j*beta*sin(x)) = sum_n J_n(beta)*exp(j*n*x)."""
    ns = np.arange(-n_max, n_max + 1)
    return ns, jv(ns, beta)


def sideband_powers(beta, n_max=10):
    """P_n = J_n(beta)^2 -- the fraction of total optical power in each
    sideband (what a spectrum analyzer actually reads, normalized)."""
    ns, c = jacobi_anger_sidebands(beta, n_max)
    return ns, c ** 2


def reconstruct_from_sidebands(t, E0, beta, fm, n_max=30):
    """Rebuild E(t) purely from the discrete Jacobi-Anger sideband sum, so it
    can be checked against phase_modulated_field's direct closed form."""
    t = np.asarray(t, float)
    ns, c = jacobi_anger_sidebands(beta, n_max)
    E = np.zeros_like(t, dtype=complex)
    for n, cn in zip(ns, c):
        E += cn * np.exp(1j * n * 2 * np.pi * fm * t)
    return E0 * E


def sideband_energy_conservation(beta, n_max=50):
    """sum_n J_n(beta)^2 = 1 for every beta -- a Bessel identity, and
    physically required: pure phase modulation is lossless, |exp(j*phi)|=1
    always, so no power can appear or vanish, only redistribute across
    sidebands."""
    _, p = sideband_powers(beta, n_max)
    return float(np.sum(p))


def carrier_suppression_beta():
    """The modulation index that fully suppresses the optical carrier
    (J_0(beta)=0, all power pushed into sidebands) -- found by root-finding
    scipy.special.jv(0, .), not hardcoded, then checked against the textbook
    value ~2.405."""
    return brentq(lambda b: jv(0, b), 2.0, 3.0)


def modulation_index_from_voltage(V_rf, V_pi):
    """beta = pi * V_rf / V_pi -- the standard EOM relation (see
    dgs.lnbo3.eo_phase_shift's own docstring: delta_phi(t) = (pi/V_pi)*V_rf*
    cos(2*pi*fm*t)), so a real device's V_pi (from dgs.lnbo3.vpi, reused
    directly below) sets a physically real modulation index."""
    return np.pi * V_rf / V_pi


if __name__ == "__main__":
    from dgs.lnbo3 import vpi

    beta = 1.5
    fm = 10e9      # 10 GHz RF drive
    E0 = 1.0
    t = np.linspace(0, 5 / fm, 20000)

    print(f"=== Jacobi-Anger sidebands at beta={beta} ===")
    ns, c = jacobi_anger_sidebands(beta, n_max=5)
    for n, cn in zip(ns, c):
        print(f"  n={n:+d}:  J_n(beta)={cn:+.5f}   power J_n^2={cn**2:.5f}")

    print(f"\n=== Direct formula vs. Jacobi-Anger reconstruction ===")
    E_direct = phase_modulated_field(t, E0, beta, fm)
    E_recon = reconstruct_from_sidebands(t, E0, beta, fm, n_max=30)
    max_err = np.max(np.abs(E_direct - E_recon))
    print(f"max |E_direct - E_reconstructed| = {max_err:.3e}")

    print(f"\n=== Energy conservation (sum J_n(beta)^2 = 1) ===")
    for b in (0.1, 1.0, 2.405, 5.0, 10.0):
        total = sideband_energy_conservation(b, n_max=80)
        print(f"  beta={b:6.3f}:  sum J_n^2 = {total:.8f}")

    print(f"\n=== Carrier suppression ===")
    beta0 = carrier_suppression_beta()
    print(f"J_0(beta)=0 at beta = {beta0:.6f}  (textbook value: 2.405)")
    _, p0 = sideband_powers(beta0, n_max=1)
    carrier_power = p0[len(p0) // 2]
    print(f"carrier power at that beta: {carrier_power:.3e}  (should be ~0)")

    print(f"\n=== Tying beta to a real LiNbO3 EOM ===")
    L_mm = 50.0
    V_pi = vpi(L_mm)
    V_rf = V_pi * beta0 / np.pi   # drive voltage that hits carrier suppression
    beta_check = modulation_index_from_voltage(V_rf, V_pi)
    print(f"L={L_mm} mm LiNbO3 modulator: V_pi = {V_pi:.3f} V")
    print(f"drive voltage for carrier suppression: V_rf = {V_rf:.3f} V")
    print(f"recovered beta from that voltage: {beta_check:.6f}  (should match {beta0:.6f})")
