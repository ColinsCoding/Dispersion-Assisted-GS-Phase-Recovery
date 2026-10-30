"""
Nitrogen-vacancy (NV) centers in diamond: a room-temperature qubit that measures magnetic fields.

In plain words
--------------
An NV center is a nitrogen atom next to a missing carbon atom in diamond. Its electrons form
a spin-1 system with three levels, m_s = 0, +1, -1. Its ground-state energy diagram is

    H / h = D S_z^2 + gamma (B . S) + E (S_x^2 - S_y^2),

with zero-field splitting D = 2.87 GHz, electron gyromagnetic ratio gamma = g mu_B / h =
28.03 GHz/T (2.803 MHz per gauss), and strain E. For a field B along the NV axis the levels
are exactly 0 and D +/- gamma B (the Zeeman effect on an energy diagram). The two microwave
transitions m_s = 0 -> +/-1 sit at

    f_+/- = D +/- gamma B   =>   B = (f_+ - f_-) / (2 gamma).

**How it's measured (ODMR, optically detected magnetic resonance):** a green laser (532 nm)
pumps the spin into m_s = 0 and makes it fluoresce red. When a microwave hits a transition,
some population moves to m_s = +/-1, which glows less. Sweep the microwave frequency and the
fluorescence shows Lorentzian dips at f_+/-. Reading their positions gives B.

**Why no dilution refrigerator:** at 300 K, kT/h is about 6 THz, far above 2.87 GHz, so
Boltzmann leaves the three levels about equally filled. The NV is *optically pumped* into
m_s = 0 instead of being cooled into it. That is the key contrast with superconducting qubits.

**The qubit:** m_s = 0 and m_s = -1 form a two-level system. A resonant microwave pulse
rotates its state around the Bloch sphere (the "qubit wheel"), and a pi pulse is a NOT gate.

Diamond has four NV orientations along the <111> body diagonals, so a field in a general
direction gives up to 8 ODMR lines, and that pattern encodes the field's direction.

Units: Hz, tesla, seconds. Angles in radians.

Sources (IEEE style)
--------------------
[1] M. W. Doherty et al., "The nitrogen-vacancy colour centre in diamond," Phys. Rep., vol.
    528, pp. 1-45, 2013 (Hamiltonian, D = 2.87 GHz, g = 2.003).
[2] J. F. Barry et al., "Sensitivity optimization for NV-diamond magnetometry," Rev. Mod.
    Phys., vol. 92, 015004, 2020 (CW-ODMR shot-noise sensitivity, Ramsey).
[3] V. M. Acosta et al., "Temperature dependence of the nitrogen-vacancy magnetic resonance
    in diamond," Phys. Rev. Lett., vol. 104, 070801, 2010 (dD/dT ~ -74 kHz/K near 300 K).
"""
import math

import numpy as np

D_HZ = 2.870e9                      # zero-field splitting
G_NV = 2.003
MU_B_OVER_H = 13.996245e9           # Hz / T
GAMMA_HZ_T = G_NV * MU_B_OVER_H     # ~28.03 GHz / T
DD_DT_HZ_K = -74.2e3                # zero-field splitting vs temperature near 300 K

_S2 = 1 / math.sqrt(2)
SX = _S2 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
SY = _S2 * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)       # basis order: m_s = +1, 0, -1

NV_AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / math.sqrt(3)


def _check_field(B):
    B = np.asarray(B, dtype=float)
    if B.shape != (3,):
        raise ValueError(f"B must be a 3-vector in tesla, got shape {B.shape}")
    return B


def hamiltonian(B_nv, D=D_HZ, E=0.0):
    """Ground-state H/h (Hz) for field B_nv = (Bx, By, Bz) in the NV frame (z along the NV axis)."""
    if D <= 0 or E < 0:
        raise ValueError(f"need D > 0 and E >= 0, got D={D}, E={E}")
    Bx, By, Bz = _check_field(B_nv)
    return (D * SZ @ SZ + GAMMA_HZ_T * (Bx * SX + By * SY + Bz * SZ)
            + E * (SX @ SX - SY @ SY))


def levels(B_nv, D=D_HZ, E=0.0):
    """Eigen-energies (Hz, ascending) and eigenvectors (columns) of the NV ground state."""
    w, v = np.linalg.eigh(hamiltonian(B_nv, D, E))
    return w, v


def axial_levels(B_axial, D=D_HZ):
    """Exact levels for a field along the NV axis: (0, D - gamma B, D + gamma B)."""
    return np.array([0.0, D - GAMMA_HZ_T * B_axial, D + GAMMA_HZ_T * B_axial])


def odmr_frequencies(B_nv, D=D_HZ, E=0.0):
    """The two transition frequencies (Hz) out of the most m_s = 0-like level."""
    w, v = levels(B_nv, D, E)
    k0 = int(np.argmax(np.abs(v[1, :]) ** 2))      # largest overlap with |m_s = 0>
    return np.sort(np.abs(np.delete(w, k0) - w[k0]))


def _nv_frame(B_lab, axis):
    """Express a lab-frame field in a frame whose z is the given NV axis."""
    z = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([B_lab @ x, B_lab @ y, B_lab @ z])


def odmr_lines_all_orientations(B_lab, D=D_HZ, E=0.0):
    """ODMR frequencies for the 4 NV orientations; returns shape (4, 2), Hz."""
    B_lab = _check_field(B_lab)
    return np.array([odmr_frequencies(_nv_frame(B_lab, ax), D, E) for ax in NV_AXES])


def field_from_splitting(f_minus, f_plus):
    """Axial field (T) from the two ODMR lines: B = (f_+ - f_-) / (2 gamma)."""
    if f_plus < f_minus:
        raise ValueError("expected f_plus >= f_minus")
    return (f_plus - f_minus) / (2 * GAMMA_HZ_T)


def odmr_spectrum(f_hz, lines_hz, linewidth_hz, contrast):
    """Normalized fluorescence: 1 - sum of Lorentzian dips (FWHM linewidth, depth contrast each)."""
    if linewidth_hz <= 0 or not 0 <= contrast < 1:
        raise ValueError("need linewidth > 0 and 0 <= contrast < 1")
    f = np.asarray(f_hz, dtype=float)
    g = linewidth_hz / 2
    dips = sum(contrast * g ** 2 / ((f - f0) ** 2 + g ** 2) for f0 in np.ravel(lines_hz))
    return 1 - dips


def cw_sensitivity(linewidth_hz, contrast, photon_rate):
    """Shot-noise-limited CW-ODMR sensitivity (T / sqrt(Hz)) for a Lorentzian line.

    eta = (4 / (3 sqrt 3)) * linewidth / (gamma * contrast * sqrt(R)).
    """
    if linewidth_hz <= 0 or not 0 < contrast < 1 or photon_rate <= 0:
        raise ValueError("need linewidth > 0, 0 < contrast < 1, photon_rate > 0")
    return 4 / (3 * math.sqrt(3)) * linewidth_hz / (GAMMA_HZ_T * contrast * math.sqrt(photon_rate))


def bloch_vector(omega_rabi, detuning, t, start=(0.0, 0.0, 1.0)):
    """Bloch vector(s) after driving for time t (angular frequencies, rad/s).

    Rotation about the axis (Omega, 0, Delta) / Omega_eff at rate Omega_eff (Rodrigues formula).
    z = +1 is m_s = 0 (bright), z = -1 is m_s = -1; P(-1) = (1 - z) / 2.
    """
    if omega_rabi <= 0:
        raise ValueError(f"omega_rabi must be > 0, got {omega_rabi}")
    oe = math.hypot(omega_rabi, detuning)
    n = np.array([omega_rabi, 0.0, detuning]) / oe
    r0 = np.asarray(start, dtype=float)
    th = oe * np.atleast_1d(np.asarray(t, dtype=float))[:, None]
    # rotation sense chosen so that a resonant drive takes +z through -y to -z
    out = (r0 * np.cos(th) + np.cross(n, r0) * np.sin(th) + n * (n @ r0) * (1 - np.cos(th)))
    return out if np.ndim(t) else out[0]


def ramsey_signal(tau_s, detuning_hz, t2star_s):
    """Ramsey fringe P(m_s = 0) = (1 + cos(2 pi detuning tau) exp(-(tau/T2*)^2)) / 2."""
    if t2star_s <= 0:
        raise ValueError(f"T2* must be > 0, got {t2star_s}")
    tau = np.asarray(tau_s, dtype=float)
    return 0.5 * (1 + np.cos(2 * math.pi * detuning_hz * tau) * np.exp(-(tau / t2star_s) ** 2))


def zero_field_splitting(T_k):
    """D(T) ~ D(300 K) + dD/dT (T - 300 K), valid near room temperature."""
    if T_k <= 0:
        raise ValueError(f"temperature must be > 0 K, got {T_k}")
    return D_HZ + DD_DT_HZ_K * (T_k - 300.0)
