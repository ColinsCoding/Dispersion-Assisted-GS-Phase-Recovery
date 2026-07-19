"""Extends dgs.electron_spin_resonance: three real, historically-linked
puzzles about electron spin, tied together with the parallel axis
theorem, charge-to-mass ratio, and special relativity.

1. CLASSICAL MODEL FAILURE (parallel-axis-theorem/moment-of-inertia route):
   If the electron's spin angular momentum (hbar/2) came from a classical
   charged sphere physically ROTATING (moment of inertia I = (2/5)mr^2 for
   a solid sphere, same formula the parallel axis theorem extends to
   off-center axes), the required equatorial surface speed comes out far
   above the speed of light. This is the real, standard textbook argument
   (Griffiths, Introduction to QM) for why electron spin is NOT literally
   classical rotation -- it's an intrinsic quantum degree of freedom.

2. GYROMAGNETIC ANOMALY: the SAME classical charge-to-mass-ratio physics
   that gives Thomson's e/m predicts a classical gyromagnetic g-factor of
   EXACTLY 1 (mu = (q/2m)*L). The real electron's g-factor is ~2.0023 --
   almost exactly double. The factor of 2 was only explained by Dirac's
   relativistic wave equation (1928); the extra 0.0023 (g-2)/2 is the QED
   anomalous magnetic moment, one of the most precisely verified numbers
   in physics.

3. THOMAS PRECESSION: a SEPARATE special-relativity effect (successive
   non-collinear Lorentz boosts don't commute) makes an orbiting electron's
   spin precess at HALF the naive rate. Historically, this exact factor of
   1/2 was the missing piece that reconciled the (g=2)-based spin-orbit
   coupling prediction with hydrogen's measured fine structure -- without
   it, the naive calculation was off by 2x.

Reuses dgs.electron_spin_resonance's real constants (MU_B, H_PLANCK,
G_FACTOR_FREE_ELECTRON) and dgs.error_propagation's Measurement class for
propagating a real resonance-frequency measurement's uncertainty into the
inferred magnetic field.
"""

import numpy as np

from dgs.electron_spin_resonance import (
    MU_B, H_PLANCK, G_FACTOR_FREE_ELECTRON, field_for_resonance_tesla,
)
from dgs.error_propagation import Measurement

C_LIGHT = 2.998e8       # m/s
Q_ELECTRON = 1.602176634e-19   # C
M_ELECTRON = 9.1093837015e-31  # kg
EPS0 = 8.8541878128e-12         # F/m
HBAR = 1.0546e-34               # J*s
ALPHA_FINE_STRUCTURE = 1.0 / 137.036


def classical_electron_radius_m():
    """r_e = q^2 / (4*pi*eps0*m*c^2) -- real value ~2.818e-15 m."""
    return Q_ELECTRON**2 / (4 * np.pi * EPS0 * M_ELECTRON * C_LIGHT**2)


def charge_to_mass_ratio_c_per_kg(q=Q_ELECTRON, m=M_ELECTRON):
    """Thomson's e/m: q/m -- real value ~1.759e11 C/kg."""
    if m <= 0:
        raise ValueError("m must be positive")
    return q / m


def moment_of_inertia_solid_sphere(mass_kg, radius_m):
    """I = (2/5) * m * r^2 for a uniform solid sphere about its own axis."""
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    return 0.4 * mass_kg * radius_m**2


def parallel_axis_theorem(I_cm_kg_m2, mass_kg, distance_m):
    """I_axis = I_cm + m*d^2 -- moment of inertia about an axis parallel
    to, and a distance d from, one through the center of mass."""
    if I_cm_kg_m2 < 0:
        raise ValueError("I_cm_kg_m2 must be non-negative")
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")
    return I_cm_kg_m2 + mass_kg * distance_m**2


def required_surface_speed_m_s(mass_kg, radius_m, target_angular_momentum=HBAR / 2):
    """If a classical solid sphere of this mass/radius carried the target
    angular momentum (default: hbar/2, the electron's real spin), what
    equatorial surface speed v = omega*r would that require? Returns
    (v_m_s, v_over_c) -- the real textbook punchline is v_over_c >> 1,
    i.e. physically impossible for a classical rotating object."""
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if target_angular_momentum <= 0:
        raise ValueError("target_angular_momentum must be positive")
    I = moment_of_inertia_solid_sphere(mass_kg, radius_m)
    omega = target_angular_momentum / I
    v = omega * radius_m
    return v, v / C_LIGHT


def classical_g_factor():
    """The classical relation mu = (q/2m)*L gives g_classical = 1 exactly
    -- no factor of 2, unlike the real electron's g ~ 2.0023."""
    return 1.0


def gyromagnetic_anomaly(g_quantum=G_FACTOR_FREE_ELECTRON):
    """How far the real electron's g-factor sits from the classical
    prediction of 1 -- the 'factor of 2' Dirac's equation explained,
    plus the small QED anomaly (g-2)/2 on top of that."""
    if g_quantum <= 0:
        raise ValueError("g_quantum must be positive")
    g_class = classical_g_factor()
    return {
        "g_classical": g_class,
        "g_quantum": g_quantum,
        "ratio": g_quantum / g_class,
        "qed_anomaly_a_e": (g_quantum - 2.0) / 2.0,
    }


def thomas_precession_frequency_rad_s(v_m_s, orbital_angular_frequency_rad_s):
    """Non-relativistic-limit Thomas precession: omega_T ~= (1/2)*(v/c)^2
    * omega_orbital. This factor of 1/2 is what reconciles the naive
    (g=2)-based spin-orbit coupling prediction with hydrogen's measured
    fine-structure splitting -- without it, the naive result is 2x too
    large."""
    if v_m_s <= 0:
        raise ValueError("v_m_s must be positive")
    if v_m_s >= C_LIGHT:
        raise ValueError("v_m_s must be less than the speed of light")
    if orbital_angular_frequency_rad_s <= 0:
        raise ValueError("orbital_angular_frequency_rad_s must be positive")
    return 0.5 * (v_m_s / C_LIGHT) ** 2 * orbital_angular_frequency_rad_s


def propagate_field_uncertainty_tesla(f_hz, sigma_f_hz, g_factor=G_FACTOR_FREE_ELECTRON):
    """Given a measured ESR resonance frequency with its uncertainty,
    propagate that into the inferred magnetic field's uncertainty using
    dgs.error_propagation's Measurement class. B = h*f/(g*mu_B) is linear
    in f, so B's RELATIVE uncertainty exactly equals f's relative
    uncertainty -- a good sanity check on the propagation machinery."""
    if f_hz <= 0:
        raise ValueError("f_hz must be positive")
    if sigma_f_hz < 0:
        raise ValueError("sigma_f_hz must be non-negative")
    if g_factor <= 0:
        raise ValueError("g_factor must be positive")
    freq = Measurement(f_hz, sigma_f_hz)
    B = freq * (H_PLANCK / (g_factor * MU_B))
    return B.value, B.sigma


if __name__ == "__main__":
    print("=== 1. Can electron spin be a classically spinning charged sphere? ===\n")
    r_e = classical_electron_radius_m()
    print(f"classical electron radius: {r_e:.3e} m (real value: ~2.818e-15 m)")
    v, v_over_c = required_surface_speed_m_s(M_ELECTRON, r_e)
    print(f"required equatorial surface speed for spin hbar/2: {v:.3e} m/s")
    print(f"v / c = {v_over_c:.1f}x the speed of light -- physically impossible.")
    print("(This is the real, standard textbook argument that spin is intrinsically")
    print(" quantum, not literal classical rotation -- Griffiths Intro to QM.)\n")

    print("=== 2. Gyromagnetic anomaly: classical g=1 vs real electron g~2.0023 ===\n")
    e_over_m = charge_to_mass_ratio_c_per_kg()
    print(f"charge-to-mass ratio q/m: {e_over_m:.4e} C/kg (real Thomson value: ~1.759e11)")
    anomaly = gyromagnetic_anomaly()
    print(f"g_classical = {anomaly['g_classical']}")
    print(f"g_quantum (real electron)  = {anomaly['g_quantum']}")
    print(f"ratio g_quantum/g_classical = {anomaly['ratio']:.4f} "
          f"(Dirac's equation explains this factor of ~2 exactly)")
    print(f"QED anomalous moment a_e = (g-2)/2 = {anomaly['qed_anomaly_a_e']:.4e} "
          f"(one of the most precisely measured numbers in physics)\n")

    print("=== 3. Thomas precession: the relativistic factor of 1/2 ===\n")
    v_1s = ALPHA_FINE_STRUCTURE * C_LIGHT   # real hydrogen 1s orbital speed scale
    omega_orbital = 4.13e16   # rad/s, representative hydrogen n=2 orbital frequency scale
    omega_T = thomas_precession_frequency_rad_s(v_1s, omega_orbital)
    print(f"hydrogen 1s electron characteristic speed (alpha*c): {v_1s:.3e} m/s")
    print(f"Thomas precession frequency (this v, representative orbital omega): "
          f"{omega_T:.3e} rad/s")
    print(f"omega_T / omega_orbital = {omega_T/omega_orbital:.3e} "
          f"(the 'extra factor of 1/2 * (v/c)^2' correction)")
    print("Historically: naive spin-orbit coupling computed with g=2 alone predicted")
    print("a splitting 2x too large vs. hydrogen's measured fine structure. Thomas")
    print("precession's factor of 1/2 is exactly what brings theory back in line with")
    print("experiment -- these are TWO SEPARATE effects (QM g-factor + SR kinematics)")
    print("that happen to combine into the right answer.\n")

    print("=== Error propagation: inferring B-field from a noisy ESR measurement ===\n")
    f_measured_hz = 9.5e9
    sigma_f_hz = 0.02e9   # 0.2 GHz measurement uncertainty, representative
    B_value, B_sigma = propagate_field_uncertainty_tesla(f_measured_hz, sigma_f_hz)
    print(f"measured resonance frequency: {f_measured_hz/1e9:.2f} +/- {sigma_f_hz/1e9:.2f} GHz")
    print(f"inferred field: {B_value:.4f} +/- {B_sigma:.4f} T")
    print(f"relative uncertainty preserved exactly: "
          f"{sigma_f_hz/f_measured_hz:.4f} (freq) vs {B_sigma/B_value:.4f} (field)")
