"""Stern-Gerlach (silver atom beam) + the anomalous Zeeman effect in
hydrogen, including spin (Lande g-factor) -- torch tensors for the
quantum-number bookkeeping, matplotlib for the resulting stick spectrum.
py-3.12 ONLY (torch), per this repo's convention.

STERN-GERLACH, 1922: a beam of silver atoms passed through an
inhomogeneous magnetic field splits into discrete spots, not a
continuous smear. Silver's single unpaired 5s electron has L=0, so the
ONLY angular momentum available is spin (S=1/2) -- and 2S+1 = 2 is an
EVEN number. This is the historically critical point: any ORBITAL
angular momentum l always gives an ODD number of components (2l+1,
including an always-present undeflected m_l=0 beam). An even split with
NO undeflected center beam is impossible for orbital angular momentum
alone -- it is the direct experimental proof that electron spin is a
new, separate kind of angular momentum.

ANOMALOUS ZEEMAN EFFECT: when spin is included, each fine-structure term
(defined by L, S, J) gets its own Lande g-factor
  g_J = 1 + [J(J+1) + S(S+1) - L(L+1)] / (2*J*(J+1))
Different terms have different g_J, so a transition between two terms
(e.g. 2P_3/2 -> 2S_1/2) splits into MORE than the naive 3-line "normal"
Zeeman pattern -- because the upper and lower sublevels shift by
different amounts. This module computes the real g-factors for
hydrogen's low terms (matches known textbook values: 2S_1/2 -> 2,
2P_1/2 -> 2/3, 2P_3/2 -> 4/3), builds the allowed transition lines under
the selection rule delta_m_J in {-1,0,1}, and plots the spectrum.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

MU_B = 9.274e-24     # J/T, Bohr magneton
H_PLANCK = 6.626e-34  # J*s
K_BOLTZMANN = 1.380649e-23  # J/K
AMU = 1.6605e-27      # kg


def lande_g_factor(L, S, J):
    """g_J = 1 + [J(J+1) + S(S+1) - L(L+1)] / (2*J*(J+1)).
    Real known hydrogen values: (L=0,S=1/2,J=1/2)->2 ; (L=1,S=1/2,J=1/2)
    -> 2/3 ; (L=1,S=1/2,J=3/2) -> 4/3."""
    if J <= 0:
        raise ValueError("J must be positive")
    if S < 0 or L < 0:
        raise ValueError("S and L must be non-negative")
    L_t = torch.tensor(float(L), dtype=torch.float64)
    S_t = torch.tensor(float(S), dtype=torch.float64)
    J_t = torch.tensor(float(J), dtype=torch.float64)
    g = 1.0 + (J_t * (J_t + 1) + S_t * (S_t + 1) - L_t * (L_t + 1)) / (2 * J_t * (J_t + 1))
    return float(g)


def zeeman_sublevel_energies_j(g_J, J, B_tesla):
    """Energy shift of each m_J sublevel: delta_E = g_J * mu_B * B * m_J,
    for m_J = -J, -J+1, ..., +J (2J+1 states -- torch tensor for the
    quantum-number sweep)."""
    if J <= 0:
        raise ValueError("J must be positive")
    if B_tesla < 0:
        raise ValueError("B_tesla must be non-negative")
    n_states = int(round(2 * J + 1))
    m_J = torch.linspace(-J, J, n_states, dtype=torch.float64)
    delta_E = g_J * MU_B * B_tesla * m_J
    return m_J, delta_E


def is_even_split(J):
    """2J+1 states: spin-1/2-only angular momentum (J=1/2) gives an EVEN
    count (2) with no undeflected center beam -- impossible for any pure
    orbital angular momentum l (which always gives an ODD count, 2l+1,
    including an undeflected m_l=0 beam). This is the real Stern-Gerlach
    historical argument for spin as a distinct kind of angular momentum."""
    if J <= 0:
        raise ValueError("J must be positive")
    n_states = int(round(2 * J + 1))
    return n_states % 2 == 0


def stern_gerlach_deflection_m(dBdz_t_per_m, magnet_length_m, drift_length_m,
                                 mass_kg, velocity_m_s, g_J=2.0, m_J=0.5):
    """Real Stern-Gerlach ballistic-deflection physics: force
    F = g_J*mu_B*m_J*(dB/dz) acts over the magnet length (parabolic
    deflection), then the atom drifts in a straight line to the screen.
    Returns total transverse deflection for one spin state; the total
    beam SPLIT is 2x this (the two m_J states deflect oppositely)."""
    if dBdz_t_per_m <= 0:
        raise ValueError("dBdz_t_per_m must be positive")
    if magnet_length_m <= 0 or drift_length_m <= 0:
        raise ValueError("magnet_length_m and drift_length_m must be positive")
    if mass_kg <= 0 or velocity_m_s <= 0:
        raise ValueError("mass_kg and velocity_m_s must be positive")
    F = g_J * MU_B * abs(m_J) * dBdz_t_per_m
    t_in_magnet = magnet_length_m / velocity_m_s
    z_in_magnet = 0.5 * (F / mass_kg) * t_in_magnet**2
    v_transverse = (F / mass_kg) * t_in_magnet
    t_drift = drift_length_m / velocity_m_s
    z_drift = v_transverse * t_drift
    return z_in_magnet + z_drift


def most_probable_beam_speed_m_s(temperature_k, mass_kg):
    """v_p = sqrt(2*k_B*T/m) -- most probable speed in an effusive
    thermal beam (Maxwell-Boltzmann), the real velocity scale for an
    oven-heated atomic beam source."""
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    return np.sqrt(2 * K_BOLTZMANN * temperature_k / mass_kg)


def zeeman_transition_lines(L_upper, S_upper, J_upper, L_lower, S_lower, J_lower,
                              B_tesla, E0_joules):
    """All allowed transition energies between two fine-structure terms
    under the electric-dipole selection rule delta_m_J in {-1, 0, +1}.
    Different Lande g-factors for the upper/lower terms make the number
    of DISTINCT line energies generally MORE than 3 (the naive "normal"
    Zeeman triplet) -- the anomalous Zeeman pattern."""
    if B_tesla < 0:
        raise ValueError("B_tesla must be non-negative")
    g_upper = lande_g_factor(L_upper, S_upper, J_upper)
    g_lower = lande_g_factor(L_lower, S_lower, J_lower)
    m_J_upper, dE_upper = zeeman_sublevel_energies_j(g_upper, J_upper, B_tesla)
    m_J_lower, dE_lower = zeeman_sublevel_energies_j(g_lower, J_lower, B_tesla)

    lines = []
    for i, mu in enumerate(m_J_upper.tolist()):
        for j, ml in enumerate(m_J_lower.tolist()):
            if abs(mu - ml) <= 1.0 + 1e-9:   # delta_m_J in {-1, 0, +1}
                E = E0_joules + float(dE_upper[i]) - float(dE_lower[j])
                lines.append({"m_J_upper": mu, "m_J_lower": ml, "energy_j": E})
    return lines, g_upper, g_lower


def plot_zeeman_spectrum(lines, save_path="stern_gerlach_zeeman_hydrogen.png"):
    """Stick spectrum of the anomalous Zeeman-split transition lines."""
    if not lines:
        raise ValueError("lines must be a non-empty list")
    energies_ev = np.array([l["energy_j"] for l in lines]) / 1.602e-19
    fig, ax = plt.subplots(figsize=(8, 4))
    for e in energies_ev:
        ax.axvline(e, ymin=0.1, ymax=0.9, color="C0", linewidth=1.5)
    ax.set_xlabel("Transition energy (eV, relative shift about line center)")
    ax.set_title(f"Anomalous Zeeman spectrum ({len(lines)} distinct lines)")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    print("=== Stern-Gerlach: silver beam, spin gives an EVEN split ===\n")
    J_silver_spin = 0.5   # silver 5s electron: L=0, S=1/2, J=1/2
    n_states_silver = int(round(2 * J_silver_spin + 1))
    print(f"silver 5s valence electron: L=0, S=1/2, J=1/2 -> "
          f"2J+1 = {n_states_silver} beam spots (even, no center spot)")
    print(f"is_even_split: {is_even_split(J_silver_spin)} "
          f"-- impossible for any pure orbital l (always odd, 2l+1)\n")

    m_Ag_kg = 107.87 * AMU
    T_oven_k = 1200.0
    v_beam = most_probable_beam_speed_m_s(T_oven_k, m_Ag_kg)
    print(f"silver oven temperature: {T_oven_k:.0f} K -> "
          f"most probable beam speed: {v_beam:.1f} m/s")

    dBdz = 1000.0    # T/m, representative Stern-Gerlach magnet gradient
    L_magnet = 0.035  # m, representative magnet length
    L_drift = 0.035   # m, representative drift to screen
    z_one_state = stern_gerlach_deflection_m(dBdz, L_magnet, L_drift, m_Ag_kg, v_beam)
    print(f"deflection of ONE spin state: {z_one_state*1e3:.3f} mm")
    print(f"total beam SPLIT (both states): {2*z_one_state*1e3:.3f} mm "
          f"(real 1922 experiment: sub-mm splitting -- same order of magnitude)\n")

    print("=== Anomalous Zeeman effect: hydrogen Lande g-factors ===\n")
    terms = [("2S_1/2", 0, 0.5, 0.5), ("2P_1/2", 1, 0.5, 0.5), ("2P_3/2", 1, 0.5, 1.5)]
    for name, L, S, J in terms:
        g = lande_g_factor(L, S, J)
        print(f"  {name}: L={L}, S={S}, J={J}  ->  g_J = {g:.4f}")
    print("  (real known values: 2S_1/2 -> 2.0000, 2P_1/2 -> 0.6667, 2P_3/2 -> 1.3333)\n")

    print("=== Full anomalous Zeeman spectrum: 2P_3/2 -> 2S_1/2 ===\n")
    B_field = 0.5   # T, representative lab field
    E0 = 1.634e-18   # J, representative Lyman-alpha-scale transition energy (unused precisely)
    lines, g_upper, g_lower = zeeman_transition_lines(
        L_upper=1, S_upper=0.5, J_upper=1.5,
        L_lower=0, S_lower=0.5, J_lower=0.5,
        B_tesla=B_field, E0_joules=E0)
    print(f"g_upper (2P_3/2) = {g_upper:.4f}, g_lower (2S_1/2) = {g_lower:.4f}")
    print(f"number of allowed (delta_m_J = 0, +-1) transition lines: {len(lines)}")
    print("(more than the 'normal' Zeeman triplet of 3 -- because g_upper != g_lower,")
    print(" this IS the anomalous Zeeman pattern.)\n")

    save_path = plot_zeeman_spectrum(lines)
    print(f"spectrum plotted and saved to: {save_path}")
