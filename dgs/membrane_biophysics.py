"""Membrane biophysics -- the neuron as an RC circuit (E&M meets biology).

A cell membrane is a lipid bilayer ~3 nm thick separating salty water. Electrically
it is three textbook E&M parts at once:

  * CAPACITOR. The thin low-dielectric bilayer is a parallel-plate capacitor,
    C = eps0 eps_r A / d, with a near-universal specific capacitance ~1 uF/cm^2.
  * BATTERIES. Each ion's concentration gradient sets an equilibrium (NERNST)
    voltage E = (RT/zF) ln(C_out/C_in) -- the voltage where diffusion balances drift.
    K+ ~ -95 mV, Na+ ~ +60 mV at body temperature.
  * RESISTOR + the RC clock. Ion channels leak current; the membrane charges with
    time constant tau = R_m C_m (~ms), the same RC as dgs.spice.

Put together (Goldman-Hodgkin-Katz) they give the resting potential ~ -70 mV. This is
electrostatics + a thermodynamic battery + an RC circuit, in a cell. NumPy. Education.
"""

import numpy as np

_R = 8.314462618          # gas constant [J/(mol K)]
_F = 96485.33212          # Faraday constant [C/mol]
_EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]


# ── ion gradients as batteries: the Nernst potential ────────────────
def nernst_potential(C_out, C_in, z=1, T=310.0):
    """Equilibrium (Nernst) potential E = (RT/zF) ln(C_out/C_in) [volts] for an ion of
    charge z. The membrane voltage at which the concentration gradient (diffusion) and
    the electric field (drift) exactly balance. RT/F = 26.7 mV at body temp (310 K)."""
    return (_R * T) / (z * _F) * np.log(C_out / C_in)


def goldman_potential(perm, C_out, C_in, T=310.0):
    """Goldman-Hodgkin-Katz resting potential from K+, Na+ (cations) and Cl- (anion):
        V = (RT/F) ln[ (P_K Ko + P_Na Nao + P_Cl Cli) / (P_K Ki + P_Na Nai + P_Cl Clo) ].
    Cl- enters with in/out swapped (it is an anion). `perm`, `C_out`, `C_in` are dicts
    keyed 'K','Na','Cl'. With physiological values this is ~ -70 mV (the resting cell)."""
    num = perm["K"] * C_out["K"] + perm["Na"] * C_out["Na"] + perm["Cl"] * C_in["Cl"]
    den = perm["K"] * C_in["K"] + perm["Na"] * C_in["Na"] + perm["Cl"] * C_out["Cl"]
    return (_R * T) / _F * np.log(num / den)


# ── the bilayer as a capacitor ──────────────────────────────────────
def specific_capacitance(thickness=2.3e-9, eps_r=2.2):
    """Membrane capacitance per unit area C/A = eps0 eps_r / d [F/m^2]. For a ~2-3 nm
    hydrophobic core this is ~1 uF/cm^2 (= 0.01 F/m^2) -- a biological near-constant."""
    return _EPS0 * eps_r / thickness


def membrane_capacitance(area, thickness=2.3e-9, eps_r=2.2):
    """Total bilayer capacitance C = eps0 eps_r A / d [farads] (parallel-plate)."""
    return specific_capacitance(thickness, eps_r) * area


# ── the RC clock: how fast the membrane responds ────────────────────
def membrane_time_constant(R_m, C_m):
    """Membrane RC time constant tau = R_m C_m [s] -- how fast the voltage responds to
    an input. Same RC as dgs.spice; ~1-20 ms for a neuron."""
    return R_m * C_m


def membrane_charging(t, V_final, tau, V0=0.0):
    """Membrane voltage charging toward V_final with time constant tau:
    V(t) = V_final + (V0 - V_final) exp(-t/tau) -- the RC curve, the neuron's response
    to a step current (reaches 63% of the way in one tau)."""
    return V_final + (V0 - V_final) * np.exp(-np.asarray(t, float) / tau)


def length_constant(R_m_specific, R_i_specific, diameter):
    """Cable length constant lambda = sqrt(R_m d / (4 R_i)) [m] -- how far a passive
    voltage signal spreads along a dendrite/axon before decaying by 1/e. R_m_specific
    [ohm m^2], R_i_specific [ohm m] (axial), diameter [m]."""
    return np.sqrt(R_m_specific * diameter / (4 * R_i_specific))


if __name__ == "__main__":
    print(f"RT/F at 310 K = {_R*310/_F*1e3:.1f} mV")
    print(f"  E_K  (4/140 mM)  = {nernst_potential(4, 140, +1)*1e3:.1f} mV")
    print(f"  E_Na (145/12 mM) = {nernst_potential(145, 12, +1)*1e3:.1f} mV")
    print(f"  E_Cl (110/10 mM) = {nernst_potential(110, 10, -1)*1e3:.1f} mV")
    V = goldman_potential({"K": 1.0, "Na": 0.04, "Cl": 0.45},
                          {"K": 5, "Na": 145, "Cl": 110}, {"K": 140, "Na": 12, "Cl": 10})
    print(f"  resting (Goldman) = {V*1e3:.1f} mV")
    print(f"specific capacitance = {specific_capacitance()*1e2:.2f} uF/cm^2  (F/m^2 * 100)")
    print(f"tau = R_m C_m (1e8 ohm * 100 pF) = {membrane_time_constant(1e8, 100e-12)*1e3:.1f} ms")
