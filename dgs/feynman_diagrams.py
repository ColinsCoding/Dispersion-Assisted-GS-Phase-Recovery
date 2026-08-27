"""Feynman diagrams for nonlinear photonics (Jalali-lab-style chi(2)/chi(3)
processes), and the measurement principle that motivates phase retrieval:

    You can only measure CHANGE -- and remember it.

A photodetector counts photons / intensity (|E|^2): a DIFFERENCE in photon
number, an ENERGY transferred. It never reads absolute phase directly -- phase
is not a "change" a detector can register, it is the thing that got thrown
away in the |.|^2. That's the whole reason GS phase retrieval (dgs.gs_core)
exists: reconstruct what intensity measurements alone cannot remember.

Each nonlinear-optics process below is a vertex (or chain of vertices) in
a perturbation series. Energy conservation at every vertex IS a "change"
statement: sum(omega_in) = sum(omega_out). The diagrams differ only in which
photons are absorbed vs emitted -- the conserved quantity is always the
total energy change across the vertex, never an absolute energy.
"""
import numpy as np
import sympy as sp


# -- Vertex / process registry --------------------------------------------------
# Each process is described by an ordered list of (label, sign) photon legs,
# sign=+1 for an emitted (output) photon, sign=-1 for an absorbed (input)
# photon. Energy conservation requires sum(sign_i * omega_i) = 0.

PROCESSES = {
    "SHG": {  # second harmonic generation, chi(2)
        "order": 2,
        "susceptibility": "chi(2)",
        "legs": [("omega", -1), ("omega", -1), ("2*omega", +1)],
        "description": "Two pump photons absorbed, one photon at 2*omega emitted",
    },
    "SFG": {  # sum frequency generation, chi(2)
        "order": 2,
        "susceptibility": "chi(2)",
        "legs": [("omega1", -1), ("omega2", -1), ("omega1+omega2", +1)],
        "description": "Two photons of different frequency absorbed, sum-frequency photon emitted",
    },
    "DFG": {  # difference frequency generation, chi(2)
        "order": 2,
        "susceptibility": "chi(2)",
        "legs": [("omega1", -1), ("omega2", +1), ("omega1-omega2", +1)],
        "description": "Pump photon absorbed, signal photon stimulated-emitted, idler photon emitted",
    },
    "THG": {  # third harmonic generation, chi(3)
        "order": 3,
        "susceptibility": "chi(3)",
        "legs": [("omega", -1), ("omega", -1), ("omega", -1), ("3*omega", +1)],
        "description": "Three pump photons absorbed, one photon at 3*omega emitted",
    },
    "FWM": {  # degenerate four-wave mixing, chi(3) -- the TS-DFT/fiber-comb workhorse
        "order": 3,
        "susceptibility": "chi(3)",
        "legs": [("omega_p", -1), ("omega_p", -1), ("omega_s", +1), ("omega_i", +1)],
        "description": "Two pump photons absorbed, signal+idler photon pair emitted "
                       "(parametric gain -- the fiber-optic process behind supercontinuum "
                       "generation and TS-DFT chirped-pulse readout)",
    },
    "SRS": {  # stimulated Raman scattering, chi(3)
        "order": 3,
        "susceptibility": "chi(3)",
        "legs": [("omega_p", -1), ("omega_p", -1), ("omega_p", +1), ("omega_s", +1)],
        "description": "Pump photon absorbed and converted to a lower-energy Stokes photon, "
                       "energy difference deposited as a phonon (vibrational change)",
    },
}


def energy_conservation_check(process_name, freq_values):
    """Substitute numeric frequencies into a process's legs and verify
    sum(sign_i * omega_i) = 0 (within float tolerance).
    freq_values: dict mapping symbol names used in PROCESSES legs to numbers,
    e.g. {"omega": 1.0} for SHG, or pre-resolved combos for sum/difference legs."""
    if process_name not in PROCESSES:
        raise ValueError(f"unknown process '{process_name}'")
    proc = PROCESSES[process_name]
    total = 0.0
    resolved = []
    for label, sign in proc["legs"]:
        expr = sp.sympify(label, locals={k: sp.Symbol(k) for k in freq_values})
        val = float(expr.subs(freq_values))
        total += sign * val
        resolved.append((label, sign, val))
    return {
        "process": process_name,
        "legs_resolved": resolved,
        "energy_change": total,
        "conserved": bool(abs(total) < 1e-9),
    }


def diagram_order(process_name):
    """Perturbation order = number of vertices = len(susceptibility tensor
    rank) - 1, i.e. chi(2) is a 3-leg (2nd order) vertex, chi(3) is 4-leg."""
    return PROCESSES[process_name]["order"]


def phase_matching_condition(process_name):
    """Symbolic momentum (k-vector) conservation -- the spatial analog of the
    energy conservation at each vertex. Diagram only contributes efficiently
    when this is (approximately) satisfied -- a "change" statement in k-space."""
    proc = PROCESSES[process_name]
    n = len(proc["legs"])
    k_syms = sp.symbols(f'k1:{n+1}')
    signs = [sign for _, sign in proc["legs"]]
    delta_k = sum(s * k for s, k in zip(signs, k_syms))
    return sp.Eq(sp.Symbol('Delta_k'), delta_k)


# -- "You can only measure change" -- the phase-retrieval connection ----------

def detector_response(E_complex):
    """A photodetector measures |E|^2 -- a change in photon flux, never the
    absolute phase of E. This is the irreversible step phase retrieval must
    undo: amplitude survives, phase information is discarded by the square."""
    E = np.asarray(E_complex, complex)
    intensity = np.abs(E) ** 2
    phase_discarded = np.angle(E)
    return {
        "intensity_measured": intensity,
        "phase_discarded": phase_discarded,
        "note": "Detector remembers |E|^2 (a change in photon count); "
                "phase is gone unless reconstructed (e.g. via GS).",
    }


def measurement_vs_memory_table():
    """Pairs of (what physically changes) vs (what gets remembered/measured)
    across the processes in this module and the broader repo."""
    return {
        "Photodetector": {"changes": "photon arrival", "measured/remembered": "intensity |E|^2, not phase"},
        "SHG/SFG/DFG vertex": {"changes": "photon energy redistribution", "measured/remembered": "energy conservation Delta(omega)=0"},
        "FWM vertex": {"changes": "two pump photons -> signal+idler pair", "measured/remembered": "phase-matching Delta(k)=0"},
        "SRS vertex": {"changes": "photon energy -> phonon (vibration)", "measured/remembered": "Stokes shift (the change IS the signal)"},
        "GS algorithm": {"changes": "estimate updated each iteration", "measured/remembered": "only intensity constraints |G_k|=sqrt(I_k); phase is inferred, never measured"},
        "common_lesson": "Every device in this list records a DIFFERENCE (energy, "
                          "intensity, photon number) -- never an absolute, unmeasurable "
                          "quantity like phase. Algorithms (GS) exist precisely to "
                          "recover what was never recorded.",
    }


def feynman_sympy_5():
    """Five symbolic equations: SHG energy conservation, FWM energy
    conservation, generic phase-matching, detector intensity (|E|^2), and the
    GS-update view of phase as the thing NOT measured."""
    omega, omega1, omega2 = sp.symbols('omega omega_1 omega_2', positive=True)
    omega_p, omega_s, omega_i = sp.symbols('omega_p omega_s omega_i', positive=True)
    k1, k2, k3 = sp.symbols('k1 k2 k3')
    E = sp.Function('E')
    t = sp.Symbol('t', real=True)

    return {
        "SHG_energy_conservation":
            sp.Eq(2 * omega, sp.Symbol('omega_2omega')),
        "FWM_energy_conservation":
            sp.Eq(2 * omega_p, omega_s + omega_i),
        "Phase_matching":
            sp.Eq(sp.Symbol('Delta_k'), k1 + k2 - k3),
        "Detector_intensity":
            sp.Eq(sp.Symbol('I'), sp.Abs(E(t))**2),
        "GS_phase_not_measured":
            sp.Eq(E(t), sp.sqrt(sp.Symbol('I')) * sp.exp(sp.I * sp.Symbol('phi_unmeasured'))),
    }


if __name__ == "__main__":
    print("=== Process registry ===")
    for name, proc in PROCESSES.items():
        print(f"  {name} ({proc['susceptibility']}, order {proc['order']}): {proc['description']}")

    print("\n=== Energy conservation check: SHG, omega=1.0 ===")
    res = energy_conservation_check("SHG", {"omega": 1.0})
    print(f"  energy_change = {res['energy_change']:.6f}, conserved = {res['conserved']}")

    print("\n=== Energy conservation check: FWM, omega_p=1.0, omega_s=1.2, omega_i=0.8 ===")
    res2 = energy_conservation_check("FWM", {"omega_p": 1.0, "omega_s": 1.2, "omega_i": 0.8})
    print(f"  energy_change = {res2['energy_change']:.6f}, conserved = {res2['conserved']}")

    print("\n=== Phase matching condition: FWM ===")
    print(f"  {phase_matching_condition('FWM')}")

    print("\n=== Detector response: E = 2*exp(i*0.7) ===")
    dr = detector_response([2 * np.exp(1j * 0.7)])
    print(f"  intensity = {dr['intensity_measured']}, phase (discarded) = {dr['phase_discarded']}")

    print("\n=== Measurement vs memory ===")
    for k, v in measurement_vs_memory_table().items():
        if isinstance(v, dict):
            print(f"  {k}: changes={v['changes']}")

    print("\n=== SymPy 5 ===")
    for k, eq in feynman_sympy_5().items():
        print(f"  {k}: {eq}")
