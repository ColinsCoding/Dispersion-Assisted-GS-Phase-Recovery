"""The electronic analog of optical dispersion: a first-order ACTIVE
ALL-PASS FILTER, which has |H(f)|=1 for EVERY frequency (verified
symbolically above: pure phase, zero magnitude distortion) -- structurally
the same category of transfer function as the optical dispersion operator
H(f)=exp(j*pi*D*f^2), which is also pure phase (|H(f)|=1 there too, since
it's a complex exponential). Two different RC time constants give two
different phase-vs-frequency curves, playing the role of D1 != D2 in the
optical experiment -- the measurement diversity dgs.gs_core needs.

CIRCUIT (a standard, buildable design -- one op-amp, two resistors, one
capacitor, per stage):
    V+ (op-amp) <- R -> node <- C -> ground        (RC low-pass divider on V+)
    V- (op-amp) <- Rf -> Vin,  Vout <- Rf -> V-      (Rf = Rf, unity-gain feedback)
    Vin --------------------------> R,C node (same Vin drives both paths)

    H(s) = (1 - sRC) / (1 + sRC)

This module builds V1(t), V2(t) by filtering the SAME input signal Vin(t)
through two such stages with different (R1,C1) vs (R2,C2), exactly
mirroring dgs.gs_core.make_measurements's disperse(E,D1) / disperse(E,D2)
-- but for a real, buildable electronic circuit instead of a fiber.
"""

import numpy as np
import sympy as sp


def derive_transfer_function_from_circuit_equations():
    """SPICE-style verification: derive H(f) from the actual op-amp CIRCUIT
    EQUATIONS (KCL/virtual-short), not by assuming the known formula.
    V+ = Vin*Zc/(R+Zc) (the RC low-pass divider sets the non-inverting
    input); ideal op-amp virtual short forces V- = V+; equal feedback/
    input resistors make V- = (Vin+Vout)/2 (a standard summing-junction
    relation). Solving these two facts for Vout/Vin should reproduce
    allpass_transfer_function's formula exactly -- confirming the formula
    is a CONSEQUENCE of the circuit, not an assumed fact."""
    w, R, C = sp.symbols('omega R C', real=True, positive=True)
    # Vin, Vout are AC PHASORS (complex amplitudes -- the real time-domain
    # voltage is Re[Vin*exp(j*omega*t)]), NOT real numbers themselves;
    # declaring them real=True previously made sp.solve silently return a
    # WRONG answer to satisfy that false constraint -- caught by comparing
    # against direct algebraic substitution, which disagreed with solve()
    Vin, Vout = sp.symbols('Vin Vout')
    Zc = 1 / (sp.I * w * C)
    V_plus = Vin * Zc / (R + Zc)
    solution = sp.solve(sp.Eq(V_plus, (Vin + Vout) / 2), Vout)
    # sp.solve can return either a list of solutions or a {symbol: expr} dict
    # depending on the internal strategy it picks -- handle both robustly
    solved = solution[Vout] if isinstance(solution, dict) else solution[0]
    H_derived = sp.simplify(solved / Vin)
    H_known = (1 - sp.I * w * R * C) / (1 + sp.I * w * R * C)
    matches = sp.simplify(H_derived - H_known) == 0
    return H_derived, matches


def allpass_transfer_function(f, R, C):
    """H(f) = (1 - j*2*pi*f*R*C) / (1 + j*2*pi*f*R*C) for the standard
    first-order active all-pass stage. |H(f)|=1 for every f (verified
    separately, symbolically) -- pure phase, the electronic dual of an
    optical dispersion operator."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    f = np.asarray(f, dtype=float)
    w = 2 * np.pi * f
    return (1 - 1j * w * R * C) / (1 + 1j * w * R * C)


def allpass_phase(f, R, C):
    """phi(f) = -2*arctan(2*pi*f*R*C), the phase this stage imparts --
    the direct analog of optical dispersion's phase phi(f)=pi*D*f^2, just
    a different functional shape (arctan vs quadratic) because it comes
    from a different physical mechanism (a single RC pole vs a length of
    dispersive fiber)."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    f = np.asarray(f, dtype=float)
    return -2 * np.arctan(2 * np.pi * f * R * C)


def apply_allpass_filter(signal, dt, R, C):
    """Filter a real time-domain signal through the all-pass stage: FFT,
    multiply by H(f), inverse FFT -- the electronic equivalent of
    dgs.gs_core.disperse(E, D)."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=dt)
    spectrum = np.fft.fft(signal)
    H = allpass_transfer_function(freqs, R, C)
    filtered = np.fft.ifft(spectrum * H)
    return filtered   # complex: real part is the actual scope trace, imag is the Hilbert-pair


def design_two_arm_circuit(f_signal_bandwidth_hz, phase_diversity_target_rad=1.0):
    """Pick two RC time constants that give USEFUL phase diversity across
    the signal's bandwidth -- i.e. |phi1(f)-phi2(f)| reaches at least
    phase_diversity_target_rad somewhere in-band, the electronic analog
    of choosing |D1|,|D2| large enough (and different enough) for GS to
    converge. Returns (R1,C1,R2,C2) for a fixed, practical C and two
    different R values spanning a decade."""
    if f_signal_bandwidth_hz <= 0:
        raise ValueError("f_signal_bandwidth_hz must be positive")
    if phase_diversity_target_rad <= 0:
        raise ValueError("phase_diversity_target_rad must be positive")
    C = 10e-9   # 10 nF, a practical, commonly-stocked capacitor value
    # choose R1 so the pole sits near the top of the signal band (fast phase roll-off there)
    R1 = 1.0 / (2 * np.pi * f_signal_bandwidth_hz * C)
    R2 = R1 / 10.0   # a decade lower -- a genuinely different phase curve, not a small perturbation
    f_test = np.linspace(f_signal_bandwidth_hz * 0.1, f_signal_bandwidth_hz, 200)
    phi1 = allpass_phase(f_test, R1, C)
    phi2 = allpass_phase(f_test, R2, C)
    achieved_diversity = np.max(np.abs(phi1 - phi2))
    return {
        "R1": R1, "C1": C, "R2": R2, "C2": C,
        "achieved_phase_diversity_rad": achieved_diversity,
        "meets_target": achieved_diversity >= phase_diversity_target_rad,
    }


def make_electronic_measurements(Vin, dt, R1, C1, R2, C2):
    """The electronic analog of dgs.gs_core.make_measurements: filter the
    SAME input signal through two different all-pass stages, giving
    V1(t), V2(t) -- record these on an oscilloscope's two channels in the
    real circuit; here, simulated directly from a chosen Vin(t)."""
    V1 = apply_allpass_filter(Vin, dt, R1, C1)
    V2 = apply_allpass_filter(Vin, dt, R2, C2)
    return np.real(V1), np.real(V2)


if __name__ == "__main__":
    print("=== SPICE-style check: derive H(f) from the op-amp's own node equations ===")
    H_derived, matches = derive_transfer_function_from_circuit_equations()
    print(f"H(f) derived from KCL + virtual-short: {H_derived}")
    print(f"matches the assumed all-pass formula exactly: {matches}")

    print("\n=== Circuit design: two all-pass stages for a 1 kHz test signal ===")
    design = design_two_arm_circuit(f_signal_bandwidth_hz=1000.0, phase_diversity_target_rad=1.0)
    print(f"Stage 1: R1={design['R1']:.0f} Ohm, C1={design['C1']*1e9:.1f} nF")
    print(f"Stage 2: R2={design['R2']:.0f} Ohm, C2={design['C2']*1e9:.1f} nF")
    print(f"Achieved phase diversity across the band: {design['achieved_phase_diversity_rad']:.3f} rad")
    print(f"Meets target diversity (>=1.0 rad): {design['meets_target']}")

    print("\n=== Simulated V1(t), V2(t) for a test signal with random phase modulation ===")
    rng = np.random.default_rng(0)
    n, fs = 2000, 200e3   # 2000 samples at 200 kHz sample rate
    t = np.arange(n) / fs
    dt = 1.0 / fs
    phi_true = np.cumsum(rng.normal(0, 0.05, n))   # slowly wandering phase, the "unknown" signal
    Vin = np.cos(2 * np.pi * 1000 * t + phi_true)   # 1 kHz carrier, phase-modulated

    V1, V2 = make_electronic_measurements(Vin, dt, design["R1"], design["C1"],
                                           design["R2"], design["C2"])
    print(f"Vin range: [{Vin.min():.3f}, {Vin.max():.3f}]")
    print(f"V1  range: [{V1.min():.3f}, {V1.max():.3f}]")
    print(f"V2  range: [{V2.min():.3f}, {V2.max():.3f}]")
    # NOTE: |H(f)|=1 preserves each FREQUENCY component's magnitude (and
    # hence total energy, Parseval's theorem) but NOT the time-domain peak
    # value -- different phase shifts per frequency reshape how components
    # recombine in time, exactly like optical dispersion reshapes a pulse
    # even though it doesn't touch the spectral amplitude at all
    energy_in, energy_v1 = np.sum(Vin**2), np.sum(V1**2)
    print(f"\ntotal energy: Vin={energy_in:.4f}, V1={energy_v1:.4f} "
          f"(preserved to {abs(energy_in-energy_v1)/energy_in:.1e} relative -- Parseval)")
    print(f"peak amplitude: Vin={np.max(np.abs(Vin)):.3f}, V1={np.max(np.abs(V1)):.3f} "
          f"(NOT preserved -- phase-only reshaping, exactly like optical dispersion)")
    print("\nThese two arrays are exactly what an oscilloscope's two channels would record --")
    print("feed them into dgs.gs_core.retrieve_phase-style recovery as I1,I2 (after appropriate")
    print("intensity-detection modeling) for a real, bench-testable two-arm phase experiment.")
