"""
ousd_alignment.py — OUSD(R&E) Critical Technology Area tagging
Maps repo components → DoD FutureG / Integrated Sensing / Trusted AI CTAs.

Reference: OUSD(R&E) Critical Technology Areas (2023)
  https://www.cto.mil/usdre-strat-vision-critical-tech-areas/

Usage
-----
    from dgs.ousd_alignment import stamp, print_alignment

    stats = {"exit_code": 0, ...}
    stats = stamp(stats, components=["td_gs", "fno", "tsdft"])
    print_alignment()
"""

from __future__ import annotations
import json
import sys
from typing import Sequence

# The alignment table uses box-drawing/star glyphs; force UTF-8 so it prints on
# a legacy Windows cp1252 console instead of raising UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

# ── CTA registry ──────────────────────────────────────────────────────────────
# Priority 1 = the CTAs this project actively targets with real, tested
# modules: FutureG, Trusted AI & Autonomy, Advanced Computing & Software,
# Integrated Sensing & Cyber, Directed Energy, Human-Machine Interfaces,
# Biotechnology (added once the repo's biotech-adjacent physics work --
# STEAM cancer-cell detection, EPR dosimetry, magnetic hyperthermia, lab-on-
# chip biosensing -- grew past "adjacent" into a real focus area), and
# Quantum Science (bumped from priority 2 once logic_timing.py's digital-logic
# control-timing work and error_propagation.py's Bayesian/MC inference work
# gave the ring-resonator/cavity-QED formalism a matching classical-control
# and readout-statistics story). Priority 2 = adjacent areas the repo touches
# without dedicated modules yet.
CTA = {
    "FutureG": {
        "priority": 1,
        "description": "Next-generation communications and sensing at optical bandwidth",
        "repo_components": ["tsdft", "gs_core", "gs_fno", "rogue_wave", "adc_timing", "griffiths_em"],
    },
    "Trusted_AI_and_Autonomy": {
        "priority": 1,
        "description": "Verified ML pipelines with physics-grounded loss functions",
        "repo_components": ["gs_fno", "fno1d", "gs_torch", "classifier", "gs_verify"],
    },
    "Advanced_Computing_and_Software": {
        "priority": 1,
        "description": "GPU-accelerated phase retrieval; SymPy analytic validation "
                       "(Maxwell -> dispersion operator H(f)=exp(i pi D f^2))",
        "repo_components": ["gs_core", "gs_fno", "gs_torch", "repl", "sympy_physics", "griffiths_em"],
    },
    "Integrated_Sensing_and_Cyber": {
        "priority": 1,
        "description": "Single-shot dispersive Fourier transform spectroscopy + secure telemetry",
        "repo_components": ["tsdft", "td_gs", "gs_monitor", "gs_backtest", "rogue_wave"],
    },
    "Directed_Energy": {
        "priority": 1,
        "description": "High-rep-rate pulsed laser characterisation; wavefront sensing",
        "repo_components": ["tsdft", "gs_core", "pic_design", "gs_surface"],
    },
    "Human_Machine_Interfaces": {
        "priority": 1,
        "description": "Real-time optical dashboard; 3-D phase visualisation; MuJoCo scanner; "
                       "quantified human-eye-vs-instrument optical performance (NA, Rayleigh "
                       "resolution, collected flux, dynamic range, temporal resolution) grounding "
                       "when a sensing task needs the instrument vs. when human vision already "
                       "suffices",
        "repo_components": ["optical_dashboard", "gs_animate", "gs_surface", "mujoco_scanner",
                            "human_vs_instrument_optics"],
    },
    "Quantum_Science": {
        "priority": 1,
        "description": "Quantum information science (not just QM formalism), theory AND the "
                       "hands-on hardware it maps to: ring resonators (optical_loops) are the "
                       "standard hardware primitive for photonic qubit generation and routing "
                       "(spontaneous four-wave mixing photon-pair sources, squeezed-light "
                       "generation) -- the bench realization is a fiber-loop or bus-plus-ring "
                       "test setup built from the same class of RF/microwave components (mixers, "
                       "directional couplers, delay lines) as the PI's EC ENGR 279AS coursework "
                       "already cited in the Phase I feasibility doc, not an unrelated skill; the "
                       "transverse/longitudinal field split (helmholtz_decomposition) is the "
                       "Coulomb-gauge decomposition underlying canonical quantization of the EM "
                       "field (only the transverse part is quantized as photons); response-"
                       "function pole structure (contour_integration_residues) is the same math "
                       "as input-output theory in cavity/circuit QED, where pole locations set "
                       "qubit/cavity decay rates -- measurable on a bench via the same S-parameter/"
                       "network-analyzer technique RF hardware characterization already uses. "
                       "Bumped to priority 1: qubit readout is a binary-outcome statistical "
                       "inference problem under noise, the same Bayesian/Monte-Carlo machinery as "
                       "error_propagation.py's Measurement class (Jacobian + MC uncertainty "
                       "propagation); the hazard-free, timing-critical control electronics that "
                       "gate/error-correction feedback loops require is the same digital-logic "
                       "discipline demonstrated in logic_timing.py (Circuit DAG, critical-path "
                       "analysis, ripple-adder hazard elimination). These are real hardware/"
                       "formalism connections, not a QM-flavored relabeling of unrelated code.",
        "repo_components": ["optical_loops", "helmholtz_decomposition", "contour_integration_residues",
                            "logic_timing", "error_propagation"],
    },
    "Microelectronics": {
        "priority": 2,
        "description": "SiC ADC front-end timing; FPGA logic synthesis for GS loop. Also the "
                       "fabrication-adjacent bridge to Quantum_Science: the microstrip/planar "
                       "transmission-line design toolkit in thz_circuits (trace geometry, "
                       "substrate height, characteristic impedance, the lambda/10 lumped-vs-"
                       "distributed boundary) is the SAME RF/microwave engineering used to wire "
                       "and control real quantum information science hardware -- superconducting "
                       "qubit drive/readout lines and photonic quantum chip RF modulators are "
                       "impedance-matched planar transmission lines, not a separate discipline "
                       "from THz circuit design. This is the same RF/microwave hardware class as "
                       "the PI's EC ENGR 279AS coursework already cited under Quantum_Science and "
                       "in the Phase I feasibility doc -- not a second, unrelated claim of skill.",
        "repo_components": ["adc_timing", "digital_logic", "firmware", "thz_circuits"],
    },
    "Biotechnology": {
        "priority": 1,
        "description": "Lab-on-chip microfluidic scanner; single-cell optical barcoding; "
                       "STEAM-based circulating-tumor-cell detection in blood; ESR/EPR "
                       "tooth-enamel radiation dosimetry; magnetic-nanoparticle hyperthermia "
                       "cancer treatment (Hund's-rules-derived Fe2+/Fe3+ magnetic moments); "
                       "CDI/Gerchberg-Saxton phase retrieval (structural biology lineage)",
        "repo_components": ["lab_on_chip", "microfluidics", "mujoco_scanner",
                            "steam_imaging", "biosensor_lab_on_chip", "esr_dosimetry",
                            "magnetic_hyperthermia", "cdi_phase_retrieval"],
    },
    "Threat_Reduction_Adjacent": {
        "priority": 2,
        "description": "NOT one of OUSD(R&E)'s 14 Critical Technology Areas — counter-WMD "
                       "threat reduction is DTRA's mission, a separate DoD component. Listed "
                       "here only because rogue-wave/anomaly detection and high-rep-rate "
                       "ADC timing are dual-use sensing capabilities DTRA-adjacent programs "
                       "could plausibly draw on (single-shot transient detection).",
        "repo_components": ["rogue_wave", "gs_monitor", "adc_timing"],
    },
}

# ── Component → CTA reverse map ───────────────────────────────────────────────
_COMP_TO_CTA: dict[str, list[str]] = {}
for _cta, _info in CTA.items():
    for _comp in _info["repo_components"]:
        _COMP_TO_CTA.setdefault(_comp, []).append(_cta)


def components_to_ctas(components: Sequence[str]) -> list[str]:
    """Return deduplicated CTA list for given component names."""
    seen: set[str] = set()
    out: list[str] = []
    for comp in components:
        for cta in _COMP_TO_CTA.get(comp, []):
            if cta not in seen:
                seen.add(cta)
                out.append(cta)
    # sort by priority then name
    out.sort(key=lambda c: (CTA[c]["priority"], c))
    return out


def stamp(stats: dict, components: Sequence[str] | None = None) -> dict:
    """
    Attach OUSD CTA metadata to an existing stats dict.

    Parameters
    ----------
    stats       : dict — your existing JSON stats block
    components  : list of component keys (see CTA registry above)
                  defaults to full repo set if None

    Returns
    -------
    stats dict with 'ousd' key added in-place
    """
    if components is None:
        components = list(_COMP_TO_CTA.keys())

    ctas = components_to_ctas(components)
    priority_1 = [c for c in ctas if CTA[c]["priority"] == 1]

    stats["ousd"] = {
        "aligned_ctas":   ctas,
        "priority_1_ctas": priority_1,
        "n_ctas":         len(ctas),
        "sbir_phase":     "Phase I — $250K (prospective; DOD/OUSD SBIR Phase I cap)",
        "program":        "Dispersion-Assisted GS Phase Recovery",
        # Honest marking: this is a PUBLIC UCLA/Jalali-Lab academic project (the
        # repo is itself a course deliverable), not government-controlled data.
        # "FOUO" was both deprecated (-> CUI, DoDI 5200.48) and wrong here.
        "classification": "UNCLASSIFIED // DISTRIBUTION A — Approved for Public Release",
        "note": "CTA tags are technology-area relevance, not a claim of DoD funding or endorsement.",
    }
    return stats


def print_alignment(components: Sequence[str] | None = None) -> None:
    """Pretty-print the OUSD CTA alignment table."""
    if components is None:
        components = list(_COMP_TO_CTA.keys())
    ctas = components_to_ctas(components)

    W = 72
    print("═" * W)
    print("  OUSD(R&E) CRITICAL TECHNOLOGY AREA ALIGNMENT")
    print("  Dispersion-Assisted GS Phase Recovery  |  UNCLASSIFIED // DIST A")
    print("═" * W)
    print(f"  {'CTA':<38} {'PRI':<5} REPO COMPONENTS")
    print("  " + "─" * (W - 2))
    for cta in ctas:
        info   = CTA[cta]
        comps  = ", ".join(info["repo_components"][:3])
        suffix = "…" if len(info["repo_components"]) > 3 else ""
        pri    = "★★" if info["priority"] == 1 else "★ "
        label  = cta.replace("_", " ")
        print(f"  {label:<38} {pri:<5} {comps}{suffix}")
    print("═" * W)
    print(f"  Total CTAs: {len(ctas)}   Priority-1: "
          f"{sum(1 for c in ctas if CTA[c]['priority']==1)}")
    print("═" * W)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_alignment()
    sample = stamp({"exit_code": 0, "status": "PASS §0"},
                   components=["td_gs", "gs_fno", "tsdft", "optical_dashboard"])
    print()
    print(json.dumps(sample["ousd"], indent=2, ensure_ascii=False))
