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
# Priority 1 = the six CTAs marked (red arrows) on the OUSD(R&E) Critical
# Technology Areas list this project targets: FutureG, Trusted AI & Autonomy,
# Advanced Computing & Software, Integrated Sensing & Cyber, Directed Energy,
# and Human-Machine Interfaces. Priority 2 = adjacent areas the repo touches.
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
        "description": "Real-time optical dashboard; 3-D phase visualisation; MuJoCo scanner",
        "repo_components": ["optical_dashboard", "gs_animate", "gs_surface", "mujoco_scanner"],
    },
    "Quantum_Science": {
        "priority": 2,
        "description": "QM formalism underpins phase space and coherence analysis",
        "repo_components": ["sympy_physics", "griffiths_qm", "wavefunction_grammar"],
    },
    "Microelectronics": {
        "priority": 2,
        "description": "SiC ADC front-end timing; FPGA logic synthesis for GS loop",
        "repo_components": ["adc_timing", "digital_logic", "firmware"],
    },
    "Biotechnology": {
        "priority": 2,
        "description": "Lab-on-chip microfluidic scanner; single-cell optical barcoding",
        "repo_components": ["lab_on_chip", "microfluidics", "mujoco_scanner"],
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
        "sbir_phase":     "Phase I — $275K (prospective)",
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
