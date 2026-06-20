"""Turn a free-text brainstorm into a torch concept-vector for THIS repo.

A stream-of-consciousness "manifesto" is mostly noise with a few real technical
kernels buried in it. This module embeds text into a fixed concept space that
spans the dispersion-assisted Gerchberg-Saxton project -- dispersion, phase
retrieval, complex/phasor, Fourier, Bessel, electrodynamics, torch, sympy,
QPSK, Hilbert spaces, photonics, hardware -- and scores how on-topic any text
(or any file in the repo) is.

Two uses:
  * vectorize a brainstorm and rank its fragments -> pull the signal out,
  * score the repo's loose files against the project theme -> informed,
    non-destructive cleanup (what actually belongs here vs strays).

torch for the vectors (py 3.12 here). Civilian optical metrology / education.
"""

import re

import numpy as np

# ── the concept space: each axis is a project theme + its trigger words ──
CONCEPTS = {
    "dispersion":      ["dispersion", "dispersive", "gvd", "group velocity", "chirp", "disperse", "absorption"],
    "phase_retrieval": ["phase retrieval", "phase recovery", "gerchberg", "saxton", "retrieve", "recover phase", "hio", "tdgs", "td-gs"],
    "complex_phasor":  ["complex", "phasor", "exponential", "argand", "imaginary", "euler", "complex number"],
    "fourier":         ["fourier", "fft", "dft", "spectrum", "spectral", "frequency", "stft", "wigner"],
    "bessel":          ["bessel"],
    "electrodynamics": ["radiation", "electromagnet", "maxwell", "radar", "antenna", "poynting", "electric field", "magnetic field"],
    "griffiths":       ["griffiths"],
    "sympy_calculus":  ["sympy", "symbolic", "derivative", "integral", "calculus", " ode", " pde", "differenti", "integrat"],
    "torch_ml":        ["torch", "tensor", "gradient", "neural", "training", "embedding", "autograd", "random forest", "decision tree"],
    "qpsk_comm":       ["qpsk", "modulation", "symbol", "carrier", "quadrature", " iq "],
    "hilbert":         ["hilbert", "orthogonal", "inner product", "basis function"],
    "hardware_fpga":   ["verilog", "vhdl", "fpga", "gate array", "testbench", "logic gate", "digital logic", "7 basic gates"],
    "cuda":            ["cuda", "gpu"],
    "photonics":       ["photonic", "laser", "optic", "photomask", "lithograph", "beam", "oscillator", "resonance", "waveguide", "fiber", "fibre"],
    "circuits_eda":    ["spice", "op amp", "opamp", "circuit", " eda ", "integrated circuit", "breadboard", "microcontroller", "ohm", "voltage", "current"],
    "quantum_nuclear": ["quantum", "qubit", "nuclear", "particle physics", "modern physics"],
    "linear_algebra":  ["linear algebra", "matrix", "eigen", "cross product", "vector space"],
    "statics_mech":    ["statics", "equilibrium", "torsion", "trajector", "robotics"],
    "chemistry":       ["chemistry", "chemical", "stoichiometry", "organic chem"],
}
ORDER = list(CONCEPTS)

# how central each theme is to *this* repo (the reference "north star" vector)
THEME_WEIGHTS = {
    "dispersion": 1.0, "phase_retrieval": 1.0, "complex_phasor": 0.9, "fourier": 0.9,
    "bessel": 0.7, "electrodynamics": 0.8, "griffiths": 0.8, "sympy_calculus": 0.7,
    "torch_ml": 0.9, "qpsk_comm": 0.7, "hilbert": 0.6, "hardware_fpga": 0.6,
    "cuda": 0.6, "photonics": 0.9, "circuits_eda": 0.5, "quantum_nuclear": 0.5,
    "linear_algebra": 0.6, "statics_mech": 0.3, "chemistry": 0.2,
}


def concept_counts(text):
    """Count keyword hits per concept (case-insensitive substring match)."""
    s = " " + text.lower() + " "
    return {c: sum(s.count(kw) for kw in kws) for c, kws in CONCEPTS.items()}


def text_to_vector(text, normalize=True):
    """Embed text as a torch vector over the concept space (len == len(ORDER))."""
    import torch
    counts = concept_counts(text)
    v = torch.tensor([float(counts[c]) for c in ORDER])
    if normalize and v.norm() > 0:
        v = v / v.norm()
    return v


def theme_vector():
    """The repo's reference concept vector (unit-normalized THEME_WEIGHTS)."""
    import torch
    v = torch.tensor([THEME_WEIGHTS[c] for c in ORDER])
    return v / v.norm()


def relevance(text):
    """Cosine similarity of text to the repo theme, in [0, 1]."""
    import torch
    v = text_to_vector(text)
    if v.norm() == 0:
        return 0.0
    return float(torch.dot(v, theme_vector()))


def matched_concepts(text):
    """Concepts that actually fired, most-hit first: the extracted signal."""
    counts = concept_counts(text)
    return sorted([(c, n) for c, n in counts.items() if n > 0],
                  key=lambda kv: -kv[1])


def rank_fragments(text, window=7, stride=3, top=12):
    """Slice text into word-windows, keep the most on-topic ones (signal mining).

    Returns a list of (score, matched_concepts, fragment_text), de-duplicated by
    the set of concepts hit so each surfaced kernel is distinct.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9+\-/]*", text)
    seen, out = set(), []
    for i in range(0, max(1, len(words) - window + 1), stride):
        frag = " ".join(words[i:i + window])
        hits = matched_concepts(frag)
        if not hits:
            continue
        key = frozenset(c for c, _ in hits)
        if key in seen:
            continue
        seen.add(key)
        out.append((relevance(frag), [c for c, _ in hits], frag))
    out.sort(key=lambda r: -r[0])
    return out[:top]


def score_file(path, head_lines=80, nb_chars=6000):
    """Theme-relevance of a repo file (name + content), for cleanup triage."""
    import json
    import pathlib
    p = pathlib.Path(path)
    text = p.name.replace("_", " ").replace("-", " ")
    try:
        if p.suffix == ".ipynb":                      # read the cell sources, not raw JSON
            nb = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            src = []
            for cell in nb.get("cells", []):
                s = cell.get("source", "")
                src.append("".join(s) if isinstance(s, list) else s)
            text += " " + " ".join(src)[:nb_chars]
        elif p.suffix in (".py", ".md", ".txt", ".v", ".vhd", ".json"):
            with open(p, encoding="utf-8", errors="ignore") as f:
                text += " " + "".join(f.readline() for _ in range(head_lines))
    except (OSError, ValueError):
        pass
    return relevance(text), [c for c, _ in matched_concepts(text)]


if __name__ == "__main__":
    import sys
    raw = open(sys.argv[1], encoding="utf-8", errors="ignore").read() if len(sys.argv) > 1 \
        else "dispersion gerchberg saxton phase retrieval in torch with bessel functions and sympy"
    print(f"overall theme relevance: {relevance(raw):.3f}")
    print("concepts present:", [c for c, _ in matched_concepts(raw)])
    print("\ntop on-topic fragments:")
    for score, cons, frag in rank_fragments(raw):
        print(f"  {score:.2f}  {','.join(cons):<28} | {frag}")
