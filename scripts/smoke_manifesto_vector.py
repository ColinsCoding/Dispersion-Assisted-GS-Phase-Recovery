"""Smoke-test manifesto_vector: concept embedding, theme relevance, fragment mining."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import manifesto_vector as mv

# 1. vector lives in the concept space and is unit-normalized
v = mv.text_to_vector("dispersion phase retrieval gerchberg saxton torch bessel")
assert v.shape == (len(mv.ORDER),)
assert abs(float(v.norm()) - 1.0) < 1e-5
assert mv.text_to_vector("xyzzy frobnicate").norm() == 0   # no concepts -> zero vector

# 2. on-topic text scores far higher than off-topic noise
on = mv.relevance("dispersion-assisted gerchberg saxton phase retrieval in torch, bessel, sympy, fourier")
off = mv.relevance("shoulder pads dancing dorm festival monogamy social security")
assert on > 0.5 and off < 0.15, (on, off)
assert on > off

# 3. matched_concepts pulls the right axes out
cons = dict(mv.matched_concepts("complex phasor exponential e^{i phi} dispersion fft"))
assert "complex_phasor" in cons and "dispersion" in cons and "fourier" in cons

# 4. fragment mining returns distinct, ranked, on-topic kernels from noisy text
noisy = ("dancing festival nonsense dispersion gvd chirp blah blah "
         "gerchberg saxton phase recovery random words verilog fpga testbench "
         "more noise sympy derivative integral griffiths electrodynamics maxwell")
frags = mv.rank_fragments(noisy, top=8)
assert len(frags) >= 3
assert all(f[0] > 0 for f in frags)                 # every surfaced fragment is on-topic
assert frags == sorted(frags, key=lambda r: -r[0])  # sorted by relevance
concept_sets = [frozenset(f[1]) for f in frags]
assert len(concept_sets) == len(set(concept_sets))  # de-duplicated by concept set

# 5. theme vector is unit length and weights the core themes highest
tv = mv.theme_vector()
assert abs(float(tv.norm()) - 1.0) < 1e-5
i_disp = mv.ORDER.index("dispersion"); i_chem = mv.ORDER.index("chemistry")
assert tv[i_disp] > tv[i_chem]

# 6. file scoring runs on a real repo file
score, cons = mv.score_file(pathlib.Path(__file__).resolve().parents[1] / "gs_core.py")
assert 0.0 <= score <= 1.0

print("SMOKE PASS")
