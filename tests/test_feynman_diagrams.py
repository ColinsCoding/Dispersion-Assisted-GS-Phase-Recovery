"""Test dgs/feynman_diagrams.py: the wavy-line squiggle actually lands on
its endpoints, every named diagram satisfies the real QED/weak vertex rule
(2 fermion lines + 1 boson line per internal vertex -- this caught a real
bug: an earlier pair_production_diagram put both photons on one vertex and
both outgoing fermions on the other, which verify_vertex_valence correctly
rejects), external legs match each process's known physics, and fermion
arrows follow the fermion-number-flow convention (reversed for
antiparticles) rather than just spatial motion."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.feynman_diagrams import (
    wavy_line_points, fermion_arrow_midpoint, vertex_line_counts,
    verify_vertex_valence, external_legs, compton_scattering_diagram,
    ee_annihilation_diagram, pair_production_diagram, beta_decay_diagram,
    ALL_DIAGRAMS,
)

# 1. wavy_line_points: an integer number of periods lands exactly on both endpoints
for n_periods in (1, 2, 5):
    xs, ys = wavy_line_points((0.0, 0.0), (1.0, 0.3), n_periods=n_periods)
    assert abs(xs[0] - 0.0) < 1e-9 and abs(ys[0] - 0.0) < 1e-9
    assert abs(xs[-1] - 1.0) < 1e-9 and abs(ys[-1] - 0.3) < 1e-9

try:
    wavy_line_points((0, 0), (1, 1), n_periods=0)
    assert False, "should have raised ValueError"
except ValueError:
    pass

try:
    wavy_line_points((0, 0), (0, 0))
    assert False, "should have raised ValueError for coincident points"
except ValueError:
    pass

# 2. fermion_arrow_midpoint: midpoint and unit direction are correct
mid, direction = fermion_arrow_midpoint((0.0, 0.0), (2.0, 0.0))
assert abs(mid[0] - 1.0) < 1e-12 and abs(mid[1]) < 1e-12
assert abs(direction[0] - 1.0) < 1e-12 and abs(direction[1]) < 1e-12
assert abs(np.linalg.norm(direction) - 1.0) < 1e-12

# 3. Every named diagram satisfies the real QED/weak vertex rule:
#    exactly 2 fermion lines + 1 boson line at every internal vertex
for title, builder in ALL_DIAGRAMS.items():
    diagram = builder()
    assert verify_vertex_valence(diagram), f"{title} failed vertex valence: {vertex_line_counts(diagram)}"

# 4. A deliberately broken topology (the actual bug this module caught:
#    both photons on one vertex, both outgoing fermions on the other) must
#    be REJECTED by verify_vertex_valence, proving the check has teeth
broken = {
    "vertices": {"g1": (0, 1), "g2": (0, -1), "A": (0.4, 0), "B": (0.7, 0),
                 "e1": (1, 1), "e2": (1, -1)},
    "lines": [
        {"from": "g1", "to": "A", "kind": "photon", "label": "γ", "external": True, "incoming": True},
        {"from": "g2", "to": "A", "kind": "photon", "label": "γ", "external": True, "incoming": True},
        {"from": "A", "to": "B", "kind": "fermion", "label": "e-", "external": False, "incoming": True},
        {"from": "B", "to": "e1", "kind": "fermion", "label": "e-", "external": True, "incoming": False},
        {"from": "B", "to": "e2", "kind": "fermion", "label": "e+", "external": True, "incoming": False, "antiparticle": True},
    ],
}
assert not verify_vertex_valence(broken)

# 5. External legs match each process's known incoming/outgoing content
compton_legs = set(external_legs(compton_scattering_diagram()))
assert compton_legs == {("e-", True), ("γ", True), ("e-", False), ("γ", False)}

ann_legs = set(external_legs(ee_annihilation_diagram()))
assert ann_legs == {("e-", True), ("e+", True), ("μ-", False), ("μ+", False)}

# (pair production has two identical incoming photon labels, so this is
# compared as a sorted multiset, not a set -- a set would silently collapse
# the duplicate ("γ", True) entry and hide a missing-photon bug)
pair_legs = sorted(external_legs(pair_production_diagram()))
assert pair_legs == sorted([("γ", True), ("γ", True), ("e+", False), ("e-", False)])

beta_legs = set(external_legs(beta_decay_diagram()))
assert beta_legs == {("d", True), ("u", False), ("e-", False), ("ν̄_e", False)}

# 6. Fermion-number-flow arrow direction: ordinary particles get the arrow
#    along their actual spatial motion; antiparticles get it reversed
ann = ee_annihilation_diagram()
v = ann["vertices"]
by_label = {line["label"]: line for line in ann["lines"]}

e_minus_line = by_label["e-"]
_, e_minus_dir = fermion_arrow_midpoint(v[e_minus_line["from"]], v[e_minus_line["to"]])
assert not e_minus_line.get("antiparticle", False)   # matter: arrow = spatial direction (into vertex)
assert e_minus_dir[1] > 0   # e- enters from below, moving up toward the vertex

e_plus_line = by_label["e+"]
assert e_plus_line.get("antiparticle", False)   # antimatter: arrow gets reversed

print("all dgs.feynman_diagrams tests passed")
