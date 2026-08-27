"""Actually DRAW the Feynman diagrams that dgs.particle_physics.feynman_diagrams()
only describes in prose (that function returns cross-section numbers, not a
picture -- vertices, propagators, and external legs as text). This module
is the missing picture: standard-notation diagrams for the four processes
already named there (Compton scattering, e+e- -> mu+mu- annihilation, pair
production, and beta decay), built from three drawing primitives:

  fermion line:  straight, solid, with an arrowhead (direction = particle
                 flow, i.e. along the arrow for particles, against it for
                 the antiparticle convention)
  photon line:   a sine-wave squiggle, perpendicular displacement from the
                 straight path, an EVEN number of half-periods so it lands
                 exactly on its endpoints (checked below, not assumed)
  weak line:     dashed, for the internal W boson in beta decay

Each diagram is data (vertices + typed lines), not hardcoded drawing code
-- draw_diagram() is the one generic renderer, and external_legs() lets the
physics content (which particles go in, which come out) be checked
independently of how it's drawn, exactly the incoming/outgoing particle
content dgs.particle_physics.feynman_diagrams()'s docstring lists for each
process.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

# Diagram labels use real Greek/combining-overline characters (gamma, mu,
# nu-bar) -- reconfigure stdout to UTF-8 so `print`ing them doesn't raise
# UnicodeEncodeError on a legacy Windows cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass


def _check_positive_int(value, name):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def wavy_line_points(p1, p2, n_periods=3, amplitude=0.09, n_points=200):
    """Photon-propagator squiggle from p1 to p2: a sine wave in the
    direction perpendicular to the straight path. n_periods FULL periods
    guarantees sin(0) = sin(2*pi*n_periods) = 0, so the squiggle's
    endpoints land exactly on p1 and p2 -- verified in
    tests/test_feynman_diagrams.py rather than just assumed from the
    formula."""
    _check_positive_int(n_periods, "n_periods")
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    d = p2 - p1
    length = np.linalg.norm(d)
    if length == 0:
        raise ValueError("p1 and p2 must be distinct points")
    along = d / length
    perp = np.array([-along[1], along[0]])
    t = np.linspace(0.0, 1.0, n_points)
    displacement = amplitude * np.sin(2 * np.pi * n_periods * t)
    pts = p1[None, :] + t[:, None] * d[None, :] + displacement[:, None] * perp[None, :]
    return pts[:, 0], pts[:, 1]


def fermion_arrow_midpoint(p1, p2):
    """Midpoint and unit direction vector of a straight fermion line, used
    to place the arrowhead that marks particle-flow direction."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    mid = (p1 + p2) / 2
    d = p2 - p1
    length = np.linalg.norm(d)
    if length == 0:
        raise ValueError("p1 and p2 must be distinct points")
    return mid, d / length


# ── generic diagram data model ──────────────────────────────────────────
# A diagram is: {"vertices": {name: (x, y)}, "lines": [line, ...]}
# A line is: {"from": name, "to": name, "kind": "fermion"|"photon"|"weak",
#             "label": str, "external": bool, "incoming": bool or None}
# "incoming" only matters for external lines: True = particle flows INTO
# the diagram (drawn from outside toward its vertex), False = flows OUT.


def vertex_line_counts(diagram):
    """Per-vertex tally of (fermion-line count, boson-line count). A real
    QED/weak vertex always has exactly 2 fermion lines meeting 1 boson
    line (photon or W); this only counts INTERNAL vertices (interaction
    points), matching _internal_vertex_names."""
    counts = {}
    for line in diagram["lines"]:
        for end in (line["from"], line["to"]):
            f, b = counts.get(end, (0, 0))
            if line["kind"] == "fermion":
                counts[end] = (f + 1, b)
            else:  # photon or weak
                counts[end] = (f, b + 1)
    internal = set(_internal_vertex_names(diagram))
    return {name: c for name, c in counts.items() if name in internal}


def verify_vertex_valence(diagram):
    """Every internal vertex must have exactly 2 fermion lines and 1 boson
    line -- the actual QED/weak-interaction vertex rule, checked here
    rather than trusted from how the diagram happened to be drawn. Returns
    True only if EVERY internal vertex satisfies this."""
    counts = vertex_line_counts(diagram)
    if not counts:
        return False
    return all(f == 2 and b == 1 for f, b in counts.values())


def external_legs(diagram):
    """The physical content check: (label, incoming) for every external
    line, independent of geometry -- this is what should be compared
    against a process's known incoming/outgoing particles, e.g. Compton
    scattering's {(e-, in), (gamma, in), (e-, out), (gamma, out)}."""
    return [
        (line["label"], line["incoming"])
        for line in diagram["lines"]
        if line["external"]
    ]


def draw_diagram(ax, diagram, title=""):
    """The one generic renderer: walks diagram['lines'], draws each by
    kind (fermion = arrowed solid line, photon = wavy_line_points, weak =
    dashed line with an arrow), places a filled dot at every INTERNAL
    vertex (a real interaction point), and labels every external leg."""
    vertices = diagram["vertices"]

    for line in diagram["lines"]:
        p1, p2 = vertices[line["from"]], vertices[line["to"]]
        kind = line["kind"]

        if kind == "photon":
            xs, ys = wavy_line_points(p1, p2)
            ax.plot(xs, ys, color="#c0392b", lw=1.8)
        elif kind == "weak":
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#8e44ad", lw=1.8, ls="--")
        else:  # fermion
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1a1a2e", lw=1.6)

        mid, direction = fermion_arrow_midpoint(p1, p2)
        if kind == "fermion" or (kind == "weak" and not line["external"]):
            # Arrow = fermion-NUMBER flow, not spatial motion. Every line's
            # (from, to) is already ordered along the true spatial motion
            # (external->vertex for incoming, vertex->external for
            # outgoing) -- so "direction" IS the right arrow for ordinary
            # particles. Antiparticles (e+, mu+, the antineutrino) carry
            # the opposite fermion number, so their arrow is drawn
            # REVERSED relative to their actual motion -- the standard
            # "positron = electron moving backward" convention -- which is
            # why this depends on "antiparticle", not "incoming".
            arrow_dir = direction if not line.get("antiparticle", False) else -direction
            ax.annotate("", xy=mid + 0.06 * arrow_dir, xytext=mid - 0.06 * arrow_dir,
                        arrowprops=dict(arrowstyle="-|>", color="#1a1a2e", lw=1.2))

        if line["external"]:
            end = p1 if line["from"] not in _internal_vertex_names(diagram) else p2
            ax.text(end[0], end[1] + (0.08 if end[1] >= 0 else -0.14), line["label"],
                    ha="center", fontsize=11)

    for name in _internal_vertex_names(diagram):
        vx, vy = vertices[name]
        ax.plot(vx, vy, "o", color="black", markersize=5, zorder=5)

    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.75, 0.75)
    ax.axis("off")
    ax.set_aspect("equal")


def _internal_vertex_names(diagram):
    """A vertex is INTERNAL if it appears as an endpoint of more than one
    line, or of an internal (non-external) line -- i.e. it's an actual
    interaction point, not just where an external leg touches the frame."""
    counts = {}
    for line in diagram["lines"]:
        counts[line["from"]] = counts.get(line["from"], 0) + 1
        counts[line["to"]] = counts.get(line["to"], 0) + 1
    internal_line_endpoints = set()
    for line in diagram["lines"]:
        if not line["external"]:
            internal_line_endpoints.add(line["from"])
            internal_line_endpoints.add(line["to"])
    return sorted(set(name for name, c in counts.items() if c > 1) | internal_line_endpoints)


# ── the four named processes ────────────────────────────────────────────

def compton_scattering_diagram():
    """gamma + e- -> gamma + e- (s-channel, tree level). Incoming electron
    (bottom-left) and incoming photon (top-left) meet at vertex A; an
    internal (virtual, off-shell) electron propagator runs to vertex B,
    where the outgoing photon (top-right) and outgoing electron
    (bottom-right) emerge."""
    vertices = {
        "e_in": (0.0, -0.4), "gamma_in": (0.0, 0.4),
        "A": (0.4, 0.0), "B": (0.7, 0.0),
        "e_out": (1.0, -0.4), "gamma_out": (1.0, 0.4),
    }
    lines = [
        {"from": "e_in", "to": "A", "kind": "fermion", "label": "e-", "external": True, "incoming": True},
        {"from": "gamma_in", "to": "A", "kind": "photon", "label": "γ", "external": True, "incoming": True},
        {"from": "A", "to": "B", "kind": "fermion", "label": "e- (virtual)", "external": False, "incoming": True},
        {"from": "B", "to": "e_out", "kind": "fermion", "label": "e-", "external": True, "incoming": False},
        {"from": "B", "to": "gamma_out", "kind": "photon", "label": "γ", "external": True, "incoming": False},
    ]
    return {"vertices": vertices, "lines": lines}


def ee_annihilation_diagram():
    """e+ + e- -> mu+ + mu- (s-channel, tree level). Incoming e- and e+
    annihilate at vertex A into an internal (virtual) photon, which
    produces the outgoing mu-/mu+ pair at vertex B -- a different final
    state than it started with, connected only by the shared photon
    propagator."""
    vertices = {
        "e_minus_in": (0.0, -0.4), "e_plus_in": (0.0, 0.4),
        "A": (0.4, 0.0), "B": (0.7, 0.0),
        "mu_minus_out": (1.0, -0.4), "mu_plus_out": (1.0, 0.4),
    }
    lines = [
        {"from": "e_minus_in", "to": "A", "kind": "fermion", "label": "e-", "external": True, "incoming": True},
        {"from": "e_plus_in", "to": "A", "kind": "fermion", "label": "e+", "external": True, "incoming": True, "antiparticle": True},
        {"from": "A", "to": "B", "kind": "photon", "label": "γ (virtual)", "external": False, "incoming": True},
        {"from": "B", "to": "mu_minus_out", "kind": "fermion", "label": "μ-", "external": True, "incoming": False},
        {"from": "B", "to": "mu_plus_out", "kind": "fermion", "label": "μ+", "external": True, "incoming": False, "antiparticle": True},
    ]
    return {"vertices": vertices, "lines": lines}


def pair_production_diagram():
    """gamma + gamma -> e+ + e-, the CROSSING-SYMMETRIC partner of Compton
    scattering. Each vertex still gets exactly the QED-required 2 fermion
    lines + 1 photon line (a real constraint, not just a drawing choice --
    an earlier version of this function put both photons on one vertex and
    both outgoing fermions on the other, which is not a valid QED vertex
    pairing): photon_in_1 and the outgoing e+ share vertex A, photon_in_2
    and the outgoing e- share vertex B, and a single internal fermion
    propagator connects A to B. Tracing the arrows (matter forward,
    antiparticle reversed) end to end shows the whole diagram is ONE
    continuous fermion-number-flow line running from the e+ leg, through
    A, through the internal propagator, through B, out to the e- leg --
    exactly the standard picture, not five independently-arrowed lines
    that happen to agree."""
    vertices = {
        "gamma_in_1": (0.0, 0.5), "gamma_in_2": (0.0, -0.5),
        "A": (0.4, 0.25), "B": (0.4, -0.25),
        "e_plus_out": (0.85, 0.5), "e_minus_out": (0.85, -0.5),
    }
    lines = [
        {"from": "gamma_in_1", "to": "A", "kind": "photon", "label": "γ", "external": True, "incoming": True},
        {"from": "gamma_in_2", "to": "B", "kind": "photon", "label": "γ", "external": True, "incoming": True},
        {"from": "A", "to": "B", "kind": "fermion", "label": "e- (virtual)", "external": False, "incoming": True},
        {"from": "A", "to": "e_plus_out", "kind": "fermion", "label": "e+", "external": True, "incoming": False, "antiparticle": True},
        {"from": "B", "to": "e_minus_out", "kind": "fermion", "label": "e-", "external": True, "incoming": False},
    ]
    return {"vertices": vertices, "lines": lines}


def beta_decay_diagram():
    """Quark-level beta-MINUS decay d -> u + W-, W- -> e- + nu_e-bar (the
    actual interacting line inside n -> p + e- + nu_e-bar; the two
    spectator quarks that don't participate are not drawn). Beta-minus
    decay produces an electron ANTIneutrino, not a neutrino -- lepton
    number is conserved (the outgoing e- has lepton number +1, so the
    outgoing neutral lepton must have lepton number -1 to balance the
    incoming d/outgoing u, which carry none), hence its arrow is drawn
    reversed like the other antiparticle legs above. The W- propagator is
    the internal WEAK line connecting the quark vertex to the leptonic
    vertex -- a genuinely different force carrier from the photon lines
    in the other three diagrams."""
    vertices = {
        "d_in": (0.0, -0.4), "A": (0.35, -0.4), "u_out": (0.7, -0.4),
        "B": (0.35, 0.15),
        "e_out": (0.7, 0.5), "nu_out": (0.7, 0.05),
    }
    lines = [
        {"from": "d_in", "to": "A", "kind": "fermion", "label": "d", "external": True, "incoming": True},
        {"from": "A", "to": "u_out", "kind": "fermion", "label": "u", "external": True, "incoming": False},
        {"from": "A", "to": "B", "kind": "weak", "label": "W- (virtual)", "external": False, "incoming": True},
        {"from": "B", "to": "e_out", "kind": "fermion", "label": "e-", "external": True, "incoming": False},
        {"from": "B", "to": "nu_out", "kind": "fermion", "label": "ν̄_e", "external": True, "incoming": False, "antiparticle": True},
    ]
    return {"vertices": vertices, "lines": lines}


ALL_DIAGRAMS = {
    "Compton scattering: γ e- -> γ e-": compton_scattering_diagram,
    "e+e- -> μ+μ- annihilation": ee_annihilation_diagram,
    "Pair production: γγ -> e+e-": pair_production_diagram,
    "Beta decay (quark level): d -> u e- ν̄_e": beta_decay_diagram,
}


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (title, builder) in zip(axes.flat, ALL_DIAGRAMS.items()):
        diagram = builder()
        draw_diagram(ax, diagram, title=title)
        print(f"{title}: external legs = {external_legs(diagram)}")
    fig.suptitle("Feynman diagrams for dgs.particle_physics.feynman_diagrams()'s named processes",
                 fontsize=13)
    fig.tight_layout()
    plt.show()
