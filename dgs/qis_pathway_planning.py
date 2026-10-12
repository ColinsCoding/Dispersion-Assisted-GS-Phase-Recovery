"""Long-term planning as computer science: a course DAG and a technology roadmap.

Two plans, one data structure (a directed acyclic graph):

  * LEARNING PATHWAY -- undergraduate computing, classical to quantum
    information. Nodes are topics, edges are prerequisites, and each topic
    points at the dgs/ or griffiths/ module that already teaches it. A
    topological sort gives a valid order; longest-path layering gives the
    minimum number of terms if every topic in a layer is taken together.
  * VENTURE ROADMAP -- from simulation to a fieldable instrument, organized by
    Technology Readiness Level (TRL 1-9, the NASA/DoD scale). Nodes carry
    durations in months; the critical path is the longest path, the same
    computation as static timing analysis in dgs.logic_timing.

All durations and dollar amounts are PLANNING ASSUMPTIONS for the user to
edit, not facts about any program, school, or funding agency. Check the
current SBIR/STTR solicitation and your own catalog before relying on them.
"""
import importlib
from collections import defaultdict, deque

# topic: (prerequisites, module that teaches it in this repo)
PATHWAY = {
    "Programming & algorithms": ((), "dgs.algorithms"),
    "Calculus & numerical methods": ((), "dgs.numerical_methods"),
    "Linear algebra & eigenvectors": (("Calculus & numerical methods",), "dgs.complex_eigen_bridge"),
    "Probability & Markov chains": (("Calculus & numerical methods",), "dgs.markov"),
    "Digital logic & timing": (("Programming & algorithms",), "dgs.logic_timing"),
    "Circuits (DC/AC, op amps)": (("Calculus & numerical methods",), "dgs.opamp_inverting"),
    "Measurement & uncertainty": (("Probability & Markov chains",), "dgs.error_propagation"),
    "Signals, Fourier & Laplace": (("Linear algebra & eigenvectors", "Circuits (DC/AC, op amps)"),
                                   "dgs.fourier_tools"),
    "Electromagnetism (Griffiths)": (("Linear algebra & eigenvectors",), "griffiths.electrodynamics"),
    "Modern physics": (("Electromagnetism (Griffiths)",), "dgs.planck_quantization_taylor"),
    "Quantum mechanics": (("Modern physics", "Linear algebra & eigenvectors"), "dgs.ehrenfest_theorem"),
    "Photonics: dispersion & group delay": (("Signals, Fourier & Laplace", "Electromagnetism (Griffiths)"),
                                            "dgs.pulse_moments_group_delay"),
    "Causality & Kramers-Kronig": (("Signals, Fourier & Laplace",), "dgs.causality"),
    "Qubits & quantum circuits": (("Quantum mechanics", "Digital logic & timing"), "dgs.qubits"),
    "Polarization qubits (Jones)": (("Qubits & quantum circuits", "Electromagnetism (Griffiths)"),
                                    "dgs.polarization_qubit"),
    "Density matrices & dephasing": (("Qubits & quantum circuits", "Probability & Markov chains"),
                                     "dgs.cell_hilbert_bridge"),
    "Phase retrieval (GS)": (("Photonics: dispersion & group delay", "Programming & algorithms"),
                             "dgs.gs_core"),
    "Tomography <-> phase retrieval": (("Density matrices & dephasing", "Phase retrieval (GS)"),
                                       "dgs.dephasing_phase_retrieval"),
    "Neural operators for instruments": (("Phase retrieval (GS)",), "dgs.gs_fno"),
    "Capstone: bench I1/I2 measurement": (("Tomography <-> phase retrieval", "Measurement & uncertainty",
                                           "Causality & Kramers-Kronig"), None),
}

# milestone: (prerequisites, months, TRL reached, kind)
ROADMAP = {
    "Theory & SymPy derivations": ((), 3, 1, "tech"),
    "Synthetic validation (this repo)": (("Theory & SymPy derivations",), 4, 3, "tech"),
    "Customer-discovery interviews": ((), 4, None, "business"),
    "Bench I1/I2 measurement": (("Synthetic validation (this repo)",), 6, 4, "tech"),
    "Uncertainty budget & calibration": (("Bench I1/I2 measurement",), 2, 4, "tech"),
    "Phase I proposal": (("Customer-discovery interviews", "Synthetic validation (this repo)"), 2, None, "business"),
    "Phase I feasibility": (("Phase I proposal", "Uncertainty budget & calibration"), 6, 5, "tech"),
    "Lab prototype in relevant environment": (("Phase I feasibility",), 9, 6, "tech"),
    "Phase II prototype & pilot users": (("Lab prototype in relevant environment",), 15, 7, "tech"),
}

# the user's own earlier planning numbers (edit freely; verify against the current solicitation)
PLANNING_BUDGET = {"Phase I": 275_000, "Phase II": 1_750_000}

TRL_LADDER = {1: "basic principles observed", 2: "concept formulated", 3: "proof of concept (analysis/simulation)",
              4: "validated in lab", 5: "validated in relevant environment (components)",
              6: "prototype demonstrated in relevant environment", 7: "prototype in operational environment",
              8: "system complete and qualified", 9: "proven in operation"}


def _prereqs(graph):
    return {k: tuple(v[0]) for k, v in graph.items()}


def topological_order(graph):
    """Kahn's algorithm; raises ValueError on a cycle or a dangling prerequisite.
    Ties are broken alphabetically so the output is deterministic."""
    pre = _prereqs(graph)
    for k, ps in pre.items():
        for p in ps:
            if p not in pre:
                raise ValueError(f"{k!r} needs unknown prerequisite {p!r}")
    indeg = {k: len(ps) for k, ps in pre.items()}
    children = defaultdict(list)
    for k, ps in pre.items():
        for p in ps:
            children[p].append(k)
    ready = deque(sorted(k for k, d in indeg.items() if d == 0))
    order = []
    while ready:
        k = ready.popleft()
        order.append(k)
        for c in sorted(children[k]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(pre):
        raise ValueError("graph has a cycle")
    return order


def layers(graph):
    """Longest-path depth of each node (0 = no prerequisites). Grouping by depth
    gives the fewest terms if every topic in a layer can run in parallel."""
    pre = _prereqs(graph)
    depth = {}
    for k in topological_order(graph):
        depth[k] = 1 + max((depth[p] for p in pre[k]), default=-1)
    out = defaultdict(list)
    for k, d in depth.items():
        out[d].append(k)
    return [sorted(out[d]) for d in sorted(out)]


def critical_path(graph=ROADMAP):
    """Earliest-finish schedule. Returns (total_months, path, finish_by_node).
    Same longest-path algorithm as a critical path through combinational logic."""
    pre = _prereqs(graph)
    finish, via = {}, {}
    for k in topological_order(graph):
        start = 0
        for p in pre[k]:
            if finish[p] > start:
                start, via[k] = finish[p], p
        finish[k] = start + graph[k][1]
    end = max(finish, key=finish.get)
    path = [end]
    while path[-1] in via:
        path.append(via[path[-1]])
    return finish[end], path[::-1], finish


def slack(graph=ROADMAP):
    """Months each milestone can slip without delaying the end date."""
    total, _, finish = critical_path(graph)
    pre = _prereqs(graph)
    children = defaultdict(list)
    for k, ps in pre.items():
        for p in ps:
            children[p].append(k)
    latest = {}
    for k in reversed(topological_order(graph)):
        latest[k] = min((latest[c] - graph[c][1] for c in children[k]), default=total)
    return {k: latest[k] - finish[k] for k in graph}


def module_available(name):
    """True if the repo module that teaches a topic imports cleanly."""
    if name is None:
        return True
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def trl_reached(graph=ROADMAP):
    """Highest TRL on the roadmap, and the milestone that reaches it."""
    best = max((v[2] or 0, k) for k, v in graph.items())
    return best[0], best[1]
