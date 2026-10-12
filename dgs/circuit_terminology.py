"""Nilsson & Riedel Table 4.1, executable: node, essential node, path, branch,
essential branch, loop, mesh, planar circuit -- computed from a netlist, then
used to WRITE the circuit equations in SymPy three ways.

Netlist element: (name, a, b, kind, value)
    'R'  resistor between a and b
    'V'  voltage source, + terminal at a:   V_a - V_b = value
    'I'  current source, current `value` flows THROUGH it from a to b

Counting rules this module makes concrete (Nilsson Sec. 4.1):
    unknown branch currents    b_e   (one per essential branch)
    KCL equations              n_e - 1
    KVL equations              b_e - (n_e - 1)   (= number of meshes if planar)

Three equation writers, all returning essential-branch currents so they can be
compared:
    branch_current_solution  KCL at n_e-1 essential nodes + KVL on a LOOP BASIS
                             (spanning-tree loops -- works for ANY circuit)
    node_voltage_solution    unknown essential-node voltages (+ a current for any
                             branch that is a bare voltage source)
    mesh_current_solution    unknown mesh currents -- needs a PLANAR circuit
                             (no meshes exist otherwise) and no current sources
Tellegen's theorem  sum_k v_k i_k = 0  is the independent energy check.
"""
from collections import namedtuple

import networkx as nx
import sympy as sp

Element = namedtuple("Element", "name a b kind value")
Branch = namedtuple("Branch", "start end steps")      # steps: [(Element, x, y), ...]
KINDS = ("R", "V", "I")


def _as_elements(elements):
    out = []
    names = set()
    for e in elements:
        e = Element(*e)
        if e.kind not in KINDS:
            raise ValueError(f"{e.name}: kind must be one of {KINDS}")
        if e.name in names:
            raise ValueError(f"duplicate element name {e.name}")
        if e.a == e.b:
            raise ValueError(f"{e.name}: both terminals on node {e.a}")
        names.add(e.name)
        out.append(e)
    if not out:
        raise ValueError("empty netlist")
    return out


# ── Table 4.1 objects ────────────────────────────────────────────────

def circuit_graph(elements):
    """MultiGraph: nodes = circuit nodes, one edge per element (key = name)."""
    g = nx.MultiGraph()
    for e in _as_elements(elements):
        g.add_edge(e.a, e.b, key=e.name, element=e)
    if not nx.is_connected(g):
        raise ValueError("circuit graph is not connected")
    return g


def nodes(elements):
    """Points where two or more elements join."""
    g = circuit_graph(elements)
    bad = [n for n, d in g.degree() if d < 2]
    if bad:
        raise ValueError(f"dangling node(s) {bad}: every node needs >= 2 elements")
    return sorted(g.nodes(), key=str)


def essential_nodes(elements):
    """Nodes where three or more elements join."""
    g = circuit_graph(elements)
    return sorted((n for n, d in g.degree() if d >= 3), key=str)


def essential_branches(elements):
    """Chains of elements joining two essential nodes through non-essential ones."""
    els = _as_elements(elements)
    g = circuit_graph(els)
    nodes(els)                                           # dangling-node check
    ess = set(essential_nodes(els))
    if not ess:                                          # one single loop
        start = sorted(g.nodes(), key=str)[0]
        ess = {start}
    used, out = set(), []
    for s in sorted(ess, key=str):
        for _, nbr, key, data in sorted(g.edges(s, keys=True, data=True), key=lambda t: t[2]):
            if key in used:
                continue
            steps, x, el = [], s, data["element"]
            while True:
                used.add(el.name)
                y = el.b if el.a == x else el.a
                steps.append((el, x, y))
                if y in ess:
                    break
                nxt = [d["element"] for _, _, k, d in g.edges(y, keys=True, data=True)
                       if k not in used]
                el, x = nxt[0], y
            out.append(Branch(s, y, steps))
    return out


def branch_label(br):
    return " - ".join(st[0].name for st in br.steps)


def _reduced_graph(branches):
    """Simple graph: each essential branch k becomes  start - ('b',k) - end
    (a self-loop branch uses two dummy nodes). Parallel branches stay distinct."""
    G = nx.Graph()
    for k, br in enumerate(branches):
        if br.start == br.end:
            G.add_edges_from([(br.start, ("b", k, 0)), (("b", k, 0), ("b", k, 1)),
                              (("b", k, 1), br.start)])
        else:
            G.add_edges_from([(br.start, ("b", k)), (("b", k), br.end)])
    return G


def _cycle_to_branches(cycle, branches):
    """Node cycle in the reduced graph -> [(branch_index, +1/-1), ...]."""
    out, L = [], len(cycle)
    for i, v in enumerate(cycle):
        if not (isinstance(v, tuple) and v and v[0] == "b"):
            continue
        prev, nxt = cycle[i - 1], cycle[(i + 1) % L]
        k = v[1]
        br = branches[k]
        if len(v) == 3:                                  # self-loop dummies
            if v[2] == 0:
                out.append((k, +1 if prev == br.start else -1))
            continue
        out.append((k, +1 if (prev == br.start and nxt == br.end) else -1))
    return out


def loop_basis(elements):
    """b_e - n_e + 1 independent loops from a spanning tree (any circuit)."""
    branches = essential_branches(elements)
    G = _reduced_graph(branches)
    return [_cycle_to_branches(c, branches) for c in nx.cycle_basis(G)]


def is_planar(elements):
    return nx.check_planarity(nx.Graph(circuit_graph(elements)))[0]


def meshes(elements):
    """Meshes = interior faces of one planar drawing (None if nonplanar).

    Faces come from networkx's planar embedding; the face with the most
    branches is taken as the outer one. Every face traversal has the same
    rotational sense, so a branch shared by two meshes appears with opposite
    signs -- exactly what mesh currents need.
    """
    branches = essential_branches(elements)
    G = _reduced_graph(branches)
    ok, emb = nx.check_planarity(G)
    if not ok:
        return None
    seen, faces = set(), []
    for u, v in emb.edges():
        if (u, v) in seen:
            continue
        face = emb.traverse_face(u, v, mark_half_edges=seen)
        faces.append(_cycle_to_branches(face, branches))
    faces.sort(key=len)
    return faces[:-1]


def table_4_1(elements):
    """Counts and one example of each Table 4.1 term."""
    els = _as_elements(elements)
    br = essential_branches(els)
    ne = len(essential_nodes(els))
    mesh = meshes(els)
    non_ess = [n for n in nodes(els) if n not in essential_nodes(els)]
    lb = loop_basis(els)
    fmt = lambda loop: " ; ".join(("" if s > 0 else "reverse ") + branch_label(br[k]) for k, s in loop)
    return {
        "n (nodes)": len(nodes(els)),
        "n_e (essential nodes)": ne,
        "b (branches = elements)": len(els),
        "b_e (essential branches)": len(br),
        "KCL equations n_e - 1": ne - 1,
        "KVL equations b_e - (n_e - 1)": len(br) - (ne - 1),
        "planar": mesh is not None,
        "meshes": None if mesh is None else len(mesh),
        "example node (non-essential)": non_ess[0] if non_ess else None,
        "example essential node": essential_nodes(els)[0] if ne else None,
        "example branch": els[0].name,
        "example essential branch": branch_label(br[0]),
        "example loop": fmt(lb[0]) if lb else None,
        "example mesh": fmt(mesh[0]) if mesh else None,
    }


# ── equation writers ─────────────────────────────────────────────────

def branch_law(br):
    """(R_total, source_total, fixed_current) with  V_start - V_end = R i + src."""
    R, src, fixed = sp.Integer(0), sp.Integer(0), None
    for el, x, _ in br.steps:
        if el.kind == "R":
            R += el.value
        elif el.kind == "V":
            src += el.value if x == el.a else -el.value
        else:
            f = el.value if x == el.a else -el.value
            if fixed is not None:
                raise ValueError("two current sources in series in one branch")
            fixed = f
    return R, src, fixed


def _kcl(node, branches, cur):
    expr = 0
    for k, br in enumerate(branches):
        if br.start == br.end:
            continue
        if br.start == node:
            expr += cur[k]
        if br.end == node:
            expr -= cur[k]
    return expr


def _ground(elements, ground):
    ess = essential_nodes(elements) or [nodes(elements)[0]]
    if ground is None:
        return ess[0]
    if ground not in ess:
        raise ValueError(f"ground {ground!r} must be an essential node {ess}")
    return ground


def branch_current_solution(elements, ground=None):
    """KCL at n_e-1 essential nodes + KVL on a loop basis. Returns
    (equations, {branch_index: current}, {branch_index: voltage})."""
    els = _as_elements(elements)
    brs = essential_branches(els)
    gnd = _ground(els, ground)
    cur, volt, unknowns = {}, {}, []
    for k, br in enumerate(brs):
        R, src, fixed = branch_law(br)
        if fixed is None:
            cur[k] = sp.Symbol(f"i_{k}")
            unknowns.append(cur[k])
            volt[k] = R * cur[k] + src
        else:
            cur[k] = fixed
            volt[k] = sp.Symbol(f"u_{k}")
            unknowns.append(volt[k])
    ess = essential_nodes(els) or [gnd]
    eqs = [sp.Eq(_kcl(n, brs, cur), 0) for n in ess if n != gnd]
    eqs += [sp.Eq(sum(s * volt[k] for k, s in loop), 0) for loop in loop_basis(els)]
    sol = sp.solve(eqs, unknowns, dict=True)
    if len(sol) != 1:
        raise ValueError("circuit equations are singular or inconsistent")
    sol = sol[0]
    return eqs, {k: sp.simplify(sp.sympify(c).subs(sol)) for k, c in cur.items()}, \
        {k: sp.simplify(sp.sympify(v).subs(sol)) for k, v in volt.items()}


def node_voltage_solution(elements, ground=None):
    """Unknown essential-node voltages (ground = 0). A branch that is only a
    voltage source gets an extra unknown current and a constraint equation."""
    els = _as_elements(elements)
    brs = essential_branches(els)
    gnd = _ground(els, ground)
    ess = essential_nodes(els) or [gnd]
    V = {n: (0 if n == gnd else sp.Symbol(f"V_{n}")) for n in ess}
    unknowns = [V[n] for n in ess if n != gnd]
    cur, eqs = {}, []
    for k, br in enumerate(brs):
        R, src, fixed = branch_law(br)
        if fixed is not None:
            cur[k] = fixed
        elif R == 0:
            cur[k] = sp.Symbol(f"i_{k}")
            unknowns.append(cur[k])
            eqs.append(sp.Eq(V[br.start] - V[br.end], src))
        else:
            cur[k] = (V[br.start] - V[br.end] - src) / R
    eqs = [sp.Eq(_kcl(n, brs, cur), 0) for n in ess if n != gnd] + eqs
    sol = sp.solve(eqs, unknowns, dict=True)
    if len(sol) != 1:
        raise ValueError("circuit equations are singular or inconsistent")
    sol = sol[0]
    return eqs, {k: sp.simplify(sp.sympify(c).subs(sol)) for k, c in cur.items()}, \
        {n: sp.sympify(v).subs(sol) for n, v in V.items()}


def mesh_current_solution(elements):
    """Unknown mesh currents j_m; branch current i_k = sum_m s_mk j_m.
    KCL is automatic; one KVL per mesh. Planar circuits without current sources."""
    els = _as_elements(elements)
    brs = essential_branches(els)
    mesh = meshes(els)
    if mesh is None:
        raise ValueError("nonplanar circuit: meshes do not exist -- use a loop basis")
    laws = [branch_law(br) for br in brs]
    if any(f is not None for _, _, f in laws):
        raise ValueError("current sources need supermeshes; use branch_current_solution")
    j = sp.symbols(f"j_0:{len(mesh)}")
    cur = {k: 0 for k in range(len(brs))}
    for m, loop in enumerate(mesh):
        for k, s in loop:
            cur[k] += s * j[m]
    volt = {k: laws[k][0] * cur[k] + laws[k][1] for k in cur}
    eqs = [sp.Eq(sum(s * volt[k] for k, s in loop), 0) for loop in mesh]
    sol = sp.solve(eqs, j, dict=True)
    if len(sol) != 1:
        raise ValueError("mesh equations are singular")
    sol = sol[0]
    return eqs, {k: sp.simplify(sp.sympify(c).subs(sol)) for k, c in cur.items()}


def tellegen_sum(elements, currents, voltages):
    """sum_k v_k i_k over essential branches: zero for any valid solution."""
    return sp.simplify(sum(voltages[k] * currents[k] for k in currents))
