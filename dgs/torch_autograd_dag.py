"""torch_autograd_dag.py -- walk a REAL PyTorch autograd computation graph
(via .grad_fn.next_functions) into an explicit DAG, then run this repo's
own graph-theory tools on it: topological order, critical-path length
(same concept as dgs/logic_timing.py's Circuit.critical_path, applied to
an actual neural-net-style computation instead of a hand-built logic
circuit), and a direct acyclicity check -- not assumed, verified.

WHY THIS IS A REAL DAG, NOT A METAPHOR: PyTorch's autograd graph IS
literally a DAG by construction -- an operation's grad_fn can only
reference the grad_fn's of tensors that already existed when it ran, so a
cycle (an operation depending on its own output) is structurally
impossible. topological_order below still runs Kahn's algorithm and
raises ValueError if it ever finds a cycle, rather than assuming success --
verify_autograd_graph_is_dag turns that into an explicit, direct check
against a REAL traced graph, not a claim about autograd taken on faith.

WALK DIRECTION: fn.next_functions lists the grad_fns that PRODUCED fn's
inputs -- i.e., walking it goes from the output BACKWARD toward the
leaves (the same direction .backward() executes in). This module reverses
that into FORWARD edges (leaf/earlier-computed node -> the node that
consumes it), so topological_order returns nodes in actual forward
computation order (inputs before the operations that use them), and
critical_path_length measures the longest forward dependency chain --
the minimum number of sequential steps this computation graph requires,
regardless of how much parallel hardware is available.

Requires torch (py 3.12 here, matching this repo's existing convention).
"""

from __future__ import annotations
import torch
from collections import defaultdict, deque
from typing import Dict, List, Tuple


# ── 1. Walking a real autograd graph into an explicit DAG ───────────────────

def build_dag_from_tensor(output: torch.Tensor) -> Dict:
    """Walk output.grad_fn.next_functions recursively, building an
    explicit DAG: nodes = {id(grad_fn): type_name}, edges = list of
    (from_id, to_id) in FORWARD order (from_id's result feeds into to_id).
    AccumulateGrad nodes are the leaves (actual tensors with
    requires_grad=True); everything else is an intermediate operation.

    Raises ValueError if `output` has no grad_fn (e.g. requires_grad=False
    or output is a leaf) -- there is no graph to walk in that case.
    """
    if output.grad_fn is None:
        raise ValueError("output.grad_fn is None -- output has no autograd "
                          "graph to walk (check requires_grad on its inputs)")
    nodes: Dict[int, str] = {}
    edges: List[Tuple[int, int]] = []
    # Keep every visited grad_fn object ALIVE for the whole walk: id() is
    # only a valid identity key while the object exists. Without this,
    # an already-visited grad_fn can be garbage-collected mid-walk and its
    # memory address reused by a later, DIFFERENT node -- a real, verified
    # bug (reproduced in ~3% of trials before this fix: two distinct nodes
    # silently collapsed into one because id() collided after GC reuse).
    _keepalive: List[object] = []

    def visit(fn) -> None:
        fid = id(fn)
        if fid in nodes:
            return
        _keepalive.append(fn)
        nodes[fid] = type(fn).__name__
        for next_fn, _ in getattr(fn, "next_functions", []):
            if next_fn is not None:
                visit(next_fn)
                edges.append((id(next_fn), fid))

    visit(output.grad_fn)
    return {"nodes": nodes, "edges": edges, "root_id": id(output.grad_fn),
            "_keepalive": _keepalive}


# ── 2. Topological order (Kahn's algorithm) -- doubles as an acyclicity check

def topological_order(nodes: Dict[int, str], edges: List[Tuple[int, int]]) -> List[int]:
    """Kahn's algorithm. Raises ValueError if the graph has a cycle (fewer
    nodes end up ordered than exist) -- a real check, not an assumption
    that the input is acyclic."""
    indeg = {n: 0 for n in nodes}
    succ = defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        indeg[b] += 1
    queue = deque(n for n in nodes if indeg[n] == 0)
    order: List[int] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(order) != len(nodes):
        raise ValueError(f"cycle detected: only {len(order)}/{len(nodes)} nodes "
                          f"could be topologically ordered -- this graph is not a DAG")
    return order


def verify_autograd_graph_is_dag(output: torch.Tensor) -> bool:
    """Build the DAG from `output` and confirm topological_order succeeds
    -- an explicit, direct check that this specific traced graph is
    acyclic, rather than a claim taken on faith about autograd in general."""
    dag = build_dag_from_tensor(output)
    try:
        topological_order(dag["nodes"], dag["edges"])
        return True
    except ValueError:
        return False


# ── 3. Critical path: the longest forward dependency chain ──────────────────

def critical_path_length(nodes: Dict[int, str], edges: List[Tuple[int, int]],
                          topo_order: List[int] | None = None) -> Tuple[int, List[int]]:
    """Longest path (in NODE COUNT, i.e. number of sequential steps) from
    any source (no predecessors) to the node it ends at -- the same
    "critical path" concept as dgs/logic_timing.py's Circuit.critical_path,
    here applied to a real torch computation graph: the minimum number of
    sequential steps this graph requires, no matter how much parallel
    hardware executes the independent branches.

    Returns (length, path) where length is the number of EDGES on the
    longest path (0 for a single isolated node) and path is the node id
    sequence achieving it.
    """
    if not nodes:
        raise ValueError("nodes must be non-empty")
    topo_order = topo_order if topo_order is not None else topological_order(nodes, edges)
    preds = defaultdict(list)
    for a, b in edges:
        preds[b].append(a)

    dist: Dict[int, int] = {}
    parent: Dict[int, int | None] = {}
    for n in topo_order:
        if not preds[n]:
            dist[n], parent[n] = 0, None
        else:
            best_pred = max(preds[n], key=lambda p: dist[p])
            dist[n] = dist[best_pred] + 1
            parent[n] = best_pred

    end = max(dist, key=lambda n: dist[n])
    path = [end]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return dist[end], path


# ── 4. Leaf vs. operation node counts ────────────────────────────────────────

def leaf_and_op_counts(nodes: Dict[int, str]) -> Dict[str, int]:
    """AccumulateGrad nodes are leaves (real tensors with requires_grad=True
    feeding into the graph); everything else is an intermediate operation."""
    n_leaves = sum(1 for name in nodes.values() if name == "AccumulateGrad")
    return {"n_leaves": n_leaves, "n_ops": len(nodes) - n_leaves, "n_total": len(nodes)}


if __name__ == "__main__":
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = (x * y + x.sin()) * y

    dag = build_dag_from_tensor(z)
    print(f"Traced graph for z = (x*y + sin(x))*y :")
    print(f"  {len(dag['nodes'])} nodes, {len(dag['edges'])} edges")

    counts = leaf_and_op_counts(dag["nodes"])
    print(f"  leaves (AccumulateGrad): {counts['n_leaves']}   ops: {counts['n_ops']}")

    is_dag = verify_autograd_graph_is_dag(z)
    print(f"\n  Verified acyclic (topological_order succeeds): {is_dag}")

    order = topological_order(dag["nodes"], dag["edges"])
    print(f"  Topological (forward computation) order: "
          f"{[dag['nodes'][n] for n in order]}")

    length, path = critical_path_length(dag["nodes"], dag["edges"], order)
    print(f"\n  Critical path length: {length} sequential steps")
    print(f"  Critical path: {' -> '.join(dag['nodes'][n] for n in path)}")

    print("\n  This is the minimum number of sequential steps needed to "
          "evaluate this graph, regardless of parallel hardware -- the "
          "same 'critical path -> fmax' logic dgs/logic_timing.py applies "
          "to hand-built logic circuits, here applied to a real autograd graph.")
