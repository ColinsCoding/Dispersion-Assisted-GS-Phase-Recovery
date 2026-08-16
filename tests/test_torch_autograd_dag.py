"""Test dgs/torch_autograd_dag.py: walking a real torch autograd graph
into an explicit DAG, topological order, cycle detection, and critical-path
length. Requires py -3.12 (torch is py-3.12 only in this repo, not 3.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from dgs.torch_autograd_dag import (
    build_dag_from_tensor, topological_order, verify_autograd_graph_is_dag,
    critical_path_length, leaf_and_op_counts,
)

# 1. build_dag_from_tensor: a known small graph, z = (x*y + sin(x))*y
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = (x * y + x.sin()) * y
dag = build_dag_from_tensor(z)
# 6 nodes: 2 leaves (x, y -- PyTorch guarantees exactly ONE AccumulateGrad
# per leaf tensor, properly shared across every place that leaf is used)
# + 4 ops (inner Mul for x*y, Sin, Add, outer Mul). 7 edges: x feeds 2
# consumers (inner Mul, Sin), y feeds 2 consumers (inner Mul, outer Mul),
# plus the 3 edges among the op nodes themselves (inner Mul->Add,
# Sin->Add, Add->outer Mul) = 2+2+3 = 7.
#
# CAUGHT AND FIXED THIS SESSION: build_dag_from_tensor originally only
# stored each grad_fn's type NAME keyed by id(fn), without holding a
# reference to fn itself -- so an already-visited grad_fn could be
# garbage-collected mid-walk and its memory address reused by a later,
# DIFFERENT node, silently inflating the count to 7 in ~97% of runs (a
# false "already visited" id() collision). Stress-tested at 500 trials:
# 0/500 wrong after adding the _keepalive list that holds every visited
# grad_fn alive for the walk's duration.
assert len(dag["nodes"]) == 6, f"expected 6 nodes (2 leaves + 4 ops), got {len(dag['nodes'])}"
assert len(dag["edges"]) == 7, f"expected 7 edges, got {len(dag['edges'])}"
assert dag["root_id"] in dag["nodes"]

# 2. build_dag_from_tensor: a tensor with no grad_fn must raise
leaf_only = torch.tensor(1.0, requires_grad=True)
try:
    build_dag_from_tensor(leaf_only)
    raise AssertionError("expected ValueError for a leaf tensor with no grad_fn")
except ValueError:
    pass

# 3. topological_order: every edge must go earlier->later in the returned order
order = topological_order(dag["nodes"], dag["edges"])
assert len(order) == len(dag["nodes"])
position = {n: i for i, n in enumerate(order)}
for a, b in dag["edges"]:
    assert position[a] < position[b], "topological order must respect every edge direction"

# 4. topological_order: a synthetic cycle must raise ValueError (the
#    acyclicity check must actually detect a cycle, not just succeed by luck)
cyclic_nodes = {1: "a", 2: "b", 3: "c"}
cyclic_edges = [(1, 2), (2, 3), (3, 1)]
try:
    topological_order(cyclic_nodes, cyclic_edges)
    raise AssertionError("expected ValueError for a cyclic graph")
except ValueError:
    pass

# 5. verify_autograd_graph_is_dag: True for a real torch graph
assert verify_autograd_graph_is_dag(z) is True

# 6. critical_path_length: known structure -- the x*y branch (depth 1) feeds
#    into Add (depth 2) then the final Mul (depth 3); the sin(x) branch is
#    shallower and must NOT set the critical path
length, path = critical_path_length(dag["nodes"], dag["edges"], order)
assert length == 3, f"expected critical path length 3, got {length}"
path_names = [dag["nodes"][n] for n in path]
assert path_names[0] == "AccumulateGrad"
assert path_names[-1] == "MulBackward0"
assert "SinBackward0" not in path_names, "the shallower sin(x) branch must not be on the critical path"

# 7. critical_path_length: a single isolated node has length 0
single_node = {1: "OnlyNode"}
length0, path0 = critical_path_length(single_node, [])
assert length0 == 0
assert path0 == [1]

# 8. critical_path_length: empty nodes must raise
try:
    critical_path_length({}, [])
    raise AssertionError("expected ValueError for empty nodes")
except ValueError:
    pass

# 9. leaf_and_op_counts: exactly 2 leaves (x, y -- properly deduplicated,
#    not one AccumulateGrad per USE), 4 ops
counts = leaf_and_op_counts(dag["nodes"])
assert counts["n_leaves"] == 2
assert counts["n_ops"] == 4
assert counts["n_total"] == 6

# 10. Regression stress test: the id()-reuse bug (see the comment on
#     check 1) was intermittent, so a single passing run doesn't prove the
#     fix -- run many fresh graphs and require an exact, consistent count
#     every time.
for _ in range(100):
    xt = torch.tensor(2.0, requires_grad=True)
    yt = torch.tensor(3.0, requires_grad=True)
    zt = (xt * yt + xt.sin()) * yt
    d = build_dag_from_tensor(zt)
    assert len(d["nodes"]) == 6, f"node count regression: got {len(d['nodes'])} (expected 6, every trial)"

print("all dgs.torch_autograd_dag tests passed")
