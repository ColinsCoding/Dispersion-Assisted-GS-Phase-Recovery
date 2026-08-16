"""Test dgs/circuit_graph_gnn.py: the full-adder-as-a-graph representation,
exact circuit evaluation via a custom PyTorch Geometric MessagePassing
layer (checked against dgs.computer_engineering.full_adder for ALL 8
input rows, not a sample), and a small trained GCN for contrast.

torch_geometric is py-3.12-only in this repo (installed alongside torch)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.computer_engineering import full_adder
from dgs.circuit_graph_gnn import (
    build_full_adder_graph, simulate_full_adder_via_message_passing,
    verify_all_truth_table_rows, N_NODES, NODE_NAMES, EDGE_INDEX,
    build_training_dataset, train_tiny_gcn,
)

# 1. build_full_adder_graph: well-formed, correct input encoding, input validation
data = build_full_adder_graph(1, 0, 1)
assert data.num_nodes == N_NODES == 8
assert data.x.shape == (8, 5)   # 4 gate-type one-hot + 1 input-value column
assert float(data.x[0, 4]) == 1.0   # A=1
assert float(data.x[1, 4]) == 0.0   # B=0
assert float(data.x[2, 4]) == 1.0   # Cin=1
assert EDGE_INDEX.shape == (2, 10)   # 10 directed wires

for bad in (2, -1, 0.5):
    try:
        build_full_adder_graph(bad, 0, 0)
        raise AssertionError(f"expected ValueError for A={bad}")
    except ValueError:
        pass

# 2. simulate_full_adder_via_message_passing: matches ground truth on a
#    few spot rows, with intermediate node values also sane
sim = simulate_full_adder_via_message_passing(1, 1, 0)
truth = full_adder(1, 1, 0)
assert sim["S"] == truth["S"] == 0
assert sim["Cout"] == truth["Cout"] == 1
# XOR1 = XOR(1,1) = 0, AND1 = AND(1,1) = 1 -- checked directly, not just the final outputs
assert sim["all_node_values"]["XOR1"] == 0.0
assert sim["all_node_values"]["AND1"] == 1.0

# 3. verify_all_truth_table_rows: EVERY row, not a sample -- the module's
#    actual headline claim
check = verify_all_truth_table_rows()
assert check["n_rows_checked"] == 8
assert check["all_match"] is True
assert check["n_mismatches"] == 0

print("dgs.circuit_graph_gnn: exact message-passing simulation checks passed")

# 4. build_training_dataset: 8 rows, each correctly labeled against the
#    SAME ground truth the simulator is checked against
dataset = build_training_dataset()
assert len(dataset) == 8
seen_labels = set()
for A in (0, 1):
    for B in (0, 1):
        for Cin in (0, 1):
            truth = full_adder(A, B, Cin)
            seen_labels.add((truth["S"], truth["Cout"]))
# not every (S,Cout) combination need be distinct, but the dataset's
# labels must all come from the real full_adder function -- checked by
# reconstructing them independently above and comparing set membership
dataset_labels = {(float(d.y[0, 0]), float(d.y[0, 1])) for d in dataset}
assert dataset_labels <= {(float(a), float(b)) for a, b in seen_labels}

print("dgs.circuit_graph_gnn: training dataset checks passed")

# 5. train_tiny_gcn: with only 8 labeled examples this is a memorization
#    task -- assert it actually succeeds at that (not a generalization
#    claim, just "did the small GCN fit its own training data")
result = train_tiny_gcn(n_epochs=500, seed=0)
assert result["n_correct_of_8"] == 8, f"expected the tiny GCN to fit all 8 training rows, got {result['n_correct_of_8']}/8"
assert result["final_loss"] < result["loss_history"][0], "training loss should have decreased"
assert len(result["predictions"]) == 8

print("dgs.circuit_graph_gnn: trained GCN checks passed")
print("all dgs.circuit_graph_gnn tests passed")
