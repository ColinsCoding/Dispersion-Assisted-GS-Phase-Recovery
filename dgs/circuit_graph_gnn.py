"""A boolean logic circuit (the full adder already implemented in
dgs.computer_engineering.full_adder) represented as a PyTorch Geometric
graph, and evaluated TWO ways: (1) exact simulation via a custom
MessagePassing layer that propagates boolean values through the DAG in
enough rounds to reach every node's true steady-state value -- not a
learned approximation, a genuine graph-structured circuit simulator --
and (2) a small GCN trained to predict the circuit's outputs, the more
typical PyTorch-Geometric-style ML use, included for contrast with (1).

WHY A CIRCUIT IS A GRAPH: a combinational logic circuit is literally a
directed acyclic graph (DAG) -- gates are nodes, wires are edges, and a
gate's output value depends only on its predecessors' values, computed in
topological order. Graph neural network MESSAGE PASSING (each node
aggregates values from its in-neighbors, applies a function, repeat) is
structurally the same operation as circuit EVALUATION (each gate reads its
input wires, applies its truth table, repeat) -- this module makes that
correspondence literal and exact, rather than using it as a metaphor.

Full adder structure (dgs.computer_engineering.full_adder's own formula,
S = XOR(XOR(A,B),Cin), Cout = majority(A,B,Cin) = OR(AND(A,B),AND(Cin,XOR(A,B)))):

    A, B, Cin (inputs)
      -> XOR1 = XOR(A,B)
      -> XOR2 = XOR(XOR1,Cin)   = S      (circuit's sum output)
      -> AND1 = AND(A,B)
      -> AND2 = AND(XOR1,Cin)
      -> OR1  = OR(AND1,AND2)   = Cout   (circuit's carry output)

8 nodes: A(0), B(1), Cin(2), XOR1(3), XOR2/S(4), AND1(5), AND2(6), OR1/Cout(7).
DAG depth is 3 (OR1 is 3 hops from the inputs), so 4 rounds of message
passing (one extra for safety margin) are enough for every node's value
to reach its true fixed point.
"""

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from dgs.computer_engineering import full_adder

GATE_TYPES = ("INPUT", "XOR", "AND", "OR")
NODE_GATE_TYPE = ("INPUT", "INPUT", "INPUT", "XOR", "XOR", "AND", "AND", "OR")
NODE_NAMES = ("A", "B", "Cin", "XOR1", "XOR2_S", "AND1", "AND2", "OR1_Cout")

EDGE_INDEX = torch.tensor([
    [0, 3], [1, 3],   # A,B -> XOR1
    [3, 4], [2, 4],   # XOR1, Cin -> XOR2 (S)
    [0, 5], [1, 5],   # A, B -> AND1
    [3, 6], [2, 6],   # XOR1, Cin -> AND2
    [5, 7], [6, 7],   # AND1, AND2 -> OR1 (Cout)
], dtype=torch.long).t().contiguous()

DAG_DEPTH = 3
N_NODES = len(NODE_NAMES)


def _gate_type_onehot() -> torch.Tensor:
    idx = torch.tensor([GATE_TYPES.index(g) for g in NODE_GATE_TYPE], dtype=torch.long)
    return torch.nn.functional.one_hot(idx, num_classes=len(GATE_TYPES)).float()


def build_full_adder_graph(A: int, B: int, Cin: int) -> Data:
    """Builds the full-adder circuit as a PyG Data object: node features
    are [gate_type_onehot (4), input_value (1)] -- input_value is only
    meaningful for the three INPUT nodes (A, B, Cin), 0 elsewhere (the
    message-passing simulator computes every other node's value from the
    graph structure, not from this feature)."""
    for name, v in (("A", A), ("B", B), ("Cin", Cin)):
        if v not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1, got {v}")
    gate_onehot = _gate_type_onehot()
    input_vals = torch.zeros(N_NODES, 1)
    input_vals[0, 0], input_vals[1, 0], input_vals[2, 0] = float(A), float(B), float(Cin)
    x = torch.cat([gate_onehot, input_vals], dim=1)
    return Data(x=x, edge_index=EDGE_INDEX, num_nodes=N_NODES)


class CircuitMessagePassing(MessagePassing):
    """Exact boolean-circuit simulation via message passing: each round,
    every node aggregates its in-neighbors' current values as a LIST
    (not summed -- summing would conflate "both inputs are 1" with "one
    input is 2", so aggregation here is 'collect', and the gate-type-
    specific reduction (XOR/AND/OR) happens in `update`), then updates its
    own value: INPUT nodes keep their fixed input value forever; XOR/AND/OR
    nodes recompute from their (0, 1, or 2) current in-neighbor values.
    Run for `DAG_DEPTH + 1` rounds so values have enough "gate delays" to
    reach every node's true steady state, the same reason a physical
    circuit needs to be given time to settle."""

    def __init__(self):
        super().__init__(aggr=None, flow="source_to_target")

    def forward(self, x_gate_type: torch.Tensor, edge_index: torch.Tensor,
                current_values: torch.Tensor) -> torch.Tensor:
        collected = self.propagate(edge_index, x=current_values.unsqueeze(-1))
        return self._apply_gate(x_gate_type, current_values, collected)

    def message(self, x_j):
        return x_j   # pass the source node's current value unchanged

    def aggregate(self, inputs, index, dim_size=None):
        # collect per-target-node incoming values as a padded (N, max_in) tensor
        max_in = 2   # every gate in this circuit has at most 2 inputs
        out = torch.full((dim_size, max_in), float('nan'))
        counts = torch.zeros(dim_size, dtype=torch.long)
        for src_val, tgt in zip(inputs, index):
            slot = counts[tgt].item()
            if slot < max_in:
                out[tgt, slot] = src_val.item()
            counts[tgt] += 1
        return out

    def _apply_gate(self, x_gate_type, current_values, collected):
        new_values = current_values.clone()
        for i in range(current_values.shape[0]):
            gate = NODE_GATE_TYPE[i]
            if gate == "INPUT":
                continue   # inputs never change
            a, b = collected[i, 0], collected[i, 1]
            if torch.isnan(a) or torch.isnan(b):
                continue   # not all predecessors have propagated a value yet
            a, b = bool(a.item() >= 0.5), bool(b.item() >= 0.5)
            if gate == "XOR":
                new_values[i] = float(a ^ b)
            elif gate == "AND":
                new_values[i] = float(a and b)
            elif gate == "OR":
                new_values[i] = float(a or b)
        return new_values


def simulate_full_adder_via_message_passing(A: int, B: int, Cin: int) -> dict:
    """Exact circuit evaluation via CircuitMessagePassing, run for
    DAG_DEPTH+1 rounds. Returns S and Cout read off nodes 4 and 7."""
    data = build_full_adder_graph(A, B, Cin)
    gate_type = data.x[:, :len(GATE_TYPES)]
    values = data.x[:, len(GATE_TYPES)]   # starts as the input values, 0 elsewhere

    layer = CircuitMessagePassing()
    for _ in range(DAG_DEPTH + 1):
        values = layer(gate_type, data.edge_index, values)

    return {"S": int(round(values[4].item())), "Cout": int(round(values[7].item())),
            "all_node_values": {name: float(v) for name, v in zip(NODE_NAMES, values.tolist())}}


def verify_all_truth_table_rows() -> dict:
    """CHECKED, not assumed: the message-passing simulator must EXACTLY
    match dgs.computer_engineering.full_adder's ground truth for ALL 8
    input combinations, not a sample."""
    mismatches = []
    for A in (0, 1):
        for B in (0, 1):
            for Cin in (0, 1):
                truth = full_adder(A, B, Cin)
                sim = simulate_full_adder_via_message_passing(A, B, Cin)
                if sim["S"] != truth["S"] or sim["Cout"] != truth["Cout"]:
                    mismatches.append({"A": A, "B": B, "Cin": Cin, "truth": truth, "sim": sim})
    return {"n_rows_checked": 8, "n_mismatches": len(mismatches), "mismatches": mismatches,
            "all_match": len(mismatches) == 0}


# ── A small, actually-trained GCN, for contrast with the exact simulator ───

def build_training_dataset() -> list:
    """All 8 (A,B,Cin) rows as PyG Data objects, each labeled with the
    TRUE (S, Cout) from dgs.computer_engineering.full_adder -- a real
    (if tiny) supervised dataset, not synthetic labels invented for this
    module."""
    dataset = []
    for A in (0, 1):
        for B in (0, 1):
            for Cin in (0, 1):
                data = build_full_adder_graph(A, B, Cin)
                truth = full_adder(A, B, Cin)
                data.y = torch.tensor([[float(truth["S"]), float(truth["Cout"])]])
                dataset.append(data)
    return dataset


class TinyCircuitGCN(torch.nn.Module):
    """A minimal 2-layer GCN (torch_geometric.nn.GCNConv) that reads the
    graph's node features and predicts (S, Cout) from a global mean-pool
    -- the "typical PyG ML" counterpart to CircuitMessagePassing's exact,
    untrained simulation."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.conv1 = GCNConv(len(GATE_TYPES) + 1, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.readout = torch.nn.Linear(hidden, 2)
        self._global_mean_pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        pooled = self._global_mean_pool(h, batch)
        return torch.sigmoid(self.readout(pooled))


def train_tiny_gcn(n_epochs: int = 500, lr: float = 0.05, seed: int = 0) -> dict:
    """Trains TinyCircuitGCN on all 8 rows (batched via
    torch_geometric.loader.DataLoader), tracks loss, and reports final
    per-row predictions vs. ground truth -- with only 8 examples this is
    closer to memorization than generalization, and that's stated plainly
    rather than oversold as a generalization result."""
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed)

    dataset = build_training_dataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = TinyCircuitGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    loss_history = []
    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

    model.eval()
    predictions = []
    with torch.no_grad():
        for A in (0, 1):
            for B in (0, 1):
                for Cin in (0, 1):
                    data = build_full_adder_graph(A, B, Cin)
                    batch = torch.zeros(N_NODES, dtype=torch.long)
                    pred = model(data.x, data.edge_index, batch)
                    truth = full_adder(A, B, Cin)
                    predictions.append({
                        "A": A, "B": B, "Cin": Cin,
                        "pred_S": float(pred[0, 0]), "pred_Cout": float(pred[0, 1]),
                        "true_S": truth["S"], "true_Cout": truth["Cout"],
                        "S_correct": bool(round(float(pred[0, 0])) == truth["S"]),
                        "Cout_correct": bool(round(float(pred[0, 1])) == truth["Cout"]),
                    })

    n_correct = sum(p["S_correct"] and p["Cout_correct"] for p in predictions)
    return {"loss_history": loss_history, "predictions": predictions,
            "n_correct_of_8": n_correct, "final_loss": loss_history[-1]}


if __name__ == "__main__":
    print("=== 1. Exact circuit simulation via graph message passing ===")
    for A in (0, 1):
        for B in (0, 1):
            for Cin in (0, 1):
                sim = simulate_full_adder_via_message_passing(A, B, Cin)
                truth = full_adder(A, B, Cin)
                match = "OK" if (sim["S"] == truth["S"] and sim["Cout"] == truth["Cout"]) else "MISMATCH"
                print(f"  A={A} B={B} Cin={Cin}: sim S={sim['S']} Cout={sim['Cout']}, "
                      f"truth S={truth['S']} Cout={truth['Cout']}  [{match}]")

    check = verify_all_truth_table_rows()
    print(f"\n  all 8 rows match: {check['all_match']} ({check['n_mismatches']} mismatches)")

    print("\n=== 2. A small trained GCN, for contrast ===")
    result = train_tiny_gcn()
    print(f"  final training loss: {result['final_loss']:.4f}")
    print(f"  correct on {result['n_correct_of_8']}/8 rows after training")
    for p in result["predictions"]:
        mark = "OK" if p["S_correct"] and p["Cout_correct"] else "WRONG"
        print(f"    A={p['A']} B={p['B']} Cin={p['Cin']}: "
              f"pred S={p['pred_S']:.2f}(true {p['true_S']}) Cout={p['pred_Cout']:.2f}(true {p['true_Cout']})  [{mark}]")

    print("\nMessage passing simulates the circuit exactly (it IS the circuit, structurally);")
    print("the trained GCN has to learn the same function from 8 labeled examples --")
    print("two very different relationships between a graph neural network and a boolean circuit.")
