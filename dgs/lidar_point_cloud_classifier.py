"""LiDAR point-cloud classification: ground vs. building vs. vegetation,
on a synthetic scene, using the SAME eigenvalue-based geometric features
real airborne/terrestrial LiDAR processing pipelines use (Weinmann, Jutzi,
Hinz & Mallet 2015 -- planarity and verticality from PCA on each point's
local k-nearest-neighborhood covariance matrix), fed into a PyTorch
Geometric graph neural network built on a k-NN graph.

k-NN GRAPH WITHOUT torch_cluster: PyG's usual `knn_graph` helper needs the
torch_cluster package, which requires compiling from source against this
exact torch/CUDA build (checked directly -- pip install fails, can't even
find torch during its own isolated build) -- the same class of Windows
build-system pain as PyTorch3D. `torch_knn_graph` below builds the same
k-nearest-neighbor edge list with plain torch.cdist + topk instead, no
extra dependency.

GEOMETRIC FEATURES, PER POINT (not arbitrary -- the standard eigenvalue
features from the point-cloud classification literature):
  height_above_local_ground : height relative to the LOWEST point in the
                               local neighborhood (a common LiDAR ground-
                               relative-height feature)
  local_density              : neighbor count within a fixed radius
  planarity   = (lambda2 - lambda3) / lambda1   -- near 1 for a flat patch
  verticality = 1 - |normal_z|                   -- near 1 for a vertical
                                                     surface (a building
                                                     facade), near 0 for a
                                                     flat roof/ground

where lambda1 >= lambda2 >= lambda3 >= 0 are the eigenvalues of the local
neighborhood's covariance matrix, and the surface normal is the
eigenvector of the SMALLEST eigenvalue (the direction of least local
spread).
"""

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, global_mean_pool

CLASS_NAMES = ("ground", "building", "vegetation")


# ── 1. Synthetic scene: ground plane + boxy buildings + noisy vegetation ───

def generate_synthetic_scene(n_ground: int = 800, n_buildings: int = 3, n_trees: int = 6,
                             points_per_building: int = 150, points_per_tree: int = 60,
                             extent: float = 40.0, seed: int = 0) -> dict:
    """A synthetic LiDAR scene: a noisy ground plane, a few box-shaped
    "buildings" (flat roofs + vertical facades -- high planarity, mixed
    verticality), and a few noisy vertical "tree" clusters (low planarity,
    scattered) -- NOT real LiDAR data, a stand-in test scene with known
    ground truth, same honesty posture as this repo's other synthetic-data
    modules."""
    rng = np.random.default_rng(seed)
    points, labels = [], []

    ground_xy = rng.uniform(-extent / 2, extent / 2, size=(n_ground, 2))
    ground_z = rng.normal(0.0, 0.03, size=n_ground)
    points.append(np.column_stack([ground_xy, ground_z]))
    labels.append(np.zeros(n_ground, dtype=int))

    for _ in range(n_buildings):
        cx, cy = rng.uniform(-extent / 2 + 5, extent / 2 - 5, size=2)
        w, d, h = rng.uniform(3, 7), rng.uniform(3, 7), rng.uniform(4, 12)
        n_roof = points_per_building // 2
        n_wall = points_per_building - n_roof
        roof_xy = np.column_stack([rng.uniform(cx - w / 2, cx + w / 2, n_roof),
                                    rng.uniform(cy - d / 2, cy + d / 2, n_roof)])
        roof_z = np.full(n_roof, h) + rng.normal(0, 0.02, n_roof)
        roof = np.column_stack([roof_xy, roof_z])

        wall_side = rng.integers(0, 4, n_wall)
        wall_t = rng.uniform(0, 1, n_wall)
        wall_z = rng.uniform(0, h, n_wall)
        wall_x = np.where(wall_side == 0, cx - w / 2, np.where(wall_side == 1, cx + w / 2,
                          cx - w / 2 + wall_t * w))
        wall_y = np.where(wall_side == 2, cy - d / 2, np.where(wall_side == 3, cy + d / 2,
                          cy - d / 2 + wall_t * d))
        wall = np.column_stack([wall_x, wall_y, wall_z]) + rng.normal(0, 0.02, (n_wall, 3))

        points.append(np.vstack([roof, wall]))
        labels.append(np.ones(points_per_building, dtype=int))

    for _ in range(n_trees):
        cx, cy = rng.uniform(-extent / 2 + 3, extent / 2 - 3, size=2)
        h = rng.uniform(3, 9)
        r = rng.uniform(0.8, 2.2)
        theta = rng.uniform(0, 2 * np.pi, points_per_tree)
        radius = rng.uniform(0, r, points_per_tree)
        z = rng.uniform(0.5, h, points_per_tree)
        x = cx + radius * np.cos(theta) + rng.normal(0, 0.15, points_per_tree)
        y = cy + radius * np.sin(theta) + rng.normal(0, 0.15, points_per_tree)
        points.append(np.column_stack([x, y, z]))
        labels.append(np.full(points_per_tree, 2, dtype=int))

    return {"points": np.vstack(points).astype(np.float32), "labels": np.concatenate(labels)}


# ── 2. k-NN graph, pure torch (no torch_cluster) ────────────────────────────

def torch_knn_graph(positions: torch.Tensor, k: int = 12) -> torch.Tensor:
    """Builds a k-nearest-neighbor edge_index (2, N*k) via torch.cdist +
    topk -- avoids torch_cluster (fails to install here, see module
    docstring)."""
    if k < 1 or k >= positions.shape[0]:
        raise ValueError(f"k must be in [1, {positions.shape[0]-1}], got {k}")
    dist = torch.cdist(positions, positions)
    knn_idx = dist.topk(k + 1, largest=False).indices[:, 1:]   # drop self (distance 0)
    src = torch.arange(positions.shape[0]).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = knn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


# ── 3. Geometric features: eigenvalue-based planarity/verticality ──────────

def compute_geometric_features(positions: torch.Tensor, edge_index: torch.Tensor,
                               k: int, density_radius: float = 1.5) -> torch.Tensor:
    """Per-point features: [height_above_local_ground, local_density,
    planarity, verticality] -- the standard eigenvalue-based geometric
    features (Weinmann et al. 2015), computed from each point's k-NN
    neighborhood covariance matrix, not arbitrary hand-picked numbers."""
    n = positions.shape[0]
    src, dst = edge_index
    features = torch.zeros(n, 4)

    dist_all = torch.cdist(positions, positions)

    for i in range(n):
        neighbor_idx = dst[src == i]
        neighborhood = positions[neighbor_idx]
        local_min_z = torch.min(torch.cat([neighborhood[:, 2], positions[i:i+1, 2]]))
        features[i, 0] = positions[i, 2] - local_min_z

        features[i, 1] = float((dist_all[i] < density_radius).sum() - 1)

        centered = neighborhood - neighborhood.mean(dim=0)
        cov = centered.T @ centered / neighborhood.shape[0]
        eigvals, eigvecs = torch.linalg.eigh(cov)   # ascending
        l3, l2, l1 = eigvals[0], eigvals[1], eigvals[2]   # ascending -> l1 largest
        l1_safe = torch.clamp(l1, min=1e-8)
        planarity = (l2 - l3) / l1_safe
        normal = eigvecs[:, 0]   # eigenvector of smallest eigenvalue
        verticality = 1.0 - torch.abs(normal[2])
        features[i, 2] = planarity
        features[i, 3] = verticality

    return features


# ── 4. GNN classifier (EdgeConv, the standard point-cloud GNN building block)

class PointCloudEdgeConvNet(torch.nn.Module):
    """Two EdgeConv layers (Wang et al. 2019 DGCNN's building block --
    message passing on the k-NN graph using EACH EDGE'S feature
    difference, not just node features) plus a per-point classifier head.
    Point-wise classification (ground/building/vegetation), not a
    single graph-level label."""

    def __init__(self, in_channels: int = 4, hidden: int = 32, n_classes: int = 3):
        super().__init__()
        mlp1 = torch.nn.Sequential(torch.nn.Linear(2 * in_channels, hidden), torch.nn.ReLU(),
                                    torch.nn.Linear(hidden, hidden))
        mlp2 = torch.nn.Sequential(torch.nn.Linear(2 * hidden, hidden), torch.nn.ReLU(),
                                    torch.nn.Linear(hidden, hidden))
        self.conv1 = EdgeConv(mlp1, aggr="max")
        self.conv2 = EdgeConv(mlp2, aggr="max")
        self.head = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        return self.head(h)


# ── 5. Train + evaluate pipeline ────────────────────────────────────────────

def build_dataset(seed: int = 0, k: int = 12) -> dict:
    scene = generate_synthetic_scene(seed=seed)
    positions = torch.as_tensor(scene["points"])
    labels = torch.as_tensor(scene["labels"], dtype=torch.long)
    edge_index = torch_knn_graph(positions, k=k)
    features = compute_geometric_features(positions, edge_index, k=k)
    return {"positions": positions, "labels": labels, "edge_index": edge_index, "features": features}


def train_classifier(n_epochs: int = 300, lr: float = 0.01, k: int = 12,
                     train_frac: float = 0.7, seed: int = 0) -> dict:
    """Trains PointCloudEdgeConvNet on a point-wise train/test split of
    the synthetic scene, tracks loss, and reports test-set accuracy and a
    confusion matrix -- a real (held-out) generalization check, not just
    training-set fit."""
    torch.manual_seed(seed)
    data = build_dataset(seed=seed, k=k)
    n = data["positions"].shape[0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask = ~train_mask

    model = PointCloudEdgeConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_history = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(data["features"], data["edge_index"])
        loss = loss_fn(logits[train_mask], data["labels"][train_mask])
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        logits = model(data["features"], data["edge_index"])
        preds = logits.argmax(dim=1)

    test_preds = preds[test_mask].numpy()
    test_labels = data["labels"][test_mask].numpy()
    accuracy = float((test_preds == test_labels).mean())

    n_classes = len(CLASS_NAMES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(test_labels, test_preds):
        confusion[t, p] += 1

    return {"loss_history": loss_history, "test_accuracy": accuracy, "confusion_matrix": confusion,
            "data": data, "predictions": preds.numpy(), "train_mask": train_mask.numpy()}


def verify_planarity_discriminates_classes(seed: int = 0) -> dict:
    """CHECKED, not assumed: ground and building-roof points (both flat)
    should have HIGHER mean planarity than vegetation points (scattered,
    noisy) -- the geometric feature actually carrying the discriminative
    signal it's supposed to, independent of whether the trained
    classifier gets it right."""
    data = build_dataset(seed=seed)
    planarity = data["features"][:, 2].numpy()
    labels = data["labels"].numpy()
    mean_ground = float(planarity[labels == 0].mean())
    mean_building = float(planarity[labels == 1].mean())
    mean_vegetation = float(planarity[labels == 2].mean())
    return {"mean_planarity_ground": mean_ground, "mean_planarity_building": mean_building,
            "mean_planarity_vegetation": mean_vegetation,
            "ground_and_building_exceed_vegetation": bool(
                mean_ground > mean_vegetation and mean_building > mean_vegetation)}


if __name__ == "__main__":
    print("=== 1. Synthetic LiDAR scene ===")
    scene = generate_synthetic_scene()
    print(f"  {len(scene['points'])} points, classes: "
          f"{[(CLASS_NAMES[c], int((scene['labels']==c).sum())) for c in range(3)]}")

    print("\n=== 2. Geometric features actually discriminate classes ===")
    check = verify_planarity_discriminates_classes()
    print(f"  mean planarity -- ground: {check['mean_planarity_ground']:.3f}, "
          f"building: {check['mean_planarity_building']:.3f}, "
          f"vegetation: {check['mean_planarity_vegetation']:.3f}")
    print(f"  flat classes exceed vegetation: {check['ground_and_building_exceed_vegetation']}")

    print("\n=== 3. Training the EdgeConv point-cloud classifier ===")
    result = train_classifier(n_epochs=300)
    print(f"  final training loss: {result['loss_history'][-1]:.4f}")
    print(f"  held-out test accuracy: {result['test_accuracy']:.1%}")
    print(f"  confusion matrix (rows=true, cols=predicted), classes={CLASS_NAMES}:")
    for i, row in enumerate(result["confusion_matrix"]):
        print(f"    {CLASS_NAMES[i]:>10}: {row}")

    print("\nGround truth from a synthetic scene, geometric features from the same PCA")
    print("eigenvalue machinery real LiDAR pipelines use, classified by a graph neural")
    print("network on a k-NN graph built without the torch_cluster package that wouldn't install.")
