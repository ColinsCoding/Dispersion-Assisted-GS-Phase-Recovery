"""Test dgs/lidar_point_cloud_classifier.py: the synthetic scene
generator, the pure-torch k-NN graph (built to avoid the torch_cluster
dependency, which fails to install here), the eigenvalue-based geometric
features actually discriminating flat vs. scattered neighborhoods, and
the trained EdgeConv classifier's held-out accuracy.

torch_geometric is py-3.12-only in this repo."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs.lidar_point_cloud_classifier import (
    generate_synthetic_scene, torch_knn_graph, compute_geometric_features,
    build_dataset, train_classifier, verify_planarity_discriminates_classes,
    CLASS_NAMES,
)

# 1. generate_synthetic_scene: well-formed, all 3 classes present, counts add up
scene = generate_synthetic_scene(n_ground=200, n_buildings=2, n_trees=3,
                                 points_per_building=60, points_per_tree=30, seed=1)
assert scene["points"].shape[1] == 3
assert scene["points"].shape[0] == len(scene["labels"])
expected_total = 200 + 2 * 60 + 3 * 30
assert scene["points"].shape[0] == expected_total
assert set(np.unique(scene["labels"])) == {0, 1, 2}

# ground points should be near z=0; building/vegetation points should have
# some points well above z=0 -- basic sanity on the generator's geometry
ground_z = scene["points"][scene["labels"] == 0, 2]
assert np.abs(ground_z).max() < 0.5
nonground_z = scene["points"][scene["labels"] != 0, 2]
assert nonground_z.max() > 2.0

print("dgs.lidar_point_cloud_classifier: synthetic scene checks passed")

# 2. torch_knn_graph: correct shape, no self-loops, valid k bounds
positions = torch.as_tensor(scene["points"])
k = 8
edge_index = torch_knn_graph(positions, k=k)
assert edge_index.shape == (2, positions.shape[0] * k)
src, dst = edge_index
assert not torch.any(src == dst), "k-NN graph should not contain self-loops"

for bad_k in (0, positions.shape[0]):
    try:
        torch_knn_graph(positions, k=bad_k)
        raise AssertionError(f"expected ValueError for k={bad_k}")
    except ValueError:
        pass

print("dgs.lidar_point_cloud_classifier: k-NN graph checks passed")

# 3. compute_geometric_features: correct shape, features in sane ranges
features = compute_geometric_features(positions, edge_index, k=k)
assert features.shape == (positions.shape[0], 4)
planarity = features[:, 2]
verticality = features[:, 3]
assert torch.all(planarity >= -1e-6) and torch.all(planarity <= 1.0 + 1e-6)
assert torch.all(verticality >= -1e-6) and torch.all(verticality <= 1.0 + 1e-6)
height_above_ground = features[:, 0]
assert torch.all(height_above_ground >= -1e-6), "height above local ground should be non-negative"

print("dgs.lidar_point_cloud_classifier: geometric feature checks passed")

# 4. verify_planarity_discriminates_classes: the actual claim -- flat
#    classes (ground, building) must exceed scattered vegetation, checked
#    across a couple of seeds, not a single lucky draw
for seed in (0, 1, 2):
    check = verify_planarity_discriminates_classes(seed=seed)
    assert check["ground_and_building_exceed_vegetation"] is True, f"seed={seed}: {check}"
    assert check["mean_planarity_ground"] > 0
    assert check["mean_planarity_vegetation"] > 0   # still positive, just smaller

print("dgs.lidar_point_cloud_classifier: planarity-discrimination checks passed")

# 5. build_dataset: consistent shapes across positions/labels/edge_index/features
data = build_dataset(seed=0, k=10)
n = data["positions"].shape[0]
assert data["labels"].shape == (n,)
assert data["features"].shape == (n, 4)
assert data["edge_index"].shape[0] == 2

# 6. train_classifier: real held-out generalization, not just training fit
result = train_classifier(n_epochs=200, seed=0)
assert result["test_accuracy"] > 0.75, f"expected reasonable held-out accuracy, got {result['test_accuracy']}"
assert result["loss_history"][-1] < result["loss_history"][0], "training loss should have decreased"
assert result["confusion_matrix"].shape == (3, 3)
assert result["confusion_matrix"].sum() == int((~result["train_mask"]).sum())

# the confusion matrix's diagonal (correct classifications) should dominate
cm = result["confusion_matrix"]
diagonal_frac = cm.diagonal().sum() / cm.sum()
assert diagonal_frac > 0.75

print("dgs.lidar_point_cloud_classifier: trained classifier checks passed")
print("all dgs.lidar_point_cloud_classifier tests passed")
