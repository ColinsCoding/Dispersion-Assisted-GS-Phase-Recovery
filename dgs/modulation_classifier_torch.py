"""
modulation_classifier_torch.py -- PyTorch classification of optical modulation format
========================================================================================

THE GAP THIS FILLS
-------------------
dgs.nn_spectral_regression trains a numpy-only MLP to REGRESS phase phi(t) from
(I1, I2) -- deliberately avoiding torch (see its own docstring: "numpy, no torch").
dgs.gs_fno trains a torch Fourier Neural Operator, also for REGRESSION. Neither
module does CLASSIFICATION.

Paper [3] (Pu & Jalali, "Neural network enabled time stretch spectral regression,"
Opt. Express 29(13), 20786, 2021) explicitly frames this distinction: prior work
(Chen et al., Sci. Rep. 6, 21471, 2016) used a neural network for "a simple binary
classification" (cancer cell / not cancer cell) before Pu & Jalali moved to
regression "into the equivalent of six thousand classes." This module deliberately
goes the other direction: given (I1, I2), CLASSIFY which of dgs.gs_core's seven
modulation formats (OOK, PAM4, QPSK, DPSK, STEAM, Soliton, 6PSK) produced it -- a
diagnostic a receiver would plausibly run BEFORE choosing which demodulation /
GS-recovery path to apply to an unrecognized incoming signal.

ARCHITECTURE
------------
Input  : (B, 2, N) -- channels [I1(t), I2(t)], same convention as dgs.gs_fno
Output : (B, 7)     -- logits over the 7 modulation-format classes

    Conv1d(2->16, k=7) -> ReLU -> MaxPool(2)
    Conv1d(16->32, k=5) -> ReLU -> MaxPool(2)
    Flatten -> Linear -> ReLU -> Linear -> 7 logits

All training/test data comes from dgs.gs_core.make_measurements -- the SAME
physics-validated forward model used by the phase-retrieval notebooks, just
relabeled by which modulation format generated each (I1, I2) pair, instead of by
the recovered phase. No new physics is introduced here; only a new task.

Grade-7 explanation
--------------------
Show the network thousands of "here's what OOK looks like," "here's what QPSK
looks like," and so on -- the same two brightness videos from gs_core, just
relabeled by which format made them -- and it learns to recognize the shape of
each format's ripples, the way you'd learn to tell handwriting apart after
seeing enough examples.

Run: py -3.13 -c "from dgs.modulation_classifier_torch import demo; demo()"
"""

import numpy as np
import torch
import torch.nn as nn

from dgs.gs_core import make_measurements

MODULATION_FORMATS = ["OOK", "PAM4", "QPSK", "DPSK", "STEAM", "SOLITON", "6PSK"]


# ── Dataset generation (reuses gs_core's physics, adds only the label) ───────

def generate_classification_dataset(n_per_format=200, n_symbols=64, sps=8,
                                     D1=-5000.0, D2=-5750.0, snr_db=25.0, seed=0):
    """Generate a labeled (I1, I2) -> modulation_format dataset.

    Every example is produced by dgs.gs_core.make_measurements -- the physics is
    identical to what the phase-retrieval notebooks use; only the label (which
    format produced this trace) is new.

    Returns
    -------
    X : float32 array, shape (n_total, 2, N) -- channels [I1(t), I2(t)]
    y : int64 array, shape (n_total,) -- class index into MODULATION_FORMATS
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for label, fmt in enumerate(MODULATION_FORMATS):
        for _ in range(n_per_format):
            trial_seed = int(rng.integers(0, 2**31 - 1))
            data = make_measurements(modulation=fmt, n_symbols=n_symbols, sps=sps,
                                      D1=D1, D2=D2, snr_db=snr_db, rng_seed=trial_seed)
            I1, I2 = data["I1"], data["I2"]
            # Per-trace normalization: classify on SHAPE, not absolute power level.
            scale = max(I1.max(), I2.max(), 1e-12)
            X_list.append(np.stack([I1 / scale, I2 / scale], axis=0))
            y_list.append(label)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    # Shuffle so class order isn't blocked (matters for minibatch training)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ── Model ──────────────────────────────────────────────────────────────────

class ModulationClassifierCNN(nn.Module):
    """1D CNN classifying modulation format from stacked (I1, I2) intensity traces."""

    def __init__(self, n_samples, n_classes=len(MODULATION_FORMATS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        flat_len = n_samples // 4  # two MaxPool1d(2) halvings
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * flat_len, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Training / evaluation ─────────────────────────────────────────────────

def train_classifier(model, X_train, y_train, X_val, y_val,
                      n_epochs=30, lr=1e-3, batch_size=32, verbose=True):
    """Standard supervised training loop (cross-entropy, Adam)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    n = len(yt)
    history = {"train_loss": [], "val_accuracy": []}
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv).argmax(dim=1)
            val_acc = float((val_pred == yv).float().mean())
        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(val_acc)
        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch+1:3d}/{n_epochs}  loss={epoch_loss:.4f}  val_acc={val_acc:.3f}")

    return history


def evaluate_classifier(model, X_test, y_test):
    """Test-set accuracy, per-class accuracy, and confusion matrix (no sklearn dependency)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    n_classes = len(MODULATION_FORMATS)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_test, y_pred):
        confusion[true_label, pred_label] += 1

    accuracy = float((y_pred == y_test).mean())
    per_class_accuracy = {
        MODULATION_FORMATS[c]: float(confusion[c, c] / max(confusion[c].sum(), 1))
        for c in range(n_classes)
    }
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "per_class_accuracy": per_class_accuracy,
        "y_pred": y_pred,
    }


# ── Self-contained demo ───────────────────────────────────────────────────

def demo():
    """End-to-end smoke test: generate data, train briefly, report accuracy."""
    print("Generating labeled (I1, I2) -> modulation-format dataset from dgs.gs_core...")
    X, y = generate_classification_dataset(n_per_format=120, n_symbols=64, sps=8, seed=0)
    n = len(y)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}  "
          f"N_samples_per_trace={X.shape[2]}  classes={MODULATION_FORMATS}")

    model = ModulationClassifierCNN(n_samples=X.shape[2])
    print("Training...")
    train_classifier(model, X_train, y_train, X_val, y_val, n_epochs=20, verbose=True)

    result = evaluate_classifier(model, X_test, y_test)
    print(f"\nTest accuracy: {result['accuracy']:.3f}")
    print("Per-class accuracy:")
    for fmt, acc in result["per_class_accuracy"].items():
        print(f"  {fmt:8s}: {acc:.3f}")
    return result


if __name__ == "__main__":
    demo()
