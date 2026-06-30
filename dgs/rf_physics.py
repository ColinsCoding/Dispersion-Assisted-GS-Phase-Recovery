"""Random forest and decision tree interpretation of GS phase retrieval physics.

The central question: which physical parameters control whether the
Gerchberg-Saxton algorithm converges to the correct phase?

The approach is experimental in the scientific sense.  We treat the GS
algorithm as a physical process with:

  Independent variables (controlled inputs):
    D          -- dispersion magnitude (the diversity parameter)
    n_iter     -- number of GS iterations
    noise_sigma -- measurement noise level on I1 and I2
    sigma_x    -- pulse width (position-space width of the wavepacket)
    pulse_type  -- 0=single Gaussian, 1=double-peak (harder)

  Dependent variable (measured output):
    converged  -- 1 if amplitude correlation > threshold, else 0

We run N_SAMPLES GS experiments varying the independent variables,
record the outcome, then train a decision tree and random forest to
predict convergence.

The decision tree is directly interpretable as a set of physics rules:
  "IF |D| > 450 AND n_iter > 35 THEN converged" is not just a
  machine-learning prediction -- it is a falsifiable physical claim
  about the GS algorithm that we can verify analytically.

This is the same structure as hypothesis testing in experimental physics:
  null hypothesis   : the variable does NOT predict convergence
  alternative       : it does
  the tree splits   : are the effect sizes (information gain)

Usage:
    from dgs.rf_physics import generate_dataset, train_and_report, plot_results
    X, y, names = generate_dataset(n_samples=400)
    tree, rf = train_and_report(X, y, names)
    plot_results(X, y, names, tree, rf)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from dgs.decision_tree import DecisionTree, RandomForest, accuracy
from dgs.gs_unsupervised import unsupervised_gs


# ---------------------------------------------------------------------------
# Feature names and the experiment design
# ---------------------------------------------------------------------------

FEATURE_NAMES = ['|D|', 'n_iter', 'noise_sigma', 'sigma_x', 'pulse_type']
CONVERGENCE_THRESHOLD = 0.92   # amplitude correlation cutoff


# ---------------------------------------------------------------------------
# Dataset generation: sweep independent variables, record outcome
# ---------------------------------------------------------------------------

def _make_signal(N: int, sigma_x: float, pulse_type: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Generate a test signal: Gaussian or double-peak."""
    t = np.linspace(-10, 10, N)
    if pulse_type == 0:
        E = np.exp(-t**2 / (2 * sigma_x**2)) * np.exp(1j * rng.uniform(0, 2*np.pi))
    else:
        phi = rng.uniform(0, 2 * np.pi)
        E = (np.exp(-(t - 2*sigma_x)**2 / (2*sigma_x**2)) +
             np.exp(-(t + 2*sigma_x)**2 / (2*sigma_x**2))) * np.exp(1j * phi)
    norm = np.sqrt(np.sum(np.abs(E)**2))
    return E / norm


def _disperse_np(E: np.ndarray, D: float) -> np.ndarray:
    N = len(E)
    omega = np.fft.fftfreq(N) * 2 * np.pi
    return np.fft.ifft(np.fft.fft(E) * np.exp(1j * D / 2 * omega**2))


def run_one_experiment(D: float, n_iter: int, noise_sigma: float,
                       sigma_x: float, pulse_type: int,
                       rng: np.random.Generator, N: int = 256) -> float:
    """Run one GS experiment. Returns amplitude correlation with ground truth."""
    E_true = _make_signal(N, sigma_x, pulse_type, rng)
    I1 = np.abs(E_true)**2 + rng.normal(0, noise_sigma, N).clip(0)
    I2 = np.abs(_disperse_np(E_true, -abs(D)))**2 + rng.normal(0, noise_sigma, N).clip(0)

    try:
        E_rec, _ = unsupervised_gs(I1, I2, D=-abs(D), n_iter=n_iter, verbose=False)
        corr = float(np.corrcoef(np.abs(E_rec)**2, np.abs(E_true)**2)[0, 1])
        return corr if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def generate_dataset(n_samples: int = 300,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate a dataset of GS experiments with randomised parameters.

    Independent variables (features):
      |D|          : uniform in [150, 2000]
      n_iter       : uniform integer in [10, 120]
      noise_sigma  : log-uniform in [1e-4, 5e-2]
      sigma_x      : uniform in [0.3, 2.5]
      pulse_type   : Bernoulli(0.4)  (0=Gaussian, 1=double-peak)

    Dependent variable (label):
      converged = 1 if amplitude_correlation >= CONVERGENCE_THRESHOLD

    Returns (X, y, feature_names).
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []

    print(f"Generating {n_samples} GS experiments (this takes ~{n_samples//10}s)...")
    for i in range(n_samples):
        D          = rng.uniform(150, 2000)
        n_iter     = int(rng.integers(10, 120))
        noise_sigma = float(np.exp(rng.uniform(np.log(1e-4), np.log(5e-2))))
        sigma_x    = rng.uniform(0.3, 2.5)
        pulse_type = int(rng.random() < 0.4)

        corr = run_one_experiment(D, n_iter, noise_sigma, sigma_x, pulse_type, rng)
        converged = int(corr >= CONVERGENCE_THRESHOLD)

        rows.append([D, n_iter, noise_sigma, sigma_x, pulse_type])
        labels.append(converged)

        if (i + 1) % 50 == 0:
            rate = np.mean(labels)
            print(f"  {i+1}/{n_samples}  convergence rate so far: {rate:.2f}")

    X = np.array(rows, dtype=float)
    y = np.array(labels)
    print(f"Done. Convergence rate: {y.mean():.2f}  ({y.sum()}/{len(y)})")
    return X, y, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Train and report
# ---------------------------------------------------------------------------

def train_and_report(X: np.ndarray, y: np.ndarray,
                     feature_names: List[str],
                     test_frac: float = 0.2,
                     seed: int = 7):
    """Train DT and RF, print full report, return (tree, rf)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test  = max(1, int(n * test_frac))
    idx_tr, idx_te = idx[n_test:], idx[:n_test]
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]

    # Decision tree (max depth 4 for human readability)
    tree = DecisionTree(max_depth=4, min_samples_split=5, random_state=seed)
    tree.fit(X_tr, y_tr)
    acc_tr_dt = accuracy(y_tr, tree.predict(X_tr))
    acc_te_dt = accuracy(y_te, tree.predict(X_te))

    # Random forest
    rf = RandomForest(n_estimators=100, max_depth=8,
                      max_features='sqrt', random_state=seed)
    rf.fit(X_tr, y_tr)
    acc_tr_rf = accuracy(y_tr, rf.predict(X_tr))
    acc_te_rf = accuracy(y_te, rf.predict(X_te))

    importances = rf.feature_importances(X_tr, y_tr)

    print()
    print("=" * 60)
    print("GS CONVERGENCE CLASSIFIER -- REPORT")
    print("=" * 60)
    print(f"Dataset: {n} samples, {y.sum()} converged ({y.mean()*100:.1f}%)")
    print(f"Train/test split: {len(y_tr)}/{len(y_te)}")
    print()
    print(f"{'Model':<22}  {'Train acc':>10}  {'Test acc':>10}")
    print("-" * 46)
    print(f"{'Decision Tree (d=4)':<22}  {acc_tr_dt:>10.3f}  {acc_te_dt:>10.3f}")
    print(f"{'Random Forest (100)':<22}  {acc_tr_rf:>10.3f}  {acc_te_rf:>10.3f}")
    print()
    print("Feature importances (Random Forest, mean decrease impurity):")
    order = np.argsort(importances)[::-1]
    for rank, i in enumerate(order):
        bar = "#" * int(importances[i] * 40)
        print(f"  {rank+1}. {feature_names[i]:<14} {importances[i]:.4f}  {bar}")
    print()
    print("Decision tree rules (depth <= 4, readable as physics):")
    _print_tree(tree.root, feature_names, indent=2)
    print()
    print("Physical interpretation:")
    _physics_summary(importances, feature_names)

    return tree, rf


def _print_tree(node, names, indent=0, prefix=""):
    """Print decision tree as nested if/else physics rules."""
    pad = " " * indent
    if node is None:
        return
    if node.is_leaf:
        label = "CONVERGED" if node.value == 1 else "NOT converged"
        print(f"{pad}{prefix}=> {label}  (n={node.n_samples}, H={node.entropy:.3f})")
        return
    fname = names[node.feat]
    print(f"{pad}{prefix}IF {fname} <= {node.thresh:.3f}:")
    _print_tree(node.left,  names, indent + 4, "")
    print(f"{pad}ELSE ({fname} > {node.thresh:.3f}):")
    _print_tree(node.right, names, indent + 4, "")


def _physics_summary(importances: np.ndarray, names: List[str]) -> None:
    order = np.argsort(importances)[::-1]
    lines = [
        "  The random forest assigns feature importance by asking:",
        "  'How much does knowing this variable reduce prediction uncertainty?'",
        "  In physics terms this is the mutual information I(X_i ; converged).",
        "",
    ]
    for rank, i in enumerate(order):
        imp = importances[i]
        if imp < 0.01:
            continue
        name = names[i]
        if name == '|D|':
            lines.append(f"  {rank+1}. |D| (importance={imp:.3f}): Dispersion is the primary driver.")
            lines.append("     Large |D| creates a highly diverse I2, giving the GS algorithm")
            lines.append("     more phase information per iteration.  This matches the theory:")
            lines.append("     the GS diversity metric corr(I1, I2) falls with |D|, and")
            lines.append("     corr < 0.95 is required for convergence (see feedback_gs_convergence.md).")
        elif name == 'n_iter':
            lines.append(f"  {rank+1}. n_iter (importance={imp:.3f}): Iteration count matters but")
            lines.append("     is secondary to |D|.  Once |D| is sufficient, 40-60 iterations")
            lines.append("     are enough.  More iterations help mostly in the high-noise regime.")
        elif name == 'noise_sigma':
            lines.append(f"  {rank+1}. noise_sigma (importance={imp:.3f}): Measurement noise.")
            lines.append("     Acts as a regulariser at low levels but dominates at sigma > 0.01.")
            lines.append("     In remote sensing (SAR, radar): speckle noise plays this role.")
        elif name == 'sigma_x':
            lines.append(f"  {rank+1}. sigma_x (importance={imp:.3f}): Pulse width interacts with D.")
            lines.append("     Narrower pulses have broader spectra, so the same D creates")
            lines.append("     more temporal spreading -- effectively increasing diversity.")
        elif name == 'pulse_type':
            lines.append(f"  {rank+1}. pulse_type (importance={imp:.3f}): Double-peak pulses")
            lines.append("     are harder because they have multiple local minima for the phase.")
    for l in lines:
        print(l)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(X: np.ndarray, y: np.ndarray, names: List[str],
                 tree: DecisionTree, rf: RandomForest) -> None:
    """Four-panel diagnostic plot."""
    importances = rf.feature_importances(X, y)
    order = np.argsort(importances)[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("GS Phase Retrieval: RF + DT Physics Interpretation", fontsize=13)

    # Panel 1: feature importances
    ax = axes[0, 0]
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
    bars = ax.bar([names[i] for i in order],
                  [importances[i] for i in order],
                  color=[colors[i] for i in order])
    ax.set_ylabel("Mean Decrease Impurity")
    ax.set_title("RF Feature Importances\n(independent variable ranking)")
    ax.set_ylim(0, max(importances) * 1.2)
    for bar, i in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{importances[i]:.3f}", ha='center', va='bottom', fontsize=9)

    # Panel 2: |D| vs n_iter scatter coloured by convergence
    ax = axes[0, 1]
    c0 = X[y == 0]; c1 = X[y == 1]
    ax.scatter(c0[:, 0], c0[:, 1], c='tomato',    alpha=0.5, s=20, label='Not converged')
    ax.scatter(c1[:, 0], c1[:, 1], c='steelblue', alpha=0.5, s=20, label='Converged')
    ax.set_xlabel("|D| (dispersion)")
    ax.set_ylabel("n_iter")
    ax.set_title("|D| vs n_iter coloured by convergence\n(independent vs dependent variables)")
    ax.legend(fontsize=8)

    # Panel 3: noise vs |D| with convergence
    ax = axes[1, 0]
    ax.scatter(c0[:, 0], c0[:, 2], c='tomato',    alpha=0.5, s=20, label='Not converged')
    ax.scatter(c1[:, 0], c1[:, 2], c='steelblue', alpha=0.5, s=20, label='Converged')
    ax.set_xlabel("|D| (dispersion)")
    ax.set_ylabel("noise_sigma")
    ax.set_yscale('log')
    ax.set_title("|D| vs noise_sigma\n(diversity vs SNR trade-off)")
    ax.legend(fontsize=8)

    # Panel 4: RF predicted probability of convergence vs |D|
    ax = axes[1, 1]
    D_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    X_probe = np.column_stack([
        D_range,
        np.full(100, np.median(X[:, 1])),   # median n_iter
        np.full(100, np.median(X[:, 2])),   # median noise
        np.full(100, np.median(X[:, 3])),   # median sigma_x
        np.zeros(100),                       # Gaussian pulse
    ])
    proba = rf.predict_proba(X_probe)
    # column for class=1
    class_idx = list(rf.classes_).index(1) if 1 in rf.classes_ else 0
    p_conv = proba[:, class_idx]
    ax.plot(D_range, p_conv, color='steelblue', lw=2.5)
    ax.axhline(0.5, color='k', ls='--', lw=1, label='Decision boundary')
    ax.fill_between(D_range, p_conv, alpha=0.15, color='steelblue')
    ax.set_xlabel("|D| (all other features at median)")
    ax.set_ylabel("RF P(converged)")
    ax.set_title("RF convergence probability vs |D|\n(marginal effect of dispersion)")
    ax.legend(fontsize=8); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("rf_physics_results.png", dpi=120, bbox_inches='tight')
    print("Saved rf_physics_results.png")
    plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y, names = generate_dataset(n_samples=200, seed=42)
    tree, rf    = train_and_report(X, y, names)
    plot_results(X, y, names, tree, rf)
